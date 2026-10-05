from __future__ import annotations

import torch
from torch import nn


class C3RMoE(nn.Module):
    """Recipient innovation exchange and held-out communication-aware coalition risk."""

    def __init__(self, input_size: int, hidden: int, market_size: int, num_experts: int = 4,
                 graph_topk: int = 2, collaboration: bool = True, credit: bool = True,
                 router_temperature: float = 0.7, innovation_exchange: bool = True,
                 router_shrinkage: float = 0.10, router_hidden_size: int = 16, **_: object):
        super().__init__()
        if num_experts != 4:
            raise ValueError("The registered design requires four role experts")
        self.num_experts, self.hidden = num_experts, hidden
        self.graph_topk, self.collaboration, self.credit = graph_topk, collaboration, credit
        self.router_temperature = router_temperature
        self.innovation_exchange = innovation_exchange
        self.router_shrinkage = router_shrinkage
        mask = torch.zeros(num_experts, input_size)
        for role, indices in enumerate(((0, 1, 4, 5, 7, 9), (1, 2, 3, 6, 8, 10),
                                        (0, 1, 2, 4, 6, 7), (5, 7, 8, 9, 10, 11))):
            mask[role, list(indices)] = 1.0
        self.register_buffer("role_mask", mask)
        self.experts = nn.ModuleList([nn.GRU(input_size, hidden, batch_first=True) for _ in range(num_experts)])
        self.temporal_attention = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(num_experts)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(num_experts)])
        self.pre_heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(num_experts)])
        self.conf_heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(num_experts)])
        self.edge_net = nn.Sequential(nn.Linear(hidden * 2 + market_size + 4, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.null_head = nn.Linear(hidden + market_size, 1)
        self.message = nn.Linear(hidden, hidden, bias=False)
        self.receiver_space = nn.Linear(hidden, hidden, bias=False)
        self.correction = nn.Sequential(nn.Linear(hidden, hidden, bias=False), nn.GELU(), nn.Linear(hidden, hidden, bias=False))
        self.update_gate = nn.Linear(hidden * 2 + market_size, 1)
        self.delta_heads = nn.ModuleList([nn.Linear(hidden, 1, bias=False) for _ in range(num_experts)])
        self.local_gate = nn.Sequential(nn.Linear(hidden + market_size, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.complementarity_scale = nn.Parameter(torch.tensor(0.1))
        # Fifteen nonempty subsets plus the complete, noncommunicating reference.
        coalition_masks = [[float(bits & (1 << i) != 0) for i in range(4)] for bits in range(1, 16)]
        self.register_buffer("coalition_masks", torch.tensor(coalition_masks))
        self.register_buffer("router_enabled", torch.tensor(False))
        self.register_buffer("static_route_prior", torch.full((16,),1.0/16))
        context_size = hidden * num_experts * 2 + market_size + 16 + num_experts * 3
        self.risk_head = nn.Sequential(nn.LayerNorm(context_size), nn.Linear(context_size, router_hidden_size),
                                       nn.GELU(), nn.Dropout(0.1), nn.Linear(router_hidden_size, 16))
        # Start from competent private predictions and weak corrections.
        nn.init.constant_(self.update_gate.bias, -2.0)
        nn.init.constant_(self.null_head.bias, 1.0)
        for head in self.delta_heads: nn.init.zeros_(head.weight)

    def _encode(self, x):
        states = []
        for i in range(self.num_experts):
            sequence, _ = self.experts[i](x * self.role_mask[i].view(1, 1, -1))
            weights = torch.softmax(self.temporal_attention[i](sequence), dim=1)
            pooled = (sequence * weights).sum(1)
            states.append(self.norms[i](0.5 * (pooled + sequence[:, -1])))
        return torch.stack(states, dim=1)

    def _private(self, hs):
        proposals = torch.stack([self.pre_heads[i](hs[:, i]).squeeze(-1) for i in range(4)], dim=1)
        log_variance = torch.stack([self.conf_heads[i](hs[:, i]).squeeze(-1) for i in range(4)], dim=1).clamp(-4.0, 4.0)
        return proposals, log_variance

    def _graph(self, hs, proposals, log_variance, market, complementarity, active, mode="learned"):
        n, e, _ = hs.shape
        src, dst = hs[:, :, None].expand(n, e, e, -1), hs[:, None, :].expand(n, e, e, -1)
        psrc, pdst = proposals[:, :, None].expand(n, e, e), proposals[:, None, :].expand(n, e, e)
        csrc, cdst = log_variance[:, :, None].expand(n, e, e), log_variance[:, None, :].expand(n, e, e)
        context = market[:, None, None].expand(n, e, e, -1)
        pair = torch.cat([src, dst, (psrc - pdst).unsqueeze(-1), (psrc - pdst).abs().unsqueeze(-1),
                          csrc.unsqueeze(-1), cdst.unsqueeze(-1), context], dim=-1)
        logits = self.edge_net(pair).squeeze(-1) + self.complementarity_scale * complementarity.view(1, e, e)
        permitted = active[:, None] & active[None, :] & ~torch.eye(e, dtype=torch.bool, device=hs.device)
        logits = logits.masked_fill(~permitted.unsqueeze(0), -1e9)
        if mode == "uniform":
            logits = torch.zeros_like(logits).masked_fill(~permitted.unsqueeze(0), -1e9)
        else:
            k = min(self.graph_topk, max(1, int(active.sum().item()) - 1))
            keep = torch.zeros_like(logits, dtype=torch.bool).scatter(1, logits.topk(k, dim=1).indices, True)
            logits = logits.masked_fill(~keep, -1e9)
        null_logits = self.null_head(torch.cat([hs, market[:, None].expand(n, e, -1)], dim=-1)).squeeze(-1)
        # A finite null alternative makes singleton/excluded receiver paths well-defined.
        probabilities = torch.softmax(torch.cat([logits, null_logits[:, None]], dim=1), dim=1)
        graph = probabilities[:, :e] * permitted.unsqueeze(0)
        null_mass = probabilities[:, e]
        null_mass = torch.where(active.view(1, e), null_mass, torch.ones_like(null_mass))
        return graph, null_mass

    def _coalition(self, hs, market, complementarity, active, communicate=True, mode="learned"):
        proposals, log_variance = self._private(hs)
        graph = hs.new_zeros((hs.shape[0], 4, 4))
        null_mass = hs.new_ones((hs.shape[0], 4))
        updated, delta = hs, torch.zeros_like(hs)
        if self.collaboration and communicate:
            graph, null_mass = self._graph(hs, proposals, log_variance, market, complementarity, active, mode)
            source = self.message(hs)[:, :, None].expand(-1, 4, 4, -1)
            receiver = self.receiver_space(hs)[:, None].expand(-1, 4, 4, -1)
            coefficient = (source * receiver).sum(-1, keepdim=True) / receiver.square().sum(-1, keepdim=True).clamp_min(1e-6)
            # Euclidean innovation in a shared latent space; no financial causality claim.
            innovation = source if mode == "raw_messages" or not self.innovation_exchange else source - coefficient * receiver
            message = (graph.unsqueeze(-1) * innovation).sum(1)
            gate = torch.sigmoid(self.update_gate(torch.cat([hs, message, market[:, None].expand(-1, 4, -1)], -1)))
            correction = torch.tanh(self.correction(message))
            delta = 0.5 * gate * correction * active.view(1, 4, 1)
            if mode == "no_messages": delta = torch.zeros_like(delta)
            updated = hs + delta
        post = proposals + torch.stack([self.delta_heads[i](delta[:, i]).squeeze(-1) for i in range(4)], 1)
        local = torch.stack([self.local_gate(torch.cat([updated[:, i], market], -1)).squeeze(-1) for i in range(4)], 1)
        local = local.masked_fill(~active.view(1, 4), -1e9)
        gates = torch.softmax(local, -1)
        return (gates * post).sum(-1), {"hidden": updated, "independent_hidden": hs, "gates": gates,
                                       "proposals": post, "private_proposals": proposals, "log_variance": log_variance,
                                       "graph": graph, "null_mass": null_mass, "message_delta": delta}

    def _from_encoded(self, hs, market, complementarity, excluded=None, output_only=False):
        active = torch.ones(4, dtype=torch.bool, device=hs.device)
        if excluded is not None: active[excluded] = False
        if not output_only:
            return self._coalition(hs, market, complementarity, active)
        _, aux = self._coalition(hs, market, complementarity, torch.ones_like(active))
        weights = aux["gates"] * active.view(1, 4)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        return (weights * aux["proposals"]).sum(-1), aux

    def routing_context(self, hs, market, complementarity):
        private, variance = self._private(hs)
        return torch.cat([hs.mean(0).flatten(), hs.std(0, unbiased=False).flatten(), market.mean(0),
                          complementarity.flatten(), private.mean(0), private.std(0, unbiased=False), variance.mean(0)])

    def coalition_bank(self, hs, market, complementarity, mode="learned"):
        if mode == "output_only":
            _, full = self._coalition(hs, market, complementarity, self.coalition_masks[-1].bool())
            bank=[]
            for mask in self.coalition_masks:
                weights=full["gates"]*mask.view(1,4)
                weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-8)
                bank.append((weights*full["proposals"]).sum(-1))
            independent,_=self._coalition(hs,market,complementarity,self.coalition_masks[-1].bool(),communicate=False)
            return torch.stack([*bank,independent],1)
        bank = [self._coalition(hs, market, complementarity, mask.bool(), mode=mode)[0]
                for mask in self.coalition_masks]
        independent, _ = self._coalition(hs, market, complementarity, self.coalition_masks[-1].bool(), communicate=False)
        return torch.stack([*bank, independent], dim=1)

    def route_cached(self, context, bank):
        predicted_risk = self.risk_head(context)
        weights = torch.softmax(-predicted_risk / self.router_temperature, dim=-1)
        weights = (1.0-self.router_shrinkage)*weights+self.router_shrinkage*self.static_route_prior
        return (bank * weights.view(1, -1)).sum(-1), predicted_risk, weights

    def forward(self, x, market, retrieval=None, complementarity=None, excluded=None, output_only=False):
        hs = self._encode(x)
        core, aux = self._from_encoded(hs, market, complementarity, excluded, output_only)
        if bool(self.router_enabled.item()) and excluded is None:
            bank = self.coalition_bank(hs, market, complementarity)
            context = self.routing_context(hs, market, complementarity)
            prediction, risk, weights = self.route_cached(context, bank)
            aux.update(route_predictions=bank, route_weights=weights, predicted_risk=risk)
            # Role inclusion, rather than misleadingly calling a gate a contribution.
            inclusion = torch.cat([self.coalition_masks, torch.ones_like(self.coalition_masks[:1])], 0)
            aux["credit_logits"] = (weights[:, None] * inclusion).sum(0)
            return prediction, aux
        aux["credit_logits"] = aux["gates"].mean(0)
        return core, aux

    @torch.no_grad()
    def coalition_credit(self, x, market, complementarity):
        hs = self._encode(x)
        full, _ = self._from_encoded(hs, market, complementarity)
        removed, output_removed = [], []
        for e in range(4):
            removed.append(self._from_encoded(hs, market, complementarity, e, False)[0])
            output_removed.append(self._from_encoded(hs, market, complementarity, e, True)[0])
        return full, torch.stack(removed, 1), torch.stack(output_removed, 1)


def build_c3r(name: str, **kwargs):
    variants = {"c3r_full": (True, True), "c3r_graph_only": (True, False),
                "c3r_credit_only": (False, True), "c3r_plain_moe": (False, False),
                "c3r_unprojected": (True, True)}
    collaboration, credit = variants[name]
    return C3RMoE(collaboration=collaboration, credit=credit, innovation_exchange=name != "c3r_unprojected", **kwargs)
