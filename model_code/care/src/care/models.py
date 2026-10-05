from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def components(raw: torch.Tensor, prior: torch.Tensor | None = None) -> torch.Tensor:
    center = raw[..., 0] if prior is None else prior[..., 1] + raw[..., 0]
    lower_prior = 0.0 if prior is None else (prior[..., 1] - prior[..., 0]).clamp_min(0)
    upper_prior = 0.0 if prior is None else (prior[..., 2] - prior[..., 1]).clamp_min(0)
    return torch.stack((center, F.softplus(lower_prior + raw[..., 1]),
                        F.softplus(upper_prior + raw[..., 2])), dim=-1)


def quantiles(parts: torch.Tensor) -> torch.Tensor:
    return torch.stack((parts[..., 0] - parts[..., 1], parts[..., 0],
                        parts[..., 0] + parts[..., 2]), dim=-1)


class GRUQuantile(nn.Module):
    def __init__(self, input_size: int, hidden: int, **_: object):
        super().__init__()
        self.encoder = nn.GRU(input_size, hidden, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 3))

    def forward(self, x, market, retrieval=None, complementarity=None):
        h = self.encoder(x)[1][-1]
        return quantiles(components(self.head(h))), {"hidden": h}


class StockMixerQuantile(nn.Module):
    """Common-protocol StockMixer-style quantile adaptation."""

    def __init__(self, input_size: int, hidden: int, lookback: int, **_: object):
        super().__init__()
        self.temporal = nn.Sequential(nn.Linear(lookback, hidden), nn.GELU(), nn.Linear(hidden, lookback))
        self.channel = nn.Sequential(nn.Linear(input_size, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.cross = nn.MultiheadAttention(hidden, 4, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 3))

    def forward(self, x, market, retrieval=None, complementarity=None):
        z = x + self.temporal(x.transpose(1, 2)).transpose(1, 2)
        h = self.channel(z).mean(dim=1)
        cross, _ = self.cross(h.unsqueeze(0), h.unsqueeze(0), h.unsqueeze(0), need_weights=False)
        h = h + cross.squeeze(0)
        return quantiles(components(self.head(h))), {"hidden": h}


class MasterQuantile(nn.Module):
    """Common-protocol market-guided temporal-attention quantile adaptation."""

    def __init__(self, input_size: int, hidden: int, market_size: int, lookback: int, **_: object):
        super().__init__()
        self.proj = nn.Linear(input_size, hidden)
        self.position = nn.Parameter(torch.zeros(1, lookback, hidden))
        layer = nn.TransformerEncoderLayer(hidden, 4, hidden * 2, dropout=0.1,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2)
        self.cross = nn.MultiheadAttention(hidden, 4, batch_first=True)
        self.market_gate = nn.Sequential(nn.Linear(market_size, hidden), nn.Sigmoid())
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 3))

    def forward(self, x, market, retrieval=None, complementarity=None):
        h = self.encoder(self.proj(x) + self.position[:, :x.shape[1]])[:, -1]
        cross, _ = self.cross(h.unsqueeze(0), h.unsqueeze(0), h.unsqueeze(0), need_weights=False)
        h = h + self.market_gate(market) * cross.squeeze(0)
        return quantiles(components(self.head(h))), {"hidden": h}


class MERAQuantile(nn.Module):
    """MERA-inspired historical-label sparse routing, adapted to three quantiles."""

    def __init__(self, input_size: int, hidden: int, num_experts: int,
                 retrieval_size: int, **_: object):
        super().__init__()
        self.channel_size = (retrieval_size - 9) // 2
        self.experts = nn.ModuleList([nn.GRU(input_size, hidden, batch_first=True)
                                      for _ in range(num_experts)])
        self.memory = nn.Sequential(nn.Linear(self.channel_size, hidden), nn.GELU())
        self.router = nn.Linear(hidden * 2, num_experts)
        self.heads = nn.ModuleList([nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 3))
                                    for _ in range(num_experts)])

    def forward(self, x, market, retrieval=None, complementarity=None):
        hidden = torch.stack([expert(x)[1][-1] for expert in self.experts], dim=1)
        memory = self.memory(retrieval[:, :self.channel_size])
        logits = self.router(torch.cat((hidden.mean(1), memory), dim=-1))
        chosen = logits.topk(k=min(2, logits.shape[-1]), dim=-1).indices
        mask = torch.full_like(logits, float("-inf")).scatter(1, chosen, 0.0)
        gate = torch.softmax(logits + mask, dim=-1)
        proposals = torch.stack([components(head(hidden[:, i]))
                                 for i, head in enumerate(self.heads)], dim=1)
        parts = torch.einsum("ne,neq->nq", gate, proposals)
        return quantiles(parts), {"gates": gate, "hidden": hidden}


class CARE(nn.Module):
    """Two matured analog distributions condition center and tail expert routes."""

    def __init__(self, input_size: int, hidden: int, retrieval_size: int,
                 market_size: int, num_experts: int, dual_memory: bool,
                 tail_quality_route: bool, **_: object):
        super().__init__()
        self.dual_memory, self.tail_quality_route = dual_memory, tail_quality_route
        self.channel_size = (retrieval_size - 9) // 2
        self.shared = nn.GRU(input_size, hidden, batch_first=True)
        self.shared_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 3))
        self.experts = nn.ModuleList([nn.GRU(input_size, hidden, batch_first=True)
                                      for _ in range(num_experts)])
        self.heads = nn.ModuleList([nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 3))
                                    for _ in range(num_experts)])
        mask = torch.zeros(num_experts, input_size)
        roles = ((0, 1, 2, 3, 6), (0, 4, 5, 7, 8), (1, 2, 9, 10, 11),
                 tuple(range(input_size)))
        for i, columns in enumerate(roles):
            mask[i, list(columns)] = 1.0
        self.register_buffer("role_mask", mask)
        self.market_memory = nn.Sequential(nn.Linear(self.channel_size, hidden), nn.GELU())
        self.stock_memory = nn.Sequential(nn.Linear(self.channel_size, hidden), nn.GELU())
        context_size = 3 * hidden + market_size + 9
        self.router = nn.Sequential(nn.Linear(context_size, hidden), nn.GELU(),
                                    nn.Linear(hidden, 3 * num_experts))
        self.adoption = nn.Sequential(nn.Linear(context_size, hidden), nn.GELU(),
                                      nn.Linear(hidden, 3))

    def forward(self, x, market, retrieval=None, complementarity=None):
        size = self.channel_size
        market_memory = retrieval[:, :size]
        stock_memory = retrieval[:, size:2 * size]
        quality = retrieval[:, 2 * size:]
        shared_h = self.shared(x)[1][-1]
        shared_parts = components(self.shared_head(shared_h))
        market_q = market_memory[:, -3:]
        stock_q = stock_memory[:, -3:] if self.dual_memory else market_q
        stock_embed = self.stock_memory(stock_memory) if self.dual_memory else torch.zeros_like(shared_h)
        route_quality = quality if self.tail_quality_route else torch.zeros_like(quality)
        if not self.dual_memory:
            route_quality = torch.cat((route_quality[:, :1], torch.zeros_like(route_quality[:, 1:2]),
                                       route_quality[:, 2:3], torch.zeros_like(route_quality[:, 3:4]),
                                       route_quality[:, 4:5], torch.zeros_like(route_quality[:, 5:])), dim=-1)
        context = torch.cat((shared_h, self.market_memory(market_memory), stock_embed,
                             market, route_quality), dim=-1)
        raw_gate = self.router(context).view(-1, 3, len(self.experts))
        raw_adoption = self.adoption(context)
        if not self.tail_quality_route:
            raw_gate = raw_gate.mean(dim=1, keepdim=True).expand_as(raw_gate)
            raw_adoption = raw_adoption.mean(dim=1, keepdim=True).expand_as(raw_adoption)
        chosen = raw_gate.topk(k=min(2, len(self.experts)), dim=-1).indices
        mask = torch.full_like(raw_gate, float("-inf")).scatter(2, chosen, 0.0)
        gates = torch.softmax(raw_gate + mask, dim=-1)
        hidden = torch.stack([expert(x * self.role_mask[i].view(1, 1, -1))[1][-1]
                              for i, expert in enumerate(self.experts)], dim=1)
        proposals = torch.stack([components(head(hidden[:, i]),
                                             market_q if i < 2 else stock_q)
                                 for i, head in enumerate(self.heads)], dim=1)
        expert_parts = torch.einsum("nqe,neq->nq", gates, proposals)
        adoption = torch.sigmoid(raw_adoption)
        final_parts = shared_parts * (1.0 - adoption) + expert_parts * adoption
        return quantiles(final_parts), {"gates": gates, "adoption": adoption,
                                        "shared_quantiles": quantiles(shared_parts),
                                        "expert_quantiles": quantiles(expert_parts),
                                        "quality": quality}


def build_model(name: str, **kwargs) -> nn.Module:
    baseline = {
        "gru_quantile": GRUQuantile,
        "stockmixer_quantile": StockMixerQuantile,
        "master_quantile": MasterQuantile,
        "mera_quantile": MERAQuantile,
    }
    cells = {
        "care_single_plain": (False, False),
        "care_dual_plain": (True, False),
        "care_single_tail": (False, True),
        "care_full": (True, True),
    }
    if name in baseline:
        return baseline[name](**kwargs)
    if name in cells:
        dual, tail = cells[name]
        return CARE(dual_memory=dual, tail_quality_route=tail, **kwargs)
    raise KeyError(f"Unknown trainable model: {name}")
