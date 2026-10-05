from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class GRUBaseline(nn.Module):
    def __init__(self, input_size: int, hidden: int, **_: object):
        super().__init__()
        self.encoder = nn.GRU(input_size, hidden, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def forward(self, x, market, retrieval=None, complementarity=None, excluded=None):
        h = self.encoder(x)[1][-1]
        return self.head(h).squeeze(-1), {"hidden": h}


class StockMixer(nn.Module):
    """Common-protocol StockMixer-inspired temporal and channel mixing network."""

    def __init__(self, input_size: int, hidden: int, lookback: int, **_: object):
        super().__init__()
        self.temporal = nn.Sequential(nn.Linear(lookback, hidden), nn.GELU(), nn.Linear(hidden, lookback))
        self.channel = nn.Sequential(nn.Linear(input_size, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.stock_mix = nn.MultiheadAttention(hidden, 4, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, market, retrieval=None, complementarity=None, excluded=None):
        z = x + self.temporal(x.transpose(1, 2)).transpose(1, 2)
        h = self.channel(z).mean(dim=1)
        cross, _ = self.stock_mix(h.unsqueeze(0), h.unsqueeze(0), h.unsqueeze(0), need_weights=False)
        h = self.norm(h + cross.squeeze(0))
        return self.head(h).squeeze(-1), {"hidden": h}


class MASTER(nn.Module):
    """Market-guided temporal/cross-sectional attention adaptation."""

    def __init__(self, input_size: int, hidden: int, market_size: int, lookback: int, **_: object):
        super().__init__()
        self.proj = nn.Linear(input_size, hidden)
        self.pos = nn.Parameter(torch.zeros(1, lookback, hidden))
        layer = nn.TransformerEncoderLayer(hidden, 4, hidden * 2, dropout=0.1, batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, 2)
        self.cross = nn.MultiheadAttention(hidden, 4, batch_first=True)
        self.market_gate = nn.Sequential(nn.Linear(market_size, hidden), nn.Sigmoid())
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def forward(self, x, market, retrieval=None, complementarity=None, excluded=None):
        h = self.temporal(self.proj(x) + self.pos[:, :x.shape[1]])[:, -1]
        cross, _ = self.cross(h.unsqueeze(0), h.unsqueeze(0), h.unsqueeze(0), need_weights=False)
        h = h + cross.squeeze(0) * self.market_gate(market)
        return self.head(h).squeeze(-1), {"hidden": h}


class ExpertAttentionModel(nn.Module):
    def __init__(self, input_size: int, hidden: int, market_size: int, num_experts: int, symmetric: bool, **_: object):
        super().__init__()
        self.num_experts, self.symmetric = num_experts, symmetric
        self.router_encoder = nn.GRU(input_size, hidden, batch_first=True)
        self.router_cross = nn.MultiheadAttention(hidden, 4, batch_first=True)
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(hidden, hidden),nn.GELU()) for _ in range(num_experts)])
        self.attn = nn.MultiheadAttention(hidden, 4, batch_first=True)
        self.proposals = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(num_experts)])
        self.gate = nn.Sequential(nn.Linear(hidden + market_size, hidden), nn.GELU(), nn.Linear(hidden, num_experts))

    def forward(self, x, market, retrieval=None, complementarity=None, excluded=None):
        router_state = self.router_encoder(x)[1][-1]
        cross,_ = self.router_cross(router_state[None],router_state[None],router_state[None],need_weights=False)
        router_state = router_state + cross.squeeze(0)
        hs = torch.stack([expert(router_state) for expert in self.experts],dim=1)
        grouped = hs.reshape(x.shape[0]*2,2,hs.shape[-1])
        mixed,weights = self.attn(grouped,grouped,grouped,need_weights=True)
        hs = hs + mixed.reshape_as(hs)
        proposals = torch.stack([self.proposals[i](hs[:, i]).squeeze(-1) for i in range(self.num_experts)], dim=1)
        logits = self.gate(torch.cat([router_state,market],dim=-1))
        selected=logits.topk(min(2,self.num_experts),dim=1).indices
        sparse=torch.full_like(logits,-1e9).scatter(1,selected,0.0)
        gate = torch.softmax(logits+sparse, dim=-1)
        pred = (gate * proposals).sum(-1)
        return pred, {"hidden": hs, "gates": gate, "proposals": proposals, "graph": weights}


class MERA(nn.Module):
    """Retrieval-augmented sparse MoE adaptation of the supplied MERA concept."""

    def __init__(self, input_size: int, hidden: int, retrieval_size: int, num_experts: int, **_: object):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([nn.GRU(input_size, hidden, batch_first=True) for _ in range(num_experts)])
        self.retrieval = nn.Sequential(nn.Linear(retrieval_size, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.gate = nn.Linear(hidden * 2, num_experts)
        self.heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(num_experts)])

    def forward(self, x, market, retrieval=None, complementarity=None, excluded=None):
        hs = torch.stack([expert(x)[1][-1] for expert in self.experts], dim=1)
        r = self.retrieval(retrieval)
        logits = self.gate(torch.cat([hs.mean(1), r], dim=-1))
        top = logits.topk(k=min(2, self.num_experts), dim=-1).indices
        mask = torch.full_like(logits, float("-inf")).scatter(1, top, 0.0)
        gates = torch.softmax(logits + mask, dim=-1)
        proposals = torch.stack([self.heads[i](hs[:, i]).squeeze(-1) for i in range(self.num_experts)], dim=1)
        return (gates * proposals).sum(-1), {"hidden": hs, "gates": gates, "proposals": proposals}


class MAFS(nn.Module):
    """Matched fixed-topology multi-agent forecasting control inspired by MAFS."""

    def __init__(self, input_size: int, hidden: int, num_experts: int, **_: object):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([nn.GRU(input_size, hidden, batch_first=True) for _ in range(num_experts)])
        role_mask = torch.zeros(num_experts, input_size)
        role_features = ((0, 1, 4, 7), (1, 2, 3, 6), (0, 1, 2, 4), (5, 8, 9, 10, 11))
        for role, indices in enumerate(role_features):
            role_mask[role, list(indices)] = 1.0
        self.register_buffer("role_mask", role_mask)
        self.message = nn.Linear(hidden, hidden)
        self.update = nn.GRUCell(hidden, hidden)
        self.vote = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(num_experts)])
        self.edge_logits = nn.Parameter(torch.zeros(num_experts, num_experts))
        ring = torch.zeros(num_experts, num_experts)
        for i in range(num_experts):
            ring[i, (i + 1) % num_experts] = 1.0
            ring[(i + 1) % num_experts, i] = 1.0
        self.register_buffer("ring", ring)

    def forward(self, x, market, retrieval=None, complementarity=None, excluded=None):
        hs = torch.stack([self.experts[i](x * self.role_mask[i].view(1, 1, -1))[1][-1]
                          for i in range(self.num_experts)], dim=1)
        learned = torch.sigmoid((self.edge_logits + self.edge_logits.T) / 2.0) * self.ring
        graph = learned / learned.sum(0, keepdim=True).clamp_min(1e-6)
        messages = torch.einsum("se,nsh->neh", graph, self.message(hs))
        updated = self.update(messages.reshape(-1, hs.shape[-1]), hs.reshape(-1, hs.shape[-1])).view_as(hs)
        proposals = torch.stack([self.heads[i](updated[:, i]).squeeze(-1) for i in range(self.num_experts)], dim=1)
        gates = torch.softmax(self.vote(updated).squeeze(-1), dim=-1)
        return (gates * proposals).sum(-1), {"hidden": updated, "gates": gates, "proposals": proposals, "graph": graph}


def build_baseline(name: str, **kwargs) -> nn.Module:
    table = {
        "gru": GRUBaseline,
        "stockmixer": StockMixer,
        "master": MASTER,
        "miga": lambda **k: ExpertAttentionModel(symmetric=False, **k),
        "mera": MERA,
        "mafs": lambda **k: MAFS(**k),
    }
    if name not in table:
        raise KeyError(f"Unknown baseline: {name}")
    return table[name](**kwargs)
