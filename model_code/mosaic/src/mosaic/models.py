from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def log_anchor(anchor: torch.Tensor) -> torch.Tensor:
    return torch.log(anchor[:, 1].clamp_min(1e-6))


class FlatMLP(nn.Module):
    """Equal-information pooled MLP control."""

    def __init__(self, features: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(features, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, seq, anchor, own, market):
        correction = self.net(x).squeeze(-1)
        return log_anchor(anchor) + correction, {"correction": correction}


class GRUModel(nn.Module):
    """Daily-sequence neural control with the same flat state at the head."""

    def __init__(self, features: int, hidden: int):
        super().__init__()
        self.encoder = nn.GRU(4, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + features, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x, seq, anchor, own, market):
        h = self.encoder(seq)[1][-1]
        correction = self.head(torch.cat((h, x), dim=-1)).squeeze(-1)
        return log_anchor(anchor) + correction, {"correction": correction}


class RGResMoE(nn.Module):
    """Common-protocol regime-gated residual MoE adaptation, not author code."""

    def __init__(self, features: int, hidden: int):
        super().__init__()
        self.base = nn.Sequential(nn.Linear(features, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(features, hidden), nn.GELU(), nn.Linear(hidden, 1))
            for _ in range(3)
        ])
        self.gate = nn.Sequential(nn.Linear(6, hidden), nn.GELU(), nn.Linear(hidden, 3))
        self.residual_scale = nn.Parameter(torch.tensor(-3.5))
        nn.init.zeros_(self.base[-1].weight)
        nn.init.zeros_(self.base[-1].bias)
        for expert in self.experts:
            nn.init.zeros_(expert[-1].weight)
            nn.init.zeros_(expert[-1].bias)

    def forward(self, x, seq, anchor, own, market):
        gate = torch.softmax(self.gate(x[:, 11:17]), dim=-1)
        proposals = torch.cat([expert(x) for expert in self.experts], dim=-1)
        residual = (gate * proposals).sum(-1)
        correction = self.base(x).squeeze(-1) + torch.sigmoid(self.residual_scale) * residual
        return log_anchor(anchor) + correction, {"correction": correction, "gate": gate}


class Mosaic(nn.Module):
    """Positive HAR anchor with separately identified downside and market branches.

    The interaction exists only when both branches are registered. Branch outputs
    describe a predictive parameterization, not causal transmission estimates.
    """

    def __init__(self, hidden: int, use_own: bool, use_market: bool):
        super().__init__()
        self.use_own, self.use_market = use_own, use_market
        self.anchor_logits = nn.Parameter(torch.tensor([0.0, 1.5, 0.5, 0.0]))
        self.bias = nn.Parameter(torch.zeros(()))
        self.own_net = nn.Sequential(nn.Linear(3, hidden // 2), nn.SiLU(),
                                     nn.Linear(hidden // 2, 1)) if use_own else None
        self.market_net = nn.Sequential(nn.Linear(4, hidden // 2), nn.SiLU(),
                                        nn.Linear(hidden // 2, 1)) if use_market else None
        self.own_gain = nn.Parameter(torch.tensor(-3.5)) if use_own else None
        self.market_gain = nn.Parameter(torch.tensor(-3.5)) if use_market else None
        self.interaction_gain = nn.Parameter(torch.tensor(-4.0)) if use_own and use_market else None

    def forward(self, x, seq, anchor, own, market):
        weights = torch.softmax(self.anchor_logits, dim=0)
        base = torch.log((anchor.clamp_min(1e-6) * weights).sum(-1)) + self.bias
        own_response = torch.zeros_like(base)
        market_response = torch.zeros_like(base)
        interaction = torch.zeros_like(base)
        if self.use_own:
            own_signal = torch.tanh(own[:, 0] * F.softplus(self.own_net(own).squeeze(-1)))
            own_response = 2.0 * torch.sigmoid(self.own_gain) * own_signal
        if self.use_market:
            market_signal = torch.tanh(market[:, 0] * market[:, 2] *
                                       F.softplus(self.market_net(market).squeeze(-1)))
            market_response = 2.0 * torch.sigmoid(self.market_gain) * market_signal
        if self.use_own and self.use_market:
            interaction = 2.0 * torch.sigmoid(self.interaction_gain) * own_signal * market_signal
        return base + own_response + market_response + interaction, {
            "base": base, "own": own_response, "market": market_response,
            "interaction": interaction, "correction": own_response + market_response + interaction,
            "anchor_weights": weights,
        }


def build_model(name: str, features: int, hidden: int) -> nn.Module:
    if name == "flat_mlp":
        # Eight hidden units give a similar parameter budget to the full cell.
        return FlatMLP(features, max(8, hidden // 4))
    if name == "gru":
        return GRUModel(features, hidden)
    if name == "rg_resmoe":
        return RGResMoE(features, hidden)
    cells = {"mosaic_base": (False, False), "mosaic_own": (True, False),
             "mosaic_market": (False, True), "mosaic_full": (True, True)}
    if name in cells:
        return Mosaic(hidden, *cells[name])
    raise KeyError(name)
