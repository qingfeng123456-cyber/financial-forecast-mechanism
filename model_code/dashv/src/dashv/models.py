from __future__ import annotations

import torch
from torch import nn


def log_anchor(anchor: torch.Tensor) -> torch.Tensor:
    return torch.log(anchor[:, 1].clamp_min(1e-6))


class _Head(nn.Module):
    def __init__(self, inputs: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inputs, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, encoded: torch.Tensor, x: torch.Tensor, anchor: torch.Tensor):
        correction = self.net(torch.cat((encoded, x), dim=-1)).squeeze(-1)
        return log_anchor(anchor) + correction, {"correction": correction}


class FlatMLP(nn.Module):
    """Equal-information pooled MLP control."""

    def __init__(self, features: int, hidden: int, lookback: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(features + 4 * lookback, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, seq, anchor, own, market):
        del own, market
        correction = self.net(torch.cat((x, seq.flatten(start_dim=1)), dim=-1)).squeeze(-1)
        return log_anchor(anchor) + correction, {"correction": correction}


class GRUModel(nn.Module):
    def __init__(self, features: int, hidden: int):
        super().__init__()
        self.encoder = nn.GRU(4, hidden, batch_first=True)
        self.head = _Head(hidden + features, hidden)

    def forward(self, x, seq, anchor, own, market):
        del own, market
        encoded = self.encoder(seq)[1][-1]
        return self.head(encoded, x, anchor)


class LSTMModel(nn.Module):
    def __init__(self, features: int, hidden: int):
        super().__init__()
        self.encoder = nn.LSTM(4, hidden, batch_first=True)
        self.head = _Head(hidden + features, hidden)

    def forward(self, x, seq, anchor, own, market):
        del own, market
        encoded = self.encoder(seq)[1][0][-1]
        return self.head(encoded, x, anchor)


class TCNModel(nn.Module):
    """Compact common-protocol temporal convolution control."""

    def __init__(self, features: int, hidden: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(4, hidden, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=2, dilation=2), nn.GELU(),
        )
        self.head = _Head(hidden + features, hidden)

    def forward(self, x, seq, anchor, own, market):
        del own, market
        encoded = self.encoder(seq.transpose(1, 2))[:, :, -1]
        return self.head(encoded, x, anchor)


class PatchTransformer(nn.Module):
    """Small patch-transformer adaptation sharing the registered information set."""

    def __init__(self, features: int, hidden: int, lookback: int, patch: int = 5):
        super().__init__()
        if lookback % patch:
            raise ValueError("lookback must be divisible by patch length")
        self.patch = patch
        self.n_patches = lookback // patch
        self.embed = nn.Linear(4 * patch, hidden)
        self.position = nn.Parameter(torch.zeros(1, self.n_patches, hidden))
        layer = nn.TransformerEncoderLayer(hidden, nhead=4, dim_feedforward=2 * hidden,
                                           dropout=0.1, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = _Head(hidden + features, hidden)

    def forward(self, x, seq, anchor, own, market):
        del own, market
        batch = seq.shape[0]
        patches = seq.reshape(batch, self.n_patches, self.patch * 4)
        encoded = self.encoder(self.embed(patches) + self.position).mean(dim=1)
        return self.head(encoded, x, anchor)


def build_model(name: str, features: int, hidden: int, lookback: int) -> nn.Module:
    if name == "flat_mlp":
        return FlatMLP(features, hidden, lookback)
    if name == "gru":
        return GRUModel(features, hidden)
    if name == "lstm":
        return LSTMModel(features, hidden)
    if name == "tcn":
        return TCNModel(features, hidden)
    if name == "patch_transformer":
        return PatchTransformer(features, hidden, lookback)
    raise KeyError(name)
