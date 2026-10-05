from __future__ import annotations

import torch


def pinball(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    levels = pred.new_tensor([0.1, 0.5, 0.9])
    error = target.unsqueeze(-1) - pred
    return torch.maximum(levels * error, (levels - 1.0) * error)


def objective(pred, target, aux, cfg):
    proper = pinball(pred, target).mean()
    total = proper
    parts = {"pinball": proper.detach()}
    gates = aux.get("gates")
    if gates is not None:
        usage = gates.mean(dim=tuple(range(gates.ndim - 1)))
        balance = ((usage - 1.0 / usage.numel()) ** 2).mean()
        total = total + cfg.balance_weight * balance
        parts["balance"] = balance.detach()
    parts["total"] = total.detach()
    return total, parts
