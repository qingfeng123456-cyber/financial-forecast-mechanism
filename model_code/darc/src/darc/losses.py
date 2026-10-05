from __future__ import annotations

import torch
import torch.nn.functional as F


def pairwise_rank_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pdiff = pred[:, None] - pred[None, :]
    ydiff = target[:, None] - target[None, :]
    mask = ydiff > 1e-6
    if not mask.any():
        return pred.new_tensor(0.0)
    return F.softplus(-pdiff[mask]).mean()


def objective(pred, target, aux, cfg):
    huber = F.smooth_l1_loss(pred, target)
    rank = pairwise_rank_loss(pred, target)
    total = huber + cfg.rank_weight * rank
    parts = {"huber": huber.detach(), "rank": rank.detach()}
    gates = aux.get("gates")
    if gates is not None:
        usage = gates.mean(0)
        balance = ((usage - 1.0 / usage.numel()) ** 2).mean()
        total = total + cfg.balance_weight * balance
        parts["balance"] = balance.detach()
    parts["total"] = total.detach()
    return total, parts
