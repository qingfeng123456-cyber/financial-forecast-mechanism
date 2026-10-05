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


def leave_one_out_rank_credit(full: torch.Tensor, alternatives: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    base = pairwise_rank_loss(full, target)
    losses = torch.stack([pairwise_rank_loss(alternatives[:, i], target) for i in range(alternatives.shape[1])])
    return (losses - base).detach()


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
    private = aux.get("private_proposals")
    if private is not None:
        expert_losses = torch.stack([F.smooth_l1_loss(private[:, e], target) + cfg.rank_weight * pairwise_rank_loss(private[:, e], target)
                                     for e in range(private.shape[1])])
        total = total + cfg.expert_weight * expert_losses.mean()
        parts["expert"] = expert_losses.mean().detach()
        variance = aux["log_variance"]
        squared = (private - target[:, None]).detach().square()
        confidence = 0.5 * (torch.exp(-variance) * squared + variance).mean()
        total = total + cfg.confidence_weight * confidence
        parts["confidence"] = confidence.detach()
    delta = aux.get("message_delta")
    if delta is not None:
        message_penalty = delta.square().mean()
        total = total + cfg.message_weight * message_penalty
        parts["message_energy"] = message_penalty.detach()
    parts["total"] = total.detach()
    return total, parts
