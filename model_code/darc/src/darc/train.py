from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .losses import objective
from .metrics import date_rank_ic


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _batch_tensor(batch, device):
    return tuple(torch.as_tensor(v, device=device) for v in
                 (batch.x, batch.market, batch.y, batch.retrieval, batch.complementarity))


@torch.no_grad()
def validate(model, panel, cfg, device) -> float:
    model.eval()
    scores = []
    for batch in panel.iter_dates("core_valid", cfg.valid_dates_per_epoch, None):
        x, market, y, retrieval, comp = _batch_tensor(batch, device)
        good = torch.isfinite(y)
        if int(good.sum()) < 8:
            continue
        pred, _ = model(x, market, retrieval, comp)
        scores.append(date_rank_ic(pred[good].cpu().numpy(), y[good].cpu().numpy()))
    return float(np.nanmean(scores)) if scores else -np.inf


def fit_forecaster(model, panel, cfg, device, output: Path, label: str) -> list[dict]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs,
                                                           eta_min=cfg.learning_rate * 0.1)
    rng = np.random.default_rng(cfg.seed)
    history = []
    best_score, best_state, stale = -np.inf, None, 0
    started = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        dates = list(panel.iter_dates("fit", cfg.dates_per_epoch, rng))
        tick = max(1, len(dates) // cfg.progress_updates_per_epoch)
        running = {}
        trained_dates = 0
        for step, batch in enumerate(dates, 1):
            x, market, y, retrieval, comp = _batch_tensor(batch, device)
            good = torch.isfinite(y)
            if int(good.sum()) < 8:
                continue
            optimizer.zero_grad(set_to_none=True)
            pred, aux = model(x, market, retrieval, comp)
            scored_aux = {"gates": aux["gates"][good]} if "gates" in aux else {}
            loss, parts = objective(pred[good], y[good], scored_aux, cfg)
            if not torch.isfinite(loss):
                raise RuntimeError(f"{label}: nonfinite loss in epoch {epoch}, date {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            trained_dates += 1
            for key, value in parts.items():
                running[key] = running.get(key, 0.0) + float(value)
            if step == 1 or step % tick == 0 or step == len(dates):
                elapsed = time.time() - started
                done = ((epoch - 1) * len(dates) + step) / max(1, cfg.epochs * len(dates))
                print(f"[{label}] epoch {epoch:02d}/{cfg.epochs} date {step:03d}/{len(dates)} "
                      f"loss={running['total']/trained_dates:.4f} elapsed={elapsed/60:.1f}m "
                      f"ETA={(elapsed/done-elapsed)/60:.1f}m", flush=True)
        if trained_dates == 0:
            raise RuntimeError(f"{label}: no date with eight finite training labels")
        scheduler.step()
        score = validate(model, panel, cfg, device)
        history.append({"epoch": epoch, "validation_rank_ic": score,
                        **{f"train_{k}": value / trained_dates for k, value in running.items()}})
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        (output / "timing.json").write_text(json.dumps({"training_seconds_so_far": time.time() - started}, indent=2) + "\n",
                                            encoding="utf-8")
        print(f"[{label}] epoch {epoch:02d} validation RankIC={score:.5f}", flush=True)
        if np.isfinite(score) and score > best_score + 1e-5:
            best_score, stale = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"state_dict": best_state, "best_validation_rank_ic": best_score}, output / "model.pt")
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    if best_state is None:
        raise RuntimeError(f"{label}: no finite checkpoint")
    model.load_state_dict(best_state)
    (output / "timing.json").write_text(json.dumps({"training_seconds": time.time() - started}, indent=2) + "\n",
                                        encoding="utf-8")
    return history
