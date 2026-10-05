from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
import torch


GLM_COLUMNS = {
    "har": (0, 1, 2, 3),
    "shar": (0, 1, 2, 3, 4, 5, 6, 7),
    "market_shar": (0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 15, 16),
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def qlike(log_mu: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    z = log_mu.clamp(-8.0, 8.0)
    return z + y * torch.exp(-z)


def tensors(batch, device):
    return tuple(torch.as_tensor(value, device=device, dtype=torch.float32) for value in
                 (batch.x, batch.seq, batch.anchor, batch.own, batch.market, batch.y))


@torch.no_grad()
def validate(model, panel, cfg, device) -> float:
    model.eval()
    scores = []
    for batch in panel.iter_dates("core_valid", cfg.valid_dates_per_epoch):
        x, seq, anchor, own, market, y = tensors(batch, device)
        good = torch.isfinite(y)
        if int(good.sum()) < 8:
            continue
        log_mu, _ = model(x, seq, anchor, own, market)
        scores.append(float(qlike(log_mu[good], y[good]).mean().cpu()))
    return float(np.mean(scores)) if scores else np.inf


def fit_model(model, panel, cfg, device, out: Path, label: str) -> list[dict]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                                  weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.learning_rate * 0.1)
    rng = np.random.default_rng(cfg.seed)
    best, best_state, stale = np.inf, None, 0
    history = []
    started = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        dates = list(panel.iter_dates("fit", cfg.dates_per_epoch, rng))
        tick = max(1, len(dates) // cfg.progress_updates_per_epoch)
        total, trained = 0.0, 0
        for step, batch in enumerate(dates, 1):
            x, seq, anchor, own, market, y = tensors(batch, device)
            good = torch.isfinite(y)
            if int(good.sum()) >= 8:
                optimizer.zero_grad(set_to_none=True)
                log_mu, _ = model(x, seq, anchor, own, market)
                loss = qlike(log_mu[good], y[good]).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError(f"{label}: nonfinite objective, epoch {epoch}, date {step}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                total += float(loss.detach().cpu())
                trained += 1
            if step == 1 or step % tick == 0 or step == len(dates):
                elapsed = time.time() - started
                fraction = ((epoch - 1) * len(dates) + step) / max(cfg.epochs * len(dates), 1)
                eta = max(0.0, elapsed / max(fraction, 1e-6) - elapsed)
                score_text = f"{total/trained:.5f}" if trained else "waiting"
                print(f"[{label}] epoch {epoch:02d}/{cfg.epochs} date {step:03d}/{len(dates)} "
                      f"QLIKE={score_text} elapsed={elapsed/60:.1f}m ETA={eta/60:.1f}m", flush=True)
        if trained == 0:
            raise RuntimeError(f"{label}: no finite target dates in fit split")
        scheduler.step()
        score = validate(model, panel, cfg, device)
        history.append({"epoch": epoch, "train_qlike": total / trained,
                        "checkpoint_qlike": score})
        (out / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        print(f"[{label}] epoch {epoch:02d} checkpoint QLIKE={score:.5f}", flush=True)
        if np.isfinite(score) and score < best - 1e-5:
            best, stale = score, 0
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
            torch.save({"state_dict": best_state, "checkpoint_qlike": best}, out / "model.pt")
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    if best_state is None:
        raise RuntimeError(f"{label}: no finite checkpoint")
    model.load_state_dict(best_state)
    (out / "timing.json").write_text(
        json.dumps({"training_seconds": time.time() - started}, indent=2) + "\n", encoding="utf-8")
    return history


def glm_design(batch, name: str) -> np.ndarray:
    values = batch.x[:, GLM_COLUMNS[name]].astype(np.float64)
    return np.column_stack((np.ones(len(values)), values))


class GLMControl:
    def __init__(self, name: str, coefficients: np.ndarray):
        self.name = name
        self.coefficients = coefficients

    def predict(self, batch) -> np.ndarray:
        return np.exp(np.clip(glm_design(batch, self.name) @ self.coefficients, -8.0, 8.0))


def fit_glm(name: str, panel, out: Path) -> GLMControl:
    matrix, targets, weights = [], [], []
    for batch in panel.iter_dates("fit"):
        good = np.isfinite(batch.y)
        if good.sum() < 8:
            continue
        x = glm_design(batch, name)[good]
        matrix.append(x)
        targets.append(batch.y[good].astype(np.float64))
        weights.append(np.full(len(x), 1.0 / len(x), dtype=np.float64))
    if not matrix:
        raise RuntimeError(f"{name}: no fit rows")
    x = np.concatenate(matrix)
    y = np.concatenate(targets)
    w = np.concatenate(weights)
    w /= w.sum()

    def objective(beta):
        raw_z = x @ beta
        z = np.clip(raw_z, -8.0, 8.0)
        ratio = y * np.exp(-z)
        ridge = 1e-3 * float(beta[1:] @ beta[1:])
        loss = float(w @ (z + ratio)) + ridge
        grad = x.T @ (w * (1.0 - ratio) * ((raw_z > -8.0) & (raw_z < 8.0)))
        grad[1:] += 2e-3 * beta[1:]
        return loss, grad

    initial = np.zeros(x.shape[1], dtype=np.float64)
    initial[0] = float(np.log(max(np.average(y, weights=w), 1e-6)))
    result = minimize(objective, initial, jac=True, method="L-BFGS-B",
                      bounds=[(-8.0, 8.0)] + [(-4.0, 4.0)] * (len(initial) - 1),
                      options={"maxiter": 180, "ftol": 1e-9})
    if not np.isfinite(result.fun):
        raise RuntimeError(f"{name}: nonfinite GLM optimization")
    record = {"name": name, "objective": float(result.fun), "converged": bool(result.success),
              "message": str(result.message), "iterations": int(result.nit),
              "coefficients": result.x.tolist(), "fit_rows": int(len(y))}
    (out / "fit.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"[{panel.spec.name}/{name}] GLM fit: {result.message}; rows={len(y)}", flush=True)
    if not result.success:
        print(f"WARNING [{panel.spec.name}/{name}]: GLM convergence flag is false; "
              "retain and inspect fit.json when interpreting this control", flush=True)
    return GLMControl(name, result.x)


def deterministic_prediction(name: str, batch, fitted: GLMControl | None) -> np.ndarray:
    if name == "persistence":
        return np.maximum(batch.anchor[:, 1], 1e-6)
    if name == "ewma":
        age = np.arange(batch.seq.shape[1] - 1, -1, -1)
        weights = np.exp(-0.1 * age)
        weights /= weights.sum()
        return np.maximum((batch.seq[:, :, 0] ** 2) @ weights, 1e-6)
    if fitted is None:
        raise ValueError(f"Missing fit for {name}")
    return fitted.predict(batch)
