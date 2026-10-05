from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


LEVELS = np.asarray([0.1, 0.5, 0.9], dtype=np.float64)


def _pinball(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    error = target[:, None] - pred
    return np.maximum(LEVELS * error, (LEVELS - 1.0) * error)


def evaluate_records(records: list[dict]) -> tuple[dict, pd.DataFrame]:
    rows = []
    for record in records:
        pred = np.asarray(record["pred"], dtype=np.float64)
        target = np.asarray(record["raw_return"], dtype=np.float64)
        good = np.isfinite(target) & np.isfinite(pred).all(axis=1)
        if good.sum() < 8:
            continue
        pred, target = pred[good], target[good]
        loss = _pinball(pred, target)
        rows.append({
            "date": record["date"], "date_index": int(record["date_index"]),
            "n_scored": int(good.sum()), "pinball": float(loss.mean()),
            "pinball_q10": float(loss[:, 0].mean()),
            "pinball_q50": float(loss[:, 1].mean()),
            "pinball_q90": float(loss[:, 2].mean()),
            "median_mae": float(np.abs(pred[:, 1] - target).mean()),
            "direction_accuracy": float(((pred[:, 1] > 0) == (target > 0)).mean()),
            "coverage80": float(((target >= pred[:, 0]) & (target <= pred[:, 2])).mean()),
            "q10_exceedance": float((target < pred[:, 0]).mean()),
            "q90_exceedance": float((target > pred[:, 2]).mean()),
            "interval_width": float((pred[:, 2] - pred[:, 0]).mean()),
        })
    daily = pd.DataFrame(rows)
    if daily.empty:
        raise ValueError("No scored decision dates")
    metrics = {name: float(daily[name].mean()) for name in (
        "pinball", "pinball_q10", "pinball_q50", "pinball_q90", "median_mae",
        "direction_accuracy", "coverage80", "q10_exceedance", "q90_exceedance",
        "interval_width")}
    metrics["n_dates"] = int(len(daily))
    metrics["n_scored_assets"] = int(daily["n_scored"].sum())
    metrics["coverage80_abs_error"] = abs(metrics["coverage80"] - 0.8)
    metrics["q10_exceedance_abs_error"] = abs(metrics["q10_exceedance"] - 0.1)
    return metrics, daily


def paired_block_bootstrap(full: pd.DataFrame, other: pd.DataFrame, draws: int,
                           seed: int, block_length: int = 20) -> dict:
    paired = full[["date", "pinball"]].merge(other[["date", "pinball"]], on="date",
                                            suffixes=("_full", "_other")).dropna()
    # Positive difference means the full model has lower loss.
    diff = (paired["pinball_other"] - paired["pinball_full"]).to_numpy()
    if len(diff) < 10:
        return {"mean_improvement": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "p_one_sided": np.nan, "n_common_dates": int(len(diff))}
    rng = np.random.default_rng(seed)
    block = min(block_length, max(2, len(diff) // 8))
    boot = np.empty(draws)
    for b in range(draws):
        starts = rng.integers(0, len(diff) - block + 1, size=int(np.ceil(len(diff) / block)))
        sample = np.concatenate([diff[s:s + block] for s in starts])[:len(diff)]
        boot[b] = sample.mean()
    centered = diff - diff.mean()
    hac = float(centered @ centered / len(diff))
    for lag in range(1, min(block, len(diff) - 1) + 1):
        hac += 2.0 * (1.0 - lag / (block + 1.0)) * float(centered[lag:] @ centered[:-lag] / len(diff))
    se = float(np.sqrt(max(hac, 0.0) / len(diff)))
    z = float(diff.mean() / se) if se > 0 else np.nan
    return {"mean_improvement": float(diff.mean()),
            "ci_low": float(np.quantile(boot, 0.025)),
            "ci_high": float(np.quantile(boot, 0.975)),
            "p_one_sided": float(norm.sf(z)) if np.isfinite(z) else np.nan,
            "hac_standard_error": se, "hac_z": z,
            "n_common_dates": int(len(diff)), "block_length": int(block)}
