from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def evaluate_records(records: list[dict], stress_threshold: float) -> tuple[dict, pd.DataFrame]:
    """Equal-weight origin-date scoring; predictions never select test assets."""
    rows = []
    for item in records:
        y = np.asarray(item["target"], dtype=np.float64)
        pred = np.asarray(item["forecast"], dtype=np.float64)
        good = np.isfinite(y) & np.isfinite(pred) & (pred > 0)
        if good.sum() < 8:
            continue
        y, pred = y[good], pred[good]
        score = np.log(pred) + y / pred
        rows.append({"date": item["date"], "date_index": item["date_index"],
                     "n_scored": int(good.sum()), "qlike": float(score.mean()),
                     "mse": float(np.square(pred - y).mean()),
                     "mae": float(np.abs(pred - y).mean()),
                     "mean_forecast": float(pred.mean()), "mean_target": float(y.mean()),
                     "stress": float(item["stress"]),
                     "regime": "high" if item["stress"] >= stress_threshold else "low"})
    daily = pd.DataFrame(rows)
    if daily.empty:
        raise RuntimeError("No scored origin dates")
    summary = {name: float(daily[name].mean()) for name in
               ("qlike", "mse", "mae", "mean_forecast", "mean_target")}
    summary["n_dates"] = int(len(daily))
    summary["n_asset_forecasts_scored"] = int(daily["n_scored"].sum())
    summary["high_stress_qlike"] = float(daily.loc[daily.regime == "high", "qlike"].mean())
    summary["low_stress_qlike"] = float(daily.loc[daily.regime == "low", "qlike"].mean())
    return summary, daily


def paired_block_bootstrap(full: pd.DataFrame, other: pd.DataFrame, draws: int,
                           seed: int, block_length: int) -> dict:
    paired = full[["date", "qlike"]].merge(other[["date", "qlike"]], on="date",
                                         suffixes=("_full", "_other")).dropna()
    improvement = (paired.qlike_other - paired.qlike_full).to_numpy()
    n = len(improvement)
    if n < 10:
        return {"n_common_dates": n, "mean_improvement": np.nan,
                "ci_low": np.nan, "ci_high": np.nan, "p_one_sided": np.nan}
    block = min(block_length, max(2, n // 8))
    rng = np.random.default_rng(seed)
    boot = np.empty(draws)
    for j in range(draws):
        starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
        boot[j] = np.concatenate([improvement[s:s + block] for s in starts])[:n].mean()
    centered = improvement - improvement.mean()
    hac = float(centered @ centered / n)
    for lag in range(1, min(block, n - 1) + 1):
        hac += 2.0 * (1.0 - lag / (block + 1.0)) * float(centered[lag:] @ centered[:-lag] / n)
    se = float(np.sqrt(max(hac, 0.0) / n))
    p = float(norm.sf(improvement.mean() / se)) if se > 0 else np.nan
    return {"n_common_dates": n, "mean_improvement": float(improvement.mean()),
            "ci_low": float(np.quantile(boot, 0.025)),
            "ci_high": float(np.quantile(boot, 0.975)),
            "p_one_sided": p, "hac_standard_error": se, "block_length": int(block)}


def holm(frame: pd.DataFrame, p_col: str = "p_one_sided") -> pd.DataFrame:
    out = frame.copy()
    out["p_holm"] = np.nan
    ordered = out[p_col].dropna().sort_values().index
    running = 0.0
    for rank, idx in enumerate(ordered):
        running = max(running, min(1.0, float(out.loc[idx, p_col]) * (len(ordered) - rank)))
        out.loc[idx, "p_holm"] = running
    return out
