from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.stats import norm


def date_rank_ic(pred: np.ndarray, target: np.ndarray) -> float:
    if len(pred) < 3 or np.nanstd(pred) < 1e-12 or np.nanstd(target) < 1e-12:
        return np.nan
    return float(spearmanr(pred, target).statistic)


def evaluate_records(records: list[dict], cost_bps: float, horizon: int,
                     forecast_error_available: bool = True) -> tuple[dict, pd.DataFrame]:
    rows, previous = [], {}
    strategy_returns, strategy_turnover = [], []
    unavailable_periods = 0
    last_trade_row = None
    next_origin = records[0]["date_index"] if records else 0
    for position, record in enumerate(records):
        pred, raw, assets = record["pred"], record["raw_return"], record["assets"]
        gross = net = turnover = np.nan
        # One disjoint sleeve: signals spaced by the registered holding horizon.
        # This avoids compounding overlapping forward returns as if independently tradable.
        score_mask = np.isfinite(raw) & np.isfinite(record["target"]) & np.isfinite(pred)
        if record["date_index"] >= next_origin:
            next_origin = record["date_index"] + horizon
            order = np.argsort(pred)
            k = max(1, min(10, len(order) // 5))
            weights = {int(assets[i]): -0.5 / k for i in order[:k]}
            weights.update({int(assets[i]): 0.5 / k for i in order[-k:]})
            union = set(previous) | set(weights)
            turnover = sum(abs(weights.get(i, 0.0) - previous.get(i, 0.0)) for i in union)
            selected = np.asarray([weights.get(int(asset), 0.0) != 0.0 for asset in assets])
            if np.isfinite(raw[selected]).all():
                simple_returns = np.expm1(raw)
                gross = sum(weights[int(assets[i])] * float(simple_returns[i]) for i in np.flatnonzero(selected))
                net = gross - cost_bps * 1e-4 * turnover
                strategy_returns.append(net); strategy_turnover.append(turnover)
                if net > -1.0:
                    previous = {int(assets[i]): weights[int(assets[i])] * (1.0 + float(simple_returns[i])) / (1.0 + net)
                                for i in np.flatnonzero(selected)}
                else:
                    unavailable_periods += 1; previous = weights
                last_trade_row = position
            else:
                unavailable_periods += 1
                previous = weights
        scored_pred, scored_raw = pred[score_mask], raw[score_mask]
        ic = float(np.corrcoef(scored_pred, scored_raw)[0,1]) if len(scored_pred)>=3 and np.std(scored_pred)>1e-12 and np.std(scored_raw)>1e-12 else np.nan
        rows.append({"date": record["date"], "rank_ic": date_rank_ic(pred[score_mask], raw[score_mask]), "gross": gross,
                     "ic":ic,"net": net, "turnover": turnover, "n_assets": len(assets), "n_scored": int(score_mask.sum())})
    if last_trade_row is not None and unavailable_periods == 0:
        # Exit the final observed holding instead of omitting terminal liquidation.
        terminal_turnover=(1.0+strategy_returns[-1])*sum(abs(w) for w in previous.values())
        terminal_fee=cost_bps*1e-4*terminal_turnover
        strategy_returns[-1]-=terminal_fee; strategy_turnover[-1]+=terminal_turnover
        rows[last_trade_row]["net"]-=terminal_fee; rows[last_trade_row]["turnover"]+=terminal_turnover
    daily = pd.DataFrame(rows)
    rank_ic = float(daily["rank_ic"].mean())
    rank_std = float(daily["rank_ic"].std(ddof=1))
    net = np.asarray(strategy_returns, dtype=float)
    scale = 252.0 / float(horizon)
    ann_return = float(np.nanmean(net) * scale)
    ann_vol = float(np.nanstd(net, ddof=1) * np.sqrt(scale))
    wealth = np.concatenate([[1.0], np.cumprod(1.0 + np.nan_to_num(net))])
    drawdown = wealth / np.maximum.accumulate(wealth) - 1.0
    economic_available = unavailable_periods == 0 and len(net) > 1
    if not economic_available:
        ann_return = ann_vol = np.nan
    metrics = {
        "rank_ic": rank_ic,
        "rank_ic_ir": rank_ic / rank_std if rank_std > 0 else np.nan,
        "ic": float(daily["ic"].mean()),
        "ic_ir": float(daily["ic"].mean()/daily["ic"].std()) if daily["ic"].std()>0 else np.nan,
        "mse": float(np.mean(np.concatenate([((r["pred"] - r["target"])[np.isfinite(r["target"])]) ** 2 for r in records]))) if forecast_error_available else np.nan,
        "mae": float(np.mean(np.abs(np.concatenate([(r["pred"] - r["target"])[np.isfinite(r["target"])] for r in records])))) if forecast_error_available else np.nan,
        "forecast_error_available": bool(forecast_error_available),
        "annual_net_return": ann_return,
        "annual_volatility": ann_vol,
        "sharpe": ann_return / ann_vol if economic_available and ann_vol > 0 else np.nan,
        "max_drawdown": float(np.nanmin(drawdown)) if economic_available else np.nan,
        "mean_turnover": float(np.mean(strategy_turnover)) if economic_available else np.nan,
        "strategy_periods": int(len(net)),
        "economic_unavailable_periods": int(unavailable_periods),
        "economic_metrics_available": bool(economic_available),
        "test_dates": int(len(daily)),
        "scored_rankic_dates":int(daily["rank_ic"].notna().sum()),
    }
    return metrics, daily


def paired_block_bootstrap(full: pd.DataFrame, other: pd.DataFrame, draws: int, seed: int, block_length: int = 20) -> dict:
    merged = full[["date", "rank_ic"]].merge(other[["date", "rank_ic"]], on="date", suffixes=("_full", "_other")).dropna()
    diff = (merged["rank_ic_full"] - merged["rank_ic_other"]).to_numpy()
    if len(diff) < 10:
        return {"mean_difference": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_one_sided": np.nan}
    rng, block = np.random.default_rng(seed), min(block_length, max(2, len(diff) // 8))
    boot = np.empty(draws)
    for b in range(draws):
        starts = rng.integers(0, len(diff) - block + 1, size=int(np.ceil(len(diff) / block)))
        sample = np.concatenate([diff[s:s + block] for s in starts])[:len(diff)]
        boot[b] = sample.mean()
    centered = diff - diff.mean()
    hac = float(centered @ centered / len(diff))
    for lag in range(1,min(block,len(diff)-1)+1):
        hac += 2.0*(1.0-lag/(block+1.0))*float(centered[lag:] @ centered[:-lag]/len(diff))
    se = np.sqrt(max(hac,0.0)/len(diff))
    z = float(diff.mean()/se) if se>0 else np.nan
    return {"mean_difference": float(diff.mean()), "ci_low": float(np.quantile(boot, 0.025)),
            "ci_high": float(np.quantile(boot, 0.975)), "p_one_sided":float(norm.sf(z)) if np.isfinite(z) else np.nan,
            "hac_standard_error":float(se),"hac_z":z,"n_common_dates":int(len(diff)),"block_length":int(block)}
