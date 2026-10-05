from __future__ import annotations

from collections import deque
import numpy as np
from scipy.stats import rankdata

from .metrics import date_rank_ic


READOUTS = ("equal_rank", "static_weighted", "recent_performance", "uniform_pair_vote",
            "global_always", "pairwise_always", "global_gated", "darc_full")


def _observed_rank_ic(pred: np.ndarray, target: np.ndarray) -> float:
    observed = np.isfinite(pred) & np.isfinite(target)
    return date_rank_ic(pred[observed], target[observed])


def _rank01(x: np.ndarray) -> np.ndarray:
    return (rankdata(x, method="average") - 1.0) / max(1, len(x) - 1)


def _softmax(x: np.ndarray, temperature: float) -> np.ndarray:
    z = (x - np.max(x)) / temperature
    p = np.exp(np.clip(z, -40, 0))
    return p / p.sum()


def _uncertain_weights(rows: list[np.ndarray], count: int, cfg) -> np.ndarray:
    neutral = np.ones(count) / count
    if len(rows) < cfg.evidence_min_dates:
        return neutral
    block = np.asarray(rows[-cfg.competence_history_dates:], dtype=float)
    mean = np.nanmean(block, axis=0)
    effective = np.isfinite(block).sum(axis=0)
    std = np.nanstd(block, axis=0, ddof=1)
    penalty = np.where(effective > 1, std / np.sqrt(np.maximum(effective, 1)), 1.0)
    advantage = np.maximum(np.nan_to_num(mean - 0.5 - penalty, nan=0.0), 0.0)
    if not np.any(advantage > 0):
        return neutral
    evidence = min(1.0, len(block) / (2 * cfg.evidence_min_dates))
    return (1.0 - evidence) * neutral + evidence * _softmax(advantage, cfg.competence_temperature)


def _block_se(values: np.ndarray, block_length: int) -> float:
    if len(values) < 2:
        return float("inf")
    centered = values - np.mean(values)
    lag_max = min(block_length, len(values) - 1)
    variance = float(centered @ centered / len(values))
    for lag in range(1, lag_max + 1):
        variance += 2 * (1 - lag / (lag_max + 1)) * float(centered[lag:] @ centered[:-lag] / len(values))
    return float(np.sqrt(max(variance, 0.0) / len(values)))


def _gate(rows: list[float], cfg) -> tuple[float, float, float]:
    block = np.asarray(rows[-cfg.advantage_history_dates:], dtype=float)
    block = block[np.isfinite(block)]
    if len(block) < cfg.evidence_min_dates:
        return 0.0, np.nan, np.nan
    mean, se = float(block.mean()), _block_se(block, cfg.bootstrap_block_length)
    # Conservative, continuous gate. This is not a no-harm guarantee.
    strength = float(np.clip(mean / (2 * (se + cfg.advantage_standard_error_floor)), -10, 10))
    return float(1 / (1 + np.exp(-strength))), mean, se


def _pair_accuracy(record: dict) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    ranks = record["ranks"]
    y = np.asarray(record["target"], dtype=float)
    i, j = np.triu_indices(len(y), 1)
    truth = np.sign(y[i] - y[j])
    valid = np.isfinite(truth) & (truth != 0)
    if not np.any(valid):
        return np.full(ranks.shape[1], np.nan), {}
    i, j, truth = i[valid], j[valid], truth[valid]
    votes = np.sign(ranks[i] - ranks[j])
    correct = 0.5 + 0.5 * votes * truth[:, None]
    split = np.minimum((votes > 0).sum(axis=1), (votes < 0).sum(axis=1))
    return correct.mean(axis=0), {int(s): correct[split == s].mean(axis=0)
                                    for s in (1, 2, 3) if np.any(split == s)}


def _candidate(ranks: np.ndarray, weights: dict[int, np.ndarray], cfg) -> tuple[np.ndarray, float]:
    n, k = ranks.shape
    pair = ranks[:, None, :] - ranks[None, :, :]
    votes = np.sign(pair)
    split = np.minimum((votes > 0).sum(axis=2), (votes < 0).sum(axis=2))
    score = np.zeros(n, dtype=float)
    for s in (0, 1, 2, 3):
        mask = split == s
        if not np.any(mask):
            continue
        # Pair-specific expert weights make this nonseparable from weighted Borda ranks.
        pair_vote = np.tanh(pair / cfg.pair_temperature) @ weights[s]
        score += np.where(mask, pair_vote, 0.0).sum(axis=1)
    score /= max(1, n - 1)
    np.fill_diagonal(split, 0)
    disputed = float(np.triu(split > 0, 1).sum() / max(1, n * (n - 1) // 2))
    return _rank01(score), disputed


def run_council(records: list[dict], cfg) -> tuple[dict[str, list[dict]], list[dict], list[dict]]:
    """Prequential readouts. Labels enter the ledger only after their return window ends."""
    records = sorted(records, key=lambda r: r["date_index"])
    if not records or any(a["date_index"] >= b["date_index"] for a, b in zip(records, records[1:])):
        raise ValueError("Council requires unique chronological origins")
    router = [r for r in records if r["split"] == "router"]
    if len(router) < 2 * cfg.evidence_min_dates:
        raise ValueError("Insufficient router origins for calibrated chronological readout")
    cut = max(cfg.evidence_min_dates, int(len(router) * cfg.calibration_fraction))
    first_live = router[cut - 1]["date_index"] + cfg.horizon + 2
    calibration = [r for r in router[:cut] if r["date_index"] + cfg.horizon + 1 < first_live]
    if len(calibration) < cfg.evidence_min_dates:
        raise ValueError("Calibration contains insufficient mature dates")
    for rec in records:
        matrix = np.asarray(rec["pred_matrix"], dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != len(cfg.baseline_models) or not np.isfinite(matrix).all():
            raise ValueError(f"Nonfinite or incomplete forecaster streams at {rec['date']}")
        rec["ranks"] = np.column_stack([_rank01(matrix[:, j]) for j in range(matrix.shape[1])])
    regime_boundary = float(np.median([r["market_volatility"] for r in calibration]))
    for rec in records:
        rec["regime"] = int(rec["market_volatility"] > regime_boundary)
    ledger: list[dict] = []
    pending: deque[dict] = deque()
    for rec in calibration:
        global_accuracy, split_accuracy = _pair_accuracy(rec)
        ledger.append({"date_index": rec["date_index"], "regime": rec["regime"],
                       "global_accuracy": global_accuracy, "split_accuracy": split_accuracy,
                       "gain_global": np.nan, "gain_pair": np.nan})
    static_rows = [row["global_accuracy"] for row in ledger]
    static_weights = _uncertain_weights(static_rows, len(cfg.baseline_models), cfg)
    outputs = {name: [] for name in READOUTS}
    diagnostics: list[dict] = []
    matured: list[dict] = []
    for rec in records:
        if rec["date_index"] < first_live:
            continue
        while pending and pending[0]["date_index"] + cfg.horizon + 1 < rec["date_index"]:
            old = pending.popleft()
            global_accuracy, split_accuracy = _pair_accuracy(old)
            base = _observed_rank_ic(old["scores"]["equal_rank"], old["target"])
            global_gain = _observed_rank_ic(old["scores"]["global_always"], old["target"]) - base
            pair_gain = _observed_rank_ic(old["scores"]["pairwise_always"], old["target"]) - base
            entry = {"date_index": old["date_index"], "regime": old["regime"],
                     "global_accuracy": global_accuracy, "split_accuracy": split_accuracy,
                     "gain_global": global_gain, "gain_pair": pair_gain}
            ledger.append(entry)
            matured.append({"date": old["date"], "date_index": old["date_index"],
                            "global_gain": global_gain, "pair_gain": pair_gain,
                            "matured_before": rec["date"]})
        history = [row for row in ledger if row["date_index"] < rec["date_index"]]
        similar = [row for row in history if row["regime"] == rec["regime"]]
        if len(similar) < cfg.evidence_min_dates:
            similar = history
        recent = similar[-cfg.competence_history_dates:]
        pooled = [row["global_accuracy"] for row in recent]
        global_weights = _uncertain_weights(pooled, len(cfg.baseline_models), cfg)
        pair_weights = {0: global_weights}
        for split in (1, 2, 3):
            band = [row["split_accuracy"][split] for row in recent if split in row["split_accuracy"]]
            pair_weights[split] = _uncertain_weights(band, len(cfg.baseline_models), cfg)
        ranks = rec["ranks"]
        equal = _rank01(ranks.mean(axis=1))
        global_score = ranks @ global_weights
        simple_recent = np.asarray([row["global_accuracy"] for row in history[-20:]], dtype=float)
        if len(simple_recent):
            recent_mean = np.nan_to_num(np.nanmean(simple_recent, axis=0), nan=0.5)
            simple_weights = _softmax(recent_mean - 0.5, cfg.competence_temperature)
        else:
            simple_weights = np.ones(len(cfg.baseline_models)) / len(cfg.baseline_models)
        pair_score, dispute = _candidate(ranks, pair_weights, cfg)
        uniform_weights = np.ones(len(cfg.baseline_models)) / len(cfg.baseline_models)
        uniform_pair, _ = _candidate(ranks, {s: uniform_weights for s in (0, 1, 2, 3)}, cfg)
        gain_rows_global = [row["gain_global"] for row in similar if np.isfinite(row["gain_global"])]
        gain_rows_pair = [row["gain_pair"] for row in similar if np.isfinite(row["gain_pair"])]
        g_global, mean_global, se_global = _gate(gain_rows_global, cfg)
        g_pair, mean_pair, se_pair = _gate(gain_rows_pair, cfg)
        # Tie-aware percentile scores share a scale; no target values enter the interpolation.
        scores = {"equal_rank": equal, "static_weighted": ranks @ static_weights,
                  "recent_performance": ranks @ simple_weights,
                  "uniform_pair_vote": uniform_pair,
                  "global_always": _rank01(global_score), "pairwise_always": pair_score,
                  "global_gated": (1 - g_global) * equal + g_global * _rank01(global_score),
                  "darc_full": (1 - g_pair) * equal + g_pair * pair_score}
        common = {key: rec[key] for key in ("date", "date_index", "assets", "target", "raw_return", "split")}
        for name in READOUTS:
            outputs[name].append({**common, "pred": np.asarray(scores[name], dtype=float)})
        diagnostics.append({"date": rec["date"], "date_index": rec["date_index"], "split": rec["split"],
                            "regime": rec["regime"], "disputed_pair_fraction": dispute,
                            "global_gate": g_global, "pair_gate": g_pair,
                            "global_gain_history": mean_global, "pair_gain_history": mean_pair,
                            "global_gain_se": se_global, "pair_gain_se": se_pair,
                            "global_weight_entropy": float(-(global_weights * np.log(global_weights + 1e-12)).sum()),
                            "mean_pair_weight_entropy": float(np.mean([-(w * np.log(w + 1e-12)).sum()
                                                                       for w in pair_weights.values()])),
                            "pair_correction": float(np.mean(np.abs(pair_score - equal))),
                            "effective_pair_correction": float(np.mean(np.abs(scores["darc_full"] - equal))),
                            **{f"weight_{name}": float(global_weights[j]) for j, name in enumerate(cfg.baseline_models)}})
        pending.append({**rec, "scores": scores})
    return outputs, diagnostics, matured
