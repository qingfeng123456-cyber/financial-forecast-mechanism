from __future__ import annotations

import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig, ax, path: Path, caption: str) -> None:
    ax.grid(alpha=0.16)
    fig.text(0.055, 0.025, textwrap.fill(caption, width=112), fontsize=7.5, va="bottom")
    fig.subplots_adjust(left=0.11, right=0.96, top=0.87, bottom=0.24)
    fig.savefig(path, format="pdf", metadata={"Title": caption[:100]})
    plt.close(fig)


def make_figures(dataset: str, diagnostics: pd.DataFrame, factorial: pd.DataFrame,
                 completed: dict, base_names: list[str], fit: dict,
                 destination: Path, horizon: int) -> None:
    """Eight module/factorial figures; literature comparisons remain tabular."""
    destination.mkdir(parents=True, exist_ok=True)
    full = diagnostics.sort_values("date_index").copy()
    if full.empty:
        raise RuntimeError(f"{dataset}: missing DASH-V diagnostics")
    dates = pd.to_datetime(full.date)
    context = (f"{dataset}; test origins {dates.min().date()}–{dates.max().date()}; "
               f"target is the next {horizon} squared daily close returns, averaged. ")
    captions = {}

    fig, ax = plt.subplots(figsize=(8, 4.5))
    static = np.asarray(fit["static_weights"])
    ax.bar(np.arange(len(base_names)), static, color="#285f8f")
    ax.set(title="Validation-trained static QLIKE pool", ylabel="Weight",
           xticks=np.arange(len(base_names)), xticklabels=base_names)
    ax.tick_params(axis="x", rotation=35, labelsize=7)
    captions["01_static"] = context + "Weights are fitted on validation forecasts and frozen before test; the plot is not test-performance evidence."
    _save(fig, ax, destination / f"{dataset}_01_static_weights.pdf", captions["01_static"])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for column, label in (("meta_fast", "Fast"), ("meta_medium", "Medium"),
                          ("meta_slow", "Slow"), ("meta_static", "Static anchor")):
        ax.plot(dates, full[column], lw=0.9, label=label)
    ax.legend(fontsize=8, ncol=2)
    ax.set(title="Delayed allocation across adaptation memories", xlabel="Decision date", ylabel="Meta weight", ylim=(0, 1))
    captions["02_meta"] = context + "Weights are formed before the current forecast and updated only from horizon-matured QLIKE losses."
    _save(fig, ax, destination / f"{dataset}_02_memory_weights.pdf", captions["02_meta"])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for column, label in (("rho_fast", "Fast"), ("rho_medium", "Medium"), ("rho_slow", "Slow")):
        ax.plot(dates, full[column], lw=0.9, label=label)
    ax.legend(fontsize=8)
    ax.set(title="Asset-specific reliability by memory", xlabel="Decision date", ylabel="Mean local reliability", ylim=(0, 1))
    captions["03_rho"] = context + "Reliability is effective matured sample size divided by that size plus the validation-estimated pooling strength."
    _save(fig, ax, destination / f"{dataset}_03_local_reliability.pdf", captions["03_rho"])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    matrix = full[[f"base_weight_{j}" for j in range(len(base_names))]].to_numpy().T
    x_edges = np.arange(matrix.shape[1] + 1)
    y_edges = np.arange(matrix.shape[0] + 1)
    image = ax.pcolormesh(x_edges, y_edges, matrix, cmap="viridis", shading="flat")
    fig.colorbar(image, ax=ax, label="Cross-asset mean effective weight")
    ax.set(title="Effective base-model allocation", xlabel="Ordered test origin", ylabel="Base forecast",
           yticks=np.arange(len(base_names)) + 0.5, yticklabels=base_names)
    ax.tick_params(axis="y", labelsize=7)
    captions["04_base"] = context + "The displayed allocation integrates memory weights, hierarchical within-memory weights, and the static anchor."
    _save(fig, ax, destination / f"{dataset}_04_base_allocation.pdf", captions["04_base"])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    raw_kappas = [fit["memory_kappa"][str(value)] for value in fit["memory_decays"]]
    kappas = [min(float(value), 1e6) for value in raw_kappas]
    labels = ["pooled" if float(value) >= 1e11 else f"{float(value):.1f}" for value in raw_kappas]
    bars = ax.bar(["Fast", "Medium", "Slow"], kappas,
                  color=["#a44936", "#d08c32", "#285f8f"])
    ax.bar_label(bars, labels=labels, fontsize=8)
    ax.set_yscale("log")
    ax.set_ylim(bottom=1.0)
    ax.set(title="Validation-estimated pooling strength", ylabel="Kappa (log scale; larger means stronger pooling)")
    captions["05_kappa"] = context + "Kappa is estimated before test from cross-asset relative-loss heterogeneity and estimation noise for each memory."
    _save(fig, ax, destination / f"{dataset}_05_pooling_strength.pdf", captions["05_kappa"])

    full_daily = completed[("test", "dashv_full")].copy()
    static_daily = completed[("test", "static_qlike")].copy()
    merged = full_daily[["date", "mean_forecast", "mean_target"]].merge(
        static_daily[["date", "mean_forecast"]], on="date", suffixes=("_full", "_static"))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ratio_full = merged.mean_forecast_full / np.maximum(merged.mean_target, 1e-8)
    ratio_static = merged.mean_forecast_static / np.maximum(merged.mean_target, 1e-8)
    ax.plot(pd.to_datetime(merged.date), ratio_full.rolling(20, min_periods=5).mean(), label="DASH-V full", lw=1.0)
    ax.plot(pd.to_datetime(merged.date), ratio_static.rolling(20, min_periods=5).mean(), label="Static pool", lw=1.0)
    ax.axhline(1.0, color="gray", ls="--", lw=0.8)
    ax.legend(fontsize=8)
    ax.set(title="Rolling forecast-to-target scale", xlabel="Decision date", ylabel="20-origin mean ratio")
    captions["06_scale"] = context + "The ratio is descriptive and uses realized targets; it diagnoses scale adaptation rather than proving superiority."
    _save(fig, ax, destination / f"{dataset}_06_scale_tracking.pdf", captions["06_scale"])

    ordered = factorial.sort_values("date").copy()
    fx = pd.to_datetime(ordered.date)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, label in (("dashv_base", "Pooled single memory"),
                        ("dashv_hier", "Hierarchy only"),
                        ("dashv_multi", "Multiple memories only")):
        diff = (ordered[name] - ordered.dashv_full).to_numpy()
        ax.plot(fx, np.cumsum(diff) / np.arange(1, len(diff) + 1), lw=1.0, label=label)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.legend(fontsize=8)
    ax.set(title="Accumulated same-date ablation effects", xlabel="Decision date", ylabel="Running mean QLIKE improvement")
    captions["07_ablation"] = context + "Each line is ablated-cell QLIKE minus full-cell QLIKE. Positive values favor the full cell; paired inference is reported separately."
    _save(fig, ax, destination / f"{dataset}_07_ablation.pdf", captions["07_ablation"])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    values = ordered.interaction.to_numpy()
    ax.plot(fx, np.cumsum(values) / np.arange(1, len(values) + 1), lw=1.0)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set(title="Pooling × memory factorial contrast", xlabel="Decision date", ylabel="Running mean QLIKE contrast")
    captions["08_factorial"] = context + "Contrast is loss(full) − loss(hierarchy only) − loss(multiple memories only) + loss(base); negative favors complementary loss reduction."
    _save(fig, ax, destination / f"{dataset}_08_factorial.pdf", captions["08_factorial"])

    (destination / f"{dataset}_captions.json").write_text(
        json.dumps(captions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
