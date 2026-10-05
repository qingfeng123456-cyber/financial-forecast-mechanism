from __future__ import annotations

import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _finish(fig, ax, path: Path, caption: str) -> None:
    ax.grid(alpha=0.18)
    fig.text(0.06, 0.025, textwrap.fill(caption, width=115), fontsize=8.0, va="bottom")
    fig.subplots_adjust(left=0.11, right=0.96, top=0.87, bottom=0.22)
    fig.savefig(path, format="pdf", metadata={"Title": caption[:80]})
    plt.close(fig)


def make_figures(dataset: str, diagnostics: pd.DataFrame, factorial: pd.DataFrame,
                 destination: Path, cfg) -> None:
    """Eight module and factorial figures per panel; literature controls stay tabular."""
    destination.mkdir(parents=True, exist_ok=True)
    full = diagnostics[(diagnostics["split"] == "test") &
                       (diagnostics["model"] == "care_full")].copy().sort_values("date_index")
    if full.empty:
        raise ValueError(f"{dataset}: no full-model diagnostics")
    full["date"] = pd.to_datetime(full["date"])
    x = full["date"]
    context = (f"{dataset}; test decision dates {x.min().date()} to {x.max().date()}; "
               f"{cfg.horizon}-date forward log returns. Historical analog outcomes matured before each decision date. ")
    captions = {}

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.plot(x, full["market_coverage"], lw=0.9, label="Market analogs")
    ax.plot(x, full["stock_coverage"], lw=0.9, label="Same-stock analogs")
    ax.legend(fontsize=8)
    ax.set(title="Availability of the two memories", xlabel="Decision date", ylabel="Mean available fraction", ylim=(-0.02, 1.02))
    captions["01_memory_support"] = context + "Fraction of the requested historical neighbors with a matured return label. This measures evidence availability, not forecast skill."
    _finish(fig, ax, destination / f"{dataset}_01_memory_support.pdf", captions["01_memory_support"])

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.plot(x, full["market_width"], lw=0.9, label="Market analog spread")
    ax.plot(x, full["stock_width"], lw=0.9, label="Same-stock analog spread")
    ax.legend(fontsize=8)
    ax.set(title="Width of retrieved return distributions", xlabel="Decision date", ylabel="Mean historical q90 − q10, normalized units")
    captions["02_memory_spread"] = context + "Each line summarizes the empirical 10th–90th percentile span of one analog source, averaged across available stocks. The unit is the training-set return scale."
    _finish(fig, ax, destination / f"{dataset}_02_memory_spread.pdf", captions["02_memory_spread"])

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    for column, label in (("lower_tail_disagreement", "Lower tail"),
                          ("median_disagreement", "Median"),
                          ("upper_tail_disagreement", "Upper tail")):
        ax.plot(x, full[column], lw=0.9, label=label)
    ax.legend(fontsize=8)
    ax.set(title="Where historical views disagree", xlabel="Decision date", ylabel="Absolute analog quantile difference, normalized units")
    captions["03_memory_disagreement"] = context + "Absolute gap between market-conditioned and same-stock empirical quantiles. A gap denotes different evidence, not proof that either source is wrong."
    _finish(fig, ax, destination / f"{dataset}_03_memory_disagreement.pdf", captions["03_memory_disagreement"])

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    for column, label in (("adoption_lower", "Lower width"),
                          ("adoption_center", "Median"),
                          ("adoption_upper", "Upper width")):
        ax.plot(x, full[column], lw=0.9, label=label)
    ax.legend(fontsize=8)
    ax.set(title="How much the model uses analog experts", xlabel="Decision date", ylabel="Mean expert-adoption weight", ylim=(-0.02, 1.02))
    captions["04_adoption"] = context + "Average learned interpolation weights between shared temporal forecast and analog-conditioned specialists, separately for median and interval widths."
    _finish(fig, ax, destination / f"{dataset}_04_adoption.pdf", captions["04_adoption"])

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    matrix = np.asarray([[full[f"gate_{part}_{expert}"].mean() for expert in range(cfg.num_experts)]
                         for part in ("lower", "center", "upper")])
    image = ax.imshow(matrix, vmin=0, vmax=max(0.5, float(matrix.max())), cmap="viridis", aspect="auto")
    ax.set(xticks=np.arange(cfg.num_experts), xticklabels=[f"E{i+1}" for i in range(cfg.num_experts)],
           yticks=np.arange(3), yticklabels=["Lower", "Median", "Upper"],
           title="Specialist participation by distribution component", xlabel="Specialist")
    fig.colorbar(image, ax=ax, label="Mean route weight", pad=0.01)
    captions["05_specialists"] = context + "Mean sparse-route weight for each specialist and distribution component on test origins. Different rows show whether tail and central routes diverge."
    _finish(fig, ax, destination / f"{dataset}_05_specialists.pdf", captions["05_specialists"])

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    for part, label in (("lower", "Lower"), ("center", "Median"), ("upper", "Upper")):
        ax.plot(x, full[f"entropy_{part}"], lw=0.9, label=label)
    ax.legend(fontsize=8)
    ax.set(title="Concentration of sparse specialist routes", xlabel="Decision date", ylabel="Entropy of mean route weights")
    captions["06_route_entropy"] = context + "Entropy of date-averaged route weights. It describes whether specialists share responsibility, without using the realized forecast outcome."
    _finish(fig, ax, destination / f"{dataset}_06_route_entropy.pdf", captions["06_route_entropy"])

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    for name, label in (("care_full", "Both modules"), ("care_dual_plain", "Dual memory only"),
                        ("care_single_tail", "Tail route only"), ("care_single_plain", "Neither")):
        part = diagnostics[(diagnostics["split"] == "test") & (diagnostics["model"] == name)].copy()
        part["date"] = pd.to_datetime(part["date"])
        ax.plot(part["date"], part["mean_correction"], lw=0.8, label=label)
    ax.legend(fontsize=7)
    ax.set(title="How the modules change the shared forecast", xlabel="Decision date", ylabel="Mean absolute quantile correction, log-return units")
    captions["07_correction"] = context + "Average absolute difference between each factorial cell's final quantiles and its own shared temporal forecast. This is a mechanism magnitude, not a performance comparison."
    _finish(fig, ax, destination / f"{dataset}_07_correction.pdf", captions["07_correction"])

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ordered = factorial.sort_values("date").copy()
    date = pd.to_datetime(ordered["date"])
    for name, label in (("care_dual_plain", "Tail-aware routing effect"),
                        ("care_single_tail", "Dual-memory effect"),
                        ("care_single_plain", "Joint versus neither")):
        difference = (ordered[name] - ordered["care_full"]).to_numpy()
        ax.plot(date, np.cumsum(difference) / np.arange(1, len(difference) + 1),
                lw=1.0, label=label)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.legend(fontsize=7)
    ax.set(title="Paired module effects over test dates", xlabel="Decision date", ylabel="Running mean pinball improvement, log-return units")
    captions["08_factorial_effects"] = context + "Running same-date difference in mean three-quantile pinball loss: comparator minus full model. Positive values favor the full cell; date-block inference is reported separately."
    _finish(fig, ax, destination / f"{dataset}_08_factorial_effects.pdf", captions["08_factorial_effects"])

    (destination / f"{dataset}_captions.json").write_text(json.dumps(captions, indent=2) + "\n",
                                                        encoding="utf-8")
