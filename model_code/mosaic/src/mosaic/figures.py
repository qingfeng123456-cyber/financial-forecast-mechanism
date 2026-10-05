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
                 destination: Path, horizon: int) -> None:
    """Eight within-model/module figures; literature-model comparisons remain tabular."""
    destination.mkdir(parents=True, exist_ok=True)
    full = diagnostics[(diagnostics["split"] == "test") &
                       (diagnostics["model"] == "mosaic_full")].sort_values("date_index").copy()
    if full.empty:
        raise RuntimeError(f"{dataset}: missing full-model diagnostics")
    dates = pd.to_datetime(full.date)
    context = (f"{dataset}; test origins {dates.min().date()}–{dates.max().date()}; "
               f"forecast target is the next {horizon} daily close-to-close squared log returns, averaged. ")
    captions = {}

    fig, ax = plt.subplots(figsize=(8, 4.5))
    weights = [full[f"anchor_weight_{j}"].mean() for j in range(4)]
    ax.bar(["1 date", "5 dates", "20 dates", "60 dates"], weights, color="#285f8f")
    ax.set(title="Learned multiscale persistence weights", ylabel="Weight", ylim=(0, 1))
    captions["01_anchor"] = context + "Softmax weights of the full model's positive variance anchor. These summarize the fitted model, not predictive superiority."
    _save(fig, ax, destination / f"{dataset}_01_anchor.pdf", captions["01_anchor"])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(dates, full.own_contribution, lw=0.9, label="Own downside")
    ax.plot(dates, full.market_contribution, lw=0.9, label="Market stress")
    ax.legend(fontsize=8)
    ax.set(title="Additive module outputs over time", xlabel="Decision date", ylabel="Mean log-variance contribution")
    captions["02_branches"] = context + "Date means of the two additive branches. They are model components and are not identified causal effects."
    _save(fig, ax, destination / f"{dataset}_02_branches.pdf", captions["02_branches"])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(dates, full.interaction_contribution, lw=0.9, color="#a44936")
    ax.set(title="Joint downside–market response", xlabel="Decision date", ylabel="Mean log-variance interaction")
    captions["03_interaction"] = context + "The full cell's explicitly constrained product branch, averaged over the contemporaneously eligible stocks."
    _save(fig, ax, destination / f"{dataset}_03_interaction.pdf", captions["03_interaction"])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = pd.qcut(full.stress.rank(method="first"), 5, labels=False)
    grouped = full.groupby(bins)[["own_contribution", "market_contribution", "interaction_contribution"]].mean()
    for column, label in (("own_contribution", "Own"), ("market_contribution", "Market"),
                          ("interaction_contribution", "Joint")):
        ax.plot(np.arange(1, 6), grouped[column], marker="o", label=label)
    ax.legend(fontsize=8)
    ax.set(title="Module outputs across past-market-stress quintiles", xlabel="Past stress quintile", ylabel="Mean log-variance contribution", xticks=np.arange(1, 6))
    captions["04_stress"] = context + "Stress groups use only information observed at the decision date; the plot describes model behavior, not group-specific accuracy."
    _save(fig, ax, destination / f"{dataset}_04_stress.pdf", captions["04_stress"])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = pd.qcut(full.mean_beta.rank(method="first"), 5, labels=False)
    grouped = full.groupby(bins)[["market_contribution", "interaction_contribution"]].mean()
    ax.plot(np.arange(1, 6), grouped.market_contribution, marker="o", label="Market")
    ax.plot(np.arange(1, 6), grouped.interaction_contribution, marker="o", label="Joint")
    ax.legend(fontsize=8)
    ax.set(title="Responses by historical market exposure", xlabel="Past beta quintile", ylabel="Mean log-variance contribution", xticks=np.arange(1, 6))
    captions["05_exposure"] = context + "Cross-date quintiles of mean rolling market exposure and corresponding module outputs; no future outcome enters the grouping."
    _save(fig, ax, destination / f"{dataset}_05_exposure.pdf", captions["05_exposure"])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    xbin = pd.qcut(full.mean_own_down.rank(method="first"), 4, labels=False)
    ybin = pd.qcut(full.stress.rank(method="first"), 4, labels=False)
    matrix = full.groupby([ybin, xbin]).interaction_contribution.mean().unstack().reindex(index=range(4), columns=range(4))
    image = ax.imshow(matrix.to_numpy(), origin="lower", aspect="auto", cmap="viridis")
    fig.colorbar(image, ax=ax, label="Mean interaction contribution")
    ax.set(title="Joint branch across observed downside states", xlabel="Own-downside quartile", ylabel="Market-stress quartile",
           xticks=range(4), xticklabels=range(1, 5), yticks=range(4), yticklabels=range(1, 5))
    captions["06_surface"] = context + "Descriptive grid of the product branch across past-own and past-market state quartiles. Empty cells remain blank."
    _save(fig, ax, destination / f"{dataset}_06_surface.pdf", captions["06_surface"])

    ordered = factorial.sort_values("date").copy()
    fx = pd.to_datetime(ordered.date)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, label in (("mosaic_base", "No branches"), ("mosaic_own", "Own only"),
                        ("mosaic_market", "Market only")):
        diff = (ordered[name] - ordered.mosaic_full).to_numpy()
        ax.plot(fx, np.cumsum(diff) / np.arange(1, len(diff) + 1), lw=1.0, label=label)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.legend(fontsize=8)
    ax.set(title="Accumulated same-date ablation effects", xlabel="Decision date", ylabel="Running mean QLIKE improvement")
    captions["07_ablation"] = context + "Each line is ablated-cell QLIKE minus full-cell QLIKE on common test dates. Positive values favor the full cell; paired date-block inference is separate."
    _save(fig, ax, destination / f"{dataset}_07_ablation.pdf", captions["07_ablation"])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    vals = ordered.interaction.to_numpy()
    ax.plot(fx, np.cumsum(vals) / np.arange(1, len(vals) + 1), lw=1.0)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set(title="Factorial loss interaction over test dates", xlabel="Decision date", ylabel="Running mean factorial QLIKE contrast")
    captions["08_factorial"] = context + "Contrast is loss(full) − loss(own only) − loss(market only) + loss(no branches). Negative means a complementary loss reduction in this registered factorial."
    _save(fig, ax, destination / f"{dataset}_08_factorial.pdf", captions["08_factorial"])

    (destination / f"{dataset}_captions.json").write_text(
        json.dumps(captions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
