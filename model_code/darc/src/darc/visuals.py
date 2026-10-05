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
    ax.grid(alpha=0.2)
    fig.text(0.06, 0.035, textwrap.fill(caption, width=115), fontsize=8.2, va="bottom")
    fig.subplots_adjust(left=0.11, right=0.97, top=0.86, bottom=0.19)
    fig.savefig(path, format="pdf", metadata={"Title": caption[:80]})
    plt.close(fig)


def make_figures(dataset: str, artifacts: dict, destination: Path, cfg) -> None:
    """Eight descriptive module-effect figures per panel; literature controls remain tables."""
    destination.mkdir(parents=True, exist_ok=True)
    diag = pd.DataFrame(artifacts["diagnostics"])
    test = diag.loc[diag["split"] == "test"].copy()
    if test.empty:
        raise ValueError(f"{dataset}: no test diagnostics for figures")
    test["date"] = pd.to_datetime(test["date"])
    x = test["date"]
    captions = {}
    context = (f"{dataset}, test origins {x.min().date()} to {x.max().date()}, "
               f"forward horizon {cfg.horizon} trading dates. ")

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.plot(x, test["disputed_pair_fraction"], lw=0.9, color="#264653")
    ax.set(title="Where the forecasters disagree", ylabel="Fraction of stock pairs", xlabel="Decision date")
    captions["01_disagreement"] = context + "Share of stock pairs on which the six frozen forecasters disagree. The pair vote split is 1–5, 2–4 or 3–3. This measures the opportunity for pair-conditional arbitration, not prediction quality."
    _finish(fig, ax, destination / f"{dataset}_01_disagreement.pdf", captions["01_disagreement"])

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.plot(x, test["global_weight_entropy"], lw=0.9, label="Global", color="#2a9d8f")
    ax.plot(x, test["mean_pair_weight_entropy"], lw=0.9, label="Pair-specific", color="#e76f51")
    ax.axhline(np.log(len(cfg.baseline_models)), color="gray", ls="--", lw=0.8, label="Equal weights")
    ax.legend(fontsize=8)
    ax.set(title="How concentrated are the expert votes?", ylabel="Weight entropy", xlabel="Decision date")
    captions["02_weight_entropy"] = context + "Entropy of global and pair-specific weights computed from already matured historical dates. Pair weights condition on a 1–5, 2–4 or 3–3 vote split. Lower entropy means a more concentrated vote; equal weights have entropy log(6)."
    _finish(fig, ax, destination / f"{dataset}_02_weight_entropy.pdf", captions["02_weight_entropy"])

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.plot(x, test["pair_gate"], label="Pairwise candidate", color="#e76f51", lw=1)
    ax.plot(x, test["global_gate"], label="Global candidate", color="#2a9d8f", lw=0.9)
    ax.legend(fontsize=8)
    ax.set(title="Matured full-list evidence controls adoption", ylabel="Gate strength", xlabel="Decision date", ylim=(-0.03, 1.03))
    captions["03_gates"] = context + "The gate is zero until enough historical lists have matured; afterwards it responds to prior full-list RankIC advantage over equal-rank aggregation. It does not use current outcomes."
    _finish(fig, ax, destination / f"{dataset}_03_gates.pdf", captions["03_gates"])

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.plot(x, test["pair_correction"], label="Ungated arbitration", lw=0.9, color="#264653")
    ax.plot(x, test["effective_pair_correction"], label="After evidence gate", lw=0.9, color="#e9c46a")
    ax.legend(fontsize=8)
    ax.set(title="Magnitude of the ranking correction", ylabel="Mean absolute score change", xlabel="Decision date")
    captions["04_correction"] = context + "Average absolute percentile-score change from equal-rank aggregation before and after the matured-advantage gate. This is a structural intervention measure, not a return estimate."
    _finish(fig, ax, destination / f"{dataset}_04_correction.pdf", captions["04_correction"])

    weight_cols = [f"weight_{name}" for name in cfg.baseline_models]
    weights = test[weight_cols].to_numpy().T
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    image = ax.imshow(weights, aspect="auto", interpolation="nearest", vmin=0, vmax=max(0.3, float(np.nanmax(weights))), cmap="viridis")
    ax.set(yticks=np.arange(len(cfg.baseline_models)), yticklabels=cfg.baseline_models,
           title="Global expert vote weights", xlabel="Chronological test origin")
    fig.colorbar(image, ax=ax, label="Weight", pad=0.01)
    captions["05_weights"] = context + "Chronological global vote weights for the six common-protocol adapted forecasters. Pair-specific arbitration additionally varies the weights by 1–5, 2–4 or 3–3 disagreement split."
    _finish(fig, ax, destination / f"{dataset}_05_weights.pdf", captions["05_weights"])

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.scatter(test["pair_gain_history"], test["pair_gate"], s=12, alpha=0.55, color="#e76f51")
    ax.axvline(0, color="gray", ls="--", lw=0.8)
    ax.set(title="Historical advantage and adoption", xlabel="Mean matured pair-candidate RankIC advantage", ylabel="Pairwise gate strength", ylim=(-0.03, 1.03))
    captions["06_gate_response"] = context + "Each point is a test decision date. The horizontal coordinate summarizes earlier completed rankings; the vertical coordinate is the resulting adoption strength."
    _finish(fig, ax, destination / f"{dataset}_06_gate_response.pdf", captions["06_gate_response"])

    factorial = artifacts["factorial"]
    cells = ["global_always", "pairwise_always", "global_gated", "darc_full"]
    values = [factorial[name].dropna().to_numpy() for name in cells]
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.boxplot(values, labels=["A0 B0", "A1 B0", "A0 B1", "A1 B1"], showfliers=False)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set(title="Four-cell module factorial", xlabel="A: pair arbitration; B: matured-advantage gate", ylabel="Test-date RankIC")
    captions["07_factorial"] = context + "Distribution of test-date RankIC in the four frozen-stream factorial cells. A activates pair-dependent arbitration; B activates the historical full-list advantage gate. Dates, not pairs, are the evaluation units."
    _finish(fig, ax, destination / f"{dataset}_07_factorial.pdf", captions["07_factorial"])

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    valid = factorial[["date", *cells]].dropna().copy()
    valid["date"] = pd.to_datetime(valid["date"])
    for control, label, color in (("pairwise_always", "Gate effect with arbitration", "#e76f51"),
                                  ("global_gated", "Arbitration effect with gate", "#2a9d8f"),
                                  ("global_always", "Joint versus neither", "#264653")):
        delta = (valid["darc_full"] - valid[control]).to_numpy()
        ax.plot(valid["date"], np.cumsum(delta) / np.arange(1, len(delta) + 1), lw=1, label=label, color=color)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.legend(fontsize=7)
    ax.set(title="Accumulating module effects", xlabel="Decision date", ylabel="Running mean paired RankIC difference")
    captions["08_module_effects"] = context + "Running mean same-date RankIC differences between the full readout and three factorial controls. Curves may cross zero; the complete paired inference table is in the evidence archive."
    _finish(fig, ax, destination / f"{dataset}_08_module_effects.pdf", captions["08_module_effects"])

    (destination / f"{dataset}_captions.json").write_text(json.dumps(captions, indent=2) + "\n", encoding="utf-8")
