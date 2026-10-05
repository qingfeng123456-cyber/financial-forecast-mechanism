# CARE AutoDL package

Upload the contents of this folder to an AutoDL working directory, install the packages in `requirements.txt` in the selected environment, then run:

```bash
python main.py
```

The command runs the fixed, chronological experiment on three bundled public equity panels and writes one downloadable `log/RUN-...-CARE.zip`. There is no automatic package installation, parameter search, retry, or time-limit kill. Training prints date-level progress, elapsed time and ETA each epoch. The archive retains source/data hashes, checkpoints, predictions, dated metrics, paired inference, all failures, and 24 vector-PDF module figures with captions.

CARE predicts the 10th, 50th and 90th percentiles of five-date forward log returns. Each decision uses two strictly past-only analog memories: earlier dates with similar market context and earlier trajectories of the same stock. Four role-specialist recurrent experts provide analog-conditioned median and interval-width proposals. A shared temporal forecaster supplies a fallback. The full model learns separate routes and fallback weights for the median and both interval widths using memory support, similarity and cross-memory tail differences. The four cells remove dual memory and tail-quality routing in a 2×2 factorial. No rule assumes that evidence disagreement means either source is wrong.

The controls are rolling same-stock empirical quantiles, market-analog empirical quantiles, and common-protocol quantile adaptations of GRU, StockMixer, MASTER and MERA. Named adaptations do not reproduce their original published protocols. The primary outcome is mean three-quantile pinball loss on test dates. Secondary outcomes are median error, directional accuracy, 80% interval coverage and 10% lower-tail exceedance. Dated paired block inference uses dates as units. Every adverse result remains in the archive; figures illustrate mechanisms and the factorial, while literature-model comparisons are tables.

This package has had local static inspection only. No local model forward, training, smoke test, backtest or empirical evaluation was run. Actual runtime and predictive value can be established only by the AutoDL run. A top-quartile journal is a research target, not an assured result.
