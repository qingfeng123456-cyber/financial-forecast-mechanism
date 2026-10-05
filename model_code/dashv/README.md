# DASH-V AutoDL package

`python main.py` runs the registered development experiment and writes one evidence
ZIP under `log/`. The program never installs dependencies, starts a parameter
sweep, retries a failed fit, or enforces a wall-clock cutoff.

The target after close on date *t* is the mean of the next five squared daily
adjusted-close log returns, multiplied by 10,000. It is a noisy daily-return
variance proxy rather than intraday realized variance. Every feature uses data
available by the forecast origin. An online update uses an issued forecast only
after its complete five-date target has matured.

## Run on AutoDL

Extract the upload ZIP so `main.py`, `config.json`, `requirements.txt`, `data/`,
`research/`, and `src/` are together. Prepare an environment containing the
packages in `requirements.txt`, then run:

```bash
python main.py
```

The code does not install packages. CUDA is recommended. Every training epoch
prints current origin progress, QLIKE, elapsed time, and ETA. When the run ends,
download the single `log/RUN-...-DASHV.zip` file and retain it unchanged.

## Registered development matrix

- Three bundled public daily-bar panels with chronological fit, checkpoint,
  validation, and test regions.
- Five statistical controls: persistence, EWMA, direct-QLIKE HAR, SHAR, and
  market-SHAR.
- Five equal-information neural controls: MLP, GRU, LSTM, TCN, and a compact
  patch-transformer adaptation. Named architectures are common-protocol
  adaptations, not reproductions of every original training protocol.
- Four matched aggregation controls and a four-cell DASH-V factorial. Every
  aggregation variant receives the same archived base forecasts.
- Fifteen neural fits in total. Aggregation replay does not retrain them.
- Equal-decision-date mean QLIKE is primary. MSE, MAE, forecast/target scale,
  past-stress results, turnover, pooling strength, memory allocation, paired
  date-block inference, and the factorial contrast are exported.
- Twenty-four vector PDF mechanism/ablation figures, eight per panel. Raw-model
  comparisons remain tables.

The returned evidence determines whether the mechanism works. These three test
panels have been examined by earlier, different mechanisms, so this package is a
development experiment rather than untouched confirmation. A positive result
still requires an independent confirmation panel before a strong submission
claim. All failures and adverse outputs remain in the evidence archive.

See `research/PROTOCOL.md` for equations, update timing, comparisons, and novelty
boundaries.
