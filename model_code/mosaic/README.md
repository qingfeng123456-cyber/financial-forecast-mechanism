# MOSAIC AutoDL package

This is a **new daily-bar variance-forecasting experiment**, not another CARE patch. `python main.py` runs the entire registered matrix and writes one downloadable evidence ZIP under `log/`. No dependency installer, sweep, retry, or time limit is built in.

The target at the close of date *t* is the mean of the next five **daily** squared adjusted-close log returns, multiplied by 10,000. It is a daily-return variance proxy; the package does not claim intraday realized variance. Forecasts and the primary QLIKE score are positive-variance forecasts. Past-only features include stock variance at four horizons, daily downside and range proxies, market stress, and rolling stock–market exposure. Test symbols are selected using observed feature history, not future labels.

## Run on AutoDL

Upload and extract the ZIP so `main.py`, `config.json`, `requirements.txt`, `data/`, and `src/` are together in one directory. In an environment with the packages in `requirements.txt` already available, run:

```bash
python main.py
```

If packages are absent, install them **manually** before running. The script will never install them. A CUDA GPU is recommended. The console prints date-level progress, elapsed time and estimated remaining time for every epoch. When it finishes, download the single `log/RUN-...-MOSAIC.zip` file. Send that entire archive for analysis, including failures and adverse results.

## Registered experiment

- Three bundled public daily-bar panels: `sp500_cc0`, `kdd17`, `acl18`, with chronological fit/checkpoint/validation/test splits.
- Statistical controls: persistence, EWMA, pooled log-link HAR, downside HAR, market-enhanced downside HAR, and its explicit linear interaction extension. These use the same daily-bar information where appropriate.
- Neural controls: an equally informed flat MLP, a 60-date GRU, and a **common-protocol adaptation** of RG-ResMoE. The adaptation is not an author-code reproduction.
- Four MOSAIC cells: positive multiscale anchor; anchor plus own-downside branch; anchor plus market-stress branch; and both branches with an explicit product interaction. Each cell is trained independently.
- Twenty-one neural fits across the three panels; twelve pooled statistical-model fits plus six parameter-free control evaluations. No automatic retuning or repeated runs.
- Primary score: equal-weight decision-date mean QLIKE. Secondary scores: MSE, MAE, forecast/target means, and an ex-ante high/low past-stress partition. Date-block paired intervals and the four-cell factorial contrast are saved. The 24 vector PDF figures describe the model and ablations; literature comparisons are in tables.

The experiment is designed to **test** whether constrained downside/market interaction adds value beyond strong same-information controls. It does not guarantee that the full cell wins. All completed and failed outputs remain in the evidence archive. The SP500 data are a current-constituent panel and must not be described as a survivorship-free historical index universe.

See `research/PROTOCOL.md` for exact equations, comparison scope, and source references.
