# Forecast Mechanism Audit

Code, evidence summaries, and manuscript sources for **From Mechanism
Activation to Forecast Value: A Multi-Study Audit on Public Equity Panels**.

This repository preserves five completed, configuration-fixed studies of
financial forecasting mechanisms: C3R, DARC, CARE, MOSAIC, and DASH-V. The
studies cover cross-sectional return ranking, conditional return quantiles, and
daily-return variance proxies. They are related development studies on the same
three public panels, not independent replications.

## Five model code packages

The complete executable code is in the root-level [`model_code/`](model_code/)
folder. These are the source files from the five historical AutoDL packages,
including each entry point, configuration, data pipeline, training loop,
metrics, inference, and figures.

| Model | Core mechanism code | Run entry point |
|---|---|---|
| C3R | [`c3r.py`](model_code/c3r/src/c3rmoe/c3r.py) | [`main.py`](model_code/c3r/main.py) |
| DARC | [`council.py`](model_code/darc/src/darc/council.py) | [`main.py`](model_code/darc/main.py) |
| CARE | [`models.py`](model_code/care/src/care/models.py) | [`main.py`](model_code/care/main.py) |
| MOSAIC | [`models.py`](model_code/mosaic/src/mosaic/models.py) | [`main.py`](model_code/mosaic/main.py) |
| DASH-V | [`aggregate.py`](model_code/dashv/src/dashv/aggregate.py) | [`main.py`](model_code/dashv/main.py) |

The neural comparator implementations are beside these files in each package's
`models.py`; `pipeline.py` and `train.py` contain the complete experimental
workflow. Nothing needs to be reconstructed from the manuscript to run the code.

## Main finding

Mechanism activation did not establish forecast superiority. Across 15
study-panel cells, no study met its broad success gate and no favorable
factorial interaction interval excluded zero. Mixed and adverse findings are
retained. The repository is intended for methodological audit and
reproducibility, not as evidence that any named mechanism is uniformly better.

## Repository layout

| Path | Contents |
|---|---|
| `model_code/` | Complete executable code for C3R, DARC, CARE, MOSAIC, and DASH-V |
| `data/` | Data provenance, schema, download sources, and expected hashes |
| `evidence/` | Consolidated tables, selected figures, and compact returned-result summaries |
| `paper/` | Flat LaTeX manuscript and supplement sources with vector figures |
| `docs/` | Reproduction, study, disclosure, and release guidance |
| `scripts/static_validate.py` | Syntax, JSON, manifest, and large-file checks only |

## Studies

| Study | Task | Primary score | Mechanism family |
|---|---|---|---|
| C3R | Return ranking | Date-level RankIC | Residual expert exchange and coalition-risk routing |
| DARC | Return ranking | Date-level RankIC | Disagreement arbitration and advantage gating |
| CARE | Conditional quantiles | Mean pinball loss | Analog memories and tail-sensitive routing |
| MOSAIC | Variance proxy | Mean QLIKE | Downside and market-stress correction branches |
| DASH-V | Variance proxy | Mean QLIKE | Delayed hierarchical pooling and multiple memories |

## Running a study

Each study keeps its original `python main.py` entry point and its own pinned
configuration. Dependencies are never installed automatically.

1. Read `data/README.md` and the selected study's `data/DATASETS.json`.
2. Place the three expected `.npz` panels in `model_code/<model>/data/`.
3. Create an environment and manually install that study's `requirements.txt`.
4. Change into the study directory and run `python main.py`.
5. Preserve the complete returned archive, including failed and adverse output.

Do not tune a mechanism against the preserved test panels and then describe
the same panels as untouched confirmation evidence.

## Static validation

Run `python scripts/static_validate.py`. It parses Python and JSON, verifies the
release manifest, and checks that large datasets, model weights, raw logs, and
archives were not committed. It does not fit a model or evaluate forecasts.

## Data and licensing

The repository does not redistribute KDD17 or ACL18 market arrays. Their source
links and expected hashes are recorded under `data/` and each study. The MIT
license applies to original code and documentation in this release, not to
third-party datasets or external implementations.

## Citation

See `CITATION.cff`. Replace the contributor entity with the final author list
before making a public archival release.
