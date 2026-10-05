from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import ExperimentConfig
from .council import READOUTS, run_council
from .data import PanelData
from .metrics import evaluate_records, paired_block_bootstrap
from .models import build_baseline
from .train import fit_forecaster, seed_everything
from .visuals import make_figures


class _Tee:
    def __init__(self, original, log):
        self.original, self.log = original, log

    def write(self, value):
        self.original.write(value)
        self.log.write(value)
        return len(value)

    def flush(self):
        self.original.flush()
        self.log.flush()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=True, default=lambda x: x.item()
                               if isinstance(x, np.generic) else str(x)) + "\n", encoding="utf-8")


def _model(name: str, panel: PanelData, cfg: ExperimentConfig):
    return build_baseline(name, input_size=panel.input_size, hidden=cfg.hidden_size,
                          market_size=6, retrieval_size=panel.retrieval_size,
                          lookback=cfg.lookback, num_experts=cfg.num_experts)


def _verify_panel(panel: PanelData, cfg: ExperimentConfig) -> dict:
    splits = panel.split_indices
    for name in ("fit", "core_valid", "router", "valid", "test"):
        if len(splits[name]) == 0 or not np.all(np.diff(splits[name]) > 0):
            raise ValueError(f"{panel.spec.name}: empty or unordered {name} split")
    sequence = ("fit", "core_valid", "router", "valid", "test")
    for earlier, later in zip(sequence, sequence[1:]):
        if int(splits[earlier][-1]) >= int(splits[later][0]):
            raise ValueError(f"{panel.spec.name}: overlapping {earlier}/{later} origins")
    if int(splits["fit"][-1]) + cfg.horizon + 1 >= int(splits["core_valid"][0]):
        raise ValueError(f"{panel.spec.name}: fit labels are not mature before checkpoint validation")
    if int(splits["core_valid"][-1]) + cfg.horizon + 1 >= int(splits["router"][0]):
        raise ValueError(f"{panel.spec.name}: checkpoint labels are not mature before calibration")
    if len(splits["router"]) * cfg.calibration_fraction < cfg.evidence_min_dates:
        raise ValueError(f"{panel.spec.name}: insufficient router calibration origins")
    return {"splits": {key: {"count": int(len(value)), "first": str(panel.dates[value[0]]),
                              "last": str(panel.dates[value[-1]])} for key, value in splits.items()},
            "feature_clock": "input features and retrieval stop at the decision origin",
            "feedback_clock": f"recorded outcomes mature after {cfg.horizon + 1} index steps",
            "test_universe": "available feature history only; target presence is not a filter"}


@torch.no_grad()
def _forecast(model, panel: PanelData, split: str, device, streams: dict, column: int, count: int) -> None:
    model.eval()
    dates = panel.split_indices[split]
    for position, t in enumerate(dates, 1):
        batch = panel.batch(int(t), require_target=False)
        if batch is None:
            continue
        x = torch.as_tensor(batch.x, device=device)
        market = torch.as_tensor(batch.market, device=device)
        retrieval = torch.as_tensor(batch.retrieval, device=device)
        complementarity = torch.as_tensor(batch.complementarity, device=device)
        pred, _ = model(x, market, retrieval, complementarity)
        values = pred.detach().cpu().numpy().astype(np.float32)
        key = (split, int(t))
        if key not in streams:
            streams[key] = {"split": split, "date": batch.date, "date_index": int(t),
                            "assets": batch.assets, "target": batch.y, "raw_return": batch.raw_return,
                            "market_volatility": float(batch.market[0, 1]),
                            "pred_matrix": np.full((len(batch.assets), count), np.nan, dtype=np.float32)}
        if not np.array_equal(streams[key]["assets"], batch.assets) or len(values) != len(batch.assets):
            raise RuntimeError(f"Forecast asset alignment changed: {split}/{batch.date}")
        streams[key]["pred_matrix"][:, column] = values
        if position == 1 or position % 100 == 0 or position == len(dates):
            print(f"  forecast {split}: {position}/{len(dates)} origins", flush=True)


def _archive_streams(records: list[dict], path: Path, column_names: list[str]) -> None:
    offsets = np.r_[0, np.cumsum([len(r["assets"]) for r in records])]
    np.savez_compressed(path, dates=np.asarray([r["date"] for r in records]),
                        date_indices=np.asarray([r["date_index"] for r in records]),
                        splits=np.asarray([r["split"] for r in records]), offsets=offsets,
                        market_volatility=np.asarray([r["market_volatility"] for r in records]),
                        model_columns=np.asarray(column_names),
                        assets=np.concatenate([r["assets"] for r in records]),
                        target=np.concatenate([r["target"] for r in records]),
                        raw_return=np.concatenate([r["raw_return"] for r in records]),
                        predictions=np.concatenate([r["pred_matrix"] for r in records]))


def _archive_readouts(readouts: dict[str, list[dict]], path: Path) -> None:
    reference = readouts["darc_full"]
    offsets = np.r_[0, np.cumsum([len(r["assets"]) for r in reference])]
    for name, rows in readouts.items():
        if len(rows) != len(reference) or any(a["date_index"] != b["date_index"] or
                                              not np.array_equal(a["assets"], b["assets"])
                                              for a, b in zip(rows, reference)):
            raise ValueError(f"Readout date/asset alignment failed: {name}")
    np.savez_compressed(path, dates=np.asarray([r["date"] for r in reference]),
                        date_indices=np.asarray([r["date_index"] for r in reference]),
                        splits=np.asarray([r["split"] for r in reference]), offsets=offsets,
                        assets=np.concatenate([r["assets"] for r in reference]),
                        score_columns=np.asarray(list(readouts)),
                        scores=np.column_stack([np.concatenate([r["pred"] for r in readouts[name]])
                                                for name in readouts]))


def _dataset(panel: PanelData, cfg: ExperimentConfig, device, output: Path, status: dict,
             started: float, dataset_no: int) -> tuple[list[dict], dict]:
    streams: dict = {}
    model_rows = []
    total = len(cfg.datasets) * len(cfg.baseline_models)
    for model_no, name in enumerate(cfg.baseline_models, 1):
        ordinal = (dataset_no - 1) * len(cfg.baseline_models) + model_no
        print(f"\n[{ordinal}/{total}] {panel.spec.name}/{name}; elapsed={(time.time()-started)/3600:.2f} h", flush=True)
        fit_dir = output / "models" / name
        fit_dir.mkdir(parents=True, exist_ok=True)
        status["attempted_fits"] += 1
        try:
            seed_everything(cfg.seed)
            model = _model(name, panel, cfg).to(device)
            parameter_count = sum(p.numel() for p in model.parameters())
            fit_forecaster(model, panel, cfg, device, fit_dir, f"{panel.spec.name}/{name}")
            for split in ("router", "valid", "test"):
                _forecast(model, panel, split, device, streams, model_no - 1, len(cfg.baseline_models))
            for split in ("valid", "test"):
                available = [rec for rec in streams.values() if rec["split"] == split
                             and np.isfinite(rec["pred_matrix"][:, model_no - 1]).all()]
                available.sort(key=lambda rec: rec["date_index"])
                if available:
                    one_model = [{**{k: rec[k] for k in ("date", "date_index", "assets", "target", "raw_return")},
                                  "pred": rec["pred_matrix"][:, model_no - 1]} for rec in available]
                    own_metrics, own_daily = evaluate_records(one_model, cfg.transaction_cost_bps, cfg.horizon)
                    _write_json(fit_dir / f"{split}_metrics.json", own_metrics)
                    own_daily.to_csv(fit_dir / f"{split}_daily.csv", index=False)
            status["completed_fits"] += 1
            seconds = json.loads((fit_dir / "timing.json").read_text(encoding="utf-8"))["training_seconds"]
            model_rows.append({"dataset": panel.spec.name, "model": name,
                               "model_scope": "common_protocol_adaptation",
                               "parameters": parameter_count, "training_seconds": seconds})
        except Exception as exc:
            failure = {"dataset": panel.spec.name, "model": name, "stage": "fit_or_forecast",
                       "error": repr(exc), "traceback": traceback.format_exc()}
            status["failures"].append(failure)
            _write_json(fit_dir / "FAILED.json", failure)
            print(f"FAILED {panel.spec.name}/{name}: {exc}", flush=True)
        finally:
            if streams:
                available_records = sorted(streams.values(), key=lambda rec: rec["date_index"])
                _archive_streams(available_records, output / "frozen_forecast_streams_partial.npz",
                                 cfg.baseline_models)
                _write_json(output / "forecast_stream_status.json", {
                    "date_count": len(available_records),
                    "complete_column_date_counts": {label: int(sum(np.isfinite(rec["pred_matrix"][:, j]).all()
                                                          for rec in available_records))
                                                    for j, label in enumerate(cfg.baseline_models)}})
            _write_json(output.parent.parent / "STATUS.json", status)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if len(model_rows) != len(cfg.baseline_models):
        raise RuntimeError(f"{panel.spec.name}: one or more forecaster fits failed; readouts cannot be scored")
    records = sorted(streams.values(), key=lambda r: r["date_index"])
    _archive_streams(records, output / "frozen_forecast_streams.npz", cfg.baseline_models)
    readouts, diagnostics, matured = run_council(records, cfg)
    _archive_readouts(readouts, output / "frozen_readout_scores.npz")
    pd.DataFrame(diagnostics).to_csv(output / "council_diagnostics.csv", index=False)
    pd.DataFrame(matured).to_csv(output / "matured_advantage_ledger.csv", index=False)
    pd.DataFrame(model_rows).to_csv(output / "model_budget.csv", index=False)
    results, daily_tables = [], {}
    for split in ("valid", "test"):
        for name in cfg.baseline_models:
            j = cfg.baseline_models.index(name)
            rows = [{**{k: rec[k] for k in ("date", "date_index", "assets", "target", "raw_return")},
                     "pred": rec["pred_matrix"][:, j]} for rec in records if rec["split"] == split]
            if not rows:
                raise RuntimeError(f"{panel.spec.name}/{split}: no {name} records")
            metrics, daily = evaluate_records(rows, cfg.transaction_cost_bps, cfg.horizon)
            daily["dataset"], daily["split"], daily["model"] = panel.spec.name, split, name
            daily_tables[(split, name)] = daily
            results.append({"dataset": panel.spec.name, "split": split, "model": name,
                            "model_scope": "common_protocol_adaptation", **metrics})
        for name in READOUTS:
            rows = [rec for rec in readouts[name] if rec["split"] == split]
            if not rows:
                raise RuntimeError(f"{panel.spec.name}/{split}: no {name} records")
            metrics, daily = evaluate_records(rows, cfg.transaction_cost_bps, cfg.horizon,
                                              forecast_error_available=False)
            daily["dataset"], daily["split"], daily["model"] = panel.spec.name, split, name
            daily_tables[(split, name)] = daily
            results.append({"dataset": panel.spec.name, "split": split, "model": name,
                            "model_scope": "frozen_readout", **metrics})
    pd.concat(daily_tables.values(), ignore_index=True).to_csv(output / "daily_metrics.csv", index=False)
    pd.DataFrame(results).to_csv(output / "metrics.csv", index=False)
    # Paired dates are the inferential unit. All comparisons, including adverse ones, are kept.
    inference = []
    for split in ("valid", "test"):
        reference = daily_tables[(split, "darc_full")]
        for name in [*cfg.baseline_models, *[n for n in READOUTS if n != "darc_full"]]:
            inference.append({"dataset": panel.spec.name, "split": split, "full_minus": name,
                              **paired_block_bootstrap(reference, daily_tables[(split, name)],
                                                       cfg.bootstrap_draws, cfg.seed, cfg.bootstrap_block_length)})
    inference_frame = pd.DataFrame(inference)
    inference_frame["p_holm"] = np.nan
    for split, group in inference_frame.groupby("split"):
        ordered = group["p_one_sided"].dropna().sort_values().index
        running = 0.0
        for rank, idx in enumerate(ordered):
            running = max(running, min(1.0, float(inference_frame.loc[idx, "p_one_sided"])
                                       * (len(ordered) - rank)))
            inference_frame.loc[idx, "p_holm"] = running
    inference_frame.to_csv(output / "paired_rankic_inference.csv", index=False)
    factorial = None
    for name in ("global_always", "pairwise_always", "global_gated", "darc_full"):
        part = daily_tables[("test", name)][["date", "rank_ic"]].rename(columns={"rank_ic": name})
        factorial = part if factorial is None else factorial.merge(part, on="date")
    factorial["interaction"] = (factorial["darc_full"] - factorial["pairwise_always"]
                                 - factorial["global_gated"] + factorial["global_always"])
    factorial.to_csv(output / "factorial_daily.csv", index=False)
    contrast = factorial[["date", "interaction"]].rename(columns={"interaction": "rank_ic"})
    zero = contrast.assign(rank_ic=0.0)
    _write_json(output / "factorial_interaction_inference.json",
                paired_block_bootstrap(contrast, zero, cfg.bootstrap_draws, cfg.seed,
                                       cfg.bootstrap_block_length))
    _write_json(output / "split_summary.json", {key: int(len(value)) for key, value in panel.split_indices.items()})
    return results, {"daily": daily_tables, "diagnostics": diagnostics, "factorial": factorial,
                     "inference": inference, "model_rows": model_rows}


def run(root: Path, cfg: ExperimentConfig, device_name: str = "auto") -> tuple[Path, bool]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device_name == "auto" else torch.device(device_name)
    run_id = "RUN-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-DARC"
    run_dir = root / "log" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "figures").mkdir()
    started = time.time()
    status = {"run_id": run_id, "registered_fits": len(cfg.datasets) * len(cfg.baseline_models),
              "attempted_fits": 0, "completed_fits": 0, "completed_datasets": 0,
              "formal_experiment_completed": False, "failures": []}
    original_out, original_err = sys.stdout, sys.stderr
    log = (run_dir / "console.log").open("w", encoding="utf-8", buffering=1)
    sys.stdout, sys.stderr = _Tee(original_out, log), _Tee(original_err, log)
    _write_json(run_dir / "resolved_config.json", cfg.to_dict())
    sources = [root / "main.py", root / "config.json", root / "requirements.txt",
               *sorted((root / "src").rglob("*.py")), *sorted((root / "research").rglob("*"))]
    _write_json(run_dir / "SOURCE_HASHES.json", [{"path": p.relative_to(root).as_posix(), "sha256": _sha256(p)}
                                                 for p in sources if p.is_file()])
    metrics = []
    try:
        if device.type != "cuda":
            print("WARNING: CUDA not selected; runtime may be substantially longer", flush=True)
        print(f"{run_id}: 3 datasets, 6 adapted forecasters, 18 fits, 8 frozen readouts per dataset", flush=True)
        print("No automatic installation, search, retry or wall-clock kill", flush=True)
        data_manifest = json.loads((root / "data" / "DATASETS.json").read_text(encoding="utf-8"))
        data_checks = []
        for item in data_manifest["datasets"]:
            data_path = root / "data" / item["file"]
            if not data_path.is_file():
                raise FileNotFoundError(f"Missing bundled panel: {data_path}")
            actual = _sha256(data_path)
            if actual != item["sha256"]:
                raise ValueError(f"Bundled panel checksum mismatch: {item['file']}")
            data_checks.append({"file": item["file"], "sha256": actual, "bytes": data_path.stat().st_size})
        _write_json(run_dir / "DATA_CHECKS.json", data_checks)
        for index, spec in enumerate(cfg.datasets, 1):
            path = root / spec.file
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"Missing bundled panel: {path}")
            print(f"\nPreparing {index}/{len(cfg.datasets)}: {spec.name}", flush=True)
            panel = PanelData(root, spec, cfg)
            output = run_dir / "datasets" / spec.name
            output.mkdir(parents=True, exist_ok=True)
            _write_json(output / "preflight.json", _verify_panel(panel, cfg))
            rows, artifacts = _dataset(panel, cfg, device, output, status, started, index)
            metrics.extend(rows)
            pd.DataFrame(metrics).to_csv(run_dir / "all_metrics.csv", index=False)
            make_figures(spec.name, artifacts, run_dir / "figures", cfg)
            status["completed_datasets"] += 1
            _write_json(run_dir / "STATUS.json", status)
            del panel
        manifest = []
        for path in sorted((run_dir / "figures").glob("*.pdf")):
            manifest.append({"file": path.relative_to(run_dir).as_posix(), "sha256": _sha256(path),
                             "bytes": path.stat().st_size})
        _write_json(run_dir / "FIGURE_MANIFEST.json", {"scope": "module effects and factorial ablations",
                                                        "expected": 24, "figures": manifest})
        if len(manifest) != 24:
            raise RuntimeError(f"Expected 24 module figures, found {len(manifest)}")
        status["formal_experiment_completed"] = (status["completed_fits"] == status["registered_fits"]
                                                  and status["completed_datasets"] == len(cfg.datasets)
                                                  and not status["failures"])
    except Exception as exc:
        status["failures"].append({"stage": "pipeline", "error": repr(exc), "traceback": traceback.format_exc()})
        print(f"PIPELINE FAILED: {exc}", flush=True)
    finally:
        status["elapsed_hours"] = (time.time() - started) / 3600
        status["finished_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(run_dir / "STATUS.json", status)
        _write_json(run_dir / "environment.json", {"python": platform.python_version(),
                                                   "platform": platform.platform(), "torch": torch.__version__,
                                                   "cuda_available": torch.cuda.is_available(),
                                                   "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
        log.flush()
        sys.stdout, sys.stderr = original_out, original_err
        log.close()
        hashes = [{"path": p.relative_to(run_dir).as_posix(), "bytes": p.stat().st_size, "sha256": _sha256(p)}
                  for p in sorted(run_dir.rglob("*")) if p.is_file() and p.name != "FILE_HASHES.json"]
        _write_json(run_dir / "FILE_HASHES.json", hashes)
        archive = Path(shutil.make_archive(str(root / "log" / run_id), "zip", root_dir=run_dir))
        print(f"Evidence bundle: {archive}", flush=True)
        print(f"formal_experiment_completed={status['formal_experiment_completed']}", flush=True)
    return archive, bool(status["formal_experiment_completed"])
