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

from .aggregate import run_aggregation, stack_records
from .config import Config
from .data import FEATURE_NAMES, Panel
from .figures import make_figures
from .metrics import evaluate_records, holm, paired_block_bootstrap
from .models import build_model
from .train import deterministic_prediction, fit_glm, fit_model, seed_everything, tensors


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


def _json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=True,
                               default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + "\n",
                    encoding="utf-8")


def _verify_panel(panel: Panel, cfg: Config) -> dict:
    splits = panel.split_indices
    names = ("fit", "core_valid", "valid", "test")
    for name in names:
        if not len(splits[name]) or not np.all(np.diff(splits[name]) > 0):
            raise ValueError(f"{panel.spec.name}: empty or unordered {name} split")
    for earlier, later in zip(names, names[1:]):
        if splits[earlier][-1] >= splits[later][0]:
            raise ValueError(f"{panel.spec.name}: split origins overlap")
    if splits["fit"][-1] + cfg.horizon >= splits["core_valid"][0]:
        raise ValueError(f"{panel.spec.name}: fit labels not mature at checkpoint date")
    return {"target": "mean of next five squared adjusted-close daily log returns times 10000",
            "forecast_origin": "after current close; all features end at or before origin",
            "aggregation_update_clock": "origin o updates states only when o+horizon <= current origin",
            "test_universe": "past-feature eligibility only, independent of forward-label presence",
            "feature_names": FEATURE_NAMES,
            "splits": {name: {"dates": int(len(splits[name])),
                               "first": str(panel.dates[splits[name][0]]),
                               "last": str(panel.dates[splits[name][-1]])} for name in names},
            "feature_mean_from_fit": panel.feature_mean.tolist(),
            "feature_std_from_fit": panel.feature_std.tolist()}


@torch.no_grad()
def _forecast(name: str, model, fitted, panel: Panel, split: str, device) -> list[dict]:
    if model is not None:
        model.eval()
    records = []
    dates = panel.split_indices[split]
    for position, t in enumerate(dates, 1):
        batch = panel.batch(int(t))
        if batch is None:
            continue
        if model is None:
            prediction = deterministic_prediction(name, batch, fitted)
        else:
            x, seq, anchor, own, market, _ = tensors(batch, device)
            log_mu, _ = model(x, seq, anchor, own, market)
            prediction = np.exp(np.clip(log_mu.detach().cpu().numpy(), -8.0, 8.0))
        if prediction.shape != (len(batch.assets),) or not np.isfinite(prediction).all() or np.any(prediction <= 0):
            raise RuntimeError(f"{panel.spec.name}/{name}/{batch.date}: invalid positive forecast")
        records.append({"date": batch.date, "date_index": int(t), "assets": batch.assets,
                        "target": batch.y, "forecast": prediction,
                        "stress": float(batch.stress[0])})
        if position == 1 or position % 100 == 0 or position == len(dates):
            print(f"  {name} {split}: {position}/{len(dates)} origins", flush=True)
    return records


def _archive_forecasts(records: list[dict], path: Path) -> None:
    offsets = np.r_[0, np.cumsum([len(row["assets"]) for row in records])]
    np.savez_compressed(path, dates=np.asarray([row["date"] for row in records]),
                        date_indices=np.asarray([row["date_index"] for row in records]),
                        offsets=offsets,
                        assets=np.concatenate([row["assets"] for row in records]),
                        target=np.concatenate([row["target"] for row in records]),
                        forecast=np.concatenate([row["forecast"] for row in records]),
                        past_stress=np.asarray([row["stress"] for row in records]))


def _dataset(panel: Panel, cfg: Config, device, output: Path, run_dir: Path,
             status: dict, started: float) -> list[dict]:
    base_names = [*cfg.deterministic_controls, *cfg.neural_controls]
    all_names = [*base_names, *cfg.aggregation_controls, *cfg.factorial_cells]
    metrics_all, completed, budget = [], {}, []
    base_records = {split: {} for split in ("valid", "test")}
    threshold = float(np.nanmedian(panel.stress[panel.split_indices["fit"]]))
    _json(output / "preflight.json", {**_verify_panel(panel, cfg),
                                      "high_stress_threshold_from_fit": threshold})

    for ordinal, name in enumerate(base_names, 1):
        print(f"\n[{panel.spec.name} base {ordinal}/{len(base_names)}] {name}; "
              f"elapsed={(time.time()-started)/3600:.2f}h", flush=True)
        model_dir = output / "models" / name
        model_dir.mkdir(parents=True, exist_ok=True)
        model, fitted = None, None
        neural = name in cfg.neural_controls
        if neural:
            status["attempted_neural_fits"] += 1
        try:
            if neural:
                seed_everything(cfg.seed)
                model = build_model(name, len(FEATURE_NAMES), cfg.hidden_size, cfg.lookback).to(device)
                fit_model(model, panel, cfg, device, model_dir, f"{panel.spec.name}/{name}")
                status["completed_neural_fits"] += 1
            elif name not in ("persistence", "ewma"):
                fitted = fit_glm(name, panel, model_dir)
            scope = "common_protocol_neural_control" if neural else "daily_bar_statistical_control"
            budget.append({"dataset": panel.spec.name, "model": name, "scope": scope,
                           "neural_fit": neural,
                           "parameters": int(sum(p.numel() for p in model.parameters()))
                           if model is not None else len(fitted.coefficients) if fitted is not None else 0})
            for split in ("valid", "test"):
                records = _forecast(name, model, fitted, panel, split, device)
                _archive_forecasts(records, model_dir / f"{split}_forecasts.npz")
                metrics, daily = evaluate_records(records, threshold)
                _json(model_dir / f"{split}_metrics.json", metrics)
                daily.to_csv(model_dir / f"{split}_daily.csv", index=False)
                metrics_all.append({"dataset": panel.spec.name, "split": split,
                                    "model": name, "scope": scope, **metrics})
                completed[(split, name)] = daily
                base_records[split][name] = records
        except Exception as exc:
            failure = {"dataset": panel.spec.name, "model": name, "stage": "fit_or_forecast",
                       "error": repr(exc), "traceback": traceback.format_exc()}
            status["failures"].append(failure)
            _json(model_dir / "FAILED.json", failure)
            print(f"FAILED {panel.spec.name}/{name}: {exc}", flush=True)
        finally:
            _json(run_dir / "STATUS.json", status)
            if model is not None:
                del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if any(name not in base_records["valid"] or name not in base_records["test"]
           for name in base_names):
        raise RuntimeError(f"{panel.spec.name}: incomplete base forecast matrix; partial evidence retained")

    print(f"\n[{panel.spec.name}] chronological delayed-feedback replay", flush=True)
    valid_rows = stack_records(base_records["valid"], base_names)
    test_rows = stack_records(base_records["test"], base_names)
    aggregation, diagnostics, fit = run_aggregation(
        valid_rows, test_rows, len(panel.symbols), base_names, cfg)
    aggregate_dir = output / "aggregation"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    _json(aggregate_dir / "fit.json", fit)
    if not fit["static_optimizer"]["converged"]:
        print(f"WARNING [{panel.spec.name}/static_qlike]: "
              f"{fit['static_optimizer']['message']}", flush=True)
    pd.DataFrame(diagnostics).to_csv(aggregate_dir / "module_diagnostics.csv", index=False)
    for name, records in aggregation.items():
        model_dir = output / "models" / name
        model_dir.mkdir(parents=True, exist_ok=True)
        _archive_forecasts(records, model_dir / "test_forecasts.npz")
        metrics, daily = evaluate_records(records, threshold)
        _json(model_dir / "test_metrics.json", metrics)
        daily.to_csv(model_dir / "test_daily.csv", index=False)
        scope = "dashv_factorial" if name in cfg.factorial_cells else "matched_aggregation_control"
        metrics_all.append({"dataset": panel.spec.name, "split": "test",
                            "model": name, "scope": scope, **metrics})
        completed[("test", name)] = daily
        budget.append({"dataset": panel.spec.name, "model": name, "scope": scope,
                       "neural_fit": False, "parameters": 0})

    validation_rows = [row for row in metrics_all if row["split"] == "valid"]
    validation_best = min(validation_rows, key=lambda row: row["qlike"])["model"]
    _json(output / "validation_selected_raw_base.json", {"model": validation_best,
                                                          "selection_split": "valid"})
    pd.DataFrame(budget).to_csv(output / "model_budget.csv", index=False)
    pd.DataFrame(metrics_all).to_csv(output / "metrics.csv", index=False)

    paired_rows = []
    reference = completed[("test", "dashv_full")]
    for name in all_names:
        if name == "dashv_full":
            continue
        paired_rows.append({"dataset": panel.spec.name, "split": "test", "full_vs": name,
                            "registered_main": name in (validation_best, "static_qlike",
                                                         "dashv_hier", "dashv_multi"),
                            **paired_block_bootstrap(reference, completed[("test", name)],
                                                     cfg.bootstrap_draws, cfg.seed,
                                                     cfg.bootstrap_block_length)})
    paired = holm(pd.DataFrame(paired_rows)).rename(columns={"p_holm": "p_holm_all_models"})
    paired.to_csv(output / "paired_qlike_inference.csv", index=False)

    factorial = None
    for name in cfg.factorial_cells:
        part = completed[("test", name)][["date", "qlike"]].rename(columns={"qlike": name})
        factorial = part if factorial is None else factorial.merge(part, on="date")
    factorial["interaction"] = (factorial.dashv_full - factorial.dashv_hier
                                - factorial.dashv_multi + factorial.dashv_base)
    factorial.to_csv(output / "factorial_daily.csv", index=False)
    interaction = factorial[["date", "interaction"]].rename(columns={"interaction": "qlike"})
    zero = interaction.assign(qlike=0.0)
    interaction_result = paired_block_bootstrap(
        interaction, zero, cfg.bootstrap_draws, cfg.seed, cfg.bootstrap_block_length)
    _json(output / "factorial_interaction_inference.json",
          {"contrast": "loss(full)-loss(hier)-loss(multi)+loss(base)",
           "negative_contrast": "complementary loss reduction",
           "mean_contrast": -float(interaction_result["mean_improvement"]),
           "contrast_ci_low": -float(interaction_result["ci_high"]),
           "contrast_ci_high": -float(interaction_result["ci_low"]),
           **interaction_result})
    main_family = paired.loc[paired.registered_main].copy()
    main_family["hypothesis"] = "full_vs_method"
    factorial_row = {column: np.nan for column in main_family.columns}
    factorial_row.update({"dataset": panel.spec.name, "split": "test",
                          "full_vs": "factorial_interaction", "registered_main": True,
                          "hypothesis": "pooling_by_memory_interaction",
                          **interaction_result})
    main_family = pd.concat([main_family, pd.DataFrame([factorial_row])], ignore_index=True)
    if "p_holm_all_models" in main_family:
        main_family = main_family.drop(columns=["p_holm_all_models"])
    main_family = holm(main_family).rename(columns={"p_holm": "p_holm_registered"})
    main_family.to_csv(output / "registered_main_inference.csv", index=False)
    make_figures(panel.spec.name, pd.DataFrame(diagnostics), factorial,
                 completed, base_names, fit, run_dir / "figures", cfg.horizon)
    return metrics_all


def run(root: Path, cfg: Config, device_name: str = "auto") -> tuple[Path, bool]:
    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if device_name == "auto" else torch.device(device_name))
    run_id = "RUN-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-DASHV"
    run_dir = root / "log" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "figures").mkdir()
    started = time.time()
    status = {"run_id": run_id,
              "registered_neural_fits": len(cfg.datasets) * len(cfg.neural_controls),
              "attempted_neural_fits": 0, "completed_neural_fits": 0,
              "completed_datasets": 0, "formal_experiment_completed": False, "failures": []}
    original_out, original_err = sys.stdout, sys.stderr
    log = (run_dir / "console.log").open("w", encoding="utf-8", buffering=1)
    sys.stdout, sys.stderr = _Tee(original_out, log), _Tee(original_err, log)
    _json(run_dir / "resolved_config.json", cfg.to_dict())
    source_files = [root / "main.py", root / "config.json", root / "requirements.txt",
                    *sorted((root / "src").rglob("*.py")), *sorted((root / "research").rglob("*"))]
    _json(run_dir / "SOURCE_HASHES.json", [{"file": p.relative_to(root).as_posix(), "sha256": _sha256(p)}
                                           for p in source_files if p.is_file()])
    metrics_all = []
    try:
        if device.type != "cuda":
            print("WARNING: CUDA not selected; run time can be much longer", flush=True)
        print(f"{run_id}: 3 public development panels, 5 statistical controls, "
              f"5 neural controls, 4 aggregation controls, 4 factorial cells, "
              f"{status['registered_neural_fits']} neural fits", flush=True)
        print("No automatic dependency installation, sweep, retry, or wall-clock kill", flush=True)
        manifest = json.loads((root / "data" / "DATASETS.json").read_text(encoding="utf-8"))
        checks = []
        for item in manifest["datasets"]:
            path = root / "data" / item["file"]
            if not path.is_file() or _sha256(path) != item["sha256"]:
                raise ValueError(f"Missing or altered bundled panel: {item['file']}")
            checks.append({"file": item["file"], "sha256": item["sha256"], "bytes": path.stat().st_size})
        _json(run_dir / "DATA_CHECKS.json", checks)
        for ordinal, spec in enumerate(cfg.datasets, 1):
            try:
                print(f"\nPreparing panel {ordinal}/{len(cfg.datasets)}: {spec.name}", flush=True)
                panel = Panel(root, spec, cfg)
                output = run_dir / "datasets" / spec.name
                output.mkdir(parents=True, exist_ok=True)
                metrics_all.extend(_dataset(panel, cfg, device, output, run_dir, status, started))
                pd.DataFrame(metrics_all).to_csv(run_dir / "all_metrics.csv", index=False)
                status["completed_datasets"] += 1
                _json(run_dir / "STATUS.json", status)
                del panel
            except Exception as exc:
                failure = {"dataset": spec.name, "stage": "dataset",
                           "error": repr(exc), "traceback": traceback.format_exc()}
                status["failures"].append(failure)
                print(f"DATASET FAILED {spec.name}: {exc}", flush=True)
        files = sorted((run_dir / "datasets").glob("*/paired_qlike_inference.csv"))
        if files:
            combined = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
            combined.to_csv(run_dir / "all_paired_qlike_inference.csv", index=False)
        main_files = sorted((run_dir / "datasets").glob("*/registered_main_inference.csv"))
        if main_files:
            main = pd.concat([pd.read_csv(path) for path in main_files], ignore_index=True)
            main = main.rename(columns={"p_holm_registered": "p_holm_dataset"})
            main = holm(main).rename(columns={"p_holm": "p_holm_all_panels"})
            main.to_csv(run_dir / "all_registered_main_inference.csv", index=False)
        figures = [{"file": p.relative_to(run_dir).as_posix(), "sha256": _sha256(p),
                    "bytes": p.stat().st_size} for p in sorted((run_dir / "figures").glob("*.pdf"))]
        _json(run_dir / "FIGURE_MANIFEST.json", {"expected": 24, "figures": figures})
        if len(figures) != 24:
            status["failures"].append({"stage": "figures", "error": f"Expected 24, got {len(figures)}"})
        status["formal_experiment_completed"] = (
            status["completed_neural_fits"] == status["registered_neural_fits"]
            and status["completed_datasets"] == len(cfg.datasets) and not status["failures"])
    except Exception as exc:
        status["failures"].append({"stage": "pipeline", "error": repr(exc),
                                   "traceback": traceback.format_exc()})
        print(f"PIPELINE FAILED: {exc}", flush=True)
    finally:
        status["elapsed_hours"] = (time.time() - started) / 3600
        status["finished_utc"] = datetime.now(timezone.utc).isoformat()
        _json(run_dir / "STATUS.json", status)
        _json(run_dir / "environment.json", {"python": platform.python_version(),
                                              "platform": platform.platform(),
                                              "torch": torch.__version__,
                                              "cuda_available": torch.cuda.is_available(),
                                              "cuda_device": torch.cuda.get_device_name(0)
                                              if torch.cuda.is_available() else None})
        log.flush()
        sys.stdout, sys.stderr = original_out, original_err
        log.close()
        _json(run_dir / "FILE_HASHES.json", [{"file": p.relative_to(run_dir).as_posix(),
                                              "bytes": p.stat().st_size, "sha256": _sha256(p)}
                                             for p in sorted(run_dir.rglob("*"))
                                             if p.is_file() and p.name != "FILE_HASHES.json"])
        archive = Path(shutil.make_archive(str(root / "log" / run_id), "zip", root_dir=run_dir))
        print(f"Evidence bundle: {archive}", flush=True)
        print(f"formal_experiment_completed={status['formal_experiment_completed']}", flush=True)
    return archive, bool(status["formal_experiment_completed"])
