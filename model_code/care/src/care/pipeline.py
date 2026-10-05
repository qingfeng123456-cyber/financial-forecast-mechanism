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
from .data import PanelData
from .metrics import evaluate_records, paired_block_bootstrap
from .models import build_model
from .train import fit_forecaster, seed_everything
from .visuals import make_figures


CELLS = ("care_single_plain", "care_dual_plain", "care_single_tail", "care_full")
STATISTICAL = ("historical", "market_analog")


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
    path.write_text(json.dumps(value, indent=2, allow_nan=True,
                               default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + "\n",
                    encoding="utf-8")


def _verify_panel(panel: PanelData, cfg: ExperimentConfig) -> dict:
    splits = panel.split_indices
    sequence = ("fit", "core_valid", "valid", "test")
    for name in sequence:
        if len(splits[name]) == 0 or not np.all(np.diff(splits[name]) > 0):
            raise ValueError(f"{panel.spec.name}: empty or unordered {name} split")
    for earlier, later in zip(sequence, sequence[1:]):
        if int(splits[earlier][-1]) >= int(splits[later][0]):
            raise ValueError(f"{panel.spec.name}: overlapping {earlier}/{later} origins")
    if int(splits["fit"][-1]) + cfg.horizon + 1 >= int(splits["core_valid"][0]):
        raise ValueError(f"{panel.spec.name}: fit labels not mature by checkpoint validation")
    return {"splits": {name: {"count": int(len(splits[name])),
                               "first": str(panel.dates[splits[name][0]]),
                               "last": str(panel.dates[splits[name][-1]])} for name in sequence},
            "target_scale_from_fit": panel.target_scale,
            "information_clock": "features at origin; retrieved outcomes end strictly before origin",
            "test_universe": "feature-history eligibility only; target presence does not select assets"}


def _statistical(name: str, batch, panel: PanelData) -> np.ndarray:
    memory = batch.retrieval
    first = panel.input_size
    size = panel.input_size + 3
    market_q = memory[:, first:first + 3]
    stock_q = memory[:, size + first:size + first + 3]
    quality = memory[:, 2 * size:]
    fallback = np.repeat(panel.training_quantiles[None, :], len(batch.assets), axis=0)
    if name == "market_analog":
        primary, available, alternate, alt_available = market_q, quality[:, 0] > 0, stock_q, quality[:, 1] > 0
    else:
        primary, available, alternate, alt_available = stock_q, quality[:, 1] > 0, market_q, quality[:, 0] > 0
    result = np.where(available[:, None], primary,
                      np.where(alt_available[:, None], alternate, fallback))
    return result * panel.target_scale


@torch.no_grad()
def _forecast(name: str, model, panel: PanelData, split: str, device) -> tuple[list[dict], list[dict]]:
    if model is not None:
        model.eval()
    records, diagnostics = [], []
    dates = panel.split_indices[split]
    for position, t in enumerate(dates, 1):
        batch = panel.batch(int(t), require_target=False)
        if batch is None:
            continue
        if model is None:
            prediction = _statistical(name, batch, panel)
        else:
            x = torch.as_tensor(batch.x, device=device)
            market = torch.as_tensor(batch.market, device=device)
            retrieval = torch.as_tensor(batch.retrieval, device=device)
            output, aux = model(x, market, retrieval)
            prediction = (output.detach().cpu().numpy() * panel.target_scale).astype(np.float32)
            if name in CELLS:
                gates = aux["gates"].detach().cpu().numpy()
                adoption = aux["adoption"].detach().cpu().numpy()
                shared = aux["shared_quantiles"].detach().cpu().numpy() * panel.target_scale
                quality = batch.retrieval[:, -9:]
                row = {"date": batch.date, "date_index": int(t), "split": split, "model": name,
                       "adoption_center": float(adoption[:, 0].mean()),
                       "adoption_lower": float(adoption[:, 1].mean()),
                       "adoption_upper": float(adoption[:, 2].mean()),
                       "mean_correction": float(np.abs(prediction - shared).mean()),
                       "market_coverage": float(quality[:, 0].mean()),
                       "stock_coverage": float(quality[:, 1].mean()),
                       "market_closeness": float(quality[:, 2].mean()),
                       "stock_closeness": float(quality[:, 3].mean()),
                       "market_width": float(quality[:, 4].mean()),
                       "stock_width": float(quality[:, 5].mean()),
                       "lower_tail_disagreement": float(quality[:, 6].mean()),
                       "median_disagreement": float(quality[:, 7].mean()),
                       "upper_tail_disagreement": float(quality[:, 8].mean())}
                for q, label in enumerate(("center", "lower", "upper")):
                    weights = gates[:, q].mean(axis=0)
                    row[f"entropy_{label}"] = float(-(weights * np.log(weights + 1e-8)).sum())
                    for expert in range(gates.shape[-1]):
                        row[f"gate_{label}_{expert}"] = float(weights[expert])
                diagnostics.append(row)
        if not np.isfinite(prediction).all() or prediction.shape != (len(batch.assets), 3):
            raise RuntimeError(f"{panel.spec.name}/{name}/{batch.date}: invalid quantile forecast")
        if np.any(prediction[:, 0] > prediction[:, 1]) or np.any(prediction[:, 1] > prediction[:, 2]):
            raise RuntimeError(f"{panel.spec.name}/{name}/{batch.date}: crossed quantiles")
        records.append({"date": batch.date, "date_index": int(t), "assets": batch.assets,
                        "raw_return": batch.raw_return, "pred": prediction})
        if position == 1 or position % 100 == 0 or position == len(dates):
            print(f"  {name} forecast {split}: {position}/{len(dates)} origins", flush=True)
    return records, diagnostics


def _archive_forecasts(records: list[dict], path: Path) -> None:
    offsets = np.r_[0, np.cumsum([len(row["assets"]) for row in records])]
    np.savez_compressed(path, dates=np.asarray([row["date"] for row in records]),
                        date_indices=np.asarray([row["date_index"] for row in records]),
                        offsets=offsets, assets=np.concatenate([row["assets"] for row in records]),
                        target=np.concatenate([row["raw_return"] for row in records]),
                        quantiles=np.concatenate([row["pred"] for row in records]))


def _dataset(panel: PanelData, cfg: ExperimentConfig, device, output: Path,
             status: dict, run_dir: Path, started: float) -> list[dict]:
    names = [*cfg.baseline_models, *CELLS]
    completed: dict[tuple[str, str], tuple[dict, pd.DataFrame]] = {}
    diagnostics = []
    budget = []
    for ordinal, name in enumerate(names, 1):
        print(f"\n[{panel.spec.name} {ordinal}/{len(names)}] {name}; elapsed={(time.time()-started)/3600:.2f} h",
              flush=True)
        model_dir = output / "models" / name
        model_dir.mkdir(parents=True, exist_ok=True)
        model = None
        trained = name not in STATISTICAL
        if trained:
            status["attempted_fits"] += 1
        try:
            if trained:
                seed_everything(cfg.seed)
                model = build_model(name, input_size=panel.input_size, hidden=cfg.hidden_size,
                                    retrieval_size=panel.retrieval_size, market_size=6,
                                    lookback=cfg.lookback, num_experts=cfg.num_experts).to(device)
                fit_forecaster(model, panel, cfg, device, model_dir, f"{panel.spec.name}/{name}")
                status["completed_fits"] += 1
            scope = ("empirical_analog_control" if name in STATISTICAL else
                     "proposed_factorial" if name in CELLS else "common_protocol_adaptation")
            budget.append({"dataset": panel.spec.name, "model": name,
                           "scope": scope,
                           "parameters": int(sum(p.numel() for p in model.parameters())) if model is not None else 0,
                           "trained": trained})
            for split in ("valid", "test"):
                records, rows = _forecast(name, model, panel, split, device)
                _archive_forecasts(records, model_dir / f"{split}_forecasts.npz")
                diagnostics.extend(rows)
                metrics, daily = evaluate_records(records)
                _json(model_dir / f"{split}_metrics.json", metrics)
                daily.to_csv(model_dir / f"{split}_daily.csv", index=False)
                completed[(split, name)] = (metrics, daily)
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
    pd.DataFrame(budget).to_csv(output / "model_budget.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(output / "module_diagnostics.csv", index=False)
    metric_rows, daily_rows = [], []
    for (split, name), (metrics, daily) in completed.items():
        scope = ("empirical_analog_control" if name in STATISTICAL else
                 "proposed_factorial" if name in CELLS else "common_protocol_adaptation")
        metric_rows.append({"dataset": panel.spec.name, "split": split, "model": name,
                            "scope": scope,
                            **metrics})
        daily_rows.append(daily.assign(dataset=panel.spec.name, split=split, model=name))
    pd.DataFrame(metric_rows).to_csv(output / "metrics.csv", index=False)
    if daily_rows:
        pd.concat(daily_rows, ignore_index=True).to_csv(output / "daily_metrics.csv", index=False)
    if all(("test", name) in completed for name in names) and all(("valid", name) in completed for name in names):
        inference = []
        for split in ("valid", "test"):
            reference = completed[(split, "care_full")][1]
            for name in names:
                if name == "care_full":
                    continue
                inference.append({"dataset": panel.spec.name, "split": split,
                                  "full_vs": name,
                                  **paired_block_bootstrap(reference, completed[(split, name)][1],
                                                           cfg.bootstrap_draws, cfg.seed,
                                                           cfg.bootstrap_block_length)})
        frame = pd.DataFrame(inference)
        frame["p_holm"] = np.nan
        for split, group in frame.groupby("split"):
            ordered = group["p_one_sided"].dropna().sort_values().index
            running = 0.0
            for rank, idx in enumerate(ordered):
                running = max(running, min(1.0, float(frame.loc[idx, "p_one_sided"]) *
                                           (len(ordered) - rank)))
                frame.loc[idx, "p_holm"] = running
        frame.to_csv(output / "paired_pinball_inference.csv", index=False)
        factorial = None
        for name in CELLS:
            part = completed[("test", name)][1][["date", "pinball"]].rename(columns={"pinball": name})
            factorial = part if factorial is None else factorial.merge(part, on="date")
        factorial["interaction"] = (factorial["care_full"] - factorial["care_dual_plain"]
                                    - factorial["care_single_tail"] + factorial["care_single_plain"])
        factorial.to_csv(output / "factorial_daily.csv", index=False)
        zero = factorial[["date", "interaction"]].rename(columns={"interaction": "pinball"}).assign(pinball=0.0)
        interaction = factorial[["date", "interaction"]].rename(columns={"interaction": "pinball"})
        _json(output / "factorial_interaction_inference.json",
              {"raw_interaction_definition": "loss_full - loss_dual_plain - loss_single_tail + loss_single_plain",
               "mean_improvement_definition": "negative raw interaction; positive favors beneficial synergy",
               **paired_block_bootstrap(interaction, zero, cfg.bootstrap_draws,
                                        cfg.seed, cfg.bootstrap_block_length)})
        make_figures(panel.spec.name, pd.DataFrame(diagnostics), factorial,
                     output.parent.parent / "figures", cfg)
    else:
        raise RuntimeError(f"{panel.spec.name}: incomplete forecast matrix; partial evidence retained")
    _json(output / "split_summary.json", {key: int(len(value)) for key, value in panel.split_indices.items()})
    return metric_rows


def run(root: Path, cfg: ExperimentConfig, device_name: str = "auto") -> tuple[Path, bool]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device_name == "auto" else torch.device(device_name)
    run_id = "RUN-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-CARE"
    run_dir = root / "log" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "figures").mkdir()
    started = time.time()
    status = {"run_id": run_id, "registered_fits": len(cfg.datasets) * (len(cfg.baseline_models) - len(STATISTICAL) + len(CELLS)),
              "attempted_fits": 0, "completed_fits": 0, "completed_datasets": 0,
              "formal_experiment_completed": False, "failures": []}
    original_out, original_err = sys.stdout, sys.stderr
    log = (run_dir / "console.log").open("w", encoding="utf-8", buffering=1)
    sys.stdout, sys.stderr = _Tee(original_out, log), _Tee(original_err, log)
    _json(run_dir / "resolved_config.json", cfg.to_dict())
    sources = [root / "main.py", root / "config.json", root / "requirements.txt",
               *sorted((root / "src").rglob("*.py")), *sorted((root / "research").rglob("*"))]
    _json(run_dir / "SOURCE_HASHES.json", [{"path": p.relative_to(root).as_posix(), "sha256": _sha256(p)}
                                           for p in sources if p.is_file()])
    metrics = []
    try:
        if device.type != "cuda":
            print("WARNING: CUDA not selected; runtime may be substantially longer", flush=True)
        print(f"{run_id}: 3 panels; 6 controls; 4 factorial cells; {status['registered_fits']} neural fits", flush=True)
        print("No automatic installation, hyperparameter search, retry, or wall-clock kill", flush=True)
        manifest = json.loads((root / "data" / "DATASETS.json").read_text(encoding="utf-8"))
        checks = []
        for item in manifest["datasets"]:
            path = root / "data" / item["file"]
            if not path.is_file() or _sha256(path) != item["sha256"]:
                raise ValueError(f"Bundled panel missing or checksum mismatch: {item['file']}")
            checks.append({"file": item["file"], "sha256": item["sha256"], "bytes": path.stat().st_size})
        _json(run_dir / "DATA_CHECKS.json", checks)
        for index, spec in enumerate(cfg.datasets, 1):
            try:
                print(f"\nPreparing {index}/{len(cfg.datasets)}: {spec.name}", flush=True)
                panel = PanelData(root, spec, cfg)
                output = run_dir / "datasets" / spec.name
                output.mkdir(parents=True, exist_ok=True)
                _json(output / "preflight.json", _verify_panel(panel, cfg))
                metrics.extend(_dataset(panel, cfg, device, output, status, run_dir, started))
                pd.DataFrame(metrics).to_csv(run_dir / "all_metrics.csv", index=False)
                status["completed_datasets"] += 1
                _json(run_dir / "STATUS.json", status)
                del panel
            except Exception as exc:
                failure = {"dataset": spec.name, "stage": "dataset", "error": repr(exc),
                           "traceback": traceback.format_exc()}
                status["failures"].append(failure)
                print(f"DATASET FAILED {spec.name}: {exc}", flush=True)
        paired_files = sorted((run_dir / "datasets").glob("*/paired_pinball_inference.csv"))
        if paired_files:
            all_paired = pd.concat((pd.read_csv(path) for path in paired_files), ignore_index=True)
            all_paired["p_holm_all_panels"] = np.nan
            for split, group in all_paired.groupby("split"):
                ordered = group["p_one_sided"].dropna().sort_values().index
                running = 0.0
                for rank, idx in enumerate(ordered):
                    running = max(running, min(1.0, float(all_paired.loc[idx, "p_one_sided"]) *
                                               (len(ordered) - rank)))
                    all_paired.loc[idx, "p_holm_all_panels"] = running
            all_paired.to_csv(run_dir / "all_paired_pinball_inference.csv", index=False)
        figures = [{"file": p.relative_to(run_dir).as_posix(), "sha256": _sha256(p),
                    "bytes": p.stat().st_size} for p in sorted((run_dir / "figures").glob("*.pdf"))]
        _json(run_dir / "FIGURE_MANIFEST.json", {"scope": "memory and factorial module effects",
                                                "expected": 24, "figures": figures})
        if len(figures) != 24:
            status["failures"].append({"stage": "figures", "error": f"Expected 24, got {len(figures)}"})
        status["formal_experiment_completed"] = (status["completed_fits"] == status["registered_fits"]
                                                  and status["completed_datasets"] == len(cfg.datasets)
                                                  and not status["failures"])
    except Exception as exc:
        status["failures"].append({"stage": "pipeline", "error": repr(exc), "traceback": traceback.format_exc()})
        print(f"PIPELINE FAILED: {exc}", flush=True)
    finally:
        status["elapsed_hours"] = (time.time() - started) / 3600
        status["finished_utc"] = datetime.now(timezone.utc).isoformat()
        _json(run_dir / "STATUS.json", status)
        _json(run_dir / "environment.json", {"python": platform.python_version(),
                                               "platform": platform.platform(), "torch": torch.__version__,
                                               "cuda_available": torch.cuda.is_available(),
                                               "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
        log.flush()
        sys.stdout, sys.stderr = original_out, original_err
        log.close()
        hashes = [{"path": p.relative_to(run_dir).as_posix(), "bytes": p.stat().st_size,
                   "sha256": _sha256(p)} for p in sorted(run_dir.rglob("*"))
                  if p.is_file() and p.name != "FILE_HASHES.json"]
        _json(run_dir / "FILE_HASHES.json", hashes)
        archive = Path(shutil.make_archive(str(root / "log" / run_id), "zip", root_dir=run_dir))
        print(f"Evidence bundle: {archive}", flush=True)
        print(f"formal_experiment_completed={status['formal_experiment_completed']}", flush=True)
    return archive, bool(status["formal_experiment_completed"])
