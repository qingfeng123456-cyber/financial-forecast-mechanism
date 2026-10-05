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

from .c3r import build_c3r
from .config import ExperimentConfig
from .data import PanelData
from .metrics import paired_block_bootstrap
from .models import build_baseline
from .preflight import run_preflight, validate_panel_splits
from .train import evaluate, fit, seed_everything
from .visuals import mechanism_figures, module_effect_figures, finalize_figure_exports, comparison_tables


class _Tee:
    def __init__(self, original, stream): self.original, self.stream = original, stream
    def write(self, value): self.original.write(value); self.stream.write(value); return len(value)
    def flush(self): self.original.flush(); self.stream.flush()
    def isatty(self): return False


def _json_default(value):
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if isinstance(value, np.ndarray): return value.tolist()
    raise TypeError(type(value).__name__)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model(name: str, panel: PanelData, cfg: ExperimentConfig):
    kwargs = dict(input_size=panel.input_size, hidden=cfg.hidden_size, market_size=6,
                  retrieval_size=panel.retrieval_size, lookback=cfg.lookback,
                  num_experts=cfg.num_experts, graph_topk=cfg.graph_topk,
                  router_temperature=cfg.router_temperature,router_shrinkage=cfg.router_shrinkage,
                  router_hidden_size=cfg.router_hidden_size)
    return build_c3r(name, **kwargs) if name.startswith("c3r_") else build_baseline(name, **kwargs)


def _display_name(name: str) -> str:
    labels = {
        "gru": "GRU", "stockmixer": "StockMixer (adapted)", "master": "MASTER (adapted)",
        "miga": "MIGA (adapted)", "mera": "MERA (adapted)", "mafs": "MAFS (adapted)",
        "c3r_full": "C3R full", "c3r_graph_only": "C3R graph only",
        "c3r_credit_only": "C3R coalition risk only", "c3r_plain_moe": "C3R task router",
        "c3r_unprojected":"C3R unprojected exchange",
    }
    return labels[name]


def run(root: Path, cfg: ExperimentConfig, device_name: str = "auto") -> tuple[Path, bool]:
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    run_id = "RUN-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-C3R"
    run_dir = root / "log" / run_id
    figures = run_dir / "figures"
    run_dir.mkdir(parents=True, exist_ok=False); figures.mkdir()
    started = time.time()
    status = {"run_id": run_id, "formal_experiment_completed": False, "failures": [], "completed": 0,"attempted":0}
    original_out,original_err=sys.stdout,sys.stderr
    console=(run_dir/"console.log").open("w",encoding="utf-8",buffering=1)
    sys.stdout,sys.stderr=_Tee(original_out,console),_Tee(original_err,console)
    (run_dir / "resolved_config.json").write_text(json.dumps(cfg.to_dict(), indent=2) + "\n", encoding="utf-8")
    source_files = [root / "main.py", root / "config.json", root / "requirements.txt",
                    *sorted((root / "src").rglob("*.py")), *sorted((root / "research").glob("*"))]
    source_inventory = [{"path": path.relative_to(root).as_posix(), "sha256": _sha256(path), "bytes": path.stat().st_size}
                        for path in source_files if path.is_file()]
    (run_dir / "SOURCE_HASHES.json").write_text(json.dumps(source_inventory, indent=2) + "\n", encoding="utf-8")
    try:
        preflight = run_preflight(root, cfg, device)
        (run_dir / "preflight.json").write_text(json.dumps(preflight, indent=2) + "\n", encoding="utf-8")
        print(f"{run_id}: runtime validation complete on {device}", flush=True)
        models = [*cfg.baseline_models, *cfg.ablation_models,*cfg.mechanism_models]
        total = len(cfg.datasets) * len(models)
        status["registered_fits"]=total
        print(f"Registered matrix: {len(cfg.datasets)} datasets x {len(models)} models = {total} backbone fits; "
              "12 additional cached readout fits; no search, automatic retry or wall-clock kill.", flush=True)
        metric_rows, daily_tables, all_histories = [], {}, {}
        for dataset_idx, spec in enumerate(cfg.datasets, 1):
            print(f"Preparing dataset {dataset_idx}/{len(cfg.datasets)}: {spec.name}", flush=True)
            panel = PanelData(root, spec, cfg)
            split_check = validate_panel_splits(panel)
            (run_dir / f"{spec.name}_split_check.json").write_text(json.dumps(split_check, indent=2) + "\n", encoding="utf-8")
            for model_idx, name in enumerate(models, 1):
                experiment_no = (dataset_idx - 1) * len(models) + model_idx
                status["attempted"]+=1
                elapsed = time.time() - started
                eta = elapsed / max(experiment_no - 1, 1) * (total - experiment_no + 1) if experiment_no > 1 else float("nan")
                print(f"\n=== Experiment {experiment_no}/{total}: {spec.name}/{name}; elapsed={elapsed/3600:.2f}h "
                      f"estimated remaining={eta/3600:.2f}h ===", flush=True)
                output = run_dir / "experiments" / spec.name / name
                output.mkdir(parents=True, exist_ok=True)
                try:
                    seed_everything(cfg.seed)
                    model = _model(name, panel, cfg).to(device)
                    parameter_count=sum(p.numel() for p in model.parameters());tick=time.time()
                    history = fit(model, panel, cfg, device, output, f"{spec.name}/{name}")
                    failure_file=output/"reference_failures.json"
                    if failure_file.is_file():
                        for failure in json.loads(failure_file.read_text(encoding="utf-8")):
                            status["failures"].append({"dataset":spec.name,"model":name,"stage":"cached_control",**failure})
                    fit_seconds=time.time()-tick;tick=time.time()
                    timing=json.loads((output/"timing.json").read_text(encoding="utf-8"))
                    metrics, daily, diagnostics = evaluate(model, panel, cfg, device, output,
                                                           collect_diagnostics=name == "c3r_full")
                    row = {"dataset": spec.name, "model": name, "display_model": _display_name(name),
                           "parameters":parameter_count,"training_seconds":timing["core_and_proposed_router_seconds"],
                           "cached_readout_seconds":timing["cached_readout_seconds"],"training_seconds_including_readouts":fit_seconds,
                           "evaluation_seconds":time.time()-tick,
                           "model_scope": "factorial" if name in cfg.ablation_models else "mechanism_control" if name in cfg.mechanism_models else "common_protocol_adaptation", **metrics}
                    metric_rows.append(row); daily_tables[(spec.name, name)] = daily
                    all_histories[(spec.name, name)] = history
                    (output / "metrics.json").write_text(json.dumps(row, indent=2, default=_json_default) + "\n", encoding="utf-8")
                    if name == "c3r_full":
                        diag_rows = []
                        for item in diagnostics:
                            row_diag = {"date": item["date"],
                                        "representation_dispersion_before": float(np.std(item["independent_hidden"], axis=1).mean()),
                                        "representation_dispersion_after": float(np.std(item["hidden"], axis=1).mean()),
                                        **{f"mean_gate_{i}": float(item["gates"][:, i].mean()) for i in range(4)},
                                        **{f"role_inclusion_{i}": item["predicted_credit"][i] for i in range(4)},
                                        **{f"grand_coalition_credit_{i}": item["coalition_credit"][i] for i in range(4)},
                                        **{f"output_only_credit_{i}": item["output_credit"][i] for i in range(4)},
                                        **{f"null_mass_{i}":float(item["null_mass"][:,i].mean()) for i in range(4)},
                                        **{f"route_weight_{k}":float(item["route_weights"][k]) for k in range(16)},
                                        **{f"predicted_risk_{k}":float(item["predicted_risk"][k]) for k in range(16)},
                                        **{f"realized_rank_loss_{k}":float(item["realized_risk"][k]) for k in range(16)},
                                        **{f"output_only_rank_loss_{k}":float(item["output_only_realized_risk"][k]) for k in range(16)},
                                        "market_dispersion":float(item["market"][1]),"mixture_rank_loss":item["mixture_rank_loss"],
                                        "weighted_bank_rank_loss":item["weighted_bank_rank_loss"],"oracle_rank_loss":item["oracle_rank_loss"]}
                            if item["graph"] is not None:
                                mean_graph = item["graph"].mean(axis=0)
                                row_diag.update({f"edge_{i}_to_{j}": float(mean_graph[i, j]) for i in range(4) for j in range(4)})
                            diag_rows.append(row_diag)
                        pd.DataFrame(diag_rows).to_csv(output / "coalition_diagnostics.csv", index=False)
                        sample_hidden = np.concatenate([d["hidden"] for d in diagnostics], axis=0)
                        sample_independent = np.concatenate([d["independent_hidden"] for d in diagnostics], axis=0)
                        sample_gates = np.concatenate([d["gates"] for d in diagnostics], axis=0)
                        sample_graph = np.concatenate([d["graph"] for d in diagnostics if d["graph"] is not None], axis=0)
                        sample_dates = np.concatenate([np.repeat(d["date"], len(d["assets"])) for d in diagnostics])
                        sample_assets = np.concatenate([d["assets"] for d in diagnostics])
                        np.savez_compressed(output / "mechanism_samples.npz", dates=sample_dates, assets=sample_assets,
                                            hidden=sample_hidden, independent_hidden=sample_independent,
                                            gates=sample_gates, graph=sample_graph,
                                            null_mass=np.concatenate([d["null_mass"] for d in diagnostics]),
                                            proposals=np.concatenate([d["proposals"] for d in diagnostics]),
                                            diagnostic_dates=np.asarray([d["date"] for d in diagnostics]),
                                            route_weights=np.stack([d["route_weights"] for d in diagnostics]),
                                            predicted_risk=np.stack([d["predicted_risk"] for d in diagnostics]),
                                            realized_rank_loss=np.stack([d["realized_risk"] for d in diagnostics]),
                                            output_only_realized_risk=np.stack([d["output_only_realized_risk"] for d in diagnostics]))
                        mechanism_figures(spec.name, diagnostics, figures, cfg.tsne_points,cfg.risk_scale_floor)
                    status["completed"] += 1
                except Exception as exc:
                    failure = {"dataset": spec.name, "model": name, "error": repr(exc), "traceback": traceback.format_exc()}
                    status["failures"].append(failure)
                    (output / "FAILED.json").write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
                    print(f"FAILED {spec.name}/{name}: {exc}", flush=True)
                finally:
                    pd.DataFrame(metric_rows).to_csv(run_dir/"all_metrics.csv",index=False)
                    (run_dir/"STATUS.json").write_text(json.dumps(status,indent=2)+"\n",encoding="utf-8")
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
        metrics_frame = pd.DataFrame(metric_rows)
        metrics_frame.to_csv(run_dir / "all_metrics.csv", index=False)
        comparisons = []
        for spec in cfg.datasets:
            key = (spec.name, "c3r_full")
            if key not in daily_tables: continue
            for name in models:
                other = (spec.name, name)
                if name == "c3r_full" or other not in daily_tables: continue
                comparisons.append({"dataset": spec.name, "full_minus": name,
                                    **paired_block_bootstrap(daily_tables[key], daily_tables[other], cfg.bootstrap_draws, cfg.seed,cfg.bootstrap_block_length)})
        inference = pd.DataFrame(comparisons)
        if not inference.empty:
            inference["p_holm"] = np.nan
            for dataset, group in inference.groupby("dataset"):
                ordered = group["p_one_sided"].dropna().sort_values().index
                running = 0.0
                m = len(group)
                for rank, idx in enumerate(ordered):
                    running = max(running, min(1.0, float(inference.loc[idx, "p_one_sided"]) * (m - rank)))
                    inference.loc[idx, "p_holm"] = running
        inference.to_csv(run_dir / "paired_rankic_inference.csv", index=False)
        interactions,yearly,control_rows=[],[],[]
        for (dataset,name),daily in daily_tables.items():
            for year,part in daily.groupby(daily["date"].str[:4]):
                yearly.append({"dataset":dataset,"model":name,"year":year,"rank_ic":part["rank_ic"].mean(),"dates":len(part)})
        for spec in cfg.datasets:
            if all((spec.name,n) in daily_tables for n in cfg.ablation_models):
                joint=None
                for n in cfg.ablation_models:
                    part=daily_tables[(spec.name,n)][["date","rank_ic"]].rename(columns={"rank_ic":n})
                    joint=part if joint is None else joint.merge(part,on="date")
                joint=joint.dropna();contrast=joint[["date"]].copy()
                contrast["rank_ic"]=joint["c3r_full"]-joint["c3r_graph_only"]-joint["c3r_credit_only"]+joint["c3r_plain_moe"]
                zero=contrast.copy();zero["rank_ic"]=0.0
                interactions.append({"dataset":spec.name,"contrast":"full-exchange-risk+plain",**paired_block_bootstrap(
                    contrast,zero,cfg.bootstrap_draws,cfg.seed,cfg.bootstrap_block_length)})
            path=run_dir/"experiments"/spec.name/"c3r_full"/"frozen_interventions.csv"
            if path.is_file():
                pivot=pd.read_csv(path).pivot(index="date",columns="intervention",values="rank_ic")
                for name in pivot.columns:
                    common=pivot[["fitted_reference",name]].dropna() if name!="fitted_reference" else pivot[[name]].dropna()
                    control_rows.append({"dataset":spec.name,"control":name,"rank_ic":common[name].mean(),
                        "full_minus_control":(common["fitted_reference"]-common[name]).mean() if name!="fitted_reference" else 0.0,
                        "common_diagnostic_dates":len(common),"inference_scope":"exploratory, adjacent sampled diagnostic dates",
                        "p_adjustment":"none; exploratory",
                        **(paired_block_bootstrap(common[["fitted_reference"]].rename(columns={"fitted_reference":"rank_ic"}).reset_index(),
                            common[[name]].rename(columns={name:"rank_ic"}).reset_index(),cfg.bootstrap_draws,cfg.seed,cfg.bootstrap_block_length)
                           if name!="fitted_reference" else {})})
        interaction_frame=pd.DataFrame(interactions);yearly_frame=pd.DataFrame(yearly);control_frame=pd.DataFrame(control_rows)
        interaction_frame.to_csv(run_dir/"factorial_interaction.csv",index=False)
        yearly_frame.to_csv(run_dir/"yearly_rankic.csv",index=False)
        control_frame.to_csv(run_dir/"cached_controls_summary.csv",index=False)
        comparison_tables(metrics_frame,run_dir/"tables",cfg.baseline_models)
        module_effect_figures(metrics_frame,figures,inference,interaction_frame,control_frame)
        figure_records=finalize_figure_exports(figures,[spec.name for spec in cfg.datasets])
        (run_dir/"FIGURE_MANIFEST.json").write_text(json.dumps({"scope":"module effects and ablations only","expected_figures":24,
            "figures":[{**r,"file":"figures/"+r["file"],"bytes":(figures/r["file"]).stat().st_size} for r in figure_records]},indent=2)+"\n",encoding="utf-8")
        status["formal_experiment_completed"] = status["completed"] == total and not status["failures"]
    except Exception as exc:
        status["failures"].append({"stage": "pipeline", "error": repr(exc), "traceback": traceback.format_exc()})
        print(f"PIPELINE FAILED: {exc}", flush=True)
    finally:
        status["elapsed_hours"] = (time.time() - started) / 3600
        status["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (run_dir / "STATUS.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        environment = {"python": platform.python_version(), "platform": platform.platform(),
                       "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
                       "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
        (run_dir / "environment.json").write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")
        console.flush();sys.stdout,sys.stderr=original_out,original_err;console.close()
        files = []
        for path in sorted(p for p in run_dir.rglob("*") if p.is_file() and p.name != "FILE_HASHES.json"):
            files.append({"path": path.relative_to(run_dir).as_posix(), "bytes": path.stat().st_size, "sha256": _sha256(path)})
        (run_dir / "FILE_HASHES.json").write_text(json.dumps(files, indent=2) + "\n", encoding="utf-8")
        archive_base = root / "log" / run_id
        archive = Path(shutil.make_archive(str(archive_base), "zip", root_dir=run_dir))
        print(f"Evidence bundle: {archive}", flush=True)
        print(f"formal_experiment_completed={status['formal_experiment_completed']}; completed={status['completed']}", flush=True)
    return archive, bool(status["formal_experiment_completed"])
