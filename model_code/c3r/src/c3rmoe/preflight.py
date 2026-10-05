from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .c3r import build_c3r
from .config import ExperimentConfig
from .models import build_baseline


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_preflight(root: Path, cfg: ExperimentConfig, device: torch.device) -> dict:
    manifest = json.loads((root / "data" / "DATASETS.json").read_text(encoding="utf-8"))
    expected = {item["file"]: item["sha256"] for item in manifest["datasets"]}
    checked = []
    for spec in cfg.datasets:
        path = root / spec.file
        if not path.is_file():
            raise FileNotFoundError(f"Bundled dataset missing: {path}")
        actual = sha256(path)
        if expected.get(path.name) != actual:
            raise RuntimeError(f"Dataset checksum mismatch: {path.name}")
        with np.load(path, allow_pickle=False) as packed:
            shape = tuple(int(v) for v in packed["values"].shape)
            if len(shape) != 3 or shape[2] != 6:
                raise ValueError(f"Unexpected panel shape for {path.name}: {shape}")
        checked.append({"file": path.name, "sha256": actual, "shape": shape})
    common = dict(input_size=12, hidden=cfg.hidden_size, market_size=6, retrieval_size=22,
                  lookback=cfg.lookback, num_experts=cfg.num_experts, graph_topk=cfg.graph_topk,
                  router_temperature=cfg.router_temperature,router_shrinkage=cfg.router_shrinkage,
                  router_hidden_size=cfg.router_hidden_size)
    torch.manual_seed(cfg.seed)
    x = torch.randn(9, cfg.lookback, 12, device=device)*0.1
    market = torch.zeros(9, 6, device=device)
    retrieval = torch.zeros(9, 22, device=device)
    comp = torch.ones(4, 4, device=device) - torch.eye(4, device=device)
    shapes = {}
    for name in [*cfg.baseline_models, *cfg.ablation_models,*cfg.mechanism_models]:
        model = (build_c3r(name, **common) if name.startswith("c3r_") else build_baseline(name, **common)).to(device)
        model.train()
        pred, aux = model(x, market, retrieval, comp)
        if pred.shape != (9,) or not torch.isfinite(pred).all():
            raise RuntimeError(f"{name}: finite forward-shape validation failed")
        pred.square().mean().backward()
        finite_grads = [torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None]
        if not finite_grads or not all(finite_grads):
            raise RuntimeError(f"{name}: gradient validation failed")
        shapes[name] = list(pred.shape)
    model = build_c3r("c3r_full", **common).to(device).eval()
    with torch.no_grad():
        hs = model._encode(x)
        removed_a, _ = model._from_encoded(hs, market, comp, excluded=1, output_only=False)
        changed = hs.clone(); changed[:, 1] += 100.0
        removed_b, _ = model._from_encoded(changed, market, comp, excluded=1, output_only=False)
        _, coalition, output_only = model.coalition_credit(x, market, comp)
        bank=model.coalition_bank(hs,market,comp)
        singleton,one_aux=model._coalition(hs,market,comp,model.coalition_masks[0].bool())
        if not torch.isfinite(bank).all() or bank.shape!=(9,16):
            raise RuntimeError("Nonfinite coalition bank")
        if not torch.allclose(one_aux["message_delta"],torch.zeros_like(one_aux["message_delta"]),atol=1e-7):
            raise RuntimeError("Null/singleton path contains a spurious message update")
        if not torch.allclose(singleton,one_aux["private_proposals"][:,0],atol=1e-6):
            raise RuntimeError("Singleton is not the private expert")
        model.router_enabled.fill_(True)
        routed,_=model(x,market,None,comp)
        permutation=torch.arange(8,-1,-1,device=device)
        permuted,_=model(x[permutation],market[permutation],None,comp)
        if not torch.allclose(routed[permutation],permuted,atol=1e-5,rtol=1e-5):
            raise RuntimeError("Date-level routing is not asset-permutation equivariant")
    model.train();model.risk_head.train()
    routed,aux=model(x,market,None,comp)
    target=torch.linspace(-1,1,9,device=device)
    from .losses import objective
    loss,_=objective(routed,target,aux,cfg)
    loss.backward()
    if not any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.risk_head.parameters()):
        raise RuntimeError("Coalition risk head has no finite gradient")
    if coalition.shape != (9, 4) or output_only.shape != (9, 4):
        raise RuntimeError("C3R coalition shape validation failed")
    if not torch.allclose(removed_a, removed_b, atol=1e-6, rtol=1e-6):
        raise RuntimeError("Excluded expert still influences the recomputed coalition")
    return {"device": str(device), "torch": torch.__version__, "cuda": torch.cuda.is_available(),
            "datasets": checked, "forward_shapes": shapes, "coalition_invariance": True,
            "singleton_null_path":True,"asset_permutation_equivariance":True,"router_gradient":True}


def validate_panel_splits(panel) -> dict:
    step = panel.cfg.horizon + 1
    boundaries = {"train": np.datetime64(panel.spec.train_end),
                  "valid": np.datetime64(panel.spec.valid_end),
                  "test": np.datetime64(panel.spec.test_end)}
    prior = None
    summary = {}
    for split in ("train", "valid", "test"):
        indices = panel.split_indices[split]
        if len(indices) == 0:
            raise RuntimeError(f"{panel.spec.name}: empty {split} split")
        if np.any(indices + step >= len(panel.dates)) or np.any(panel.dates[indices + step] > boundaries[split]):
            raise RuntimeError(f"{panel.spec.name}: {split} label matures beyond its boundary")
        if prior is not None and panel.dates[indices[0]] <= prior:
            raise RuntimeError(f"{panel.spec.name}: chronological split overlap")
        prior = boundaries[split]
        active = panel.feature_eligible[indices].sum(axis=1)
        summary[split] = {"origins": int(len(indices)), "first": str(panel.dates[indices[0]]),
                          "last": str(panel.dates[indices[-1]]), "last_label_end": str(panel.dates[indices[-1] + step]),
                          "feature_eligible_assets_min": int(active.min()),
                          "feature_eligible_assets_median": float(np.median(active)),
                          "feature_eligible_assets_max": int(active.max())}
    for earlier,later in (("fit","core_valid"),("core_valid","router")):
        a,b=panel.split_indices[earlier],panel.split_indices[later]
        if len(a)==0 or len(b)==0 or a[-1]+step>=b[0]:
            raise RuntimeError(f"{panel.spec.name}: staged {earlier}/{later} label overlap")
    for split in ("fit","core_valid","router"):
        indices=panel.split_indices[split]
        summary[split]={"origins":int(len(indices)),"first":str(panel.dates[indices[0]]),
                        "last":str(panel.dates[indices[-1]]),"last_label_end":str(panel.dates[indices[-1]+step])}
    summary["scaler_last_observation"]=str(panel.dates[panel.split_indices["fit"][-1]])
    return summary
