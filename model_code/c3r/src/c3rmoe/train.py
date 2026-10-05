from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .c3r import C3RMoE
from .losses import leave_one_out_rank_credit, objective, pairwise_rank_loss
from .metrics import date_rank_ic, evaluate_records


def seed_everything(seed):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _tensor_batch(batch, device):
    return tuple(torch.as_tensor(value, device=device) for value in
                 (batch.x, batch.market, batch.y, batch.retrieval, batch.complementarity))


@torch.no_grad()
def validate(model, panel, cfg, device, split="valid"):
    model.eval(); values = []
    for batch in panel.iter_dates(split, cfg.valid_dates_per_epoch, None):
        x, market, y, retrieval, comp = _tensor_batch(batch, device)
        prediction, _ = model(x, market, retrieval, comp)
        good = torch.isfinite(y)
        if int(good.sum()) >= 8:
            values.append(date_rank_ic(prediction[good].cpu().numpy(), y[good].cpu().numpy()))
    return float(np.nanmean(values)) if values else -np.inf


@torch.no_grad()
def _router_cache(model, panel, cfg, device, split, limit):
    model.eval(); cache = [];started=time.time()
    total=min(limit,len(panel.split_indices[split])) if limit else len(panel.split_indices[split])
    tick=max(1,total//cfg.progress_updates_per_epoch)
    for step,batch in enumerate(panel.iter_dates(split, limit, None),1):
        x, market, y, _, comp = _tensor_batch(batch, device)
        good = torch.isfinite(y)
        if int(good.sum()) < 8: continue
        hs = model._encode(x)
        bank = model.coalition_bank(hs, market, comp)
        context = model.routing_context(hs, market, comp)
        risks = torch.stack([pairwise_rank_loss(bank[good, k], y[good]) for k in range(16)])
        scale = risks.std(unbiased=False).clamp_min(cfg.risk_scale_floor)
        normalized = (risks - risks.mean()) / scale
        output_bank=model.coalition_bank(hs,market,comp,mode="output_only")[good]
        output_losses=torch.stack([pairwise_rank_loss(output_bank[:,k],y[good]) for k in range(16)])
        output_risk=(output_losses-output_losses.mean())/output_losses.std(unbiased=False).clamp_min(cfg.risk_scale_floor)
        cache.append({"date": batch.date, "context": context.cpu(), "bank": bank[good].cpu(),
                      "target": y[good].cpu(), "risk": normalized.cpu(), "raw_risk": risks.cpu(),
                      "output_bank":output_bank.cpu(),"output_risk":output_risk.cpu()})
        if step==1 or step%tick==0 or step==total:
            elapsed=time.time()-started
            print(f"[{panel.spec.name}/{split}] coalition cache {step}/{total} elapsed={elapsed/60:.1f}m "
                  f"ETA={elapsed/step*(total-step)/60:.1f}m",flush=True)
    if not cache: raise RuntimeError(f"Empty {split} coalition-risk cache")
    return cache


def _fit_router(model, panel, cfg, device, output, label):
    # The backbone never sees this later training interval during fitting or selection.
    cache = _router_cache(model, panel, cfg, device, "router", cfg.router_cache_dates)
    valid_cache = _router_cache(model, panel, cfg, device, "valid", cfg.valid_dates_per_epoch)
    for parameter in model.parameters(): parameter.requires_grad_(False)
    for parameter in model.risk_head.parameters(): parameter.requires_grad_(True)
    seed_everything(cfg.seed)
    for layer in model.risk_head.modules():
        if isinstance(layer,(torch.nn.Linear,torch.nn.LayerNorm)):layer.reset_parameters()
    model.router_enabled.fill_(True)
    mean_risk=torch.stack([item["risk"] for item in cache]).mean(0).to(device)
    model.static_route_prior.copy_(torch.softmax(-mean_risk/cfg.router_temperature,-1))
    optimizer = torch.optim.AdamW(model.risk_head.parameters(), lr=cfg.router_learning_rate, weight_decay=cfg.weight_decay)
    best_score, best_state, stale, history = -np.inf, None, 0, []
    rng = np.random.default_rng(cfg.seed)
    start = time.time()
    risk_table = [{"date": item["date"], **{f"route_{k}_rank_loss": float(item["raw_risk"][k]) for k in range(16)}} for item in cache]
    import pandas as pd
    pd.DataFrame(risk_table).to_csv(output / "heldout_coalition_risks.csv", index=False)
    for epoch in range(1, cfg.router_epochs + 1):
        model.eval(); model.risk_head.train()
        totals = []
        for step, idx in enumerate(rng.permutation(len(cache)), 1):
            item = cache[idx]
            context, bank, y, target_risk = [item[k].to(device) for k in ("context", "bank", "target", "risk")]
            optimizer.zero_grad(set_to_none=True)
            prediction, predicted_risk, weights = model.route_cached(context, bank)
            task = F.smooth_l1_loss(prediction, y) + cfg.rank_weight * pairwise_rank_loss(prediction, y)
            loss = task
            if model.credit:
                target_prob = torch.softmax(-target_risk / cfg.router_temperature, -1).detach()
                risk_loss = F.smooth_l1_loss(predicted_risk, target_risk.detach())
                alignment = -(target_prob * weights.clamp_min(1e-8).log()).sum()
                loss = loss + cfg.credit_weight * (risk_loss + 0.1 * alignment)
            if not torch.isfinite(loss):raise RuntimeError(f"{label}: nonfinite router loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.risk_head.parameters(), 2.0); optimizer.step()
            totals.append(float(loss.detach()))
        model.eval(); scores = []
        with torch.no_grad():
            for item in valid_cache:
                prediction, _, _ = model.route_cached(item["context"].to(device), item["bank"].to(device))
                scores.append(date_rank_ic(prediction.cpu().numpy(), item["target"].numpy()))
        score = float(np.nanmean(scores)); elapsed = time.time() - start
        history.append({"stage": "router", "epoch": epoch, "train_loss": float(np.mean(totals)), "validation_rank_ic": score})
        print(f"[{label}] router epoch {epoch:02d}/{cfg.router_epochs} loss={np.mean(totals):.4f} "
              f"validation RankIC={score:.5f} elapsed={elapsed/60:.1f}m ETA={elapsed/epoch*(cfg.router_epochs-epoch)/60:.1f}m", flush=True)
        if score > best_score + 1e-5:
            best_score, stale = score, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.risk_head.state_dict().items()}
        else:
            stale += 1
            if stale >= cfg.patience: break
    if best_state is None: raise RuntimeError(f"{label}: no finite router checkpoint")
    model.risk_head.load_state_dict(best_state)
    if label.endswith("/c3r_full"):
        model.reference_cache=(cache,valid_cache)
    return history


def _fit_reference_routers(model,cache,valid_cache,cfg,device,output,label):
    """Matched, inexpensive readouts on the same frozen coalition cache."""
    import copy
    model.reference_routers={}; model.reference_priors={};control_history={}
    mask=torch.cat([model.coalition_masks,torch.ones_like(model.coalition_masks[:1])],0)
    normalized_mask=mask/mask.sum(1,keepdim=True)
    singleton=torch.tensor([0,1,3,7],device=device)
    def fit_one(name):
        seed_everything(cfg.seed)
        head=torch.nn.Parameter(torch.zeros(16,device=device)) if name=="static_mixture" else copy.deepcopy(model.risk_head)
        if isinstance(head,torch.nn.Module):
            # Start each comparator afresh rather than transferring the proposed scorer.
            for layer in head.modules():
                if isinstance(layer,(torch.nn.Linear,torch.nn.LayerNorm)):layer.reset_parameters()
            parameters=list(head.parameters())
        else:parameters=[head]
        optimizer=torch.optim.AdamW(parameters,lr=cfg.router_learning_rate,weight_decay=cfg.weight_decay)
        best,best_state,stale=-np.inf,None,0
        risk_key="output_risk" if name=="output_only_fitted" else "risk"
        mean_risk=torch.stack([item[risk_key] for item in cache]).mean(0).to(device)
        if name=="four_expert_gate":mean_risk=mean_risk[singleton]
        prior=torch.softmax(-mean_risk/cfg.router_temperature,-1)
        started=time.time()
        def readout(item):
            ctx=item["context"].to(device)
            score=head if name=="static_mixture" else head(ctx)
            bank=item["output_bank" if name=="output_only_fitted" else "bank"].to(device)
            if name=="four_expert_gate":
                risk=score[:4]; bank=bank[:,singleton]
            elif name=="additive_risk_fitted":
                risk=normalized_mask@score[:4]+(mask.sum(1)/4.0)*score[4]
                risk=risk+torch.cat([torch.zeros(15,device=device),score[5].view(1)])
            else:risk=score
            weights=torch.softmax(-risk/cfg.router_temperature,-1)
            weights=(1-cfg.router_shrinkage)*weights+cfg.router_shrinkage*prior
            return (bank*weights.view(1,-1)).sum(1),risk,weights
        rng=np.random.default_rng(cfg.seed)
        epochs=[]
        for epoch in range(1,cfg.router_epochs+1):
            if isinstance(head,torch.nn.Module):head.train()
            train_losses=[]
            for idx in rng.permutation(len(cache)):
                item=cache[idx];optimizer.zero_grad(set_to_none=True)
                pred,risk,weights=readout(item);y=item["target"].to(device)
                loss=F.smooth_l1_loss(pred,y)+cfg.rank_weight*pairwise_rank_loss(pred,y)
                if name in ("additive_risk_fitted","output_only_fitted"):
                    target=item[risk_key].to(device)
                    target_prob=torch.softmax(-target/cfg.router_temperature,-1)
                    alignment=-(target_prob*weights.clamp_min(1e-8).log()).sum()
                    loss=loss+cfg.credit_weight*(F.smooth_l1_loss(risk,target)+0.1*alignment)
                if not torch.isfinite(loss):raise RuntimeError(f"{label}: nonfinite cached control loss {name}")
                loss.backward();torch.nn.utils.clip_grad_norm_(parameters,2.0);optimizer.step()
                train_losses.append(float(loss.detach()))
            if isinstance(head,torch.nn.Module):head.eval()
            with torch.no_grad():
                score=float(np.nanmean([date_rank_ic(readout(i)[0].cpu().numpy(),i["target"].numpy()) for i in valid_cache]))
            elapsed=time.time()-started
            epochs.append({"epoch":epoch,"train_loss":float(np.mean(train_losses)),"validation_rank_ic":score,"elapsed_seconds":elapsed})
            print(f"[{label}] control {name} epoch {epoch:02d}/{cfg.router_epochs} loss={np.mean(train_losses):.4f} validation RankIC={score:.5f} "
                  f"elapsed={elapsed/60:.1f}m ETA={elapsed/epoch*(cfg.router_epochs-epoch)/60:.1f}m",flush=True)
            if score>best+1e-5:
                best,stale=score,0
                best_state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()} if isinstance(head,torch.nn.Module) else head.detach().cpu().clone()
            else:
                stale+=1
                if stale>=cfg.patience:break
        if best_state is None:raise RuntimeError(f"{label}: no {name} checkpoint")
        if isinstance(head,torch.nn.Module):head.load_state_dict(best_state);head.eval()
        else:head.data.copy_(best_state.to(device))
        print(f"[{label}] cached control {name}: validation RankIC={best:.5f}; no new backbone fit",flush=True)
        return head,prior,{"best_validation_rank_ic":best,"parameters":sum(p.numel() for p in parameters),"epochs":epochs}
    failures=[]
    for name in ("additive_risk_fitted","four_expert_gate","output_only_fitted","static_mixture"):
        try:
            head,prior,record=fit_one(name)
            model.reference_routers[name]=head;model.reference_priors[name]=prior;control_history[name]=record
        except Exception as exc:
            failures.append({"control":name,"error":repr(exc),"traceback":traceback.format_exc()})
            print(f"[{label}] FAILED cached control {name}: {exc}",flush=True)
        # A later readout failure cannot erase an earlier completed readout.
        torch.save({"heads":{k:({n:v.detach().cpu() for n,v in h.state_dict().items()} if isinstance(h,torch.nn.Module) else h.detach().cpu())
                    for k,h in model.reference_routers.items()},"priors":{k:v.cpu() for k,v in model.reference_priors.items()}},output/"reference_routers.pt")
        (output/"reference_history.json").write_text(json.dumps(control_history,indent=2)+"\n",encoding="utf-8")
        (output/"reference_failures.json").write_text(json.dumps(failures,indent=2)+"\n",encoding="utf-8")


def fit(model, panel, cfg, device, output: Path, label: str):
    is_c3r = isinstance(model, C3RMoE)
    train_split, valid_split = ("fit", "core_valid") if is_c3r else ("train", "valid")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.learning_rate * 0.1)
    rng, started = np.random.default_rng(cfg.seed), time.time()
    best_score, best_state, stale, history = -np.inf, None, 0, []
    for epoch in range(1, cfg.epochs + 1):
        model.train(); batches = list(panel.iter_dates(train_split, cfg.dates_per_epoch, rng)); running = {}
        tick = max(1, len(batches) // cfg.progress_updates_per_epoch)
        for step, batch in enumerate(batches, 1):
            x, market, y, retrieval, comp = _tensor_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            prediction, aux = model(x, market, retrieval, comp)
            loss, parts = objective(prediction, y, aux, cfg)
            if is_c3r:
                active=model.coalition_masks[((epoch-1)*cfg.dates_per_epoch+step-1)%15].bool()
                partial,_=model._coalition(aux["independent_hidden"],market,comp,active)
                loss=loss+cfg.coalition_weight*(F.smooth_l1_loss(partial,y)+cfg.rank_weight*pairwise_rank_loss(partial,y))
            parts["total"]=loss.detach()
            if not torch.isfinite(loss): raise RuntimeError(f"Nonfinite training loss: {label}/{epoch}/{step}")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); optimizer.step()
            for key, value in parts.items(): running[key] = running.get(key, 0.0) + float(value)
            if step == 1 or step % tick == 0 or step == len(batches):
                elapsed = time.time() - started
                fraction = ((epoch - 1) * len(batches) + step) / max(1, cfg.epochs * len(batches))
                print(f"[{label}] core epoch {epoch:02d}/{cfg.epochs} date {step:03d}/{len(batches)} "
                      f"loss={running['total']/step:.4f} elapsed={elapsed/60:.1f}m ETA={(elapsed/fraction-elapsed)/60:.1f}m", flush=True)
        scheduler.step(); score = validate(model, panel, cfg, device, valid_split)
        history.append({"stage": "core", "epoch": epoch, "validation_rank_ic": score,
                        **{f"train_{k}": value/max(1,len(batches)) for k,value in running.items()}})
        print(f"[{label}] core epoch {epoch:02d} validation RankIC={score:.5f}", flush=True)
        if score > best_score + 1e-5:
            best_score, stale = score, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= cfg.patience: break
    if best_state is None: raise RuntimeError(f"{label}: no finite core checkpoint")
    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "best_core_validation_rank_ic": best_score}, output / "backbone.pt")
    if is_c3r: history.extend(_fit_router(model, panel, cfg, device, output, label))
    torch.save({"state_dict": {k:v.detach().cpu() for k,v in model.state_dict().items()},
                "parameters": sum(p.numel() for p in model.parameters()),
                "best_core_validation_rank_ic": best_score}, output / "model.pt")
    (output / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    timing={"core_and_proposed_router_seconds":time.time()-started,"cached_readout_seconds":0.0}
    if is_c3r and label.endswith("/c3r_full"):
        readout_started=time.time()
        cache,valid_cache=model.reference_cache
        _fit_reference_routers(model,cache,valid_cache,cfg,device,output,label)
        del model.reference_cache
        timing["cached_readout_seconds"]=time.time()-readout_started
    (output/"timing.json").write_text(json.dumps(timing,indent=2)+"\n",encoding="utf-8")
    return history


@torch.no_grad()
def evaluate(model, panel, cfg, device, output: Path, collect_diagnostics=False):
    model.eval(); records, diagnostics, intervention_records = [], [], {}
    dates = panel.split_indices["test"]
    diagnostic_dates = set(dates[np.linspace(0, len(dates)-1, min(96, len(dates))).round().astype(int)].tolist())
    for batch in panel.iter_dates("test", None, None):
        x, market, y, retrieval, comp = _tensor_batch(batch, device)
        prediction, aux = model(x, market, retrieval, comp)
        good = torch.isfinite(y)
        records.append({"date": batch.date, "date_index": batch.date_index, "assets": batch.assets,
                        "pred": prediction.cpu().numpy(), "target": batch.y, "raw_return": batch.raw_return})
        if not isinstance(model, C3RMoE) or not collect_diagnostics or batch.date_index not in diagnostic_dates or int(good.sum()) < 8: continue
        full, coalition, output_only = model.coalition_credit(x, market, comp)
        coalition_credit = leave_one_out_rank_credit(full[good], coalition[good], y[good]).cpu().numpy()
        output_credit = leave_one_out_rank_credit(full[good], output_only[good], y[good]).cpu().numpy()
        bank, weights, risk = aux["route_predictions"], aux["route_weights"], aux["predicted_risk"]
        realized_risks = torch.stack([pairwise_rank_loss(bank[good,k],y[good]) for k in range(16)])
        hs, context = aux["independent_hidden"], model.routing_context(aux["independent_hidden"], market, comp)
        # No fitted-model search: each intervention holds the learned weights fixed.
        controls = {"grand_coalition": bank[:,14], "independent_reference": bank[:,15], "uniform_coalitions": bank.mean(1)}
        singleton_risk = risk[torch.tensor([0,1,3,7], device=device)]
        additive_risk = (model.coalition_masks * singleton_risk).sum(1) / model.coalition_masks.sum(1)
        additive_risk = torch.cat([additive_risk, singleton_risk.mean().view(1)])
        additive_weights = torch.softmax(-additive_risk/cfg.router_temperature, -1)
        controls["additive_risk"] = (bank * additive_weights.view(1,-1)).sum(1)
        for mode in ("no_messages", "uniform", "raw_messages", "output_only"):
            alternative_bank = model.coalition_bank(hs, market, comp, mode=mode)
            controls[mode] = (alternative_bank * weights.view(1,-1)).sum(1)
            if mode=="output_only":
                output_only_risks=torch.stack([pairwise_rank_loss(alternative_bank[good,k],y[good]) for k in range(16)])
        for name,head in getattr(model,"reference_routers",{}).items():
            score=head if name=="static_mixture" else head(context)
            reference_bank=bank
            if name=="four_expert_gate":
                score=score[:4];reference_bank=bank[:,torch.tensor([0,1,3,7],device=device)]
            elif name=="additive_risk_fitted":
                masks=torch.cat([model.coalition_masks,torch.ones_like(model.coalition_masks[:1])],0)
                score=(masks/masks.sum(1,keepdim=True))@score[:4]+(masks.sum(1)/4.0)*score[4]+torch.cat([torch.zeros(15,device=device),score[5].view(1)])
            elif name=="output_only_fitted":reference_bank=model.coalition_bank(hs,market,comp,mode="output_only")
            ref_weights=torch.softmax(-score/cfg.router_temperature,-1)
            ref_weights=(1-cfg.router_shrinkage)*ref_weights+cfg.router_shrinkage*model.reference_priors[name]
            controls[name]=(reference_bank*ref_weights.view(1,-1)).sum(1)
        for name, values in controls.items():
            intervention_records.setdefault(name, []).append({"date":batch.date,"date_index":batch.date_index,
                "assets":batch.assets,"pred":values.cpu().numpy(),"target":batch.y,"raw_return":batch.raw_return})
        intervention_records.setdefault("fitted_reference", []).append(records[-1])
        sample = np.linspace(0,len(batch.assets)-1,min(32,len(batch.assets))).round().astype(int)
        diagnostics.append({"date":batch.date,"assets":batch.assets[sample],"market":batch.market[0],
            "hidden":aux["hidden"][sample].cpu().numpy(),"independent_hidden":hs[sample].cpu().numpy(),
            "gates":aux["gates"][sample].cpu().numpy(),"proposals":aux["proposals"][sample].cpu().numpy(),
            "graph":aux["graph"][sample].cpu().numpy(),"null_mass":aux["null_mass"][sample].cpu().numpy(),
            "predicted_credit":aux["credit_logits"].cpu().numpy(),"coalition_credit":coalition_credit,"output_credit":output_credit,
            "route_weights":weights.cpu().numpy(),"predicted_risk":risk.cpu().numpy(),"realized_risk":realized_risks.cpu().numpy(),
            "output_only_realized_risk":output_only_risks.cpu().numpy(),
            "mixture_rank_loss":float(pairwise_rank_loss(prediction[good],y[good])),
            "weighted_bank_rank_loss":float((weights*realized_risks).sum()),
            "oracle_rank_loss":float(realized_risks.min())})
    metrics, daily = evaluate_records(records, cfg.transaction_cost_bps, cfg.horizon)
    daily.to_csv(output / "daily_metrics.csv", index=False)
    # Flat numeric arrays avoid pickle and preserve each date's inference universe.
    counts = np.asarray([len(r["assets"]) for r in records])
    np.savez_compressed(output / "predictions.npz", dates=np.asarray([r["date"] for r in records]),
        offsets=np.concatenate([[0],np.cumsum(counts)]),assets=np.concatenate([r["assets"] for r in records]),
        predictions=np.concatenate([r["pred"] for r in records]),targets=np.concatenate([r["target"] for r in records]),
        raw_returns=np.concatenate([r["raw_return"] for r in records]))
    if intervention_records:
        import pandas as pd
        rows=[]
        for name, rec in intervention_records.items():
            # These sparse dates are not a continuously traded strategy; report ranking only.
            for r in rec:
                finite=np.isfinite(r["target"])
                rows.append({"date":r["date"],"intervention":name,
                             "rank_ic":date_rank_ic(r["pred"][finite],r["target"][finite])})
        pd.DataFrame(rows).to_csv(output / "frozen_interventions.csv",index=False)
    return metrics,daily,diagnostics
