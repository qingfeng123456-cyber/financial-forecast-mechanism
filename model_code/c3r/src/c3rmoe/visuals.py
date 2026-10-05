from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


ROLE_NAMES = ["Short trend", "Medium trend", "Reversal", "Risk/liquidity"]
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
FIGURE_TYPES = ["factorial_ablation", "message_interventions", "communication_and_abstention",
                "paired_representation", "communication_credit", "coalition_risk",
                "routing_behavior", "router_ablations"]
plt.rcParams.update({"font.size":10,"font.family":"serif","font.serif":["DejaVu Serif"],
    "axes.spines.top":False,"axes.spines.right":False,"legend.fontsize":10,
    "savefig.dpi":300,"pdf.fonttype":42,"text.usetex":False,"mathtext.fontset":"stix"})


def _save(fig, figures:Path, dataset:str, number:int, caption:str):
    name=f"M{number:02d}_{dataset}_{FIGURE_TYPES[number-1]}.pdf"
    # Preserve about 8-point labels when wide panels are included at a 6.4-inch manuscript width.
    scale=max(1.0,float(fig.get_size_inches()[0])/8.0)
    for text in fig.findobj(Text):text.set_fontsize(text.get_fontsize()*scale)
    fig.tight_layout();fig.savefig(figures/name,bbox_inches="tight");plt.close(fig)
    with (figures/"CAPTIONS.jsonl").open("a",encoding="utf-8") as stream:
        stream.write(json.dumps({"file":name,"dataset":dataset,"module_analysis":FIGURE_TYPES[number-1],"caption":caption})+"\n")


def _intervals(ax, rows, labels):
    ticks=[]
    for i,(row,label) in enumerate(zip(rows,labels)):
        n=row.get("n_common_dates",row.get("common_diagnostic_dates",0))
        ticks.append(f"{label} (n={int(n) if pd.notna(n) else 0})")
        mean=row.get("mean_difference",np.nan)
        if not np.isfinite(mean):mean=row.get("full_minus_control",np.nan)
        low,high=row.get("ci_low",np.nan),row.get("ci_high",np.nan)
        if np.isfinite(mean):ax.plot(mean,i,"o",color=COLORS[0])
        else:ax.text(0,i,"unavailable",fontsize=8,va="center")
        if np.isfinite(low) and np.isfinite(high):ax.hlines(i,low,high,color=COLORS[0],lw=2)
    ax.set_yticks(range(len(ticks)),ticks);ax.axvline(0,color="0.4",lw=.8);ax.invert_yaxis()
    ax.set_xlabel("RankIC difference; paired date-block 95% interval")


def _row(frame, column, value):
    if frame.empty or column not in frame:return {}
    selected=frame[frame[column]==value]
    return selected.iloc[0].to_dict() if len(selected) else {}


def module_effect_figures(metrics, figures, inference, interactions, controls):
    """Three result-based module figures per dataset; literature models never enter."""
    for dataset in metrics["dataset"].unique() if not metrics.empty else []:
        sub=metrics[metrics["dataset"]==dataset]
        pairs=inference[inference["dataset"]==dataset] if not inference.empty else pd.DataFrame()
        cached=controls[controls["dataset"]==dataset] if not controls.empty else pd.DataFrame()
        cell_names=["c3r_plain_moe","c3r_graph_only","c3r_credit_only","c3r_full"]
        cell_labels=["Neither","Exchange","Risk","Both"]
        values=[_row(sub,"model",name) for name in cell_names]
        fig,axes=plt.subplots(1,2,figsize=(12,4.2))
        axes[0].bar(range(4),[v.get("rank_ic",np.nan) for v in values],color=COLORS)
        counts=[int(v.get("scored_rankic_dates",0)) for v in values]
        axes[0].set_xticks(range(4),[f"{label}\nn={n}" for label,n in zip(cell_labels,counts)])
        axes[0].axhline(0,color="0.4",lw=.8);axes[0].set_ylabel("Mean test-date RankIC")
        rows=[_row(pairs,"full_minus",name) for name in ["c3r_graph_only","c3r_credit_only","c3r_plain_moe"]]
        rows.append(_row(interactions,"dataset",dataset))
        _intervals(axes[1],rows,["Full - exchange only","Full - risk only","Full - neither","Factorial interaction"])
        _save(fig,figures,dataset,1,
            f"{dataset}: four independently trained factorial cells. Left: mean available scored-date RankIC and date counts. "
            "Right: common-date paired effects and moving-date-block intervals; interaction is full - exchange-only - risk-only + neither. "
            "Intervals describe chronological variation, not fit variability. Interaction is a distinct contrast, not the combined improvement.")
        fig,axes=plt.subplots(1,2,figsize=(13,4))
        names=["no_messages","uniform","raw_messages","output_only"]
        _intervals(axes[0],[_row(cached,"control",n) for n in names],
            ["No messages","Uniform edges","Unprojected messages","Output-only removal"])
        _intervals(axes[1],[_row(pairs,"full_minus","c3r_unprojected")],["Matched unprojected fit"])
        _save(fig,figures,dataset,2,
            f"{dataset}: exchange interventions. Left: full-minus-intervention effects with frozen learned route weights on matched "
            "sampled diagnostic dates (exploratory intervals). Date-level mixture weights are held fixed; local gates recompute under "
            "each intervention. Right: full-minus-unprojected control independently trained with matched "
            "capacity and protocol, on common full-test dates. The two panels use different evaluation grids and are not pooled.")
        fig,ax=plt.subplots(figsize=(10,3.8))
        names=["additive_risk_fitted","four_expert_gate","output_only_fitted","static_mixture"]
        _intervals(ax,[_row(cached,"control",name) for name in names],
            ["Additive risk scorer","Private-output task gate","Output-only risk scorer","Static fitted mixture"])
        _save(fig,figures,dataset,8,
            f"{dataset}: full-minus-routing-control RankIC on matched sampled diagnostic dates. The four controls share the frozen "
            "full-model backbone and use their own trained readouts, the same training/validation cache, temperature and shrinkage. "
            "Blocks contain adjacent sampled diagnostic observations, not consecutive trading days; intervals are exploratory. "
            "They do not establish continuous trading performance. Missing effects are marked unavailable.")


def mechanism_figures(dataset, diagnostics, figures, max_points, risk_scale_floor=0.02):
    """Five paired/descriptive mechanism figures per dataset, from actual inference."""
    if not diagnostics:return
    n_dates=len(diagnostics)
    graph=np.concatenate([d["graph"] for d in diagnostics]);null=np.concatenate([d["null_mass"] for d in diagnostics])
    fig,axes=plt.subplots(1,2,figsize=(11,4.3))
    im=axes[0].imshow(graph.mean(0),cmap="Blues",vmin=0,vmax=1)
    axes[0].set_xticks(range(4),ROLE_NAMES,rotation=25,ha="right");axes[0].set_yticks(range(4),ROLE_NAMES)
    axes[0].set(xlabel="Receiving role",ylabel="Sending role");fig.colorbar(im,ax=axes[0],label="Mean message weight")
    axes[1].boxplot([null[:,i] for i in range(4)],showfliers=False)
    axes[1].set_xticks(range(1,5),ROLE_NAMES,rotation=25,ha="right");axes[1].set_ylabel("No-message probability")
    _save(fig,figures,dataset,3,
        f"{dataset}: grand-coalition directed communication and recipient abstention across {n_dates} diagnostic dates and "
        f"{len(null)} sampled asset-date observations. Columns receive messages; missing edge mass includes the null choice. "
        "Boxplots are descriptive distributions; asset-date observations are not independent experimental repetitions.")
    before=np.concatenate([d["independent_hidden"] for d in diagnostics]);after=np.concatenate([d["hidden"] for d in diagnostics])
    # Select asset-date rows, then include all four roles on both sides without aliasing role indices.
    selected=np.linspace(0,len(before)-1,min(len(before),max(1,max_points//8))).round().astype(int)
    flat_before=before[selected].reshape(-1,before.shape[-1]);flat_after=after[selected].reshape(-1,after.shape[-1])
    joint=np.concatenate([flat_before,flat_after]);role_labels=np.tile(np.arange(4),len(selected))
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    if len(joint)>=20:
        reduced=PCA(n_components=min(20,joint.shape[1],len(joint)-1)).fit_transform(joint)
        embedding=TSNE(n_components=2,perplexity=min(30,max(5,len(joint)//20)),init="pca",learning_rate="auto",random_state=2026).fit_transform(reduced)
        for ax,points,label in zip(axes[:2],[embedding[:len(flat_before)],embedding[len(flat_before):]], ["Before messages","After messages"]):
            for role in range(4):
                take=role_labels==role;ax.scatter(points[take,0],points[take,1],s=8,alpha=.5,color=COLORS[role],label=ROLE_NAMES[role])
            ax.set_xlim(embedding[:,0].min(),embedding[:,0].max());ax.set_ylim(embedding[:,1].min(),embedding[:,1].max())
            ax.set_xlabel(f"{label}; t-SNE coordinate 1");ax.set_ylabel("t-SNE coordinate 2")
        axes[0].legend(frameon=False)
    else:
        for ax in axes[:2]:ax.text(.5,.5,"Too few observations for embedding",ha="center",transform=ax.transAxes)
    shift=np.std(after,axis=1).mean(1)-np.std(before,axis=1).mean(1)
    axes[2].hist(shift,bins=30,color=COLORS[0],alpha=.8);axes[2].axvline(0,color="0.4",lw=.8)
    axes[2].set(xlabel="After-minus-before role dispersion",ylabel="Sampled asset-date count")
    _save(fig,figures,dataset,4,
        f"{dataset}: paired grand-coalition representation changes on {n_dates} diagnostic dates. Both embeddings use the same "
        f"{len(selected)} asset-date rows and four roles, fitted jointly into one coordinate system. The right panel shows paired "
        "changes in cross-role representation dispersion for all sampled rows. Geometry alone does not establish prediction gains.")
    coalition=np.stack([d["coalition_credit"] for d in diagnostics]);output=np.stack([d["output_credit"] for d in diagnostics])
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    positions=np.arange(4)
    axes[0].bar(positions-.18,coalition.mean(0),.36,label="Recomputed removal",color=COLORS[0])
    axes[0].bar(positions+.18,output.mean(0),.36,label="Output-only removal",color=COLORS[1])
    axes[0].set_xticks(positions,ROLE_NAMES,rotation=25,ha="right");axes[0].set_ylabel("Mean grand-coalition rank-loss credit")
    axes[0].axhline(0,color="0.4",lw=.8);axes[0].legend(frameon=False)
    mediated=coalition-output
    axes[1].boxplot([mediated[:,i] for i in range(4)],showfliers=False)
    axes[1].set_xticks(range(1,5),ROLE_NAMES,rotation=25,ha="right");axes[1].axhline(0,color="0.4",lw=.8)
    axes[1].set_ylabel("Recomputed minus output-only removal credit")
    _save(fig,figures,dataset,5,
        f"{dataset}: grand-coalition removal diagnostics on {n_dates} dates. Credit equals removed-candidate rank loss minus "
        "grand-coalition rank loss; positive values indicate that removal hurts that grand coalition. Recomputed removal cuts "
        "incident message paths; output-only removal retains them. These are computational diagnostics, not Shapley values, "
        "financial causal effects, or credits of the final routed mixture.")
    risk=np.stack([d["predicted_risk"] for d in diagnostics]);raw=np.stack([d["realized_risk"] for d in diagnostics])
    output_risk=np.stack([d["output_only_realized_risk"] for d in diagnostics])
    normalized=(raw-raw.mean(1,keepdims=True))/np.maximum(raw.std(1,keepdims=True),risk_scale_floor)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    axes[0].scatter(risk.flatten(),normalized.flatten(),s=7,alpha=.25,color=COLORS[0])
    axes[0].set(xlabel="Predicted relative coalition rank-risk",ylabel="Realized relative coalition rank-risk")
    difference=output_risk[:,:15]-raw[:,:15]
    axes[1].boxplot([difference[:,i] for i in range(15)],showfliers=False)
    axes[1].set_xticks(range(1,16),range(1,16));axes[1].axhline(0,color="0.4",lw=.8)
    axes[1].set(xlabel="Active-role subset bitmask",ylabel="Output-only minus recomputed rank loss")
    _save(fig,figures,dataset,6,
        f"{dataset}: risk scoring and communication recomputation on {n_dates} held-out diagnostic dates. Left: all 16 correlated "
        "candidates per date; targets are centered and scaled per date with the registered floor. Right: paired candidate-loss "
        "differences for the 15 nonempty subsets using the same frozen backbone; positive values favor recomputation for that "
        "candidate. The identical independent reference is omitted from the right panel. Dates, not candidate points, are replication units.")
    weights=np.stack([d["route_weights"] for d in diagnostics]);inclusion=np.stack([d["predicted_credit"] for d in diagnostics])
    dispersion=np.asarray([d["market"][1] for d in diagnostics]);groups=np.digitize(dispersion,np.quantile(dispersion,[.33,.67]))
    counts=[int((groups==i).sum()) for i in range(3)]
    allocation=np.stack([inclusion[groups==i].mean(0) if counts[i] else np.full(4,np.nan) for i in range(3)])
    sizes=np.asarray([bin(k).count("1") for k in range(1,16)]+[4])
    entropy=-(weights*np.log(np.maximum(weights,1e-8))).sum(1)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    im=axes[0].imshow(allocation.T,cmap="Blues",vmin=0,vmax=1,aspect="auto")
    axes[0].set_xticks(range(3),[f"{name}\nn={n}" for name,n in zip(["Low","Middle","High"],counts)])
    axes[0].set_yticks(range(4),ROLE_NAMES);axes[0].set_xlabel("Retrospective dispersion group")
    fig.colorbar(im,ax=axes[0],label="Role inclusion probability")
    axes[1].plot(weights@sizes,color=COLORS[0]);axes[1].set(xlabel="Chronological diagnostic date index",ylabel="Expected active-role count",ylim=(1,4))
    axes[2].plot(entropy,color=COLORS[1]);axes[2].set(xlabel="Chronological diagnostic date index",ylabel="Coalition allocation entropy",ylim=(0,np.log(16)))
    _save(fig,figures,dataset,7,
        f"{dataset}: routing behavior on {n_dates} diagnostic dates. Dispersion groups are retrospective descriptive bins with "
        "shown date counts, not model-selection rules. Inclusion and expected size include the independent four-role reference; "
        "dispersion is the cross-sectional standard deviation of the observed one-session log returns (market field 1), standardized "
        "and clipped under the training scaler. Entropy measures the 16 route candidates, including the independent reference. "
        "These quantities describe adaptation without asserting that it is beneficial.")


def comparison_tables(metrics, directory, baseline_names):
    directory.mkdir(parents=True,exist_ok=True)
    if metrics.empty:return
    table=metrics[metrics["model"].isin([*baseline_names,"c3r_full"])].copy()
    columns=["dataset","display_model","model_scope","ic","rank_ic","rank_ic_ir","mae","sharpe","parameters","training_seconds"]
    table[columns].to_csv(directory/"baseline_comparison.csv",index=False)
    rows=[]
    for _,row in table.iterrows():
        values=[str(row["dataset"]).replace("_",r"\_"),str(row["display_model"])]
        values.extend([f"{float(row[k]):.4f}" if np.isfinite(row[k]) else "--" for k in ["ic","rank_ic","rank_ic_ir","mae","sharpe"]])
        rows.append(" & ".join(values)+r" \\")
    lines=[r"\begin{table}[tb]",r"\centering",r"\caption{Common-protocol adapted literature controls and the full model. Economic metrics are fee scenarios; unavailable values are shown as dashes.}",
           r"\begin{tabular}{llrrrrr}",r"\hline",r"Dataset & Model & IC & RankIC & RankICIR & MAE & Sharpe \\",r"\hline",*rows,
           r"\hline",r"\end{tabular}",r"\end{table}"]
    (directory/"baseline_comparison.tex").write_text("\n".join(lines)+"\n",encoding="utf-8")


def finalize_figure_exports(figures, datasets):
    path=figures/"CAPTIONS.jsonl"
    records=[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.is_file() else []
    records.sort(key=lambda r:(r["dataset"],r["file"]))
    expected={f"M{i:02d}_{dataset}_{FIGURE_TYPES[i-1]}.pdf" for dataset in datasets for i in range(1,9)}
    actual={r["file"] for r in records}
    if actual!=expected or len(records)!=len(expected):
        raise RuntimeError(f"Module figure collection incomplete: expected {len(expected)}, received {len(records)}")
    captions=[];includes=[]
    for r in records:
        if not (figures/r["file"]).is_file():raise FileNotFoundError(r["file"])
        captions.extend([f"## {r['file']}",r["caption"],""])
        escaped=r["caption"].replace("_",r"\_").replace("%",r"\%")
        includes.append("\n".join([r"\begin{figure}[tb]",r"\centering",
            r"\includegraphics[width=\textwidth]{figures/"+r["file"]+"}",
            r"\caption{"+escaped+"}",r"\label{fig:"+Path(r["file"]).stem+"}",r"\end{figure}"]))
    (figures/"CAPTIONS.md").write_text("\n".join(captions),encoding="utf-8")
    (figures/"latex_includes.tex").write_text("\n\n".join(includes)+"\n",encoding="utf-8")
    return records
