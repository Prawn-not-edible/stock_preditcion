"""Create comparison tables and charts from completed development/recent runs."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
STOCKS=["sh.601088","sh.601225","sz.000983"]
MODELS=["DLinear","iTransformer","TimeXer","LastCloseBaseline"]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics-dir",type=Path,default=ROOT/"results/metrics")
    p.add_argument("--analysis-dir",type=Path,default=ROOT/"results/analysis")
    a=p.parse_args(); m=a.metrics_dir; out=a.analysis_dir
    dev=pd.read_csv(m/"development_summary.csv")
    recent=pd.read_csv(m/"recent_summary.csv")
    with (m/"model_selection.json").open(encoding="utf-8") as f: selection=json.load(f)["stocks"]
    dcols={c:f"development_{c}" for c in dev.columns if c not in ("stock","model")}
    rcols={c:f"recent_{c}" for c in recent.columns if c not in ("stock","model")}
    comparison=dev.rename(columns=dcols).merge(recent.rename(columns=rcols),on=["stock","model"],validate="one_to_one")
    for period in ("development","recent"):
        base=comparison[comparison.model=="LastCloseBaseline"][["stock",f"{period}_mean_nmse"]].rename(columns={f"{period}_mean_nmse":f"{period}_baseline_nmse"})
        comparison=comparison.merge(base,on="stock",validate="many_to_one")
        comparison[f"{period}_relative_nmse_improvement_vs_baseline"]=(comparison[f"{period}_baseline_nmse"]-comparison[f"{period}_mean_nmse"])/comparison[f"{period}_baseline_nmse"]
    comparison["selected_model"]=comparison.stock.map({k:v["selected_model"] for k,v in selection.items()})
    m.mkdir(parents=True,exist_ok=True); comparison.to_csv(m/"model_comparison.csv",index=False)

    compdir=out/"stock_comparison"; compdir.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(15,4.8),sharey=False)
    for ax,stock in zip(axes,STOCKS):
        g=comparison[comparison.stock==stock].set_index("model").reindex(MODELS)
        x=range(len(MODELS)); w=.36
        ax.bar([i-w/2 for i in x],g["development_mean_nmse"],w,label="Development")
        ax.bar([i+w/2 for i in x],g["recent_mean_nmse"],w,label="Recent 180d")
        ax.set_title(f"{stock} | selected: {selection[stock]['selected_model']}")
        ax.set_xticks(list(x),["DLinear","iTransformer","TimeXer","Last Close"],rotation=25,ha="right")
        ax.set_ylabel("Mean NMSE (lower is better)"); ax.grid(axis="y",alpha=.2)
    axes[0].legend(frameon=False); fig.tight_layout()
    fig.savefig(compdir/"development_vs_recent_nmse.png",dpi=180,bbox_inches="tight"); plt.close(fig)

    recent_runs=pd.read_csv(m/"recent_training_runs.csv",parse_dates=["retrain_origin"])
    train_dir=out/"training_curves"; train_dir.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),sharey=False)
    for ax,(stock,part) in zip(axes,recent_runs.groupby("stock",sort=False)):
        for model,grp in part.groupby("model",sort=False):
            grp=grp.sort_values("retrain_origin")
            ax.plot(grp["retrain_origin"],grp["validation_loss"],marker="o",label=f"{model} validation")
        ax.set_title(stock); ax.set_ylabel("Best validation MSE (scaled target)"); ax.tick_params(axis="x",rotation=35); ax.grid(alpha=.2)
    axes[0].legend(fontsize=8,frameon=False); fig.tight_layout()
    fig.savefig(train_dir/"recent_refit_validation_loss.png",dpi=180,bbox_inches="tight"); plt.close(fig)
    print(f"Wrote {m/'model_comparison.csv'} and charts under {out}")

if __name__=="__main__": main()
