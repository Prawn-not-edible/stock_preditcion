"""Leakage-safe shared Residual training experiments for Phase 4 coal stocks."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import residual_phase3 as p3

TARGETS = ["sh.601088", "sh.601225", "sz.000983"]
MODELS = p3.MODELS
ALPHAS = p3.ALPHAS
AS_OF = "2026-10-30"
POOL_PROCESSED = ROOT / "data/coal_pool/processed"
OUT = ROOT / "results/phase4_joint_training"


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def read_processed(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(path)
    if df.columns.tolist() != ["date", *p3.COLS]:
        raise ValueError(f"unexpected processed columns {path}: {df.columns.tolist()}")
    raw = df[p3.COLS].to_numpy(dtype=np.float32)
    if not np.isfinite(raw).all(): raise ValueError(f"non-finite model data {path}")
    return df, raw


def stock_files(pool: str) -> dict[str, Path]:
    paths = {s: ROOT / "data/processed" / f"{s.replace('.', '_')}.csv" for s in TARGETS}
    if pool == "P2":
        paths.update({p.stem.replace("_", "."): p for p in POOL_PROCESSED.glob("*.csv")})
    if pool == "P1": return {s: paths[s] for s in TARGETS}
    return paths


def membership_for(asof: str, pool: str) -> list[str]:
    if pool in ("P0", "P1"): return TARGETS.copy()
    manifest = ROOT / "data/coal_pool/manifests/industry_membership.csv"
    if not manifest.exists(): raise FileNotFoundError(f"missing pool manifest: {manifest}")
    df = pd.read_csv(manifest, dtype=str).fillna("")
    if df.classification_asof.eq("current_only").all() and asof < AS_OF:
        # Never use present-day membership to claim historical eligibility.
        return TARGETS.copy()
    df["date_key"] = df.classification_asof.where(df.classification_asof.ne("current_only"), AS_OF)
    usable = df[df.date_key.le(asof)].sort_values("date_key").drop_duplicates("code", keep="last")
    usable = usable[usable.industry.astype(str).str.contains("煤炭|煤矿|采掘", regex=True, na=False) |
                    usable.code.isin(TARGETS)]
    allowed = set(usable.code)
    result = [s for s in stock_files("P2") if s in allowed]
    missing_targets = set(TARGETS) - set(result)
    if missing_targets: raise ValueError(f"target(s) missing from dated P2 membership at {asof}: {missing_targets}")
    return result


def common_sequences(files: dict[str, Path], stocks: list[str]) -> list[tuple[str, tuple[str, ...]]]:
    all_maps = []
    for stock in TARGETS:
        df, _ = read_processed(files[stock]); dates = df.date.astype(str).tolist()
        mapping = {}
        for origin in range(p3.SEQ_LEN, len(dates) - p3.PRED_LEN + 1):
            mapping[tuple(dates[origin:origin + p3.PRED_LEN])] = dates[origin]
        all_maps.append(mapping)
    shared = set(all_maps[0]).intersection(*(set(x) for x in all_maps[1:]))
    return sorted(((all_maps[0][seq], seq) for seq in shared), key=lambda x: x[0])


def common_sessions(files: dict[str, Path]) -> list[str]:
    date_sets=[]
    for stock in TARGETS:
        df,_=read_processed(files[stock]); date_sets.append(set(df.date.astype(str)))
    return sorted(set.intersection(*date_sets))


def choose_dev_dates(files: dict[str, Path]) -> list[tuple[str, tuple[str, ...]]]:
    shared = common_sequences(files, TARGETS)
    selected = []
    for year in (2019, 2022, 2025):
        options = [x for x in shared if x[0].startswith(f"{year}-")]
        if not options: raise ValueError(f"no same-date 5-day target window for year {year}")
        anchor_day = pd.Timestamp(f"{year}-07-01")
        selected.append(min(options, key=lambda x: (abs(pd.Timestamp(x[0]) - anchor_day), x[0])))
    if len({x[0] for x in selected}) != 3: raise AssertionError("duplicate historical dates")
    return selected


def choose_recent_dates(files: dict[str, Path]) -> list[tuple[str, tuple[str, ...]]]:
    shared = [x for x in common_sequences(files, TARGETS) if x[0] <= AS_OF]
    by_origin={origin:(origin,seq) for origin,seq in shared}
    sessions=[d for d in common_sessions(files) if d<=AS_OF]
    tail=sessions[-40:]
    if len(tail) != 40: raise ValueError(f"only {len(tail)} aligned sessions in recent tail")
    starts=tail[::5]
    picked=[by_origin[d] for d in starts]
    if len(picked) != 8: raise AssertionError(f"expected 8 recent windows, got {len(picked)}")
    return picked


class MultiStockResidualDataset(Dataset):
    """Samples remain stock-local; windows are batched together, never concatenated."""
    def __init__(self, files: dict[str, Path], members: list[str], cutoff: str,
                 validation_start: str | None = None, origin_date: str | None = None):
        self.x, self.y, self.stock_ids, self.origin_dates, self.target_dates = [], [], [], [], []
        self.anchors, self.per_stock_stats, self.sample_counts = [], {}, defaultdict(int)
        self.train_cutoffs = {}
        for stock in members:
            if stock not in files: continue
            df, raw = read_processed(files[stock]); dates = df.date.astype(str).tolist()
            train_end = int(np.searchsorted(dates, cutoff, side="left"))
            if validation_start is None:
                split = train_end
                origins = range(p3.SEQ_LEN, split - p3.PRED_LEN + 1)
                self.train_cutoffs[stock] = dates[split - 1] if split else ""
            else:
                val_start = int(np.searchsorted(dates, validation_start, side="left"))
                test_start = int(np.searchsorted(dates, origin_date, side="left"))
                if not (p3.SEQ_LEN < val_start < min(test_start, len(dates))):
                    raise ValueError(f"invalid split for {stock}: train_end={val_start} test={test_start}")
                split = val_start
                # validation_start is the first known validation row; labels must end strictly before D.
                origins = range(val_start, test_start - p3.PRED_LEN + 1)
                self.train_cutoffs[stock] = dates[val_start - 1]
            if split <= p3.SEQ_LEN:
                continue
            vm, vs = p3.fit_volume_stats(raw, split)
            self.per_stock_stats[stock] = {"mean": vm, "scale": vs, "train_end_index": split,
                                           "train_cutoff": self.train_cutoffs[stock]}
            for o in origins:
                if o + p3.PRED_LEN > split and validation_start is None:
                    raise AssertionError("training label crosses training boundary")
                if validation_start is not None and dates[o + p3.PRED_LEN - 1] >= origin_date:
                    raise AssertionError("validation label reaches prediction date")
                x, anchor = p3.build_features(raw, o, vm, vs); y, _ = p3.build_target(raw, o)
                self.x.append(x); self.y.append(y); self.stock_ids.append(stock)
                self.origin_dates.append(dates[o]); self.target_dates.append(tuple(dates[o:o + p3.PRED_LEN]))
                self.anchors.append(anchor); self.sample_counts[stock] += 1
        if not self.x: raise ValueError(f"empty MultiStockResidualDataset before {cutoff}")
        self.x = np.stack(self.x); self.y = np.stack(self.y)
        self.stock_ids = np.asarray(self.stock_ids); self.anchors = np.asarray(self.anchors, dtype=np.float64)

    def __len__(self): return len(self.y)
    def __getitem__(self, index): return torch.from_numpy(self.x[index]), torch.from_numpy(self.y[index]), index

    def balanced_sampler(self, seed: int):
        counts = self.sample_counts
        weights = torch.as_tensor([1.0 / counts[s] for s in self.stock_ids], dtype=torch.double)
        generator = torch.Generator().manual_seed(seed)
        return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=generator)


def boundary_dates(files: dict[str, Path], origin: str, val_rows: int = 126) -> tuple[str, dict[str, str]]:
    # Per-security row windows end before the same calendar origin, preserving a common cutoff date.
    val_dates = {}
    for stock, path in files.items():
        df, _ = read_processed(path); dates = df.date.astype(str).tolist()
        test_start = int(np.searchsorted(dates, origin, side="left"))
        if test_start == 0: continue
        val_start = max(p3.SEQ_LEN + p3.PRED_LEN, min(test_start, len(dates)) - val_rows)
        if val_start >= len(dates): continue
        val_dates[stock] = dates[val_start]
    if not val_dates: raise ValueError("could not establish common train/validation boundaries")
    # A single timestamp applies to the entire pool; any later row is still excluded by per-stock searchsorted.
    return max(val_dates.values()), val_dates


def make_loaders(files, members, cutoff, origin, batch, seed, val_rows=126):
    val_start, starts = boundary_dates({s: files[s] for s in members if s in files}, origin, val_rows)
    # The global calendar split is the latest per-stock 126-row start, so every validation label remains pre-D.
    train = MultiStockResidualDataset(files, members, val_start)
    val = MultiStockResidualDataset(files, members, val_start, val_start, origin)
    sampler = train.balanced_sampler(seed)
    tr = DataLoader(train, batch_size=batch, sampler=sampler, pin_memory=torch.cuda.is_available())
    va = DataLoader(val, batch_size=batch, shuffle=False, pin_memory=torch.cuda.is_available())
    return train, val, tr, va, val_start, starts


def fit_shared(model_name, files, members, target_stock, origin, opts, checkpoint=None, small=False):
    model_files = {s: files[s] for s in members if s in files}
    # validation interval is the latest 126 pre-origin observations, defined on a common calendar cutoff.
    _, val_starts = boundary_dates(model_files, origin, opts.val_rows)
    val_start = max(val_starts.values())
    # Training data includes history strictly before the validation start; validation labels end before origin.
    train_ds = MultiStockResidualDataset(model_files, members, val_start)
    val_ds = MultiStockResidualDataset(model_files, members, val_start, val_start, origin)
    train_loader = DataLoader(train_ds, batch_size=opts.batch_size, sampler=train_ds.balanced_sampler(opts.seed),
                              pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=opts.batch_size, shuffle=False, pin_memory=torch.cuda.is_available())
    dev = torch.device("cuda" if opts.device == "auto" and torch.cuda.is_available() else ("cpu" if opts.device == "auto" else opts.device))
    if dev.type == "cuda": torch.cuda.reset_peak_memory_stats(dev)
    seed_all(opts.seed)
    model = p3.make_model(model_name, small=small).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr)
    best, best_epoch, stale, best_state = float("inf"), 0, 0, None
    start = time.time(); tr_loss = float("nan")
    for epoch in range(1, opts.epochs + 1):
        model.train(); losses=[]
        for x, y, _ in train_loader:
            x=x.to(dev, non_blocking=dev.type=="cuda"); y=y.to(dev, non_blocking=dev.type=="cuda")
            pred=p3.output_return(model,x); loss=torch.nn.functional.mse_loss(pred,y)
            if not torch.isfinite(loss): raise FloatingPointError("non-finite joint train loss")
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()): raise FloatingPointError("non-finite gradient")
            optimizer.step(); losses.append(loss.item())
        tr_loss=float(np.mean(losses)); model.eval(); by_stock=defaultdict(list)
        with torch.no_grad():
            for x,y,idx in val_loader:
                pred=p3.output_return(model,x.to(dev,non_blocking=dev.type=="cuda"))
                batch_loss=torch.nn.functional.mse_loss(pred,y.to(dev),reduction="none").mean(1).cpu().numpy()
                for i,l in zip(idx.tolist(),batch_loss): by_stock[val_ds.stock_ids[i]].append(float(l))
        val_loss=float(np.mean([np.mean(v) for v in by_stock.values()]))
        if val_loss < best - 1e-9:
            best,best_epoch,stale=val_loss,epoch,0; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else: stale+=1
        if stale>=opts.patience: break
    if best_state is None: raise RuntimeError("no valid joint checkpoint")
    model.load_state_dict(best_state); model.eval()
    target_predictions={}
    for target in (TARGETS if all(s in model_files for s in TARGETS) else [target_stock]):
        df,raw=read_processed(model_files[target]); idx=df.date.astype(str).tolist().index(origin)
        stats=train_ds.per_stock_stats.get(target)
        if stats is None: raise ValueError(f"target {target} has no legal train data")
        # Every stock keeps its own train-only log volume/amount scaler.
        x,anchor=p3.build_features(raw,idx,stats["mean"],stats["scale"])
        with torch.no_grad(): ret=p3.output_return(model,torch.from_numpy(x[None]).to(dev)).cpu().numpy()[0]
        if ret.shape!=(p3.PRED_LEN,) or not np.isfinite(ret).all(): raise ValueError("invalid pooled prediction")
        target_predictions[target]={"pred_return":ret,"anchor":anchor}
    sha=""
    if checkpoint:
        checkpoint.parent.mkdir(parents=True,exist_ok=True)
        torch.save({"state_dict":best_state,"model":model_name,"config":{"seed":opts.seed,"seq_len":60,"pred_len":5,
          "batch_size":opts.batch_size,"lr":opts.lr,"epochs":opts.epochs,"patience":opts.patience,"members":members,
          "train_cutoff_by_stock":train_ds.train_cutoffs,"volume_stats":{k:{"mean":v["mean"].tolist(),"scale":v["scale"].tolist()} for k,v in train_ds.per_stock_stats.items()}},
          "target_volume_stats":{"mean":stats["mean"],"scale":stats["scale"]}},checkpoint)
        sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    peak=torch.cuda.max_memory_allocated(dev)/(1024**2) if dev.type=="cuda" else 0.0
    return {"target_predictions":target_predictions,"best_epoch":best_epoch,"train_loss":tr_loss,"validation_loss":best,
      "runtime_seconds":time.time()-start,"peak_gpu_memory_mb":peak,"device":str(dev),"params":sum(p.numel() for p in model.parameters()),
      "checkpoint":str(checkpoint or ""),"checkpoint_sha256":sha,"training_samples":len(train_ds),
      "validation_samples":len(val_ds),"pool_members":members,"sample_counts":dict(train_ds.sample_counts),
      "validation_start":val_start,"train_cutoff_by_stock":train_ds.train_cutoffs}


def evaluate_one(pred_return, actual_return, anchor):
    return p3.residual_metrics(np.asarray(pred_return), np.asarray(actual_return), anchor)


def run_smoke(out: Path):
    cfg=json.loads((ROOT/"config/stocks.json").read_text(encoding="utf-8"))["stocks"][0]
    df,raw=read_processed(ROOT/"data/processed"/cfg["output_file"])
    train_end=200; vm,vs=p3.fit_volume_stats(raw,train_end)
    members=MultiStockResidualDataset({cfg["query_code"]:ROOT/"data/processed"/cfg["output_file"]},[cfg["query_code"]],df.date.iloc[train_end])
    fit=p3.finite_checkpoint_fit("DLinear",raw,160,vm,vs,"cpu",small=True)
    result={"status":"passed","dataset_samples":len(members),"train_cutoff":members.train_cutoffs,
      "balanced_sampler":len(members.balanced_sampler(2026).weights),"smoke_model":fit}
    out.mkdir(parents=True,exist_ok=True); (out/"cpu_smoke.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode",choices=["smoke","development","recent"],required=True)
    ap.add_argument("--device",default="auto"); ap.add_argument("--epochs",type=int,default=50)
    ap.add_argument("--batch-size",type=int,default=128); ap.add_argument("--patience",type=int,default=7)
    ap.add_argument("--lr",type=float,default=0.001); ap.add_argument("--seed",type=int,default=2026)
    ap.add_argument("--val-rows",type=int,default=126)
    ap.add_argument("--out",type=Path,default=OUT); ap.add_argument("--coal-membership",type=Path,default=ROOT/"data/coal_pool/manifests/industry_membership.csv")
    ap.add_argument("--test-max-fits",type=int,default=0,help="optional capped run for feasibility profiling")
    a=ap.parse_args()
    if a.mode=="smoke": return run_smoke(a.out/"smoke")
    stage=a.out/("development" if a.mode=="development" else "recent_40d")
    stage.mkdir(parents=True,exist_ok=True)
    files=stock_files("P2")
    if a.mode=="development": dates=choose_dev_dates({s:files[s] for s in TARGETS})
    else: dates=choose_recent_dates({s:files[s] for s in TARGETS})
    a.out.mkdir(parents=True,exist_ok=True)
    configs=[]; predictions=[]; runrows=[]; fit_count=0
    for pool in (("P0","P1","P2") if a.mode=="development" else ("P0","P1","P2")):
      for origin,target_dates in dates:
        members=membership_for(origin,pool)
        for model_name in MODELS:
          # P0 remains stock-specific; P1/P2 fit one shared model per model/date.
          shared_fit=None
          for target in TARGETS:
            if pool=="P0" and a.mode=="recent":
              p0=load_phase3_recent(target,model_name,target_dates)
              append_rows(predictions,a.mode,pool,target,model_name,origin,target_dates,p0,None)
              continue
            if shared_fit is None or pool=="P0":
              fit_members=[target] if pool=="P0" else members
              shared_fit=fit_shared(model_name,files,fit_members,target,origin,SimpleNamespace(**vars(a)),
                    a.out/"checkpoints"/f"{a.mode}_{pool}_{target.replace('.', '_')+'_' if pool=='P0' else ''}{model_name}_{origin}.pt")
              fit_count+=1
            target_fit=shared_fit["target_predictions"][target]
            df,raw=read_processed(files[target]); origin_idx=list(df.date.astype(str)).index(origin)
            actual,anchor=p3.build_target(raw,origin_idx)
            if not math.isclose(anchor,target_fit["anchor"],abs_tol=1e-8): raise AssertionError("anchor mismatch")
            date_list=tuple(df.date.iloc[origin_idx:origin_idx+5].astype(str))
            if date_list!=target_dates: raise AssertionError(f"target dates mismatch {target}")
            for alpha in ALPHAS:
              gated=alpha*target_fit["pred_return"]
              metrics=evaluate_one(gated,actual,anchor)
              predictions.append({"phase":a.mode,"pool":pool,"stock":target,"model":model_name,"origin_date":origin,
                "target_dates":";".join(date_list),"alpha":alpha,"anchor_close":anchor,
                "predicted_return":p3.strvec(gated),"actual_return":p3.strvec(actual),
                "predicted_open":p3.strvec(p3.restore_open(anchor,gated)),"actual_open":p3.strvec(raw[origin_idx:origin_idx+5,5]),**metrics,
                "best_epoch":shared_fit["best_epoch"],"train_loss":shared_fit["train_loss"],"validation_loss":shared_fit["validation_loss"],
                "runtime_seconds":shared_fit["runtime_seconds"],"params":shared_fit["params"],"training_samples":shared_fit["training_samples"],
                "checkpoint_sha256":shared_fit["checkpoint_sha256"]})
            runrows.append({"phase":a.mode,"pool":pool,"target_stock":target if pool=="P0" else "shared",
              "origin_date":origin,"model":model_name,**{k:v for k,v in shared_fit.items() if k not in ("target_predictions",)}})
            print(f"{a.mode} {pool} {model_name} {origin} -> {target}: epoch={shared_fit['best_epoch']} val={shared_fit['validation_loss']:.6g} sec={shared_fit['runtime_seconds']:.1f}",flush=True)
            if a.test_max_fits and fit_count>=a.test_max_fits: break
          if a.test_max_fits and fit_count>=a.test_max_fits: break
        if a.test_max_fits and fit_count>=a.test_max_fits: break
      if a.test_max_fits and fit_count>=a.test_max_fits: break
    pd.DataFrame(predictions).to_csv(stage/"joint_predictions.csv",index=False)
    run_df=pd.DataFrame(runrows)
    if not run_df.empty:
        run_df=run_df.drop_duplicates(["pool","target_stock","origin_date","model"],keep="first")
    run_df.to_csv(stage/"training_runs.csv",index=False)
    (stage/"selected_dates.json").write_text(json.dumps([{"origin":d,"target_dates":list(t)} for d,t in dates],indent=2),encoding="utf-8")
    hashes={}
    for s,path in files.items():
        if path.exists(): hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        import subprocess
        commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    except Exception: commit="unknown"
    (a.out/"configs").mkdir(parents=True,exist_ok=True)
    membership_config={}
    for d,_ in dates:
        membership_config[d]={}
        for pool in ("P1","P2"):
            try: membership_config[d][pool]=membership_for(d,pool)
            except FileNotFoundError:
                if not a.test_max_fits: raise
                membership_config[d][pool]=TARGETS
    (a.out/"configs"/f"{a.mode}.json").write_text(json.dumps({"phase":4,"mode":a.mode,"code_commit":commit,
        "origin_dates":[d for d,_ in dates],"pool_version":"dated BaoStock industry membership",
        "memberships_by_origin":membership_config,
        "seq_len":60,"pred_len":5,"target":"future_open / close[origin-1] - 1",
        "features":p3.COLS,"transform":"price relative anchor; per-stock log1p volume/amount scaler fit on train rows only",
        "loss":"MSE residual return","optimizer":"Adam","lr":a.lr,"epochs":a.epochs,"patience":a.patience,
        "batch_size":a.batch_size,"seed":a.seed,"device":a.device,"data_sha256":hashes},ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"completed fits={fit_count}; rows={len(predictions)}")


def load_phase3_recent(stock, model, target_dates):
    source=pd.read_csv(ROOT/"results/residual_phase3/recent_v1_windows.csv")
    rows=source[(source.stock==stock)&(source.model==model)&(source.target_dates==";".join(target_dates))]
    if len(rows)!=1: raise ValueError(f"Phase 3 P0 exact-date row missing for {stock} {model} {target_dates}")
    row=rows.iloc[0]
    return {"pred_return":p3.parse_vec(row.predicted_return),"anchor":float(row.anchor_close),
            "fit":row.to_dict()}


def append_rows(predictions, phase, pool, target, model, origin, target_dates, p0, fit):
    actual=p3.parse_vec(p0["fit"]["actual_return"]); anchor=p0["anchor"]
    for alpha in ALPHAS:
        ret=alpha*p0["pred_return"]
        predictions.append({"phase":phase,"pool":pool,"stock":target,"model":model,"origin_date":origin,
          "target_dates":";".join(target_dates),"alpha":alpha,"anchor_close":anchor,
          "predicted_return":p3.strvec(ret),"actual_return":p3.strvec(actual),
          "predicted_open":p3.strvec(p3.restore_open(anchor,ret)),"actual_open":p0["fit"]["actual_open"],
          **p3.residual_metrics(ret,actual,anchor),"best_epoch":"","train_loss":"","validation_loss":"",
          "runtime_seconds":"","params":"","training_samples":"","checkpoint_sha256":""})


if __name__=="__main__": main()
