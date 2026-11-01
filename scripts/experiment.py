"""Leakage-aware official-model trainer for CPU smoke and expanding windows."""
from __future__ import annotations
import argparse, csv, hashlib, importlib, json, math, os, random, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "Time-Series-Library"
sys.path.insert(0, str(LIB))
COLS = ["high", "low", "close", "volume", "amount", "open"]
TARGET = 5
MODELS = ["DLinear", "iTransformer", "TimeXer"]

def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def read_stocks() -> list[dict]:
    return json.loads((ROOT/"config/stocks.json").read_text(encoding="utf-8"))["stocks"]

def make_model(name: str, small: bool=False) -> torch.nn.Module:
    cfg = SimpleNamespace(task_name="long_term_forecast", features="MS", seq_len=60, label_len=30,
        pred_len=5, enc_in=6, dec_in=6, c_out=6, d_model=16 if small else 128,
        d_ff=32 if small else 256, e_layers=1 if small else 2, n_heads=4,
        dropout=0.1, patch_len=12, moving_avg=25, factor=3, embed="timeF", freq="d",
        activation="gelu", use_norm=True, output_attention=False, individual=False)
    return importlib.import_module(f"models.{name}").Model(cfg)

def matrix(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(path)
    if df.columns.tolist() != ["date", *COLS]:
        raise ValueError(f"Unexpected model columns in {path}: {df.columns.tolist()}")
    return df, df[COLS].to_numpy(dtype=np.float32)

def fit_scale(arr: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    mean=arr.mean(0); scale=arr.std(0); scale[scale < 1e-8]=1.0
    return mean.astype(np.float32),scale.astype(np.float32)

def windows(scaled: np.ndarray, origins: range) -> TensorDataset:
    xs=np.stack([scaled[o-60:o] for o in origins])
    ys=np.stack([scaled[o:o+5] for o in origins])
    return TensorDataset(torch.from_numpy(xs),torch.from_numpy(ys))

def output_open(model, x: torch.Tensor) -> torch.Tensor:
    dec=torch.zeros((x.shape[0],35,6),dtype=x.dtype,device=x.device)
    y=model(x,None,dec,None)
    if y is None or y.shape[1] != 5 or y.shape[2] not in (1,6):
        raise ValueError(f"Unexpected {model.__class__.__name__} output shape {None if y is None else tuple(y.shape)}")
    return y[..., -1]

def fit_evaluate(name: str, raw: np.ndarray, train_end: int, val_start: int,
                 test_origin: int|None, opts, checkpoint: Path|None=None) -> dict:
    mean,scale=fit_scale(raw[:train_end]); scaled=((raw-mean)/scale).astype(np.float32)
    train_origins=range(60,train_end-5+1)
    val_origins=range(val_start, (test_origin if test_origin is not None else len(raw))-5+1)
    if len(train_origins)<1 or len(val_origins)<1: raise ValueError("No legal train/validation windows")
    train_dl=DataLoader(windows(scaled,train_origins),batch_size=opts.batch_size,shuffle=True)
    val_dl=DataLoader(windows(scaled,val_origins),batch_size=opts.batch_size,shuffle=False)
    device=torch.device("cuda" if opts.device=="auto" and torch.cuda.is_available() else ("cpu" if opts.device=="auto" else opts.device))
    seed_all(opts.seed)
    if device.type=="cuda": torch.cuda.reset_peak_memory_stats(device)
    model=make_model(name,opts.small).to(device)
    nparams=sum(p.numel() for p in model.parameters())
    opt=torch.optim.Adam(model.parameters(),lr=opts.lr)
    best=float("inf"); best_epoch=0; stale=0; best_state=None; train_loss=float("nan"); start=time.time()
    for epoch in range(1,opts.epochs+1):
        model.train(); losses=[]
        for x,y in train_dl:
            x=x.to(device); target=y[:,:,-1].to(device)
            pred=output_open(model,x)
            loss=torch.nn.functional.mse_loss(pred,target)
            if not torch.isfinite(loss): raise FloatingPointError("Non-finite training loss")
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("Non-finite model gradient")
            losses.append(loss.item())
        train_loss=float(np.mean(losses)); model.eval(); vals=[]
        with torch.no_grad():
            for x,y in val_dl:
                pred=output_open(model,x.to(device)); target=y[:,:,-1].to(device)
                vals.append(torch.nn.functional.mse_loss(pred,target).item())
        val_loss=float(np.mean(vals))
        if val_loss < best-1e-9:
            best=val_loss; best_epoch=epoch; stale=0
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else: stale+=1
        if stale >= opts.patience: break
    if best_state is None: raise RuntimeError("No finite validation checkpoint")
    model.load_state_dict(best_state)
    if any(not torch.isfinite(p).all() for p in model.parameters()): raise FloatingPointError("Non-finite model parameter")
    prediction=None
    if test_origin is not None:
        model.eval()
        with torch.no_grad(): prediction=output_open(model,torch.from_numpy(scaled[test_origin-60:test_origin][None]).to(device)).cpu().numpy()[0]
        prediction=prediction*scale[-1]+mean[-1]
    if checkpoint:
        checkpoint.parent.mkdir(parents=True,exist_ok=True)
        torch.save({"state_dict":best_state,"model":name,"best_epoch":best_epoch,"mean":mean,"scale":scale,"config":{"seed":opts.seed,"small":opts.small}},checkpoint)
    peak_mem=torch.cuda.max_memory_allocated(device)/(1024**2) if device.type=="cuda" else 0.0
    return {"prediction":prediction,"best_epoch":best_epoch,"train_loss":train_loss,"val_loss":best,
            "params":nparams,"runtime_seconds":time.time()-start,"peak_gpu_memory_mb":peak_mem,"device":str(device),"mean":mean,"scale":scale}

def fit_final(name: str, raw: np.ndarray, epochs: int, opts, checkpoint: Path) -> dict:
    mean,scale=fit_scale(raw); scaled=((raw-mean)/scale).astype(np.float32)
    origins=range(60,len(raw)-5+1)
    dl=DataLoader(windows(scaled,origins),batch_size=opts.batch_size,shuffle=True)
    device=torch.device("cuda" if opts.device=="auto" and torch.cuda.is_available() else ("cpu" if opts.device=="auto" else opts.device))
    seed_all(opts.seed); model=make_model(name,opts.small).to(device); opt=torch.optim.Adam(model.parameters(),lr=opts.lr)
    losses=[]; start=time.time()
    for epoch in range(1,epochs+1):
        model.train(); ep=[]
        for x,y in dl:
            x=x.to(device); target=y[:,:,-1].to(device); pred=output_open(model,x)
            loss=torch.nn.functional.mse_loss(pred,target)
            if not torch.isfinite(loss): raise FloatingPointError("Non-finite final training loss")
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); ep.append(loss.item())
        losses.append(float(np.mean(ep)))
    checkpoint.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"state_dict":model.state_dict(),"model":name,"epochs":epochs,"mean":mean,"scale":scale},checkpoint)
    model.eval()
    with torch.no_grad(): pred=output_open(model,torch.from_numpy(scaled[-60:][None]).to(device)).cpu().numpy()[0]
    pred=pred*scale[-1]+mean[-1]
    return {"prediction":pred,"mean":mean,"scale":scale,"params":sum(p.numel() for p in model.parameters()),
            "train_loss":losses[-1],"runtime_seconds":time.time()-start,"checkpoint":str(checkpoint)}

def predict_checkpoint(name: str, raw: np.ndarray, origin: int, checkpoint: Path, device: str="cpu") -> np.ndarray:
    saved=torch.load(checkpoint,map_location="cpu",weights_only=False)
    mean=np.asarray(saved["mean"],dtype=np.float32); scale=np.asarray(saved["scale"],dtype=np.float32)
    scaled=((raw-mean)/scale).astype(np.float32)
    model=make_model(name,small=bool(saved.get("config",{}).get("small",False)))
    model.load_state_dict(saved["state_dict"]); model.to(device).eval()
    with torch.no_grad(): pred=output_open(model,torch.from_numpy(scaled[origin-60:origin][None]).to(device)).cpu().numpy()[0]
    return pred*scale[-1]+mean[-1]

def nmse_metrics(pred: np.ndarray,true: np.ndarray,baseline: float) -> dict:
    err=pred-true; return {"nmse":float(np.mean((err/baseline)**2)),"mae":float(np.mean(np.abs(err))),"rmse":float(np.sqrt(np.mean(err**2)))}

def get_opts(args):
    return SimpleNamespace(batch_size=args.batch_size,epochs=args.epochs,patience=args.patience,
        lr=args.lr,seed=args.seed,device=args.device,small=args.small)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=["smoke","development","recent","final"],default="smoke")
    p.add_argument("--stock",default="sh.601088"); p.add_argument("--epochs",type=int,default=50)
    p.add_argument("--batch-size",type=int,default=32); p.add_argument("--patience",type=int,default=7)
    p.add_argument("--lr",type=float,default=.001); p.add_argument("--seed",type=int,default=2026)
    p.add_argument("--device",default="auto"); p.add_argument("--small",action="store_true")
    args=p.parse_args()
    if args.mode == "smoke":
        args.epochs=1; args.batch_size=8; args.patience=1; args.small=True
        if args.device == "auto": args.device="cpu"
    opts=get_opts(args)
    paths={s["query_code"]:ROOT/"data/processed"/s["output_file"] for s in read_stocks()}
    stock_cfg=[s for s in read_stocks() if args.mode!="smoke" or s["query_code"]==args.stock]
    if args.mode == "final":
        selection=json.loads((ROOT/"results/metrics/model_selection.json").read_text(encoding="utf-8"))
        dates=json.loads((ROOT/"config/prediction_dates.json").read_text(encoding="utf-8"))["dates"]
        details=[]; metadata=[]
        for s in stock_cfg:
            df,raw=matrix(paths[s["query_code"]]); model_name=selection["stocks"][s["query_code"]]["selected_model"]
            if df.date.iloc[-1] != "2026-10-30": raise ValueError(f"{s['query_code']}: processed history does not end on 2026-10-30")
            if model_name=="LastCloseBaseline":
                pred=np.full(5,float(raw[-1,2])); ckpt=""
            else:
                best=[r["best_epoch"] for r in pd.read_csv(ROOT/"results/metrics/development_windows.csv").to_dict("records") if r["stock"]==s["query_code"] and r["model"]==model_name]
                epochs=max(1,int(np.median(best))) if best else 1
                ckpt=ROOT/"results/checkpoints"/f"final_{s['query_code'].replace('.','_')}_{model_name}.pt"
                result=fit_final(model_name,raw,epochs,opts,ckpt); pred=result["prediction"]
                np.savez(ckpt.with_suffix(".npz"),mean=result["mean"],scale=result["scale"])
                ckpt=str(ckpt)
            if not np.isfinite(pred).all() or (pred<=0).any(): raise ValueError(f"Invalid final predictions for {s['query_code']}")
            for d,v in zip(dates,pred): details.append({"stock":s["query_code"],"date":d,"open":float(v)})
            ckpt_hash=hashlib.sha256(Path(ckpt).read_bytes()).hexdigest() if ckpt else ""
            metadata.append({"stock":s["query_code"],"model":model_name,"prediction_source":"LastCloseBaseline" if model_name=="LastCloseBaseline" else "trained_model",
                "training_cutoff":df.date.iloc[-1],"last_input_date":df.date.iloc[-1],"last_known_close":float(raw[-1,2]),
                "source_csv_sha256":hashlib.sha256(paths[s["query_code"]].read_bytes()).hexdigest(),
                "checkpoint":ckpt,"checkpoint_sha256":ckpt_hash,"predicted_dates":dates})
        out=ROOT/"results/predictions"; out.mkdir(parents=True,exist_ok=True)
        pd.DataFrame(details).to_csv(out/"prediction_details.csv",index=False)
        (out/"metadata.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
        print(f"Final inference written to {out/'prediction_details.csv'}")
        return
    rows=[]
    if args.mode == "recent":
        # Locked architecture/config; a new model is fitted at day 0,20,... .
        training_runs=[]
        for s in stock_cfg:
            df,raw=matrix(paths[s["query_code"]]); n=len(raw); start=n-180
            if start < 60+126: raise ValueError(f"Not enough pre-holdout history for {s['query_code']}")
            origin_list=list(range(start,n-5+1,5)); fold_rows=[]
            for model_name in MODELS:
                saved=None; train_origin=None; ckpt=None
                for wi,origin in enumerate(origin_list):
                    if wi % 4 == 0:
                        train_end=origin-126
                        ckpt=ROOT/"results/checkpoints"/f"recent_{s['query_code'].replace('.','_')}_{model_name}_{df.date.iloc[origin]}.pt"
                        saved=fit_evaluate(model_name,raw,train_end,train_end,origin,opts,ckpt)
                        train_origin=origin
                        training_runs.append({"stock":s["query_code"],"model":model_name,"retrain_origin":df.date.iloc[origin],
                            "train_cutoff":df.date.iloc[train_end-1],"validation_start":df.date.iloc[train_end],
                            "validation_end":df.date.iloc[origin-1],"best_epoch":saved["best_epoch"],
                            "train_loss":saved["train_loss"],"validation_loss":saved["val_loss"],
                            "parameter_count":saved["params"],"runtime_seconds":saved["runtime_seconds"],
                            "peak_gpu_memory_mb":saved["peak_gpu_memory_mb"],"device":saved["device"],"checkpoint":str(ckpt),
                            "checkpoint_sha256":hashlib.sha256(ckpt.read_bytes()).hexdigest()})
                    else:
                        saved["prediction"]=predict_checkpoint(model_name,raw,origin,ckpt,opts.device if opts.device!="auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
                    true=raw[origin:origin+5,-1]; baseline=float(raw[origin-1,2])
                    metrics=nmse_metrics(saved["prediction"],true,baseline)
                    fold_rows.append({"stock":s["query_code"],"model":model_name,"origin_date":df.date.iloc[origin],
                        "target_dates":";".join(df.date.iloc[origin:origin+5]),"actual_open":";".join(f"{x:.6f}" for x in true),
                        "predicted_open":";".join(f"{x:.6f}" for x in saved["prediction"]),"baseline_close":baseline,
                        "training_cutoff":df.date.iloc[train_origin-127],"retrain_origin":df.date.iloc[train_origin],"checkpoint":str(ckpt),
                        "best_epoch":saved["best_epoch"],"train_loss":saved["train_loss"],"validation_loss":saved["val_loss"],
                        "retraining_runtime_seconds":saved["runtime_seconds"],"peak_gpu_memory_mb":saved["peak_gpu_memory_mb"],**metrics})
            for wi,origin in enumerate(origin_list):
                true=raw[origin:origin+5,-1]; baseline=float(raw[origin-1,2]); bpred=np.full(5,baseline)
                fold_rows.append({"stock":s["query_code"],"model":"LastCloseBaseline","origin_date":df.date.iloc[origin],
                    "target_dates":";".join(df.date.iloc[origin:origin+5]),"actual_open":";".join(f"{x:.6f}" for x in true),
                    "predicted_open":";".join(f"{x:.6f}" for x in bpred),"baseline_close":baseline,
                    "training_cutoff":df.date.iloc[origin-1],"retrain_origin":"","checkpoint":"",**nmse_metrics(bpred,true,baseline)})
            rows.extend(fold_rows)
            print(f"{s['query_code']}: recent rolling windows {len(origin_list)} per model",flush=True)
        out=ROOT/"results/metrics"; out.mkdir(parents=True,exist_ok=True)
        result_df=pd.DataFrame(rows); result_df.to_csv(out/"recent_windows.csv",index=False)
        result_df.groupby(["stock","model"],as_index=False).agg(mean_nmse=("nmse","mean"),median_nmse=("nmse","median"),std_nmse=("nmse","std"),mean_mae=("mae","mean"),mean_rmse=("rmse","mean"),windows=("nmse","count")).to_csv(out/"recent_summary.csv",index=False)
        pd.DataFrame(training_runs).to_csv(out/"recent_training_runs.csv",index=False)
        return

    rows=[]
    for s in stock_cfg:
        df,raw=matrix(paths[s["query_code"]]); n=len(raw)
        if args.mode=="smoke":
            manifest=json.loads((ROOT/"data/splits"/(Path(s["output_file"]).stem+".json")).read_text(encoding="utf-8"))
            origin=manifest["splits"][0]["origin_index"]; train_end=origin-126
            folds=[{"fold_id":"cpu_smoke","origin":origin,"val_start":train_end}]
        else:
            manifest=json.loads((ROOT/"data/splits"/(Path(s["output_file"]).stem+".json")).read_text(encoding="utf-8"))
            folds=[{"fold_id":f["fold_id"],"origin":f["origin_index"],"val_start":f["segments"]["val"][0]} for f in manifest["splits"]]
        for fold in folds:
            o,vs=fold["origin"],fold["val_start"]
            true=raw[o:o+5,-1]; baseline=float(raw[o-1,2])
            for model_name in MODELS:
                ckpt=ROOT/"results/checkpoints"/f"{args.mode}_{s['query_code'].replace('.','_')}_{model_name}_{fold['fold_id']}.pt"
                result=fit_evaluate(model_name,raw,vs,vs,o,opts,ckpt)
                metrics=nmse_metrics(result["prediction"],true,baseline)
                reload_error=float("nan")
                if args.mode=="smoke":
                    reloaded=predict_checkpoint(model_name,raw,o,ckpt,"cpu")
                    reload_error=float(np.max(np.abs(reloaded-result["prediction"])))
                rows.append({"stock":s["query_code"],"model":model_name,"fold":fold["fold_id"],
                    "origin_date":df.date.iloc[o],"target_dates":";".join(df.date.iloc[o:o+5]),
                    "actual_open":";".join(map(lambda x:f"{x:.6f}",true)),
                    "predicted_open":";".join(map(lambda x:f"{x:.6f}",result["prediction"])),
                    "baseline_close":baseline,"train_cutoff":df.date.iloc[vs-1],"best_epoch":result["best_epoch"],
                    "train_loss":result["train_loss"],"validation_loss":result["val_loss"],"parameter_count":result["params"],
                    "runtime_seconds":result["runtime_seconds"],"peak_gpu_memory_mb":result["peak_gpu_memory_mb"],"device":result["device"],
                    "checkpoint":str(ckpt),"checkpoint_reload_max_abs_error":reload_error,**metrics})
                print(f"{s['query_code']} {model_name} {fold['fold_id']}: NMSE={metrics['nmse']:.6g}, best_epoch={result['best_epoch']}",flush=True)
            bpred=np.full(5,baseline,dtype=float); bm=nmse_metrics(bpred,true,baseline)
            rows.append({"stock":s["query_code"],"model":"LastCloseBaseline","fold":fold["fold_id"],
                "origin_date":df.date.iloc[o],"target_dates":";".join(df.date.iloc[o:o+5]),
                "actual_open":";".join(map(lambda x:f"{x:.6f}",true)),"predicted_open":";".join(map(lambda x:f"{x:.6f}",bpred)),
                "baseline_close":baseline,"train_cutoff":df.date.iloc[o-1],"best_epoch":"","train_loss":"","validation_loss":"",
                "parameter_count":0,"runtime_seconds":0,"checkpoint":"",**bm})
    out=ROOT/"results/metrics"; out.mkdir(parents=True,exist_ok=True)
    result_df=pd.DataFrame(rows)
    outname={"smoke":"smoke_windows.csv","development":"development_windows.csv","recent":"recent_windows.csv"}[args.mode]
    result_df.to_csv(out/outname,index=False)
    if args.mode=="development":
        summary=result_df.groupby(["stock","model"],as_index=False).agg(mean_nmse=("nmse","mean"),median_nmse=("nmse","median"),std_nmse=("nmse","std"),mean_mae=("mae","mean"),mean_rmse=("rmse","mean"),windows=("nmse","count"))
        summary["std_nmse"]=summary["std_nmse"].fillna(0.0)
        summary.to_csv(out/"development_summary.csv",index=False)
        baseline_summary=summary[summary.model=="LastCloseBaseline"][["stock","mean_nmse"]].rename(columns={"mean_nmse":"baseline_mean_nmse"})
        comparison=summary.merge(baseline_summary,on="stock",how="left")
        comparison["relative_nmse_improvement_vs_baseline"]=(comparison["baseline_mean_nmse"]-comparison["mean_nmse"])/comparison["baseline_mean_nmse"]
        comparison.to_csv(out/"model_comparison.csv",index=False)
        decisions={}
        for stock,group in summary.groupby("stock"):
            baseline=float(group.loc[group.model=="LastCloseBaseline","mean_nmse"].iloc[0])
            deep=group[group.model.isin(MODELS)].sort_values(["mean_nmse","std_nmse"])
            chosen=deep.iloc[0] if len(deep) and float(deep.iloc[0].mean_nmse)<baseline else group[group.model=="LastCloseBaseline"].iloc[0]
            decisions[stock]={"selected_model":str(chosen.model),"mean_nmse":float(chosen.mean_nmse),"baseline_mean_nmse":baseline,
                "rule":"lowest development mean NMSE; choose LastCloseBaseline if all deep models are worse"}
        (out/"model_selection.json").write_text(json.dumps({"selection_basis":"development only; recent holdout not used","stocks":decisions},ensure_ascii=False,indent=2),encoding="utf-8")
    if args.mode=="recent":
        result_df.groupby(["stock","model"],as_index=False).agg(mean_nmse=("nmse","mean"),median_nmse=("nmse","median"),std_nmse=("nmse","std"),mean_mae=("mae","mean"),mean_rmse=("rmse","mean"),windows=("nmse","count")).to_csv(out/"recent_summary.csv",index=False)
    if args.mode=="smoke": print("Local CPU/GPU smoke window output written; this is not the full experiment.")

if __name__=="__main__": main()
