"""Train Phase6 accepted P2_ID model(s) through 2026-10-30 and create draft forecasts.

Fallback stocks retain Phase5 submitted predictions. Export/validate separately using
existing scripts/export_submission.py and scripts/validate_submission.py.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import phase4_joint as p4
import phase5_joint as p5
import residual_phase3 as p3
import phase6_stock_id as p6
from phase6_id_model import batch_ids


def train_final(name, members, vocab, files, opts, epochs, path):
    ds = p4.MultiStockResidualDataset(files, members, "2026-10-31")
    dev = p5.device_from(opts.device)
    p5.set_seed(opts.seed)
    loader = DataLoader(ds, batch_size=opts.batch_size, sampler=ds.balanced_sampler(opts.seed),
                        pin_memory=(dev.type == "cuda"))
    model = p6.make_model(name, vocab, opts.embedding_dim).to(dev)
    optim = torch.optim.Adam(model.parameters(), lr=opts.lr)
    started = time.time()
    for epoch in range(1, epochs+1):
        model.train(); total=count=0
        for x,y,idx in loader:
            stock_ids = batch_ids(ds.stock_ids, idx, vocab, dev)
            x=x.to(dev,non_blocking=dev.type=="cuda"); y=y.to(dev,non_blocking=dev.type=="cuda")
            output=model(x,stock_ids)
            mse=torch.nn.functional.mse_loss(output,y)
            loss=mse + opts.id_l2*model.id_correction(stock_ids).square().mean()
            if not torch.isfinite(loss).item(): raise FloatingPointError("non-finite Phase6 final loss")
            optim.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optim.step()
            total+=float(mse.item())*len(x); count+=len(x)
        print(f"final P2_ID {name} {epoch}/{epochs} MSE={total/count:.8g}",flush=True)
    checkpoint = {"state_dict": {k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                  "model": name, "pool": "P2_ID", "origin": "2026-11-02", "vocab": vocab,
                  "embedding_dim": opts.embedding_dim, "id_l2": opts.id_l2,
                  "members": members, "epochs": epochs, "seed": opts.seed,
                  "data_version": "phase5-targetaware",
                  "volume_stats": {s:{"mean":v["mean"].tolist(),"scale":v["scale"].tolist()}
                                   for s,v in ds.per_stock_stats.items()}}
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(checkpoint,path)
    return {"checkpoint":str(path), "checkpoint_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
            "epochs":epochs,"training_windows":len(ds),"seconds":time.time()-started}


def forecast_final(ckpt, files, stock, device):
    saved = torch.load(ckpt,map_location="cpu",weights_only=False)
    df,raw=p4.read_processed(files[stock]); last=str(df.date.iloc[-1])
    if last!="2026-10-30": raise ValueError(f"Unexpected final input date {stock}: {last}")
    vm=saved["volume_stats"][stock]
    x,anchor=p3.build_features(raw,len(raw),np.asarray(vm["mean"]),np.asarray(vm["scale"]))
    dev=p5.device_from(device)
    model=p6.make_model(saved["model"],saved["vocab"],saved["embedding_dim"]).to(dev)
    model.load_state_dict(saved["state_dict"]); model.eval()
    with torch.no_grad():
        ret=model(torch.from_numpy(x[None]).to(dev),
                  torch.tensor([saved["vocab"].index(stock)],dtype=torch.long,device=dev)).cpu().numpy()[0]
    return anchor,ret


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device",default="auto")
    ap.add_argument("--student-id",required=True)
    ap.add_argument("--out",type=Path,default=p6.OUT_DEFAULT)
    opts=ap.parse_args(); root=opts.out
    selection=json.loads((root/"final_model_selection.json").read_text(encoding="utf8"))["stocks"]
    dev_info=pd.read_csv(root/"development/training_runs.csv")
    cfg=json.loads((root/"development/run_config.json").read_text(encoding="utf8"))
    cfg_recent=json.loads((root/"recent/run_config.json").read_text(encoding="utf8"))
    for prop in ("seed","lr","batch_size","embedding_dim","id_l2"):
        if cfg[prop]!=cfg_recent[prop]: raise ValueError(f"Dev/recent config mismatch {prop}")
    run_opts=argparse.Namespace(seed=cfg["seed"],lr=cfg["lr"],batch_size=cfg["batch_size"],
                                embedding_dim=cfg["embedding_dim"],id_l2=cfg["id_l2"],device=opts.device)
    files=p4.stock_files("P2")
    members=p4.membership_for("2026-10-30","P2")
    vocab=p6.get_vocab()
    previous=pd.read_csv(ROOT/"results/phase5_targetaware/final/prediction_details.csv")
    original_meta=json.loads((ROOT/"results/phase5_targetaware/final/metadata.json").read_text(encoding="utf8"))
    dates=json.loads((ROOT/"config/prediction_dates.json").read_text(encoding="utf8"))["dates"]
    assert dates==["2026-11-02","2026-11-03","2026-11-04","2026-11-05","2026-11-06"]
    fitting={}; result=[]; meta=[]
    for stock in p4.TARGETS:
        choice=selection[stock]
        if choice["pool"]=="P2_ID":
            name=choice["model"]
            if name not in fitting:
                q=dev_info[dev_info.model.eq(name)]
                if len(q)!=3: raise ValueError(f"Missing dev fits for {name}")
                epochs=max(1,int(np.median(q.best_epoch.astype(int))))
                path=root/"final/checkpoints"/f"P2_ID_{name}_2026-10-30.pt"
                fitting[name]=train_final(name,members,vocab,files,run_opts,epochs,path)
            fitted=fitting[name]
            anchor,ret=forecast_final(fitted["checkpoint"],files,stock,opts.device)
            values=p3.restore_open(anchor,float(choice["alpha"])*ret)
            item={"stock":stock,"method":"Phase6-P2_ID","pool":"P2_ID","model":name,
                  "alpha":float(choice["alpha"]),"prediction_source":"trained_model",
                  "checkpoint":fitted["checkpoint"],"checkpoint_sha256":fitted["checkpoint_sha256"]}
        else:
            p=previous[previous.stock.eq(stock)].set_index("date").loc[dates]
            values=p.open.to_numpy(dtype=float)
            item=next(m.copy() for m in original_meta if m["stock"]==stock)
            item["fallback_from_phase5"]=True
        if len(values)!=5 or not np.isfinite(values).all() or np.any(values<=0):
            raise ValueError(f"Invalid forecast for {stock}")
        for date,value in zip(dates,values):
            result.append({"stock":stock,"date":date,"open":float(value)})
        item.update({"training_cutoff":"2026-10-30","last_input_date":"2026-10-30","predicted_dates":dates})
        meta.append(item)
    final=root/"final";final.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(result).to_csv(final/"prediction_details.csv",index=False)
    (final/"metadata.json").write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding="utf8")
    print(f"Created {final/'prediction_details.csv'}; run export_submission and validate_submission")

if __name__=="__main__":main()
