"""Fit selected Phase 4 joint model on all legal history and export isolated submissions."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import phase4_joint as p4
import residual_phase3 as p3

OUT = p4.OUT
DATES = json.loads((ROOT / "config/prediction_dates.json").read_text(encoding="utf-8"))["dates"]


def main():
    selection = json.loads((OUT / "final_model_selection.json").read_text(encoding="utf-8"))["stocks"]
    candidates = [s for s, choice in selection.items() if choice["pool"] in ("P1", "P2")]
    if not candidates:
        print("No Phase 4 candidate passed selection; Phase 3 predictions remain authoritative.")
        return
    if len(set((selection[s]["pool"], selection[s]["model"]) for s in candidates)) != 1:
        raise ValueError("selected joint candidates require more than one final shared model")

    first = candidates[0]
    choice = selection[first]
    pool, model_name = choice["pool"], choice["model"]
    files = p4.stock_files("P2")
    members = p4.membership_for(p4.AS_OF, pool)
    run_path = OUT / "development" / "training_runs.csv"
    runs = pd.read_csv(run_path)
    dev = runs[(runs.pool == pool) & (runs.model == model_name)]
    epochs = max(1, int(np.median(dev.best_epoch.astype(int))))
    dataset = p4.MultiStockResidualDataset(files, members, "2026-10-31")
    opts = SimpleNamespace(seed=2026, device="cuda" if torch.cuda.is_available() else "cpu",
                           batch_size=512, lr=0.001)
    p4.seed_all(opts.seed)
    loader = DataLoader(dataset, batch_size=opts.batch_size,
                        sampler=dataset.balanced_sampler(opts.seed),
                        pin_memory=torch.cuda.is_available())
    device = torch.device(opts.device)
    model = p3.make_model(model_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr)
    losses = []
    started = time.time()
    for epoch in range(1, epochs + 1):
        model.train(); batch_losses = []
        for x, y, _ in loader:
            x=x.to(device, non_blocking=device.type == "cuda")
            y=y.to(device, non_blocking=device.type == "cuda")
            pred=p3.output_return(model,x); loss=torch.nn.functional.mse_loss(pred,y)
            if not torch.isfinite(loss): raise FloatingPointError("non-finite final joint loss")
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("non-finite final joint gradient")
            optimizer.step(); batch_losses.append(float(loss.item()))
        losses.append(float(np.mean(batch_losses)))
        print(f"final {pool} {model_name} epoch={epoch}/{epochs} loss={losses[-1]:.7g}",flush=True)

    ckpt = OUT / "checkpoints" / f"final_{pool}_{model_name}_2026-10-30.pt"
    ckpt.parent.mkdir(parents=True,exist_ok=True)
    stats={s:{"mean":v["mean"].tolist(),"scale":v["scale"].tolist()} for s,v in dataset.per_stock_stats.items()}
    torch.save({"state_dict":model.state_dict(),"model":model_name,"pool":pool,
                "config":{"seed":opts.seed,"seq_len":60,"pred_len":5,"batch_size":opts.batch_size,
                          "learning_rate":opts.lr,"epochs":epochs,"train_cutoff":"2026-10-30",
                          "pool_members":members,"train_windows":dict(dataset.sample_counts),
                          "volume_stats":stats}},ckpt)
    checkpoint_sha=hashlib.sha256(ckpt.read_bytes()).hexdigest()

    model_rows=[]; final_rows=[]; metadata=[]
    for stock in p4.TARGETS:
        df,raw=p4.read_processed(files[stock]); stats_stock=dataset.per_stock_stats[stock]
        x,anchor=p3.build_features(raw,len(raw),stats_stock["mean"],stats_stock["scale"])
        model.eval()
        with torch.no_grad():
            pred_return=p3.output_return(model,torch.from_numpy(x[None]).to(device)).cpu().numpy()[0]
        if not np.isfinite(pred_return).all(): raise FloatingPointError(f"invalid final prediction for {stock}")
        model_rows.append({"stock":stock,"anchor_close":anchor,
                           "predicted_return":p3.strvec(pred_return),
                           "predicted_open":p3.strvec(p3.restore_open(anchor,pred_return))})
        selected=selection[stock]
        if selected["pool"] in ("P1","P2"):
            alpha=float(selected["alpha"])
            pred_open=p3.restore_open(anchor,alpha*pred_return)
            method=f"{selected['pool']}-Joint-Residual"
            used_model=model_name; used_alpha=alpha; source="trained_model"
            stock_checkpoint=ckpt.relative_to(ROOT).as_posix(); stock_sha=checkpoint_sha; epoch_count=epochs
        else:
            phase3_meta=json.loads((ROOT/"results/residual_phase3/final/metadata.json").read_text(encoding="utf-8"))
            m=next(x for x in phase3_meta if x["stock"]==stock)
            phase3_detail=pd.read_csv(ROOT/"results/residual_phase3/final/prediction_details.csv")
            prior=phase3_detail[phase3_detail.stock.eq(stock)].set_index("date").loc[DATES]
            pred_open=prior.open.to_numpy(dtype=float)
            method=m["method"]; used_model=m["model"]; used_alpha=m["alpha"]
            source="LastCloseBaseline"; stock_checkpoint=""; stock_sha=""; epoch_count=None
        for day,value in zip(DATES,pred_open):
            final_rows.append({"stock":stock,"date":day,"open":float(value)})
        metadata.append({"stock":stock,"method":method,"model":used_model,"alpha":used_alpha,
                         "prediction_source":source,"training_cutoff":"2026-10-30","last_input_date":"2026-10-30",
                         "last_known_close":float(raw[-1,2]),"checkpoint":stock_checkpoint,
                         "checkpoint_sha256":stock_sha,"epoch_reference_median":epoch_count,
                         "predicted_dates":DATES,
                         "source_csv_sha256":hashlib.sha256(files[stock].read_bytes()).hexdigest()})
    final_dir=OUT/"final"; final_dir.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(model_rows).to_csv(final_dir/"joint_model_raw_predictions.csv",index=False)
    pd.DataFrame(final_rows).to_csv(final_dir/"prediction_details.csv",index=False)
    (final_dir/"metadata.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
    config={"model":model_name,"pool":pool,"pool_members":members,"alpha_by_stock":{s:selection[s]["alpha"] for s in p4.TARGETS},
            "train_cutoff":"2026-10-30","epoch_source":"median of three historical development fits; recent labels not used",
            "epochs":epochs,"batch_size":opts.batch_size,"learning_rate":opts.lr,"seed":opts.seed,
            "seq_len":60,"pred_len":5,"features":p3.COLS,"loss":"MSE residual return",
            "training_windows_by_stock":dict(dataset.sample_counts),"checkpoint_sha256":checkpoint_sha,
            "runtime_seconds":time.time()-started,"device":str(device),"data_sha256":{
                s:hashlib.sha256(path.read_bytes()).hexdigest() for s,path in files.items() if s in members}}
    (OUT/"final_training_config.json").write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"selected":candidates,"checkpoint_sha256":checkpoint_sha,
                      "prediction_details":str(final_dir/"prediction_details.csv"),
                      "runtime_seconds":config["runtime_seconds"]},ensure_ascii=False,indent=2))


if __name__=="__main__": main()
