"""Optional: train final Phase 5 selected model(s) through 2026-10-30.

Run after phase5_analyze.py:
  python scripts/phase5_final.py --device cuda --student-id STUDENT_ID
  python scripts/export_submission.py --student-id STUDENT_ID \
      --input results/phase5_targetaware/final/prediction_details.csv \
      --metadata results/phase5_targetaware/final/metadata.json \
      --output-dir results/phase5_targetaware/final/submission
  python scripts/validate_submission.py --student-id STUDENT_ID \
      --metadata results/phase5_targetaware/final/metadata.json \
      --output-dir results/phase5_targetaware/final/submission \
      --report results/phase5_targetaware/final/submission_validation.json
"""
from __future__ import annotations

import argparse
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
import phase5_joint as p5


def full_train(pool, model_name, members, files, epochs, opts, checkpoint, pretrained=None):
    ds = p4.MultiStockResidualDataset(files, members, "2026-10-31")
    dl = DataLoader(ds, batch_size=opts.batch_size,
                    sampler=ds.balanced_sampler(opts.seed), pin_memory=(opts.device.type == "cuda"))
    p5.set_seed(opts.seed)
    model = p3.make_model(model_name).to(opts.device)
    if pretrained is not None:
        upstream = torch.load(pretrained, map_location="cpu", weights_only=False)
        model.load_state_dict(upstream["state_dict"])
    optim = torch.optim.Adam(model.parameters(), lr=opts.finetune_lr if pretrained else opts.lr)
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        samples = 0
        for x, y, _ in dl:
            x, y = x.to(opts.device), y.to(opts.device)
            pred = p3.output_return(model, x)
            loss = torch.nn.functional.mse_loss(pred, y)
            if not torch.isfinite(loss).item():
                raise FloatingPointError("Non-finite full-history training loss")
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            loss_sum += float(loss.item()) * len(x)
            samples += len(x)
        print(f"final {pool}/{model_name} epoch={epoch}/{epochs} loss={loss_sum/max(samples,1):.8g}", flush=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    stats = {s: {"mean": v["mean"].tolist(), "scale": v["scale"].tolist()}
             for s, v in ds.per_stock_stats.items()}
    torch.save({"state_dict": state, "model": model_name, "pool": pool,
                "origin": "2026-11-02", "volume_stats": stats,
                "full_history_until": "2026-10-30", "fine_tune_from": str(pretrained or ""),
                "epochs": epochs, "samples": len(ds), "seed": opts.seed}, checkpoint)
    return {"path": checkpoint, "sha": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "epochs": epochs, "runtime_seconds": time.time() - t0, "samples": len(ds)}


def forecast(ckpt: Path, files: dict, stock: str, device: torch.device):
    saved = torch.load(ckpt, map_location="cpu", weights_only=False)
    df, raw = p4.read_processed(files[stock])
    if df.date.iloc[-1] != "2026-10-30":
        raise ValueError(f"{stock} history unexpectedly ends {df.date.iloc[-1]}")
    vm = saved["volume_stats"][stock]
    x, anchor = p3.build_features(raw, len(raw), np.asarray(vm["mean"]), np.asarray(vm["scale"]))
    model = p3.make_model(saved["model"]).to(device)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    with torch.no_grad():
        ret = p3.output_return(model, torch.from_numpy(x[None]).to(device)).cpu().numpy()[0]
    return anchor, ret


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--student-id", required=True)
    ap.add_argument("--out", type=Path, default=p5.OUT_DEFAULT)
    opts = ap.parse_args()
    out = opts.out
    sel = json.loads((out / "final_model_selection.json").read_text(encoding="utf-8"))["stocks"]
    dev = pd.read_csv(out / "development/training_runs.csv")
    cfg = json.loads((out / "development/run_config.json").read_text(encoding="utf-8"))
    files = p4.stock_files("P2")
    dates = json.loads((ROOT / "config/prediction_dates.json").read_text(encoding="utf-8"))["dates"]
    if len(dates) != 5 or dates[0] != "2026-11-02" or dates[-1] != "2026-11-06":
        raise ValueError(f"Unexpected official forecast dates {dates}")
    device = p5.device_from(opts.device)
    options = SimpleNamespace(device=device, seed=int(cfg["seed"]),
                              batch_size=int(cfg["batch_size"]),
                              lr=float(cfg["lr"]), finetune_lr=float(cfg["finetune_lr"]))
    cache: dict[tuple, dict] = {}

    def median_best(pool, model, stock=None):
        q = dev[dev.pool.eq(pool) & dev.model.eq(model)]
        if stock is not None:
            q = q[q.stock.eq(stock)]
        if len(q) != 3:
            raise ValueError(f"Expected 3 development fits for {pool}/{model}/{stock}; got {len(q)}")
        return max(0, int(np.median(q.best_epoch.astype(int))))

    def ensure_fit(pool, model, stock=None):
        key = (pool, model, stock if pool in ("P0", "P3") else "shared")
        if key in cache:
            return cache[key]
        if pool == "P3":
            upstream = ensure_fit("P2", model)
            members = [stock]
            pretrained = upstream["path"]
        elif pool == "P0":
            members, pretrained = [stock], None
        else:
            members, pretrained = p4.membership_for("2026-10-30", pool), None
        epochs = median_best(pool, model, stock if pool in ("P0", "P3") else "shared")
        ckpt = out / "final/checkpoints" / f"{pool}_{model}_{stock.replace('.', '_') if stock else 'shared'}.pt"
        result = full_train(pool, model, members, files, epochs, options, ckpt, pretrained)
        cache[key] = result
        return result

    rows, metadata = [], []
    for stock in p4.TARGETS:
        choice = sel[stock]
        pool, model, alpha = choice["pool"], choice["model"], float(choice["alpha"])
        df, raw = p4.read_processed(files[stock])
        if pool == "LastCloseBaseline":
            anchor = float(raw[-1, 2])
            predictions = np.full(5, anchor)
            ckpt, digest, epochs = "", "", None
        else:
            if pool not in p5.POOLS:
                raise ValueError(f"Unsupported final pool {pool}")
            fitted = ensure_fit(pool, model, stock)
            anchor, ret = forecast(fitted["path"], files, stock, device)
            predictions = p3.restore_open(anchor, alpha * ret)
            ckpt, digest, epochs = str(fitted["path"]), fitted["sha"], fitted["epochs"]
        if not np.isfinite(predictions).all() or np.any(predictions <= 0):
            raise ValueError(f"Invalid future predictions for {stock}: {predictions}")
        for day, price in zip(dates, predictions):
            rows.append({"stock": stock, "date": day, "open": float(price)})
        metadata.append({"stock": stock, "method": "LastCloseBaseline" if pool == "LastCloseBaseline" else "Phase5-Residual",
                         "pool": pool, "model": model, "alpha": alpha,
                         "prediction_source": "LastCloseBaseline" if pool == "LastCloseBaseline" else "trained_model",
                         "training_cutoff": "2026-10-30", "last_input_date": "2026-10-30",
                         "last_known_close": float(raw[-1, 2]), "checkpoint": ckpt,
                         "checkpoint_sha256": digest, "epoch_reference_median": epochs,
                         "source_csv_sha256": hashlib.sha256(files[stock].read_bytes()).hexdigest(),
                         "predicted_dates": dates})
    dst = out / "final"
    dst.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(dst / "prediction_details.csv", index=False)
    (dst / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Phase 5 final forecasts saved to {dst}; export + validate before submission.")


if __name__ == "__main__":
    main()
