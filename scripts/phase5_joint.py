"""Phase 5: target-aware joint Residual training and target-specific fine-tuning.

Drop into scripts/ alongside phase4_joint.py and residual_phase3.py.
Runs a fair P0 (solo), P1 (3 stocks), P2 (coal pool), P3 (P2 -> solo fine-tune)
comparison with identical dates, batch size, inputs, labels, and alpha selection.

Usage:
  python scripts/phase5_joint.py --mode smoke --device cpu
  python scripts/phase5_joint.py --mode development --device cuda --models DLinear,TimeXer
  python scripts/phase5_joint.py --mode recent --device cuda --models DLinear,TimeXer
  python scripts/phase5_analyze.py
"""
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
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import phase4_joint as p4
import residual_phase3 as p3

OUT_DEFAULT = ROOT / "results/phase5_targetaware"
POOLS = ("P0", "P1", "P2", "P3")
ALPHAS = tuple(float(a) for a in p3.ALPHAS)


def score_by_stock(losses: dict[str, list[float]], required: list[str]) -> tuple[float, float, dict[str, float]]:
    """Validation checkpoint objective: target-stock macro mean, not coal-pool mean."""
    means = {s: float(np.mean(v)) for s, v in losses.items() if v}
    missing = sorted(set(required) - set(means))
    if missing:
        raise ValueError(f"required target(s) have no validation samples: {missing}")
    return float(np.mean([means[s] for s in required])), float(np.mean(list(means.values()))), means


def device_from(name: str) -> torch.device:
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available()
                        else ("cpu" if name == "auto" else name))


def set_seed(seed: int):
    p4.seed_all(seed)


def loaders(files, members, validation_start, origin, opts):
    train = p4.MultiStockResidualDataset(files, members, validation_start)
    val = p4.MultiStockResidualDataset(files, members, validation_start, validation_start, origin)
    pin = device_from(opts.device).type == "cuda"
    tr_dl = DataLoader(train, batch_size=opts.batch_size,
                       sampler=train.balanced_sampler(opts.seed), pin_memory=pin)
    va_dl = DataLoader(val, batch_size=opts.batch_size, shuffle=False, pin_memory=pin)
    return train, val, tr_dl, va_dl


def evaluate_val(model, val_ds, loader, device, required):
    model.eval()
    by_stock: dict[str, list[float]] = defaultdict(list)
    with torch.no_grad():
        for x, y, indices in loader:
            pred = p3.output_return(model, x.to(device, non_blocking=device.type == "cuda"))
            per_sample = ((pred - y.to(device, non_blocking=device.type == "cuda")) ** 2).mean(dim=1)
            for idx, loss in zip(indices.tolist(), per_sample.cpu().numpy().tolist()):
                by_stock[str(val_ds.stock_ids[idx])].append(float(loss))
    return score_by_stock(by_stock, required)


def save_checkpoint(path: Path, model_name: str, pool: str, origin: str, state,
                    train_ds, selection: dict, args, fine_tune_from: str = ""):
    path.parent.mkdir(parents=True, exist_ok=True)
    stats = {s: {"mean": v["mean"].tolist(), "scale": v["scale"].tolist()}
             for s, v in train_ds.per_stock_stats.items()}
    torch.save({"state_dict": state, "model": model_name, "pool": pool,
                "origin": origin, "volume_stats": stats,
                "selection": selection, "fine_tune_from": fine_tune_from,
                "training_config": {"seed": args.seed, "batch_size": args.batch_size,
                                    "lr": args.lr if pool != "P3" else args.finetune_lr,
                                    "seq_len": p3.SEQ_LEN, "pred_len": p3.PRED_LEN}}, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit_model(model_name: str, pool: str, members: list[str], files: dict,
              origin: str, val_start: str, opts, checkpoint: Path,
              required_targets: list[str], pretrained: Path | None = None):
    """P0/P1/P2 train from scratch; P3 starts from same-origin P2 best checkpoint.

    P3 evaluates epoch-0 pretraining weights as a legal fine-tuning candidate.
    The same validation period (strictly before origin) is used for selection.
    """
    dev = device_from(opts.device)
    set_seed(opts.seed)
    tr_ds, va_ds, tr_dl, va_dl = loaders(files, members, val_start, origin, opts)
    model = p3.make_model(model_name, small=False).to(dev)
    if pretrained is not None:
        upstream = torch.load(pretrained, map_location="cpu", weights_only=False)
        if upstream["model"] != model_name or upstream["origin"] != origin or upstream["pool"] != "P2":
            raise ValueError("P3 initialization is not the matching-origin P2 checkpoint")
        model.load_state_dict(upstream["state_dict"])
    opt_lr = opts.finetune_lr if pretrained else opts.lr
    max_epochs = opts.finetune_epochs if pretrained else opts.epochs
    patience = opts.finetune_patience if pretrained else opts.patience
    optim = torch.optim.Adam(model.parameters(), lr=opt_lr)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    started = time.time()
    best, best_epoch, stale, best_state = float("inf"), -1, 0, None
    best_pool_loss, best_by_stock = float("nan"), {}
    if pretrained is not None:
        initial_target, initial_pool, initial_per_stock = evaluate_val(model, va_ds, va_dl, dev, required_targets)
        best, best_pool_loss, best_by_stock = initial_target, initial_pool, initial_per_stock
        best_epoch = 0
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    train_loss = float("nan")
    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss, total_count = 0.0, 0
        for x, y, _ in tr_dl:
            x = x.to(dev, non_blocking=dev.type == "cuda")
            y = y.to(dev, non_blocking=dev.type == "cuda")
            pred = p3.output_return(model, x)
            loss = torch.nn.functional.mse_loss(pred, y)
            if not torch.isfinite(loss).item():
                raise FloatingPointError("non-finite training loss")
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if any(p.grad is not None and not torch.isfinite(p.grad).all().item() for p in model.parameters()):
                raise FloatingPointError("non-finite training gradient")
            optim.step()
            total_loss += float(loss.item()) * len(x)
            total_count += len(x)
        train_loss = total_loss / max(total_count, 1)
        target_val, pool_val, per_stock = evaluate_val(model, va_ds, va_dl, dev, required_targets)
        if not np.isfinite(target_val):
            raise FloatingPointError("non-finite target validation")
        if target_val < best - 1e-9:
            best, best_epoch, stale = target_val, epoch, 0
            best_pool_loss, best_by_stock = pool_val, per_stock
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("no best checkpoint")
    info = {"target_val_nmse": best, "pool_val_nmse": best_pool_loss,
            "val_by_stock": best_by_stock, "best_epoch": best_epoch,
            "train_loss_last_epoch": train_loss, "max_epochs": max_epochs,
            "training_samples": len(tr_ds), "validation_samples": len(va_ds),
            "members": members, "sample_counts": dict(tr_ds.sample_counts),
            "required_targets": required_targets, "validation_start": val_start,
            "runtime_seconds": time.time() - started,
            "params": sum(p.numel() for p in model.parameters()),
            "gpu_peak_mib": (torch.cuda.max_memory_allocated(dev) / (1024**2)) if dev.type == "cuda" else 0.0}
    sha = save_checkpoint(checkpoint, model_name, pool, origin, best_state, tr_ds,
                          info, opts, str(pretrained) if pretrained else "")
    info.update({"checkpoint": str(checkpoint), "sha256": sha})
    return info


def predict_from_checkpoint(ckpt: Path, files: dict, stock: str, origin: str, device: str):
    saved = torch.load(ckpt, map_location="cpu", weights_only=False)
    if stock not in saved["volume_stats"]:
        raise ValueError(f"checkpoint {ckpt} lacks normalization for stock {stock}")
    df, raw = p4.read_processed(files[stock])
    dates = df.date.astype(str).tolist()
    if origin not in dates:
        raise ValueError(f"prediction date {origin} is absent for {stock}")
    o = dates.index(origin)
    stats = saved["volume_stats"][stock]
    x, anchor = p3.build_features(raw, o, np.asarray(stats["mean"]), np.asarray(stats["scale"]))
    dev = device_from(device)
    model = p3.make_model(saved["model"]).to(dev)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    with torch.no_grad():
        ret = p3.output_return(model, torch.from_numpy(x[None]).to(dev)).cpu().numpy()[0]
    if not np.isfinite(ret).all():
        raise ValueError("non-finite model prediction")
    target, target_anchor = p3.build_target(raw, o)
    if not math.isclose(anchor, target_anchor, abs_tol=1e-7, rel_tol=0):
        raise AssertionError("anchor mismatch")
    target_dates = tuple(dates[o:o + p3.PRED_LEN])
    return ret, target, anchor, target_dates


def record_predictions(pool, model, stock, origin, ckpt, files, opts,
                       expected_dates, records):
    pred, actual, anchor, target_dates = predict_from_checkpoint(ckpt, files, stock, origin, opts.device)
    if target_dates != expected_dates:
        raise ValueError(f"target calendar mismatch {pool}/{stock}/{origin}")
    for alpha in ALPHAS:
        gated = alpha * pred
        metrics = p3.residual_metrics(gated, actual, anchor)
        records.append({"pool": pool, "model": model, "stock": stock,
                        "origin_date": origin, "target_dates": ";".join(target_dates),
                        "alpha": alpha, "anchor_close": anchor,
                        "actual_return": p3.strvec(actual),
                        "predicted_return": p3.strvec(gated),
                        "actual_open": p3.strvec(p3.restore_open(anchor, actual)),
                        "predicted_open": p3.strvec(p3.restore_open(anchor, gated)),
                        "nmse": metrics["nmse"], "mae": metrics["mae"], "rmse": metrics["rmse"],
                        "checkpoint": str(ckpt)})


def run_stage(opts, stage: str):
    out = Path(opts.out)
    files = p4.stock_files("P2")
    target_files = {s: files[s] for s in p4.TARGETS}
    selected_dates = (p4.choose_dev_dates(target_files) if stage == "development"
                      else p4.choose_recent_dates(target_files))
    groups = [[pair] for pair in selected_dates] if stage == "development" else [selected_dates[i:i + 4] for i in (0, 4)]
    out_stage = out / stage
    out_stage.mkdir(parents=True, exist_ok=True)
    (out_stage / "selected_dates.json").write_text(json.dumps(
        [{"origin": d, "target_dates": list(dates)} for d, dates in selected_dates], indent=2), encoding="utf-8")
    model_names = opts.models
    run_rows, pred_rows = [], []
    for group_index, group in enumerate(groups):
        fit_origin = group[0][0]
        validation_start, _ = p4.boundary_dates(target_files, fit_origin, opts.val_rows)
        print(f"{stage} group={group_index + 1} origin={fit_origin} val_start={validation_start}", flush=True)
        for model in model_names:
            fitted: dict[tuple[str, str], Path] = {}
            # P0: one from-scratch model per target; P1/P2: one shared model each.
            for pool in ("P0", "P1", "P2"):
                members = p4.membership_for(fit_origin, pool)
                target_subset = p4.TARGETS if pool != "P0" else p4.TARGETS
                for stock in target_subset:
                    if pool != "P0" and stock != p4.TARGETS[0]:
                        continue
                    actual_members = [stock] if pool == "P0" else members
                    required = [stock] if pool == "P0" else list(p4.TARGETS)
                    name = stock.replace(".", "_") if pool == "P0" else "shared"
                    ckpt = out_stage / "checkpoints" / f"{pool}_{model}_{fit_origin}_{name}.pt"
                    fit = fit_model(model, pool, actual_members, files, fit_origin, validation_start,
                                    opts, ckpt, required)
                    fitted[(pool, stock if pool == "P0" else "shared")] = ckpt
                    run_rows.append({"stage": stage, "pool": pool, "model": model,
                                     "stock": stock if pool == "P0" else "shared", "origin_date": fit_origin,
                                     **{k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                                        for k, v in fit.items()}})
                    print(f"fit {pool} {model} {name}: val_target={fit['target_val_nmse']:.7g} "
                          f"val_pool={fit['pool_val_nmse']:.7g} epoch={fit['best_epoch']}", flush=True)
            # P3: fine-tune one P2 checkpoint for each target, same fit origin.
            source = fitted[("P2", "shared")]
            for stock in p4.TARGETS:
                ckpt = out_stage / "checkpoints" / f"P3_{model}_{fit_origin}_{stock.replace('.', '_')}.pt"
                fit = fit_model(model, "P3", [stock], files, fit_origin, validation_start,
                                opts, ckpt, [stock], pretrained=source)
                fitted[("P3", stock)] = ckpt
                run_rows.append({"stage": stage, "pool": "P3", "model": model,
                                 "stock": stock, "origin_date": fit_origin,
                                 **{k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                                    for k, v in fit.items()}})
                print(f"fine-tune P3 {model} {stock}: val={fit['target_val_nmse']:.7g} "
                      f"epoch={fit['best_epoch']}", flush=True)
            # Identical checkpoints within each 20-trading-day cycle; inputs update per origin.
            for origin, expected_dates in group:
                for stock in p4.TARGETS:
                    for pool in POOLS:
                        key = (pool, stock) if pool in ("P0", "P3") else (pool, "shared")
                        record_predictions(pool, model, stock, origin, fitted[key], files,
                                           opts, expected_dates, pred_rows)
    if not pred_rows:
        raise RuntimeError("no Phase 5 predictions")
    pd.DataFrame(pred_rows).to_csv(out_stage / "predictions.csv", index=False)
    pd.DataFrame(run_rows).to_csv(out_stage / "training_runs.csv", index=False)
    config = {"phase": 5, "stage": stage, "models": model_names, "pools": list(POOLS),
              "seq_len": p3.SEQ_LEN, "pred_len": p3.PRED_LEN,
              "batch_size": opts.batch_size, "lr": opts.lr, "seed": opts.seed,
              "epochs": opts.epochs, "patience": opts.patience,
              "val_rows": opts.val_rows, "finetune_lr": opts.finetune_lr,
              "finetune_epochs": opts.finetune_epochs,
              "finetune_patience": opts.finetune_patience,
              "checkpoint_selection": "equal weight macro validation NMSE of the 3 target stocks",
              "P3_pretraining": "same-fit-origin P2 target-aware checkpoint",
              "alpha_selection": "development only; locked for recent",
              "origin_dates": [d for d, _ in selected_dates],
              "coal_pool_manifest": "data/coal_pool/manifests/industry_membership.csv"}
    (out_stage / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(pred_rows)} predictions and {len(run_rows)} fits to {out_stage}", flush=True)


def smoke():
    set_seed(2026)
    assert abs(p3.residual_metrics(np.zeros(5), np.zeros(5), 20.0)["nmse"]) < 1e-12
    required, pool, means = score_by_stock({"sh.601088": [0.002, 0.004],
                                            "sh.601225": [0.001], "sz.000983": [0.003],
                                            "sh.600121": [0.5]}, list(p4.TARGETS))
    assert required < pool, (required, pool, means)
    x = torch.randn(2, p3.SEQ_LEN, 6)
    for model_name in p3.MODELS:
        model = p3.make_model(model_name, small=True)
        p = p3.output_return(model, x)
        assert tuple(p.shape) == (2, 5) and torch.isfinite(p).all().item()
    print("Phase 5 smoke passed: target macro checkpoint, residual zero, model IO")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("smoke", "development", "recent"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--models", default="DLinear,TimeXer",
                        help="Comma-separated subset of DLinear,iTransformer,TimeXer; use all for all three")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--val-rows", type=int, default=126)
    parser.add_argument("--finetune-lr", type=float, default=0.0001)
    parser.add_argument("--finetune-epochs", type=int, default=10)
    parser.add_argument("--finetune-patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    opts = parser.parse_args()
    if opts.models.lower() == "all":
        opts.models = list(p3.MODELS)
    else:
        opts.models = [m.strip() for m in opts.models.split(",") if m.strip()]
    if not opts.models or any(m not in p3.MODELS for m in opts.models):
        parser.error("Invalid --models; expected DLinear,iTransformer,TimeXer")
    if opts.batch_size <= 0 or opts.finetune_epochs < 1:
        parser.error("Batch size and fine-tuning epochs must be positive")
    if opts.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but not available")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    if opts.mode == "smoke":
        smoke()
    else:
        run_stage(opts, opts.mode)


if __name__ == "__main__":
    main()
