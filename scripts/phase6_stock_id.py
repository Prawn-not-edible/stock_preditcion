"""Phase 6: P2 + explicit Stock ID embedding, matched with Phase 5 P2 predictions.

Runs 3 development shared fits and 2 recent shared fits per architecture (no P2 retraining).
    python scripts/phase6_stock_id.py --mode smoke --device cpu
    python scripts/phase6_stock_id.py --mode development --device cuda --models TimeXer,DLinear
    python scripts/phase6_stock_id.py --mode recent --device cuda --models TimeXer,DLinear
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
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
from phase6_id_model import StockIDResidual, stable_vocab, batch_ids

OUT_DEFAULT = ROOT / "results/phase6_stock_id"
ALPHAS = tuple(float(v) for v in p3.ALPHAS)


def get_vocab() -> list[str]:
    cfg = json.loads((ROOT / "config/coal_stock_pool.json").read_text(encoding="utf8"))
    return stable_vocab(cfg["security_codes"])


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_phase5_reference(opts, files: dict[str, Path]) -> dict:
    """Reject a comparison if Phase5 used incompatible training settings/data."""
    ref_root = ROOT / "results/phase5_targetaware"
    required = {"batch_size": opts.batch_size, "seed": opts.seed, "lr": opts.lr,
                "epochs": opts.epochs, "patience": opts.patience,
                "val_rows": opts.val_rows, "seq_len": p3.SEQ_LEN,
                "pred_len": p3.PRED_LEN}
    refs = {}
    for stage in ("development", "recent"):
        config_path = ref_root / stage / "run_config.json"
        prediction_path = ref_root / stage / "predictions.csv"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        for key, value in required.items():
            if config.get(key) != value:
                raise ValueError(f"Frozen Phase5 {stage} {key}={config.get(key)!r}, expected {value!r}")
        if not set(opts.models).issubset(config.get("models", [])):
            raise ValueError(f"Phase5 {stage} does not contain all requested model controls")
        refs[stage] = {"config_sha256": sha256(config_path),
                       "predictions_sha256": sha256(prediction_path)}

    # Phase5's final metadata stores hashes for all three target input files.
    metadata_path = ref_root / "final/metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for row in metadata:
        actual = sha256(files[row["stock"]])
        if actual != row["source_csv_sha256"]:
            raise ValueError(f"Target data hash changed since Phase5 for {row['stock']}")
    refs["final_metadata_sha256"] = sha256(metadata_path)
    return refs


def make_model(name: str, vocab: list[str], emb_dim: int) -> StockIDResidual:
    # Preserve the CPU RNG state at the point where Phase5's no-ID model has
    # finished initialization. Initializing ID parameters must not shift the
    # base model's subsequent dropout RNG stream during the controlled ablation.
    backbone = p3.make_model(name, small=False)
    state_after_backbone = torch.get_rng_state()
    model = StockIDResidual(backbone, p3.output_return,
                            len(vocab), embedding_dim=emb_dim, horizon=p3.PRED_LEN)
    torch.set_rng_state(state_after_backbone)
    return model


def evaluate(model, ds, loader, dev, vocab):
    model.eval()
    by_stock = defaultdict(list)
    with torch.no_grad():
        for x, y, idx in loader:
            sid = batch_ids(ds.stock_ids, idx, vocab, dev)
            pred = model(x.to(dev, non_blocking=dev.type == "cuda"), sid)
            losses = ((pred - y.to(dev, non_blocking=dev.type == "cuda")) ** 2).mean(1)
            for k, error in zip(idx.tolist(), losses.cpu().tolist()):
                by_stock[str(ds.stock_ids[k])].append(float(error))
    return p5.score_by_stock(by_stock, list(p4.TARGETS))


def fit_id(model_name, origin, opts, members, vocab, files, out_file):
    dev = p5.device_from(opts.device)
    p5.set_seed(opts.seed)
    target_files = {s: files[s] for s in p4.TARGETS}
    val_start, _ = p4.boundary_dates(target_files, origin, opts.val_rows)
    train_ds, val_ds, train_dl, val_dl = p5.loaders(files, members, val_start, origin, opts)
    if not set(members).issubset(vocab):
        raise ValueError("Member not present in stable stock vocabulary")
    model = make_model(model_name, vocab, opts.embedding_dim).to(dev)
    # make_model already restores the post-backbone RNG state.
    optim = torch.optim.Adam(model.parameters(), lr=opts.lr)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    best = math.inf
    best_epoch = 0
    best_state = None
    best_pool = math.nan
    best_stock_losses = {}
    stalled = 0
    last_train_mse = math.nan
    started = time.time()
    for epoch in range(1, opts.epochs + 1):
        model.train()
        loss_total = count = 0
        for x, y, idx in train_dl:
            sid = batch_ids(train_ds.stock_ids, idx, vocab, dev)
            x = x.to(dev, non_blocking=dev.type == "cuda")
            y = y.to(dev, non_blocking=dev.type == "cuda")
            pred = model(x, sid)
            mse = torch.nn.functional.mse_loss(pred, y)
            reg = model.id_correction(sid).square().mean()
            loss = mse + opts.id_l2 * reg
            if not torch.isfinite(loss).item():
                raise FloatingPointError("Non-finite Phase6 training loss")
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            count += len(x)
            loss_total += mse.item() * len(x)
        last_train_mse = loss_total / count
        target_nmse, pool_nmse, stock_losses = evaluate(model, val_ds, val_dl, dev, vocab)
        if not np.isfinite(target_nmse):
            raise FloatingPointError("Non-finite validation NMSE")
        if target_nmse < best - 1e-9:
            best, best_epoch, best_pool, best_stock_losses = target_nmse, epoch, pool_nmse, stock_losses
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stalled = 0
        else:
            stalled += 1
        if stalled >= opts.patience:
            break
    if best_state is None:
        raise RuntimeError("No successful checkpoint")
    stats = {s: {"mean": v["mean"].tolist(), "scale": v["scale"].tolist()}
             for s, v in train_ds.per_stock_stats.items()}
    out_file.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "state_dict": best_state,
        "model": model_name,
        "pool": "P2_ID",
        "origin": origin,
        "vocab": vocab,
        "embedding_dim": opts.embedding_dim,
        "id_l2": opts.id_l2,
        "volume_stats": stats,
        "members": members,
        "validation_start": val_start,
        "best_epoch": best_epoch,
        "selection_target_nmse": best,
        "seed": opts.seed,
        "data_version": "phase5-targetaware",
    }
    torch.save(obj, out_file)
    result = {
        "pool": "P2_ID", "model": model_name, "origin_date": origin,
        "members": json.dumps(members, ensure_ascii=False),
        "n_members": len(members), "embedding_dim": opts.embedding_dim,
        "id_l2": opts.id_l2, "seed": opts.seed,
        "best_epoch": best_epoch,
        "train_mse_last": last_train_mse,
        "target_val_nmse": best,
        "pool_val_nmse": best_pool,
        "val_by_stock": json.dumps(best_stock_losses, ensure_ascii=False),
        "train_samples": len(train_ds), "validation_samples": len(val_ds),
        "training_cutoff": val_start,
        "params": sum(p.numel() for p in model.parameters()),
        "gpu_peak_mib": torch.cuda.max_memory_allocated(dev)/(1024**2) if dev.type == "cuda" else 0,
        "seconds": time.time() - started,
        "checkpoint": str(out_file), "checkpoint_sha256": hashlib.sha256(out_file.read_bytes()).hexdigest()
    }
    return result


def infer(ckpt: Path, files: dict, stock: str, origin: str, device: str):
    saved = torch.load(ckpt, map_location="cpu", weights_only=False)
    if saved["pool"] != "P2_ID" or stock not in saved["vocab"] or stock not in saved["volume_stats"]:
        raise ValueError("Checkpoint lacks P2_ID stock or scaler")
    df, raw = p4.read_processed(files[stock]); dates = df.date.astype(str).tolist()
    if origin not in dates:
        raise ValueError(f"No trading day {origin} in {stock}")
    idx = dates.index(origin)
    vm = saved["volume_stats"][stock]
    x, anchor = p3.build_features(raw, idx, np.asarray(vm["mean"]), np.asarray(vm["scale"]))
    dev = p5.device_from(device)
    model = make_model(saved["model"], saved["vocab"], saved["embedding_dim"]).to(dev)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    with torch.no_grad():
        ret = model(torch.from_numpy(x[None]).to(dev),
                    torch.tensor([saved["vocab"].index(stock)], dtype=torch.long, device=dev)).cpu().numpy()[0]
    y, a2 = p3.build_target(raw, idx)
    if not math.isclose(anchor, a2, abs_tol=1e-8, rel_tol=0):
        raise AssertionError("Anchor mismatch")
    return ret, y, anchor, tuple(dates[idx:idx+p3.PRED_LEN])


def emit_prediction(rows, ckpt, origin, dates, files, opts, model):
    for stock in p4.TARGETS:
        ret, y, anchor, actual_dates = infer(ckpt, files, stock, origin, opts.device)
        if actual_dates != dates:
            raise ValueError(f"Different prediction dates for {stock}: {actual_dates} != {dates}")
        for alpha in ALPHAS:
            gated = alpha * ret
            m = p3.residual_metrics(gated, y, anchor)
            rows.append({
                "pool": "P2_ID", "model": model, "stock": stock,
                "origin_date": origin, "target_dates": ";".join(actual_dates),
                "alpha": alpha, "anchor_close": anchor,
                "actual_return": p3.strvec(y), "predicted_return": p3.strvec(gated),
                "actual_open": p3.strvec(p3.restore_open(anchor, y)),
                "predicted_open": p3.strvec(p3.restore_open(anchor, gated)),
                "nmse": m["nmse"], "mae": m["mae"], "rmse": m["rmse"],
                "checkpoint": str(ckpt),
            })


def run(opts):
    out = Path(opts.out); stage = opts.mode
    stage_dir = out / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    files = p4.stock_files("P2")
    phase5_reference = validate_phase5_reference(opts, files)
    data_hashes = {stock: sha256(path) for stock, path in sorted(files.items())}
    target_files = {s: files[s] for s in p4.TARGETS}
    origins = (p4.choose_dev_dates(target_files) if stage == "development"
               else p4.choose_recent_dates(target_files))
    groups = [[t] for t in origins] if stage == "development" else [origins[:4], origins[4:]]
    vocab = get_vocab()
    records, fits = [], []
    for group in groups:
        fit_origin = group[0][0]
        members = p4.membership_for(fit_origin, "P2")
        for model in opts.models:
            ckpt = stage_dir / "checkpoints" / f"P2_ID_{model}_{fit_origin}.pt"
            info = fit_id(model, fit_origin, opts, members, vocab, files, ckpt)
            fits.append(info)
            print(f"{stage} P2_ID {model} {fit_origin}: "
                  f"target_val={info['target_val_nmse']:.8g}, epoch={info['best_epoch']}", flush=True)
            for origin, target_dates in group:
                emit_prediction(records, ckpt, origin, target_dates, files, opts, model)
    pd.DataFrame(records).to_csv(stage_dir / "predictions.csv", index=False)
    pd.DataFrame(fits).to_csv(stage_dir / "training_runs.csv", index=False)
    (stage_dir / "run_config.json").write_text(json.dumps({
        "phase": 6, "stage": stage, "models": opts.models, "batch_size": opts.batch_size,
        "seed": opts.seed, "epochs": opts.epochs, "patience": opts.patience,
        "lr": opts.lr, "val_rows": opts.val_rows,
        "embedding_dim": opts.embedding_dim, "id_l2": opts.id_l2,
        "vocab": vocab, "origin_dates": [s[0] for s in origins],
        "alpha_grid": list(ALPHAS),
        "reference": "Phase5 P2 shared model predictions, same target-aware checkpoint selection",
        "checkpoint_selection": "3 target stocks equal weight macro validation NMSE",
        "phase5_reference_sha256": phase5_reference,
        "input_data_sha256": data_hashes,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(records)} predictions and {len(fits)} fits to {stage_dir}")


def smoke():
    dev = torch.device("cpu")
    class Toy(torch.nn.Module):
        def forward(self, x):
            return x[:, -1, :5]
    model = StockIDResidual(Toy(), lambda net, x: net(x), 3, 8, 5)
    x = torch.randn(4, 60, 6)
    ids = torch.tensor([0, 2, 1, 0], dtype=torch.long)
    torch.testing.assert_close(model(x, ids), model.backbone(x), atol=0, rtol=0)
    loss = model(x, ids).square().mean(); loss.backward()
    assert model.id_head.weight.grad is not None
    assert batch_ids(np.array(["sh.601088", "sh.601225"]), torch.tensor([1, 0]),
                     ["sh.601088", "sh.601225"], dev).tolist() == [1, 0]
    print("Phase6 smoke passed: zero head, gradient, stock-name ID alignment")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["smoke", "development", "recent"], required=True)
    ap.add_argument("--models", default="TimeXer,DLinear")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--val-rows", type=int, default=126)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--embedding-dim", type=int, default=8)
    ap.add_argument("--id-l2", type=float, default=0.001)
    opts = ap.parse_args()
    opts.models = [s.strip() for s in opts.models.split(",") if s.strip()]
    if any(m not in p3.MODELS for m in opts.models) or not opts.models:
        ap.error("--models must list available model names")
    if opts.embedding_dim < 1 or opts.id_l2 < 0 or opts.batch_size < 1:
        ap.error("invalid hyperparameter")
    if opts.device == "cuda" and not torch.cuda.is_available():
        ap.error("CUDA is not available")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    if opts.mode == "smoke":
        smoke()
    else:
        run(opts)


if __name__ == "__main__":
    main()
