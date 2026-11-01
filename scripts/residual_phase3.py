"""Phase 3 residual-return experiments, kept separate from Phase 2 outputs."""
from __future__ import annotations

import argparse
import csv
import csv
import hashlib
import importlib
import json
import math
import random
import subprocess
import sys
import time
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
MODELS = ["DLinear", "iTransformer", "TimeXer"]
DEV_FOLDS = ["origin_02", "origin_04", "origin_06"]
ALPHAS = [0.0, 0.25, 0.5, 1.0]
SEQ_LEN, PRED_LEN = 60, 5


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_model(name: str, small: bool = False) -> torch.nn.Module:
    """Same official model family/configuration used by Phase 2."""
    cfg = SimpleNamespace(
        task_name="long_term_forecast", features="MS", seq_len=SEQ_LEN,
        label_len=30, pred_len=PRED_LEN, enc_in=6, dec_in=6, c_out=6,
        d_model=16 if small else 128, d_ff=32 if small else 256,
        e_layers=1 if small else 2, n_heads=4, dropout=0.1,
        patch_len=12, moving_avg=25, factor=3, embed="timeF", freq="d",
        activation="gelu", use_norm=True, output_attention=False,
        individual=False,
    )
    return importlib.import_module(f"models.{name}").Model(cfg)


def fit_volume_stats(raw: np.ndarray, train_end: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit log1p volume/amount stats only on rows visible to the train split."""
    if train_end <= 0:
        raise ValueError("train_end must be positive")
    values = np.log1p(np.maximum(raw[:train_end, 3:5], 0.0))
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def build_features(raw: np.ndarray, origin: int, volume_mean: np.ndarray,
                   volume_scale: np.ndarray) -> tuple[np.ndarray, float]:
    """Build [60,6] features, anchored to close[origin-1]."""
    if origin < SEQ_LEN or origin > len(raw):
        raise ValueError(f"invalid origin {origin} for N={len(raw)}")
    anchor = float(raw[origin - 1, 2])
    if not math.isfinite(anchor) or anchor <= 0:
        raise ValueError(f"anchor must be finite and > 0, got {anchor}")
    hist = raw[origin - SEQ_LEN:origin]
    price = hist[:, [0, 1, 2, 5]] / anchor - 1.0
    volume = (np.log1p(np.maximum(hist[:, 3:5], 0.0)) - volume_mean) / volume_scale
    features = np.column_stack((price[:, 0], price[:, 1], price[:, 2],
                                volume[:, 0], volume[:, 1], price[:, 3]))
    if features.shape != (SEQ_LEN, 6) or not np.isfinite(features).all():
        raise ValueError("invalid residual input features")
    return features.astype(np.float32), anchor


def build_target(raw: np.ndarray, origin: int) -> tuple[np.ndarray, float]:
    """Build five future Open returns against only the last known Close."""
    if origin < 1 or origin + PRED_LEN > len(raw):
        raise ValueError(f"no complete target horizon at origin {origin}")
    anchor = float(raw[origin - 1, 2])
    if not math.isfinite(anchor) or anchor <= 0:
        raise ValueError(f"anchor must be finite and > 0, got {anchor}")
    target = raw[origin:origin + PRED_LEN, 5] / anchor - 1.0
    if not np.isfinite(target).all():
        raise ValueError("non-finite target returns")
    return target.astype(np.float32), anchor


def restore_open(anchor: float, pred_return: np.ndarray) -> np.ndarray:
    if not math.isfinite(float(anchor)) or anchor <= 0:
        raise ValueError("anchor must be finite and > 0")
    pred_return = np.asarray(pred_return, dtype=np.float64)
    return float(anchor) * (1.0 + pred_return)


def residual_metrics(pred_return: np.ndarray, actual_return: np.ndarray,
                     anchor: float) -> dict:
    pred_return = np.asarray(pred_return, dtype=np.float64)
    actual_return = np.asarray(actual_return, dtype=np.float64)
    pred_open = restore_open(anchor, pred_return)
    actual_open = restore_open(anchor, actual_return)
    err = pred_open - actual_open
    nmse = float(np.mean((err / anchor) ** 2))
    residual_mse = float(np.mean((pred_return - actual_return) ** 2))
    if not math.isclose(nmse, residual_mse, rel_tol=1e-10, abs_tol=1e-14):
        raise AssertionError(f"residual MSE {residual_mse} != price NMSE {nmse}")
    return {"nmse": nmse, "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))), "residual_mse": residual_mse}


def matrix(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(path)
    if df.columns.tolist() != ["date", *COLS]:
        raise ValueError(f"Unexpected model columns in {path}: {df.columns.tolist()}")
    # Phase 2 trained/evaluated on float32 arrays; match its exact stored Open precision for pairing.
    raw = df[COLS].to_numpy(dtype=np.float32)
    if not np.isfinite(raw).all():
        raise ValueError(f"non-finite numeric input in {path}")
    return df, raw


def stock_configs() -> list[dict]:
    return json.loads((ROOT / "config/stocks.json").read_text(encoding="utf-8"))["stocks"]


def manifest_for(stock: dict) -> dict:
    return json.loads((ROOT / "data/splits" / f"{Path(stock['output_file']).stem}.json").read_text(encoding="utf-8"))


def history_dataset(raw: np.ndarray, train_end: int, val_start: int,
                    test_origin: int, device: str, opts, small: bool = False,
                    checkpoint: Path | None = None) -> dict:
    """Train one V1 fit with expanding train range and rolling 126-day validation."""
    if not (60 + 5 <= train_end <= val_start < test_origin):
        raise ValueError(f"invalid train/validation boundaries {train_end}, {val_start}, {test_origin}")
    vm, vs = fit_volume_stats(raw, train_end)

    def samples(origins: range) -> tuple[torch.Tensor, torch.Tensor]:
        xs, ys = [], []
        for o in origins:
            x, _ = build_features(raw, o, vm, vs)
            y, _ = build_target(raw, o)
            xs.append(x); ys.append(y)
        return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))

    tr_o = range(SEQ_LEN, train_end - PRED_LEN + 1)
    va_o = range(val_start, test_origin - PRED_LEN + 1)
    if not len(tr_o) or not len(va_o):
        raise ValueError("empty train or validation windows")
    tr_x, tr_y = samples(tr_o)
    va_x, va_y = samples(va_o)
    dev = torch.device("cuda" if opts.device == "auto" and torch.cuda.is_available()
                       else ("cpu" if opts.device == "auto" else opts.device))
    pin_memory = dev.type == "cuda"
    train_dl = DataLoader(TensorDataset(tr_x, tr_y), batch_size=opts.batch_size,
                          shuffle=True, pin_memory=pin_memory)
    val_dl = DataLoader(TensorDataset(va_x, va_y), batch_size=opts.batch_size,
                        shuffle=False, pin_memory=pin_memory)
    seed_all(opts.seed)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    model = make_model(opts.model, small=small).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr)
    best, best_epoch, stale, best_state = float("inf"), 0, 0, None
    train_loss, start = float("nan"), time.time()
    for epoch in range(1, opts.epochs + 1):
        model.train(); losses = []
        for x, y in train_dl:
            x, y = x.to(dev, non_blocking=pin_memory), y.to(dev, non_blocking=pin_memory)
            pred = output_return(model, x)
            loss = torch.nn.functional.mse_loss(pred, y)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite residual training MSE")
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("non-finite residual model gradient")
            optimizer.step(); losses.append(float(loss.item()))
        train_loss = float(np.mean(losses)); model.eval(); val_losses = []
        with torch.no_grad():
            for x, y in val_dl:
                pred = output_return(model, x.to(dev, non_blocking=pin_memory))
                val_losses.extend(torch.nn.functional.mse_loss(pred, y.to(dev), reduction="none").mean(1).cpu().tolist())
        val_loss = float(np.mean(val_losses))
        if not math.isfinite(val_loss):
            raise FloatingPointError("non-finite residual validation MSE")
        if val_loss < best - 1e-9:
            best, best_epoch, stale = val_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= opts.patience:
            break
    if best_state is None:
        raise RuntimeError("no finite residual validation checkpoint")
    model.load_state_dict(best_state); model.eval()
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError("non-finite model parameter")
    x_test, anchor = build_features(raw, test_origin, vm, vs)
    with torch.no_grad():
        pred_return = output_return(model, torch.from_numpy(x_test[None]).to(dev)).cpu().numpy()[0]
    if pred_return.shape != (PRED_LEN,) or not np.isfinite(pred_return).all():
        raise ValueError(f"invalid output returns: {pred_return}")
    ckpt_hash = ""
    if checkpoint:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": best_state, "model": opts.model,
                    "volume_mean": vm, "volume_scale": vs,
                    "config": {"seed": opts.seed, "seq_len": SEQ_LEN,
                               "pred_len": PRED_LEN, "features": COLS,
                               "epochs": opts.epochs, "patience": opts.patience,
                               "lr": opts.lr, "train_end": train_end,
                               "val_start": val_start, "small": small}}, checkpoint)
        ckpt_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    peak = torch.cuda.max_memory_allocated(dev) / (1024 ** 2) if dev.type == "cuda" else 0.0
    return {"prediction_return": pred_return, "anchor": anchor,
            "best_epoch": best_epoch, "train_loss": train_loss,
            "validation_loss": best, "params": sum(p.numel() for p in model.parameters()),
            "runtime_seconds": time.time() - start, "peak_gpu_memory_mb": peak,
            "device": str(dev), "checkpoint": str(checkpoint or ""),
            "checkpoint_sha256": ckpt_hash, "volume_mean": vm, "volume_scale": vs}


def output_return(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    # Keep the official models unchanged. TimeXer remains features=MS with Open last.
    dec = torch.zeros((x.shape[0], 35, 6), dtype=x.dtype, device=x.device)
    y = model(x, None, dec, None)
    if y is None or y.ndim != 3 or y.shape[1] != PRED_LEN or y.shape[2] not in (1, 6):
        raise ValueError(f"unexpected model output shape {None if y is None else tuple(y.shape)}")
    return y[..., -1]


def finite_checkpoint_fit(model_name: str, raw: np.ndarray, origin: int,
                          volume_mean: np.ndarray, volume_scale: np.ndarray,
                          device: str = "cpu", small: bool = False) -> dict:
    """Tiny CPU smoke: one supervised residual sample, one optimizer step."""
    x_np, anchor = build_features(raw, origin, volume_mean, volume_scale)
    y_np, target_anchor = build_target(raw, origin)
    if anchor != target_anchor:
        raise AssertionError("input and label anchors differ")
    seed_all(2026)
    model = make_model(model_name, small=small).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.from_numpy(x_np[None]).to(device)
    y = torch.from_numpy(y_np[None]).to(device)
    pred = output_return(model, x)
    loss = torch.nn.functional.mse_loss(pred, y)
    optimizer.zero_grad(set_to_none=True); loss.backward()
    grads_finite = all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters())
    optimizer.step(); model.eval()
    with torch.no_grad():
        pred_after = output_return(model, x).cpu().numpy()[0]
    return {"model": model_name, "shape": list(pred.shape), "loss": float(loss.item()),
            "finite_gradients": bool(grads_finite), "finite_predictions": bool(np.isfinite(pred_after).all()),
            "baseline_zero_residual": restore_open(anchor, np.zeros(PRED_LEN)).tolist(),
            "anchor": anchor, "pred_return": pred_after.tolist(),
            "pred_open": restore_open(anchor, pred_after).tolist()}


def train_final_all_history(model_name: str, raw: np.ndarray, epochs: int,
                            opts, checkpoint: Path) -> dict:
    """Train the selected final model on every legal labeled window through cutoff."""
    n = len(raw)
    vm, vs = fit_volume_stats(raw, n)
    origins = range(SEQ_LEN, n - PRED_LEN + 1)
    xs, ys = [], []
    for origin in origins:
        x, _ = build_features(raw, origin, vm, vs)
        y, _ = build_target(raw, origin)
        xs.append(x); ys.append(y)
    dev = torch.device("cuda" if opts.device == "auto" and torch.cuda.is_available()
                       else ("cpu" if opts.device == "auto" else opts.device))
    pin_memory = dev.type == "cuda"
    dl = DataLoader(TensorDataset(torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))),
                    batch_size=opts.batch_size, shuffle=True, pin_memory=pin_memory)
    seed_all(opts.seed)
    model = make_model(model_name, small=False).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr)
    losses = []; start = time.time()
    for _ in range(epochs):
        model.train(); ep = []
        for x, y in dl:
            x, y = x.to(dev, non_blocking=pin_memory), y.to(dev, non_blocking=pin_memory)
            pred = output_return(model, x); loss = torch.nn.functional.mse_loss(pred, y)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite final residual loss")
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("non-finite final residual gradient")
            optimizer.step(); ep.append(float(loss.item()))
        losses.append(float(np.mean(ep)))
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "model": model_name,
                "volume_mean": vm, "volume_scale": vs,
                "config": {"seed": opts.seed, "seq_len": SEQ_LEN, "pred_len": PRED_LEN,
                           "epochs": epochs, "batch_size": opts.batch_size, "lr": opts.lr,
                           "train_end": n, "features": COLS, "features_mode": "MS"}}, checkpoint)
    x, anchor = build_features(raw, n, vm, vs)
    model.eval()
    with torch.no_grad():
        pred_return = output_return(model, torch.from_numpy(x[None]).to(dev)).cpu().numpy()[0]
    return {"pred_return": pred_return, "anchor": anchor, "epochs": epochs,
            "train_loss": losses[-1], "volume_mean": vm.tolist(), "volume_scale": vs.tolist(),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "runtime_seconds": time.time() - start, "device": str(dev),
            "training_samples": len(origins)}


def strvec(values: np.ndarray) -> str:
    return ";".join(f"{float(x):.8f}" for x in np.asarray(values).ravel())


def phase2_rows(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return df


def parse_vec(value: str) -> np.ndarray:
    return np.asarray([float(x) for x in value.split(";")], dtype=np.float64)


def assert_pair(stock: str, model: str, origin_date: str, target_dates: str,
                actual: np.ndarray, v0_row: pd.Series) -> None:
    if v0_row["origin_date"] != origin_date or v0_row["target_dates"] != target_dates:
        raise ValueError(f"Phase 2 date mismatch for {stock} {model} {origin_date}")
    v0_actual = parse_vec(v0_row["actual_open"])
    # Compare exactly the six-decimal strings stored by Phase 2 (including float32 rounding).
    canonical = lambda values: tuple(f"{float(x):.6f}" for x in values)
    if canonical(v0_actual) != canonical(actual):
        raise ValueError(f"Phase 2 actual Open mismatch for {stock} {model} {origin_date}")


def evaluate_records(stock: str, model: str, origin_date: str, target_dates: str,
                     anchor: float, actual_open: np.ndarray, pred_open: np.ndarray,
                     pred_return: np.ndarray, actual_return: np.ndarray,
                     version: str, alpha: float | None, fit: dict | None,
                     v0_row: pd.Series | None = None) -> list[dict]:
    met = residual_metrics(pred_return, actual_return, anchor)
    base_open = np.full(PRED_LEN, anchor, dtype=np.float64)
    base_ret = np.zeros(PRED_LEN, dtype=np.float64)
    base_met = residual_metrics(base_ret, actual_return, anchor)
    common = {"stock": stock, "model": model, "origin_date": origin_date,
              "target_dates": target_dates, "anchor_close": anchor,
              "actual_open": strvec(actual_open), "actual_return": strvec(actual_return)}
    records = []
    if v0_row is not None:
        v0_pred = parse_vec(v0_row["predicted_open"])
        v0_return = v0_pred / anchor - 1.0
        v0m = residual_metrics(v0_return, actual_return, anchor)
        # Preserve the saved Phase 2 aggregate scores; only the vectors are rounded in its CSV.
        for key in ("nmse", "mae", "rmse"):
            if key in v0_row and str(v0_row[key]) != "":
                v0m[key] = float(v0_row[key])
        records.append({**common, "version": "V0", "alpha": "", "predicted_return": strvec(v0_return),
                        "predicted_open": strvec(v0_pred), **v0m, "best_epoch": "", "train_loss": "",
                        "validation_loss": "", "runtime_seconds": "", "peak_gpu_memory_mb": "",
                        "checkpoint": "", "checkpoint_sha256": ""})
    records.append({**common, "version": version, "alpha": "" if alpha is None else alpha,
                    "predicted_return": strvec(pred_return), "predicted_open": strvec(pred_open), **met,
                    "best_epoch": "" if fit is None else fit["best_epoch"],
                    "train_loss": "" if fit is None else fit["train_loss"],
                    "validation_loss": "" if fit is None else fit["validation_loss"],
                    "runtime_seconds": "" if fit is None else fit["runtime_seconds"],
                    "peak_gpu_memory_mb": "" if fit is None else fit["peak_gpu_memory_mb"],
                    "checkpoint": "" if fit is None else fit["checkpoint"],
                    "checkpoint_sha256": "" if fit is None else fit["checkpoint_sha256"]})
    records.append({**common, "version": "LastCloseBaseline", "alpha": 0.0,
                    "predicted_return": strvec(base_ret), "predicted_open": strvec(base_open), **base_met,
                    "best_epoch": "", "train_loss": "", "validation_loss": "",
                    "runtime_seconds": "", "peak_gpu_memory_mb": "", "checkpoint": "",
                    "checkpoint_sha256": ""})
    return records


def load_v0_dev(stock: str, model: str, fold: str, phase2: pd.DataFrame) -> pd.Series:
    sel = phase2[(phase2.stock == stock) & (phase2.model == model) & (phase2.fold == fold)]
    if len(sel) != 1:
        raise ValueError(f"expected one Phase 2 development row for {stock} {model} {fold}, got {len(sel)}")
    return sel.iloc[0]


def load_v0_recent(stock: str, model: str, target_dates: str, phase2: pd.DataFrame) -> pd.Series:
    sel = phase2[(phase2.stock == stock) & (phase2.model == model) & (phase2.target_dates == target_dates)]
    if len(sel) != 1:
        raise ValueError(f"expected one Phase 2 recent row for {stock} {model} {target_dates}, got {len(sel)}")
    return sel.iloc[0]


def csv_write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows to write to {path}")
    pd.DataFrame(rows).to_csv(path, index=False)


def run_smoke(out: Path) -> None:
    stock = stock_configs()[0]
    df, raw = matrix(ROOT / "data/processed" / stock["output_file"])
    train_end, origin = 100, 160
    vm, vs = fit_volume_stats(raw, train_end)
    checks = []
    for name in MODELS:
        row = finite_checkpoint_fit(name, raw, origin, vm, vs, "cpu", small=True)
        if row["shape"] != [1, PRED_LEN] or not row["finite_gradients"] or not row["finite_predictions"]:
            raise AssertionError(f"CPU residual smoke failed: {row}")
        checks.append(row)
    artifact = {"status": "passed", "device": "cpu", "models": checks,
                "source_stock": stock["query_code"], "origin_date": df.date.iloc[origin],
                "history_start": df.date.iloc[origin - SEQ_LEN],
                "target_dates": df.date.iloc[origin:origin + PRED_LEN].tolist(),
                "scaler_fit_range_rows": [0, train_end], "alpha_candidates": ALPHAS,
                "alpha_zero_is_last_close_baseline": True}
    out.mkdir(parents=True, exist_ok=True)
    gated_rows = []
    for result in checks:
        for alpha in ALPHAS:
            pred = restore_open(result["anchor"], alpha * np.asarray(result["pred_return"]))
            for day, value in zip(df.date.iloc[origin:origin + PRED_LEN], pred):
                gated_rows.append({"stock": stock["query_code"], "model": result["model"],
                                   "alpha": alpha, "date": str(day), "open": float(value)})
    pd.DataFrame(gated_rows).to_csv(out / "smoke_gated_variants.csv", index=False)
    preview = next(r for r in checks if r["model"] == "TimeXer")
    preview_path = out / "smoke_submission_preview.csv"
    with preview_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n"); writer.writerow(["Date", "Open"])
        writer.writerows((str(day), f"{float(value):.6f}") for day, value in zip(
            df.date.iloc[origin:origin + PRED_LEN], preview["pred_open"]))
    with preview_path.open("r", encoding="utf-8", newline="") as f:
        exported = list(csv.reader(f))
    if exported[0] != ["Date", "Open"] or len(exported) != 6 or any(float(row[1]) <= 0 for row in exported[1:]):
        raise AssertionError("CPU smoke prediction export validation failed")
    artifact["prediction_export"] = {"file": preview_path.name, "rows": len(exported) - 1,
                                     "header": exported[0], "format_validation": "passed"}
    (out / "cpu_smoke.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(artifact, ensure_ascii=False, indent=2))


def run_development(opts, out: Path, v0_path: Path) -> None:
    p2 = phase2_rows(v0_path)
    v1_rows, alpha_scores, alpha_selection, comparisons = [], [], [], []
    for stock_cfg in stock_configs():
        stock = stock_cfg["query_code"]
        df, raw = matrix(ROOT / "data/processed" / stock_cfg["output_file"])
        manifest = manifest_for(stock_cfg)
        folds = {f["fold_id"]: f for f in manifest["splits"]}
        for model in MODELS:
            for fold_id in DEV_FOLDS:
                fold = folds[fold_id]; origin = int(fold["origin_index"])
                val_start = int(fold["segments"]["val"][0]); train_end = val_start
                checkpoint = out / "checkpoints" / f"development_{stock.replace('.', '_')}_{model}_{fold_id}.pt"
                fit_opts = SimpleNamespace(**vars(opts), model=model)
                fit = history_dataset(raw, train_end, val_start, origin, opts.device, fit_opts, False, checkpoint)
                print(f"development {stock} {model} {fold_id}: best_epoch={fit['best_epoch']} "
                      f"train_mse={fit['train_loss']:.8g} val_mse={fit['validation_loss']:.8g} "
                      f"runtime={fit['runtime_seconds']:.2f}s", flush=True)
                actual_return, anchor = build_target(raw, origin)
                if not math.isclose(anchor, fit["anchor"], rel_tol=0, abs_tol=1e-10):
                    raise AssertionError("fit and target anchors differ")
                pred_return = fit["prediction_return"]
                pred_open = restore_open(anchor, pred_return)
                actual_open = raw[origin:origin + PRED_LEN, 5]
                origin_date = str(df.date.iloc[origin])
                target_dates = ";".join(df.date.iloc[origin:origin + PRED_LEN].astype(str))
                v0 = load_v0_dev(stock, model, fold_id, p2)
                assert_pair(stock, model, origin_date, target_dates, actual_open, v0)
                common = {"stock": stock, "model": model, "fold": fold_id,
                          "origin_date": origin_date, "target_dates": target_dates,
                          "train_end_index_exclusive": train_end,
                          "train_cutoff": str(df.date.iloc[train_end - 1]),
                          "validation_start": str(df.date.iloc[val_start]),
                          "validation_end": str(df.date.iloc[origin - 1]),
                          "anchor_close": anchor, "predicted_return": strvec(pred_return),
                          "actual_return": strvec(actual_return), "predicted_open": strvec(pred_open),
                          "actual_open": strvec(actual_open), "best_epoch": fit["best_epoch"],
                          "train_loss": fit["train_loss"], "validation_loss": fit["validation_loss"],
                          "runtime_seconds": fit["runtime_seconds"],
                          "peak_gpu_memory_mb": fit["peak_gpu_memory_mb"],
                          "checkpoint": fit["checkpoint"], "checkpoint_sha256": fit["checkpoint_sha256"]}
                v1_rows.append({**common, **residual_metrics(pred_return, actual_return, anchor)})
                for alpha in ALPHAS:
                    gated = alpha * pred_return
                    gated_open = restore_open(anchor, gated)
                    metrics = residual_metrics(gated, actual_return, anchor)
                    alpha_scores.append({**common, "alpha": alpha,
                                         "gated_return": strvec(gated),
                                         "gated_open": strvec(gated_open), **metrics})
                comparisons.extend(evaluate_records(stock, model, origin_date, target_dates, anchor,
                    actual_open, pred_open, pred_return, actual_return, "V1", None, fit, v0))
            stock_scores = [r for r in alpha_scores if r["stock"] == stock and r["model"] == model]
            means = {a: float(np.mean([r["nmse"] for r in stock_scores if r["alpha"] == a])) for a in ALPHAS}
            selected = min(ALPHAS, key=lambda a: (means[a], a))
            # Attach V2 with development-selected alpha for each paired origin.
            for row in [r for r in v1_rows if r["stock"] == stock and r["model"] == model]:
                pr = parse_vec(row["predicted_return"]); ar = parse_vec(row["actual_return"])
                anchor = float(row["anchor_close"]); p_open = restore_open(anchor, selected * pr)
                actual_open = parse_vec(row["actual_open"])
                v0 = load_v0_dev(stock, model, row["fold"], p2)
                comparisons.extend(evaluate_records(stock, model, row["origin_date"], row["target_dates"], anchor,
                    actual_open, p_open, selected * pr, ar, "V2", selected, None, v0)[1:2])
            alpha_scores_for_key = [r for r in stock_scores]
            alpha_scores_for_key.sort(key=lambda r: (r["fold"], r["alpha"]))
            # Save a separate compact lock record. The full score table retains every candidate.
            alpha_selection.append({"stock": stock, "model": model,
                                    "selected_alpha": selected, "development_mean_nmse": means[selected],
                                    "candidate_mean_nmse_json": json.dumps(means, sort_keys=True)})
    csv_write(out / "development_v1_windows.csv", v1_rows)
    csv_write(out / "development_alpha_scores.csv", alpha_scores)
    csv_write(out / "development_alpha_selection.csv", alpha_selection)
    locks = alpha_selection
    grouped_locks: dict[str, dict] = {}
    for r in locks:
        grouped_locks.setdefault(r["stock"], {})[r["model"]] = {
            "selected_alpha": r["selected_alpha"],
            "candidate_mean_nmse": json.loads(r["candidate_mean_nmse_json"]),
            "development_mean_nmse": r["development_mean_nmse"]}
    (out / "locked_alphas.json").write_text(json.dumps({"selection_basis": "development only",
        "alphas": ALPHAS, "stocks": grouped_locks},
        ensure_ascii=False, indent=2), encoding="utf-8")
    csv_write(out / "development_comparison.csv", comparisons)
    print(f"Development complete: {len(v1_rows)} fits; locked {len(locks)} stock-model alphas")


def read_alpha_locks(path: Path) -> dict[tuple[str, str], float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {(stock, model): float(info[model]["selected_alpha"])
            for stock, info in data["stocks"].items() for model in info}


def run_recent(opts, out: Path, v0_path: Path) -> None:
    p2 = phase2_rows(v0_path)
    locks = read_alpha_locks(out / "locked_alphas.json")
    v1_rows, comparisons, train_rows = [], [], []
    for stock_cfg in stock_configs():
        stock = stock_cfg["query_code"]
        df, raw = matrix(ROOT / "data/processed" / stock_cfg["output_file"])
        n = len(raw)
        origins = list(range(n - 40, n - PRED_LEN + 1, PRED_LEN))
        if len(origins) != 8:
            raise AssertionError(f"expected 8 recent origins, got {len(origins)}")
        for model in MODELS:
            alpha = locks[(stock, model)]
            for cycle in range(2):
                group = origins[cycle * 4:(cycle + 1) * 4]
                retrain_origin = group[0]
                val_start = retrain_origin - 126
                train_end = val_start
                checkpoint = out / "checkpoints" / f"recent_{stock.replace('.', '_')}_{model}_cycle{cycle + 1}.pt"
                fit_opts = SimpleNamespace(**vars(opts), model=model)
                fit = history_dataset(raw, train_end, val_start, retrain_origin, opts.device, fit_opts, False, checkpoint)
                print(f"recent {stock} {model} cycle={cycle + 1}: best_epoch={fit['best_epoch']} "
                      f"train_mse={fit['train_loss']:.8g} val_mse={fit['validation_loss']:.8g} "
                      f"runtime={fit['runtime_seconds']:.2f}s", flush=True)
                train_rows.append({"stock": stock, "model": model, "cycle": cycle + 1,
                    "retrain_origin": str(df.date.iloc[retrain_origin]),
                    "train_cutoff": str(df.date.iloc[train_end - 1]),
                    "validation_start": str(df.date.iloc[val_start]),
                    "validation_end": str(df.date.iloc[retrain_origin - 1]),
                    "best_epoch": fit["best_epoch"], "train_loss": fit["train_loss"],
                    "validation_loss": fit["validation_loss"], "runtime_seconds": fit["runtime_seconds"],
                    "peak_gpu_memory_mb": fit["peak_gpu_memory_mb"], "device": fit["device"],
                    "parameter_count": fit["params"], "checkpoint": fit["checkpoint"],
                    "checkpoint_sha256": fit["checkpoint_sha256"], "locked_alpha": alpha})
                # Fixed weights in this 20-trading-day cycle; only the known 60-row input moves.
                dev = torch.device(opts.device if opts.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
                saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
                model_obj = make_model(model, small=False).to(dev)
                model_obj.load_state_dict(saved["state_dict"]); model_obj.eval()
                for origin in group:
                    vm, vs = fit["volume_mean"], fit["volume_scale"]
                    x, anchor = build_features(raw, origin, vm, vs)
                    with torch.no_grad():
                        pred_return = output_return(model_obj, torch.from_numpy(x[None]).to(dev, non_blocking=dev.type == "cuda")).cpu().numpy()[0]
                    actual_return, target_anchor = build_target(raw, origin)
                    if not math.isclose(anchor, target_anchor, rel_tol=0, abs_tol=1e-10):
                        raise AssertionError("recent input/target anchor mismatch")
                    actual_open = raw[origin:origin + PRED_LEN, 5]
                    pred_open = restore_open(anchor, pred_return)
                    origin_date = str(df.date.iloc[origin])
                    target_dates = ";".join(df.date.iloc[origin:origin + PRED_LEN].astype(str))
                    v0 = load_v0_recent(stock, model, target_dates, p2)
                    assert_pair(stock, model, origin_date, target_dates, actual_open, v0)
                    fit_at_origin = {**fit, "best_epoch": fit["best_epoch"],
                                     "train_loss": fit["train_loss"], "validation_loss": fit["validation_loss"]}
                    v1_rows.append({"stock": stock, "model": model, "cycle": cycle + 1,
                        "origin_date": origin_date, "target_dates": target_dates,
                        "retrain_origin": str(df.date.iloc[retrain_origin]),
                        "train_cutoff": str(df.date.iloc[train_end - 1]),
                        "validation_start": str(df.date.iloc[val_start]),
                        "validation_end": str(df.date.iloc[retrain_origin - 1]),
                        "anchor_close": anchor, "locked_alpha": alpha,
                        "predicted_return": strvec(pred_return), "actual_return": strvec(actual_return),
                        "predicted_open": strvec(pred_open), "actual_open": strvec(actual_open),
                        "best_epoch": fit["best_epoch"], "train_loss": fit["train_loss"],
                        "validation_loss": fit["validation_loss"],
                        "runtime_seconds": fit["runtime_seconds"],
                        "peak_gpu_memory_mb": fit["peak_gpu_memory_mb"],
                        "checkpoint": fit["checkpoint"], "checkpoint_sha256": fit["checkpoint_sha256"],
                        **residual_metrics(pred_return, actual_return, anchor)})
                    comparisons.extend(evaluate_records(stock, model, origin_date, target_dates, anchor,
                        actual_open, pred_open, pred_return, actual_return, "V1", None, fit_at_origin, v0))
                    gated_return = alpha * pred_return
                    comparisons.extend(evaluate_records(stock, model, origin_date, target_dates, anchor,
                        actual_open, restore_open(anchor, gated_return), gated_return, actual_return,
                        "V2", alpha, None, v0)[1:2])
    csv_write(out / "recent_v1_windows.csv", v1_rows)
    csv_write(out / "recent_training_runs.csv", train_rows)
    csv_write(out / "recent_comparison.csv", comparisons)
    print(f"Recent complete: {len(train_rows)} fits; {len(v1_rows)} residual windows")


def run_final(opts, out: Path) -> None:
    """Retrain a selected neural predictor only when recent evidence justifies it."""
    summary = pd.read_csv(out / "candidate_summary.csv")
    config = {"student_id": opts.student_id, "as_of": "2026-10-30",
              "optimizer": "Adam", "learning_rate": opts.lr,
              "epoch_cap": opts.epochs, "batch_size": opts.batch_size,
              "early_stopping_patience": opts.patience, "seed": opts.seed,
              "seq_len": SEQ_LEN, "pred_len": PRED_LEN,
              "features_order": COLS, "features_mode": "MS",
              "models": {}}
    detail_rows, metadata = [], []
    for stock_cfg in stock_configs():
        stock = stock_cfg["query_code"]
        df, raw = matrix(ROOT / "data/processed" / stock_cfg["output_file"])
        sub = summary[summary.stock == stock]
        baseline = sub[sub.version == "LastCloseBaseline"]
        choices = sub[sub.version.isin(["V1", "V2"])].copy()
        if choices.empty:
            method, model_name, alpha = "LastCloseBaseline", "LastCloseBaseline", 0.0
        else:
            choices["improvement_vs_baseline"] = 1.0 - choices.mean_nmse / float(baseline.mean_nmse.min())
            viable = choices[(choices.improvement_vs_baseline >= 0.05) &
                             (choices.median_nmse < float(baseline.median_nmse.min())) &
                             (choices.beats_baseline_windows >= 5)]
            if viable.empty:
                method, model_name, alpha = "LastCloseBaseline", "LastCloseBaseline", 0.0
            else:
                best = viable.sort_values(["mean_nmse", "median_nmse"]).iloc[0]
                method, model_name = str(best.version), str(best.model)
                alpha = 1.0 if method == "V1" else float(best.alpha)
        config["models"][stock] = {"method": method, "model": model_name, "alpha": alpha}
        dates = json.loads((ROOT / "config/prediction_dates.json").read_text(encoding="utf-8"))["dates"]
        if model_name == "LastCloseBaseline":
            last_close = float(raw[-1, 2]); pred = np.full(PRED_LEN, last_close)
            checkpoint = ""; checkpoint_sha256 = ""; cutoff = str(df.date.iloc[-1]); epoch = None
        else:
            # Epoch count is locked from matching development fits; all history through cutoff is train data.
            dev = pd.read_csv(out / "development_v1_windows.csv")
            eps = dev[(dev.stock == stock) & (dev.model == model_name)].best_epoch.astype(int)
            epoch = max(1, int(np.median(eps)))
            final_opts = SimpleNamespace(**vars(opts))
            fit = train_final_all_history(model_name, raw, epoch, final_opts,
                out / "checkpoints" / f"final_{stock.replace('.', '_')}_{model_name}.pt")
            pred = restore_open(fit["anchor"], fit["pred_return"] * alpha)
            checkpoint = fit["checkpoint"]; checkpoint_sha256 = fit["checkpoint_sha256"]
            cutoff = str(df.date.iloc[-1])
            config["models"][stock].update({"epochs": epoch,
                "training_samples": fit["training_samples"], "train_loss": fit["train_loss"],
                "volume_log1p_mean": fit["volume_mean"], "volume_log1p_scale": fit["volume_scale"],
                "checkpoint_sha256": checkpoint_sha256, "runtime_seconds": fit["runtime_seconds"],
                "device": fit["device"]})
        for day, value in zip(dates, pred):
            detail_rows.append({"stock": stock, "date": day, "open": float(value)})
        metadata.append({"stock": stock, "method": method, "model": model_name, "alpha": alpha,
                         "prediction_source": "LastCloseBaseline" if model_name == "LastCloseBaseline" else "trained_model",
                         "training_cutoff": cutoff, "last_input_date": str(df.date.iloc[-1]),
                         "last_known_close": float(raw[-1, 2]), "checkpoint": checkpoint,
                         "checkpoint_sha256": checkpoint_sha256,
                         "epoch_reference_median": epoch, "predicted_dates": dates,
                         "source_csv_sha256": hashlib.sha256((ROOT / "data/processed" / stock_cfg["output_file"]).read_bytes()).hexdigest()})
    final_dir = out / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(detail_rows).to_csv(final_dir / "prediction_details.csv", index=False)
    (final_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "final_training_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"prediction_details": str(final_dir / "prediction_details.csv"),
                      "metadata": str(final_dir / "metadata.json"), "methods": config["models"]}, ensure_ascii=False, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["smoke", "development", "recent", "final"], required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--student-id", default="")
    p.add_argument("--phase-dir", type=Path, default=ROOT / "results/residual_phase3")
    p.add_argument("--v0-development", type=Path, default=ROOT / "results/remote_gpu_run/results/metrics/development_windows.csv")
    p.add_argument("--v0-recent", type=Path, default=ROOT / "results/remote_gpu_run/results/metrics/recent_windows.csv")
    a = p.parse_args()
    if a.mode == "smoke":
        run_smoke(a.phase_dir / "smoke"); return
    opts = SimpleNamespace(device=a.device, epochs=a.epochs, batch_size=a.batch_size,
                           patience=a.patience, lr=a.lr, seed=a.seed)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    a.phase_dir.mkdir(parents=True, exist_ok=True)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        commit_file = ROOT / "UPDATE_COMMIT.txt"
        commit = commit_file.read_text(encoding="utf-8").strip() if commit_file.exists() else "unknown"
    input_hashes = {}
    for stock_cfg in stock_configs():
        src = ROOT / "data/processed" / stock_cfg["output_file"]
        input_hashes[str(src.relative_to(ROOT))] = hashlib.sha256(src.read_bytes()).hexdigest()
    experiment_config = {"phase": 3, "mode": a.mode, "code_commit": commit,
        "as_of": "2026-10-30", "stocks": [s["query_code"] for s in stock_configs()],
        "processed_sha256": input_hashes, "seq_len": SEQ_LEN, "pred_len": PRED_LEN,
        "features_order": COLS, "price_features": "historical price / close[origin-1] - 1",
        "volume_amount_transform": "log1p, fit mean/std on raw[:train_end] only",
        "target": "future_open / close[origin-1] - 1", "alphas": ALPHAS,
        "development_folds": DEV_FOLDS, "recent_holdout_tail_trading_days": 40,
        "recent_windows_per_stock": 8, "recent_retrain_period_trading_days": 20,
        "recent_validation_trading_days": 126, "epochs": a.epochs,
        "early_stopping_patience": a.patience, "optimizer": "Adam",
        "learning_rate": a.lr, "batch_size": a.batch_size, "seed": a.seed,
        "device": a.device, "cuda_float32_matmul_precision": "high" if torch.cuda.is_available() else "default",
        "pin_memory": a.device == "cuda" or (a.device == "auto" and torch.cuda.is_available()),
        "model_configs": {"d_model": 128, "d_ff": 256,
            "e_layers": 2, "n_heads": 4, "dropout": 0.1, "label_len": 30,
            "features": "MS", "use_norm": True},
        "phase2_v0_files": [str(a.v0_development), str(a.v0_recent)]}
    config_path = a.phase_dir / f"experiment_config_{a.mode}.json"
    config_path.write_text(json.dumps(experiment_config, ensure_ascii=False, indent=2), encoding="utf-8")
    if a.mode == "development":
        run_development(opts, a.phase_dir, a.v0_development)
    elif a.mode == "recent":
        run_recent(opts, a.phase_dir, a.v0_recent)
    else:
        if not a.student_id:
            raise ValueError("--student-id is required for final predictions")
        opts.student_id = a.student_id
        run_final(opts, a.phase_dir)


if __name__ == "__main__":
    main()
