#!/usr/bin/env python3
"""BaoStock daily CSV -> strictly audited model CSV + expanding-window split manifests.

This is a *data-only* preprocessing tool. It deliberately does not fit scalers,
create future labels, impute data, or assume trading days from the calendar.
The split JSON describes exclusive row boundaries for an external model adapter.

Example (run from the stock_preditcion repository root):
  python scripts/prepare.py --input-dir data/raw --output-dir data/processed \
      --splits-dir data/splits --audit-dir results/metrics --as-of 2026-10-30
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

MODEL_COLS = ["date", "high", "low", "close", "volume", "amount", "open"]
NUM_COLS = MODEL_COLS[1:]
REQUIRED_RAW = ["date", "code", "open", "high", "low", "close", "volume", "amount", "tradestatus", "adjustflag"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_folds(n: int, seq_len: int = 60, pred_len: int = 5,
              validation_size: int = 126, recent_holdout: int = 180,
              minimum_train: int = 756, count: int = 6) -> list[dict[str, Any]]:
    """Six expanding historical windows, leaving the recent 180 rows untouched.

    Each origin is the first test-target row. Validation is the 126 known rows
    immediately before it; training ends before that validation interval.
    """
    first = max(seq_len + pred_len, minimum_train + validation_size)
    last = n - recent_holdout - pred_len
    if last < first:
        raise ValueError(f"{n} rows insufficient for a 3-year train, validation, test and recent holdout")
    import numpy as np
    origins = sorted(set(int(x) for x in np.linspace(first, last, min(count, last-first+1))))
    folds = []
    for i, origin in enumerate(origins, 1):
        val_start = origin - validation_size
        test_end = origin + pred_len
        folds.append({
            "fold_id": f"origin_{i:02d}",
            "origin_index": origin,
            "segments": {"train": [0, val_start], "val": [val_start, origin], "test": [origin, test_end]},
            "valid_origin_ranges_inclusive": {
                "train": [seq_len, val_start-pred_len],
                "val": [val_start, origin-pred_len],
                "test": [origin, test_end-pred_len],
            },
            "train_window_count_stride1": val_start-seq_len-pred_len+1,
            "val_window_count_stride1": validation_size-pred_len+1,
            "test_window_count_stride1": 1,
        })
    return folds


def clean_one(path: Path, code: str, as_of: date | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    # Read strings to preserve BaoStock's raw values for validation.
    raw = pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    missing = sorted(set(REQUIRED_RAW) - set(raw.columns))
    if missing:
        raise ValueError(f"{path}: missing columns: {missing}")
    if raw.empty:
        raise ValueError(f"{path}: empty CSV")
    codes = raw["code"].str.strip().str.lower()
    if not codes.eq(code).all():
        raise ValueError(f"{path}: code mismatch; expected {code}; got {sorted(codes.unique())[:10]}")
    status = raw["tradestatus"].str.strip()
    invalid_status = sorted(set(status) - {"0", "1"})
    if invalid_status:
        raise ValueError(f"{path}: unsupported tradestatus values: {invalid_status}")
    statuses = raw["adjustflag"].str.strip().unique().tolist()
    if statuses != ["3"]:
        if sorted(statuses) != ["3"]:
            raise ValueError(f"{path}: adjustflag is {statuses}, expected only '3' (unadjusted)")
    raw_dates = pd.to_datetime(raw["date"], format="%Y-%m-%d", errors="coerce")
    if raw_dates.isna().any():
        raise ValueError(f"{path}: {int(raw_dates.isna().sum())} invalid date(s)")
    raw = raw.copy()
    raw["date"] = raw_dates
    if as_of is not None:
        later_mask = raw["date"].dt.date > as_of
        excluded_later = int(later_mask.sum())
        excluded_later_dates = raw.loc[later_mask, "date"].dt.strftime("%Y-%m-%d").tolist()
        raw = raw.loc[~later_mask].copy()
    else:
        excluded_later = 0
        excluded_later_dates = []
    if raw["date"].duplicated().any():
        dupes = raw.loc[raw["date"].duplicated(keep=False), "date"].dt.strftime("%Y-%m-%d").tolist()[:20]
        raise ValueError(f"{path}: duplicate raw dates; investigate, examples: {dupes}")
    halted = int(raw["tradestatus"].eq("0").sum())
    halted_dates = raw.loc[raw["tradestatus"].eq("0"), "date"].dt.strftime("%Y-%m-%d").tolist()
    traded = raw.loc[raw["tradestatus"].eq("1")].copy()
    if traded.empty:
        raise ValueError(f"{path}: no active trading rows after filtering")
    if traded["date"].duplicated().any():
        dupes = traded.loc[traded["date"].duplicated(keep=False), "date"].dt.strftime("%Y-%m-%d").tolist()[:10]
        raise ValueError(f"{path}: duplicate active trade dates; investigate, examples: {dupes}")
    for col in NUM_COLS:
        traded[col] = pd.to_numeric(traded[col], errors="coerce")
    finite = np.isfinite(traded[NUM_COLS].to_numpy(dtype=float))
    if not finite.all():
        bad = np.argwhere(~finite)
        examples = [{"row": int(i), "date": traded["date"].iloc[int(i)].strftime("%Y-%m-%d"), "column": NUM_COLS[int(j)]} for i,j in bad[:20]]
        raise ValueError(f"{path}: non-finite active-row values; examples: {examples}")
    if not (traded[["open", "close", "high", "low"]] > 0).all().all():
        raise ValueError(f"{path}: non-positive OHLC price")
    if (traded[["volume", "amount"]] < 0).any().any():
        raise ValueError(f"{path}: negative volume or amount")
    # Numeric comparisons allow tiny rounding differences, not material anomalies.
    eps = 1e-6
    if ((traded["high"] + eps < traded[["open", "close", "low"]].max(axis=1)) |
        (traded["low"] - eps > traded[["open", "close", "high"]].min(axis=1))).any():
        raise ValueError(f"{path}: inconsistent OHLC high/low bounds")
    traded = traded.sort_values("date").reset_index(drop=True)
    if len(traded) < 200:
        raise ValueError(f"{path}: too few trading rows ({len(traded)})")
    out = traded[MODEL_COLS].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    if not math.isfinite(out[NUM_COLS].to_numpy(dtype=float).sum()):
        raise ValueError(f"{path}: non-finite numeric values")
    audit = {
        "stock": code,
        "source": str(path),
        "source_sha256": sha256(path),
        "raw_rows": int(len(pd.read_csv(path, usecols=["date"], encoding="utf-8-sig"))),
        "excluded_after_as_of": excluded_later,
        "excluded_after_as_of_dates": excluded_later_dates,
        "excluded_halted_tradestatus_0": halted,
        "halted_dates": halted_dates,
        "processed_rows": int(len(out)),
        "first_trading_date": out["date"].iloc[0],
        "last_trading_date": out["date"].iloc[-1],
        "adjustflag": "3",
        "columns": MODEL_COLS,
        "zero_volume_traded_rows": int((traded["volume"] == 0).sum()),
    }
    return out, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/stocks.json"))
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--audit-dir", type=Path, default=Path("results/metrics"))
    parser.add_argument("--as-of", default="2026-10-30", help="exclusive of rows later than this date (YYYY-MM-DD)")
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--pred-len", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true", help="explicitly overwrite processed output")
    args = parser.parse_args()
    cutoff = date.fromisoformat(args.as_of) if args.as_of else None
    if args.seq_len < 1 or args.pred_len < 1:
        parser.error("seq_len and pred_len must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.splits_dir.mkdir(parents=True, exist_ok=True)
    args.audit_dir.mkdir(parents=True, exist_ok=True)
    stocks_cfg = json.loads(args.config.read_text(encoding="utf-8"))["stocks"]
    reports = []
    for stock in stocks_cfg:
        code, filename = stock["query_code"], stock["output_file"]
        src = args.input_dir / filename
        if not src.exists():
            raise FileNotFoundError(f"Missing original CSV: {src}")
        dest = args.output_dir / filename
        split_dest = args.splits_dir / f"{Path(filename).stem}.json"
        if not args.overwrite and (dest.exists() or split_dest.exists()):
            raise FileExistsError(f"Existing processed output for {code}; use --overwrite after review")
        df, audit = clean_one(src, code, cutoff)
        folds = get_folds(len(df), args.seq_len, args.pred_len)
        # Attach dates for each exclusive boundary; never guess weekend/holiday rows.
        for fold in folds:
            bounds = fold["segments"]
            fold["date_segments"] = {
                name: [df["date"].iloc[a], df["date"].iloc[b-1]]
                for name, (a, b) in bounds.items()
            }
        manifest = {
            "stock": code,
            "processed_csv": str(dest),
            "N": len(df),
            "seq_len": args.seq_len,
            "pred_len": args.pred_len,
            "boundary_convention": "half_open_0_based_row_indices",
            "origin_convention": "origin o predicts [o,o+pred_len); X uses [o-seq_len,o)",
            "splits": folds,
        }
        df.to_csv(dest, index=False, encoding="utf-8")
        split_dest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        audit["output_path"] = str(dest)
        audit["output_sha256"] = sha256(dest)
        audit["fold_count"] = len(folds)
        reports.append(audit)
        print(f"{code} -> {dest}: {len(df)} rows, {len(folds)} folds")
    (args.audit_dir / "preprocessing_audit.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Audit written: {args.audit_dir / 'preprocessing_audit.json'}")


if __name__ == "__main__":
    main()
