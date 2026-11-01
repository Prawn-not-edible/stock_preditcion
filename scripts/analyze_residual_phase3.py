"""Pair V0/V1/V2 Phase 3 outputs and summarize recent-window evidence."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/residual_phase3"


def vectors(value: str) -> np.ndarray:
    return np.asarray([float(x) for x in str(value).split(";")], dtype=np.float64)


def paired(path: Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path, dtype={"stock": str, "model": str, "version": str})
    required = {"stock", "model", "version", "origin_date", "target_dates", "actual_open",
                "predicted_open", "actual_return", "predicted_return", "anchor_close", "nmse"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    audit = {"file": str(path.relative_to(ROOT)), "groups_checked": 0, "dates_and_actuals_equal": True}
    keys = ["stock", "model", "origin_date"]
    for key, rows in df.groupby(keys, dropna=False):
        if len(rows) != rows.version.nunique():
            raise ValueError(f"duplicate version in paired group {key}")
        if "V0" not in set(rows.version) or "V1" not in set(rows.version) or "V2" not in set(rows.version):
            raise ValueError(f"missing V0/V1/V2 in paired group {key}")
        dates = rows.target_dates.unique()
        actual = rows.actual_open.unique()
        if len(dates) != 1 or len(actual) != 1:
            raise ValueError(f"paired truth/date mismatch for {key}")
        audit["groups_checked"] += 1
    return df, audit


def summarize(df: pd.DataFrame, horizon: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = (df[df.version == "LastCloseBaseline"]
            .set_index(["stock", "model", "origin_date"])["nmse"])
    v0 = (df[df.version == "V0"]
          .groupby(["stock", "model"], as_index=True).nmse.mean())
    rows = []
    for (stock, model, version), g in df.groupby(["stock", "model", "version"]):
        if version == "LastCloseBaseline":
            continue
        base_vals, wins = [], 0
        for r in g.itertuples():
            b = float(base.loc[(stock, model, r.origin_date)])
            base_vals.append(b)
            wins += int(float(r.nmse) < b)
        mean = float(g.nmse.mean()); bmean = float(np.mean(base_vals))
        v0mean = float(v0.loc[(stock, model)])
        rows.append({"horizon": horizon, "stock": stock, "model": model, "version": version,
            "alpha": ";".join(sorted(set(str(x) for x in g.alpha.dropna()))) if "alpha" in g else "",
            "mean_nmse": mean, "median_nmse": float(g.nmse.median()),
            "std_nmse": float(g.nmse.std(ddof=1)) if len(g) > 1 else 0.0,
            "windows": int(len(g)), "beats_baseline_windows": wins,
            "relative_improvement_vs_baseline": (bmean - mean) / bmean if bmean else 0.0,
            "relative_improvement_vs_v0": (v0mean - mean) / v0mean if v0mean else 0.0,
            "baseline_mean_nmse": bmean, "v0_mean_nmse": v0mean})
    summary = pd.DataFrame(rows)
    # Carry one explicit LastClose row for each stock/model so the final decision is auditable.
    base_rows = []
    for (stock, model), g in df[df.version == "LastCloseBaseline"].groupby(["stock", "model"]):
        base_rows.append({"horizon": horizon, "stock": stock, "model": model,
            "version": "LastCloseBaseline", "alpha": "0.0", "mean_nmse": float(g.nmse.mean()),
            "median_nmse": float(g.nmse.median()), "std_nmse": float(g.nmse.std(ddof=1)),
            "windows": len(g), "beats_baseline_windows": 0,
            "relative_improvement_vs_baseline": 0.0, "relative_improvement_vs_v0": 0.0,
            "baseline_mean_nmse": float(g.nmse.mean()),
            "v0_mean_nmse": float(v0.loc[(stock, model)])})
    summary = pd.concat([summary, pd.DataFrame(base_rows)], ignore_index=True)
    step_rows, corr_rows = [], []
    for (stock, model, version), g in df[df.version != "LastCloseBaseline"].groupby(["stock", "model", "version"]):
        pred_all, actual_all = [], []
        for step in range(5):
            errors = []
            for r in g.itertuples():
                pr, ar = vectors(r.predicted_return), vectors(r.actual_return)
                errors.append(float((pr[step] - ar[step]) ** 2))
                if step == 0:
                    pred_all.extend(pr.tolist()); actual_all.extend(ar.tolist())
            step_rows.append({"horizon": horizon, "stock": stock, "model": model,
                              "version": version, "step": step + 1, "nmse": float(np.mean(errors))})
        corr = float(np.corrcoef(pred_all, actual_all)[0, 1]) if len(pred_all) > 1 and np.std(pred_all) > 0 and np.std(actual_all) > 0 else float("nan")
        corr_rows.append({"horizon": horizon, "stock": stock, "model": model,
                          "version": version, "return_correlation": corr,
                          "return_pairs": len(pred_all)})
    return summary, pd.DataFrame(step_rows), pd.DataFrame(corr_rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dev, dev_audit = paired(OUT / "development_comparison.csv")
    recent, recent_audit = paired(OUT / "recent_comparison.csv")
    dev_sum, _, _ = summarize(dev, "development")
    recent_sum, steps, corrs = summarize(recent, "recent_40d")
    dev_sum.to_csv(OUT / "development_summary.csv", index=False)
    recent_sum.to_csv(OUT / "candidate_summary.csv", index=False)
    recent_sum.to_csv(OUT / "recent_summary.csv", index=False)
    steps.to_csv(OUT / "recent_step_nmse.csv", index=False)
    corrs.to_csv(OUT / "recent_return_correlations.csv", index=False)
    audit = {"phase2_v0_pairing": [dev_audit, recent_audit],
             "recent_windows_per_stock_model": 8,
             "all_expected_recent_windows": bool((recent_sum[recent_sum.version == "V0"].windows == 8).all()),
             "alpha_zero_is_last_close_baseline": True,
             "recent_is_fast_model_selection_data_not_independent_final_test": True}
    (OUT / "alignment_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    try:
        import matplotlib.pyplot as plt
        plot = recent_sum[recent_sum.version.isin(["V0", "V1", "V2", "LastCloseBaseline"])]
        stocks = sorted(plot.stock.unique()); models = sorted(plot.model.unique())
        fig, axes = plt.subplots(len(stocks), 1, figsize=(12, 4 * len(stocks)), squeeze=False)
        for ax, stock in zip(axes[:, 0], stocks):
            sub = plot[plot.stock == stock]
            labels = [f"{m}\n{v}" for m, v in zip(sub.model, sub.version)]
            ax.bar(np.arange(len(sub)), sub.mean_nmse)
            ax.set_xticks(np.arange(len(sub)), labels, rotation=45, ha="right")
            ax.set_title(stock); ax.set_ylabel("Recent 40-day mean NMSE (8 windows)")
            ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(OUT / "recent_nmse_comparison.png", dpi=160)
        plt.close(fig)
    except ImportError:
        audit["plot_status"] = "matplotlib unavailable"
        (OUT / "alignment_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
