"""Audited Phase6 Stock-ID vs Phase5 P2 no-ID, identical dates and alpha search.

Uses Phase5 saved P2 predictions as a frozen control, NOT a re-fit using new settings.
The same 3 development origins are used for alpha selection, and the same
8 recent windows for comparison (NOT an independent test set).
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import phase4_joint as p4
import residual_phase3 as p3

OUT_DEFAULT = ROOT / "results/phase6_stock_id"
PHASE5 = ROOT / "results/phase5_targetaware"
STOCKS = list(p4.TARGETS)
ALPHAS = list(map(float, p3.ALPHAS))


def validate_pair(no_id: pd.DataFrame, with_id: pd.DataFrame, stage: str):
    idx = ["stock", "model", "origin_date", "alpha"]
    for label, frame in [("P2", no_id), ("P2_ID", with_id)]:
        if frame.duplicated(idx).any():
            raise ValueError(f"Duplicate {stage} {label} prediction")
        if frame[idx].isna().any().any():
            raise ValueError(f"Missing {stage} {label} pairing fields")
    a = no_id.set_index(idx).sort_index()
    b = with_id.set_index(idx).sort_index()
    if not a.index.equals(b.index):
        raise ValueError(f"{stage}: missing/unmatched P2 vs ID predictions")
    for ca in ("target_dates", "actual_open", "anchor_close"):
        if ca == "anchor_close":
            if not np.allclose(a[ca].astype(float), b[ca].astype(float), rtol=0, atol=1e-5):
                raise ValueError(f"{stage}: anchor mismatch")
        elif ca == "actual_open":
            for va, vb in zip(a[ca], b[ca]):
                pa, pb = p3.parse_vec(va), p3.parse_vec(vb)
                if pa.shape != (5,) or not np.allclose(pa, pb, rtol=0, atol=1e-5):
                    raise ValueError(f"{stage}: actual Open mismatch")
        elif not (a[ca].astype(str) == b[ca].astype(str)).all():
            raise ValueError(f"{stage}: target calendar mismatch")
    for (stock, model, origin), group in with_id.groupby(["stock", "model", "origin_date"]):
        if set(np.round(group.alpha.astype(float), 8)) != set(ALPHAS):
            raise ValueError(f"{stage}: alpha grid missing for {stock}/{model}/{origin}")
    return {"stage": stage, "paired_rows": len(no_id),
            "paired_groups": no_id.groupby(["stock", "model", "origin_date"]).ngroups,
            "dates_and_truth_equal": True}


def alpha_locks(dev: pd.DataFrame):
    locks = {}
    for (pool, model, stock), g in dev.groupby(["pool", "model", "stock"]):
        if g.origin_date.nunique() != 3:
            raise ValueError(f"Expected exactly 3 dev origins for {pool}/{model}/{stock}")
        score = g.groupby("alpha").nmse.mean().sort_values()
        # deterministic small-alpha tie break
        best = min(ALPHAS, key=lambda a: (round(float(score.loc[a]), 14), a))
        locks[(pool, model, stock)] = best
    return locks


def lock_predictions(df, locks):
    frames = []
    for (pool, model, stock), g in df.groupby(["pool", "model", "stock"]):
        alpha = locks[(pool, model, stock)]
        frames.append(g[np.isclose(g.alpha.astype(float), alpha, rtol=0, atol=1e-10)])
    return pd.concat(frames, ignore_index=True)


def summarize(df, baseline):
    b_lookup = {(r.stock, r.origin_date): r.nmse for r in baseline.itertuples()}
    output = []
    for (stock, model, pool), g in df.groupby(["stock", "model", "pool"]):
        bas = baseline[baseline.stock.eq(stock)]
        diffs = np.array([float(r.nmse) - float(b_lookup[(stock, r.origin_date)]) for r in g.itertuples()])
        eps = 1e-12 + 1e-8 * np.array([b_lookup[(stock, r.origin_date)] for r in g.itertuples()])
        output.append({"stock": stock, "model": model, "pool": pool, "alpha": float(g.alpha.iloc[0]),
                       "windows": len(g), "mean_nmse": float(g.nmse.mean()),
                       "median_nmse": float(g.nmse.median()),
                       "baseline_mean_nmse": float(bas.nmse.mean()),
                       "baseline_median_nmse": float(bas.nmse.median()),
                       "improvement_vs_baseline": 1 - float(g.nmse.mean()) / float(bas.nmse.mean()),
                       "wins": int((diffs < -eps).sum()), "ties": int((abs(diffs) <= eps).sum()),
                       "losses": int((diffs > eps).sum())})
    tab = pd.DataFrame(output)
    basep = tab[tab.pool.eq("P2")].set_index(["stock", "model"])
    tab["improvement_vs_no_id_same_model"] = [
        1 - row.mean_nmse / basep.loc[(row.stock, row.model), "mean_nmse"]
        for row in tab.itertuples()]
    return tab


def incumbent_predictions(rec5: pd.DataFrame):
    choices = json.loads((PHASE5 / "final_model_selection.json").read_text(encoding="utf8"))["stocks"]
    rows = []
    for stock in STOCKS:
        c = choices[stock]
        if c["pool"] == "LastCloseBaseline":
            g = rec5[(rec5.stock == stock) & (rec5.alpha == 0) & (rec5.pool == "P2")]
            # P2 is used only to supply identical target dates and baseline scores.
            g = g.drop_duplicates(["origin_date"])
        else:
            g = rec5[(rec5.stock == stock) & (rec5.pool == c["pool"])
                     & (rec5.model == c["model"])
                     & np.isclose(rec5.alpha.astype(float), float(c["alpha"]))]
        if len(g) != 8:
            raise ValueError(f"Phase5 incumbent incomplete for {stock}: {len(g)}")
        for r in g.itertuples():
            rows.append({"stock": stock, "origin_date": r.origin_date, "nmse": float(r.nmse)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    pairs = []
    stages = {}
    run_configs = {}
    for stage in ("development", "recent"):
        run_configs[stage] = json.loads((args.out / stage / "run_config.json").read_text(encoding="utf8"))
        p6 = pd.read_csv(args.out / stage / "predictions.csv")
        p5 = pd.read_csv(PHASE5 / stage / "predictions.csv")
        reference = p5[(p5.pool == "P2") & p5.model.isin(p6.model.unique())].copy()
        reference["pool"] = "P2"
        audit = validate_pair(reference, p6, stage)
        pairs.append(audit)
        stages[stage] = pd.concat([reference, p6], ignore_index=True)
    for field in ("input_data_sha256", "phase5_reference_sha256"):
        if run_configs["development"].get(field) != run_configs["recent"].get(field):
            raise ValueError(f"Development/recent {field} differs")
    dev, recent = stages["development"], stages["recent"]
    locks = alpha_locks(dev)
    # Phase5 P2 locked-alpha decisions must match frozen Phase5 development outcomes.
    old_locks = json.loads((PHASE5 / "locked_alphas.json").read_text(encoding="utf8"))
    for x in old_locks:
        if x["pool"] == "P2" and ("P2", x["model"], x["stock"]) in locks:
            if abs(float(x["alpha"]) - locks[("P2", x["model"], x["stock"])]) > 1e-10:
                raise ValueError("Frozen Phase5 P2 alpha differs; audit input data")
    (args.out / "locked_alphas.json").write_text(json.dumps([
        {"pool": p, "model": m, "stock": s, "alpha": a}
        for (p,m,s),a in sorted(locks.items())], indent=2, ensure_ascii=False), encoding="utf8")
    locked = lock_predictions(recent, locks)
    locked.to_csv(args.out / "selected_recent_predictions.csv", index=False)
    baseline = recent[(recent.pool == "P2") & (recent.alpha == 0)].drop_duplicates(["stock", "origin_date"])
    metrics = summarize(locked, baseline)
    metrics.to_csv(args.out / "comparison.csv", index=False)
    past = pd.read_csv(PHASE5 / "recent" / "predictions.csv")
    incumbent = incumbent_predictions(past)
    incumbent_mean = incumbent.groupby("stock").nmse.mean().to_dict()
    incumbent_median = incumbent.groupby("stock").nmse.median().to_dict()
    picks = {}
    for s in STOCKS:
        allowed = metrics[(metrics.stock == s) & metrics.pool.eq("P2_ID")].copy()
        # Require clear improvement over the prior *selected* Phase5 model as well.
        viable = allowed[(allowed.improvement_vs_baseline >= 0.05)
                         & (allowed.median_nmse < allowed.baseline_median_nmse)
                         & (allowed.wins >= 5)
                         & (allowed.mean_nmse < 0.98 * incumbent_mean[s])]
        if viable.empty:
            picks[s] = {"pool": "Phase5-incumbent", "reason": "No P2_ID candidate cleared preset gates"}
        else:
            r = viable.sort_values(["mean_nmse", "median_nmse"]).iloc[0]
            picks[s] = {"pool": "P2_ID", "model": str(r.model), "alpha": float(r.alpha),
                        "recent_mean_nmse": float(r.mean_nmse),
                        "improvement_vs_phase5_incumbent": 1 - r.mean_nmse / incumbent_mean[s],
                        "reason": "Met Phase6 conservative selection gates"}
    audit = {"comparison": pairs, "phase5_ref": "phase5-targetaware saved P2 forecasts",
             "stocks": STOCKS, "strict_date_truth_pairing": True,
             "phase5_reference_sha256": run_configs["development"]["phase5_reference_sha256"],
             "input_data_sha256": run_configs["development"]["input_data_sha256"],
             "historical_development_for_alpha": 3, "recent_windows": 8,
             "recent_is_independent_test": False}
    (args.out / "pairing_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf8")
    (args.out / "final_model_selection.json").write_text(json.dumps({
        "as_of": "2026-10-30", "recent_is_independent_test": False,
        "phase5_incumbent_mean_nmse": incumbent_mean,
        "phase5_incumbent_median_nmse": incumbent_median,
        "stocks": picks}, indent=2, ensure_ascii=False), encoding="utf8")
    text = ["# Phase6 Stock ID Embedding: controlled ablation", "",
            "P2: Phase5 frozen no-ID shared model; P2_ID: identical P2 pool + stock-specific learned 5-day bias.",
            "All alpha choices come from three historical development origins. Recent 8 windows are reused development data, NOT independent testing.",
            "", "## Paired recent-window results", "",
            "|Stock|Model|Pool|Alpha|Mean NMSE ↓|Median NMSE ↓|Baseline relative|Wins/Ties/Losses|",
            "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in metrics.sort_values(["stock", "model", "pool"]).itertuples():
        text.append(f"|{r.stock}|{r.model}|{r.pool}|{r.alpha:.2f}|{r.mean_nmse:.8f}|{r.median_nmse:.8f}|{r.improvement_vs_baseline:+.1%}|{r.wins}/{r.ties}/{r.losses}|")
    text += ["", "## Phase5 incumbent comparison / selection", ""]
    for s in STOCKS:
        text.append(f"- {s}: Phase5 incumbent mean={incumbent_mean[s]:.8f}; Phase6 selection={picks[s]}")
    text += ["", "## Limitations", "",
             "1. ID correction is static per stock; cannot model state-dependent cross-stock interactions.",
             "2. No-ID control is reused from Phase5 instead of retrained; compare seeds/inputs/configs for any differences.",
             "3. Improvement in this repeatedly viewed recent period is not proof of out-of-sample performance.",
             "4. Company actions in unadjusted data and survivorship bias remain as previously audited."]
    (args.out / "REPORT.md").write_text("\n".join(text), encoding="utf8")
    print(json.dumps({"audit": audit, "selection": picks}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
