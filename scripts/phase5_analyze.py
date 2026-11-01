"""Compare Phase 5 P0/P1/P2/P3 using development-locked alphas and paired dates.

Run only after BOTH development and recent have completed successfully:
  python scripts/phase5_analyze.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import phase4_joint as p4
import residual_phase3 as p3


def choose_alphas(dev: pd.DataFrame) -> dict[tuple[str, str, str], float]:
    locks = {}
    for key, group in dev.groupby(["pool", "model", "stock"]):
        # Deterministic tie-break: smaller alpha, favoring the conservative baseline.
        tab = group.groupby("alpha", as_index=False).nmse.mean().sort_values(["nmse", "alpha"])
        if group.origin_date.nunique() < 3:
            raise ValueError(f"Only {group.origin_date.nunique()} development windows for {key}")
        locks[key] = float(tab.iloc[0].alpha)
    return locks


def validate_pairing(raw: pd.DataFrame):
    expected = set(p4.TARGETS)
    if set(raw.stock) != expected:
        raise ValueError(f"Unexpected target stock set {set(raw.stock)}")
    if set(raw.pool) != {"P0", "P1", "P2", "P3"}:
        raise ValueError("Missing pool(s)")
    if raw.duplicated(["pool", "model", "stock", "origin_date", "alpha"]).any():
        raise ValueError("Duplicate per-pool forecast")
    for (stock, model, origin), block in raw.groupby(["stock", "model", "origin_date"]):
        if block.pool.nunique() != 4:
            raise ValueError(f"Unpaired pool {stock}/{model}/{origin}")
        if block.actual_open.nunique() != 1 or block.target_dates.nunique() != 1:
            raise ValueError(f"Mismatched truth or dates {stock}/{model}/{origin}")
        if block.anchor_close.max() - block.anchor_close.min() > 1e-5:
            raise ValueError(f"Mismatched anchors {stock}/{model}/{origin}")
        for pool in ("P0", "P1", "P2", "P3"):
            sl = block[block.pool.eq(pool)]
            if set(np.round(sl.alpha.astype(float).values, 6)) != set(p3.ALPHAS):
                raise ValueError(f"Missing alpha candidates {stock}/{pool}/{origin}")


def locked_subset(raw: pd.DataFrame, locks):
    selected = []
    for key, group in raw.groupby(["pool", "model", "stock"]):
        alpha = locks[key]
        subset = group[np.isclose(group.alpha, alpha, atol=1e-10)]
        if subset.origin_date.nunique() != group.origin_date.nunique():
            raise ValueError(f"Wrong subset for {key} alpha={alpha}")
        selected.append(subset.copy())
    return pd.concat(selected, ignore_index=True)


def compare_rows(df: pd.DataFrame, baseline_source: pd.DataFrame | None = None):
    source = df if baseline_source is None else baseline_source
    baseline = source[source.alpha.eq(0)].drop_duplicates(["stock", "origin_date"])
    lookup = {(r.stock, r.origin_date): float(r.nmse) for r in baseline.itertuples()}
    rows = []
    for key, group in df.groupby(["pool", "model", "stock"]):
        pool, model, stock = key
        diffs = np.array([float(r.nmse) - lookup[(stock, r.origin_date)] for r in group.itertuples()])
        eps = np.array([1e-12 + 1e-8 * lookup[(stock, r.origin_date)] for r in group.itertuples()])
        wins = int(np.sum(diffs < -eps))
        ties = int(np.sum(np.abs(diffs) <= eps))
        losses = int(np.sum(diffs > eps))
        b = baseline[baseline.stock.eq(stock)]
        price = float(group.nmse.mean())
        denom = float(b.nmse.mean())
        rows.append({"pool": pool, "model": model, "stock": stock,
                     "alpha": float(group.alpha.iloc[0]), "windows": len(group),
                     "mean_nmse": price, "median_nmse": float(group.nmse.median()),
                     "std_nmse": float(group.nmse.std(ddof=0)),
                     "baseline_mean_nmse": denom,
                     "baseline_median_nmse": float(b.nmse.median()),
                     "improvement_vs_baseline": float(1 - price / denom),
                     "wins": wins, "ties": ties, "losses": losses})
    result = pd.DataFrame(rows)
    solo = result[result.pool.eq("P0")].set_index(["model", "stock"])
    result["improvement_vs_P0"] = [
        1 - float(r.mean_nmse) / float(solo.loc[(r.model, r.stock), "mean_nmse"])
        for r in result.itertuples()
    ]
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "results/phase5_targetaware")
    ap.add_argument("--min-improvement", type=float, default=0.05)
    ap.add_argument("--min-wins", type=int, default=5)
    opts = ap.parse_args()
    dev_path = opts.out / "development" / "predictions.csv"
    rec_path = opts.out / "recent" / "predictions.csv"
    dev = pd.read_csv(dev_path)
    recent = pd.read_csv(rec_path)
    validate_pairing(dev)
    validate_pairing(recent)
    dev_dates = sorted(dev.origin_date.unique().tolist())
    rec_dates = sorted(recent.origin_date.unique().tolist())
    if len(dev_dates) != 3 or len(rec_dates) != 8:
        raise ValueError(f"Expected 3 development and 8 recent dates, got {len(dev_dates)}, {len(rec_dates)}")
    if set(dev_dates).intersection(rec_dates):
        raise ValueError("Development dates overlap recent dates")
    locks = choose_alphas(dev)
    opts.out.mkdir(parents=True, exist_ok=True)
    (opts.out / "locked_alphas.json").write_text(json.dumps(
        [{"pool": pool, "model": model, "stock": stock, "alpha": alpha,
          "source": "three pre-recent development origins"}
         for (pool, model, stock), alpha in sorted(locks.items())], ensure_ascii=False, indent=2), encoding="utf-8")
    selected = locked_subset(recent, locks)
    selected.to_csv(opts.out / "selected_recent_predictions.csv", index=False)
    # Alpha locks are per pool/model/stock; Last Close must come from raw alpha=0
    # rows, independent of each candidate's selected alpha.
    baseline = recent[recent.alpha.eq(0)].drop_duplicates(["stock", "origin_date"])
    metrics = compare_rows(selected, baseline)
    metrics.to_csv(opts.out / "comparison.csv", index=False)
    picks = {}
    for stock in p4.TARGETS:
        b = baseline[baseline.stock.eq(stock)]
        base_mean = float(b.nmse.mean())
        base_median = float(b.nmse.median())
        models = metrics[metrics.stock.eq(stock)].copy()
        viable = models[(models.improvement_vs_baseline >= opts.min_improvement)
                        & (models.median_nmse < base_median)
                        & (models.wins >= opts.min_wins)]
        if viable.empty:
            picks[stock] = {"pool": "LastCloseBaseline", "model": "LastCloseBaseline", "alpha": 0.0,
                            "recent_mean_nmse": base_mean,
                            "reason": "No candidate met mean/median/paired-win selection rule"}
        else:
            r = viable.sort_values(["mean_nmse", "median_nmse", "pool"]).iloc[0]
            picks[stock] = {"pool": r.pool, "model": r.model, "alpha": float(r.alpha),
                            "recent_mean_nmse": float(r.mean_nmse), "baseline_mean_nmse": base_mean,
                            "wins": int(r.wins), "reason": "Met predefined recent comparison rule"}
    sel_info = {"as_of": "2026-10-30", "not_independent_test": True,
                "alpha_selection": "development 3 dates only",
                "batch_size_comparable": True,
                "checkpoints_selected_by": "target-stock macro validation NMSE",
                "recent_windows": 8, "stocks": picks}
    (opts.out / "final_model_selection.json").write_text(json.dumps(sel_info, ensure_ascii=False, indent=2), encoding="utf-8")
    report = ["# Phase 5 target-aware training & fine-tuning report", "",
              "P0 单股从头训练；P1 三股联合；P2 煤炭池联合；P3 同日期 P2 权重初始化后单股微调。",
              "所有方案使用相同 Batch Size、统一训练/验证日历边界、相同输入及目标定义。",
              "共享模型 checkpoint 由三只目标股验证 NMSE 宏平均选择；P3 由自身目标股验证 NMSE 选择。",
              "Alpha 仅从历史开发 3 起点选择，近期 8 窗口仅作快速模型选择，不是独立测试集。",
              "", "## 近期同日期指标", "",
              "|股票|训练方式|模型|Alpha|Mean NMSE ↓|Median NMSE ↓|对比 Last Close|Win/Tie/Loss|",
              "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in metrics.sort_values(["stock", "pool", "model"]).itertuples():
        report.append(f"|{r.stock}|{r.pool}|{r.model}|{r.alpha:.2f}|{r.mean_nmse:.8f}|"
                      f"{r.median_nmse:.8f}|{r.improvement_vs_baseline:+.1%}|{r.wins}/{r.ties}/{r.losses}|")
    report += ["", "## 最终候选（仅是近期开发集选择）", ""]
    for stock, choice in picks.items():
        report.append(f"- {stock}: {choice['pool']} / {choice['model']}; alpha={choice['alpha']}; "
                      f"近期 mean NMSE={choice['recent_mean_nmse']:.8f}")
    report += ["", "## 解释限制", "",
               "1. 仅三个历史开发窗口锁定 Alpha，结果仍可能不稳定。",
               "2. 最近 40 天在先前阶段已被观察，不构成全新独立测试集。",
               "3. P3 使用同日期 P2 checkpoint，微调与预训练使用相同时间切分，但不使用未来标签。",
               "4. 如果模型预测不改善，应保留 Last Close；不能因追求神经网络而修改选模规则。", ""]
    (opts.out / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Report: {opts.out / 'REPORT.md'}")
    print(json.dumps(picks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
