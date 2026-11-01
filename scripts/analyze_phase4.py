"""Lock development alphas and summarize strictly paired Phase 4 forecasts."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import residual_phase3 as p3
import phase4_joint as p4
OUT=p4.OUT


def main():
    dev=pd.read_csv(OUT/"development/joint_predictions.csv")
    recent=pd.read_csv(OUT/"recent_40d/joint_predictions.csv")
    if dev.empty or recent.empty: raise ValueError("development or recent predictions are empty")
    # P0 alpha is the locked Phase 3 alpha; never retune P0 on Phase 4 dates.
    p3lock=json.loads((ROOT/"results/residual_phase3/locked_alphas.json").read_text(encoding="utf-8"))["stocks"]
    locks={}
    for (pool,stock,model),group in dev[dev.pool.isin(["P1","P2"])].groupby(["pool","stock","model"]):
        score=group.groupby("alpha",as_index=False).nmse.mean().sort_values(["nmse","alpha"])
        locks[(pool,stock,model)]=float(score.iloc[0].alpha)
    for stock in p4.TARGETS:
      for model in p4.MODELS:
        locks[("P0",stock,model)]=float(p3lock[stock][model]["selected_alpha"])
    lock_rows=[{"pool":k[0],"stock":k[1],"model":k[2],"alpha":v,
                "selection_source":"Phase 3 development lock" if k[0]=="P0" else "Phase 4 historical development windows"}
               for k,v in locks.items()]
    (OUT/"locked_alphas.json").write_text(json.dumps(lock_rows,ensure_ascii=False,indent=2),encoding="utf-8")
    selected=[]
    for (pool,stock,model,origin),group in recent.groupby(["pool","stock","model","origin_date"]):
        alpha=locks[(pool,stock,model)]
        row=group[np.isclose(group.alpha,alpha)]
        if len(row)!=1: raise ValueError(f"missing locked recent alpha {pool}/{stock}/{model}/{origin}")
        selected.append(row.iloc[0].to_dict())
    detail=pd.DataFrame(selected)
    # Date pairing: actual dates and values must match across P0/P1/P2 for every stock/model/window.
    for (stock,model,origin),group in detail.groupby(["stock","model","origin_date"]):
        if set(group.pool)!={"P0","P1","P2"}: raise ValueError(f"incomplete pool pairing {stock} {model} {origin}")
        if group.target_dates.nunique()!=1 or group.actual_open.nunique()!=1:
            raise ValueError(f"date/actual Open mismatch {stock} {model} {origin}")
    baselines=[]
    for _,r in detail.drop_duplicates(["stock","origin_date"]).iterrows():
        actual=p3.parse_vec(r.actual_return); anchor=float(r.anchor_close); zero=np.zeros(5)
        baselines.append({"stock":r.stock,"origin_date":r.origin_date,"target_dates":r.target_dates,"model":"LastCloseBaseline",
          "pool":"Baseline","alpha":0.0,"anchor_close":anchor,"actual_open":r.actual_open,
          "actual_return":r.actual_return,"predicted_return":p3.strvec(zero),"predicted_open":p3.strvec(np.full(5,anchor)),
          **p3.residual_metrics(zero,actual,anchor)})
    baseline=pd.DataFrame(baselines)
    detail=pd.concat([detail,pd.DataFrame(baselines)],ignore_index=True,sort=False)
    detail.to_csv(OUT/"per_window_predictions.csv",index=False)
    base_lookup={(r.stock,r.origin_date):r.nmse for _,r in baseline.iterrows()}
    rows=[]
    for (pool,stock,model),g in pd.DataFrame(selected).groupby(["pool","stock","model"]):
        wins=sum(float(r.nmse)<base_lookup[(stock,r.origin_date)] for _,r in g.iterrows())
        steps=np.zeros(5); actuals=[]; preds=[]
        for _,r in g.iterrows():
            pr=p3.parse_vec(r.predicted_return); ar=p3.parse_vec(r.actual_return)
            steps+=(pr-ar)**2; actuals.extend(ar.tolist()); preds.extend(pr.tolist())
        k=(pool,stock,model)
        rows.append({"pool":pool,"stock":stock,"model":model,"alpha":locks[k],"windows":len(g),
            "mean_nmse":g.nmse.mean(),"median_nmse":g.nmse.median(),"std_nmse":g.nmse.std(ddof=0),
            "mean_mae":g.mae.mean(),"mean_rmse":g.rmse.mean(),"beats_baseline_windows":wins,
            "relative_improvement_vs_baseline":1-g.nmse.mean()/baseline[baseline.stock.eq(stock)].nmse.mean(),
            "step1_nmse":steps[0]/len(g),"step2_nmse":steps[1]/len(g),"step3_nmse":steps[2]/len(g),
            "step4_nmse":steps[3]/len(g),"step5_nmse":steps[4]/len(g),
            "return_correlation":float(np.corrcoef(preds,actuals)[0,1]) if np.std(preds)>0 and np.std(actuals)>0 else np.nan})
    summaries=pd.DataFrame(rows)
    p0=summaries[summaries.pool.eq("P0")].set_index(["stock","model"])
    summaries["relative_improvement_vs_P0"]=[1-r.mean_nmse/p0.loc[(r.stock,r.model),"mean_nmse"] if r.pool!="P0" else 0.0 for _,r in summaries.iterrows()]
    summaries.to_csv(OUT/"per_stock_summary.csv",index=False)
    # Three stocks carry equal weight regardless of available training windows.
    comparisons=[]
    for (pool,model),g in summaries.groupby(["pool","model"]):
        comparisons.append({"pool":pool,"model":model,"alpha_equal_weight_mean":g.alpha.mean(),
          "equal_weight_mean_nmse":g.mean_nmse.mean(),"equal_weight_median_nmse":g.median_nmse.mean(),
          "mean_baseline_wins_per_stock":g.beats_baseline_windows.mean(),"n_stock":g.stock.nunique()})
    pd.DataFrame(comparisons).to_csv(OUT/"model_comparison.csv",index=False)
    # Conservative candidate per stock: must improve both Baseline and P0 with stable windows.
    picks={}
    for stock in p4.TARGETS:
        sg=summaries[summaries.stock.eq(stock)]
        candidates=sg[sg.pool.isin(["P1","P2"])].copy()
        candidates["improvement_vs_p0"]=candidates.relative_improvement_vs_P0
        viable=candidates[(candidates.relative_improvement_vs_baseline>=0.05)&
                          (candidates.relative_improvement_vs_P0>=0.05)&
                          (candidates.beats_baseline_windows>=5)]
        if viable.empty:
            chosen={"pool":"P3-LastCloseBaseline","model":"LastCloseBaseline","alpha":0.0,
                    "reason":"No joint candidate met >=5% improvement vs Baseline and P0 plus >=5/8 Baseline wins."}
        else:
            r=viable.sort_values(["mean_nmse","median_nmse"]).iloc[0]
            chosen={"pool":r.pool,"model":r.model,"alpha":float(r.alpha),"reason":"Met conservative joint selection rule."}
        picks[stock]=chosen
    (OUT/"final_model_selection.json").write_text(json.dumps({"as_of":AS_OF,"recent_windows":8,
        "not_independent_test":True,"stocks":picks},ensure_ascii=False,indent=2),encoding="utf-8")
    report=["# Phase 4 多股票联合 Residual 训练实验报告","",
      "P0 是 Phase 3 单股票 Residual；P1 是三只目标煤炭股共享模型；P2 使用 BaoStock 煤炭行业池共享模型。全部结果以相同目标日期配对。近期 40 个交易日已用于开发和选模，不能称为独立测试集。","",
      "## 近期同日期结果","","|股票|Baseline NMSE|P0 Residual|P1 Joint|P2 Coal Pool|","|---|---:|---:|---:|---:|"]
    for stock in p4.TARGETS:
        sg=summaries[summaries.stock.eq(stock)]
        vals={pool:float(sg[sg.pool.eq(pool)].mean_nmse.mean()) for pool in ("P0","P1","P2")}
        base=float(baseline[baseline.stock.eq(stock)].nmse.mean())
        report.append(f"|{stock}|{base:.8f}|{vals['P0']:.8f}|{vals['P1']:.8f}|{vals['P2']:.8f}|")
    report += ["","逐模型、逐股票 alpha、窗口数、步长 NMSE、MAE/RMSE、Baseline 胜场、变化率相关性见 `per_stock_summary.csv`。Alpha 在开发集锁定并在近期固定；P0 alpha 直接继承 Phase 3 锁定值。","",
      "## 最终选择","", "未通过保守规则的股票保留 Phase 3 正式预测，不覆盖已验证提交 CSV。", ""]
    for stock,choice in picks.items(): report.append(f"- {stock}: `{choice['pool']}` / `{choice['model']}` alpha={choice['alpha']} — {choice['reason']}")
    report += ["","## 数据、隔离和运行产物","","股票池查询、原始/处理记录、停牌数、异常跳变、训练窗口量、训练参数量、耗时和 checkpoint 哈希分别见 `pool_audit/`、`development/` 与 `recent_40d/`。每个训练窗口的标签均严格截至各自统一日期边界；逐股票成交量/额 scaler 只在训练范围拟合。","","Phase 3 正式预测与 Phase 2 历史结果保持原样。",""]
    (OUT/"EXPERIMENT_REPORT.md").write_text("\n".join(report),encoding="utf-8")
    print(f"wrote {OUT}: {len(summaries)} stock/model/pool summaries; selection={picks}")

if __name__=="__main__": main()
