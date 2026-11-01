# Phase 4 多股票联合 Residual 训练实验报告

P0 是 Phase 3 单股票 Residual；P1 是三只目标煤炭股共享模型；P2 使用 BaoStock 煤炭行业池共享模型。全部结果以相同目标日期配对。近期 40 个交易日已用于开发和选模，不能称为独立测试集。

三个统一历史开发起点是 2019-07-01、2022-07-01、2025-07-01。Phase 3 的历史开发起点按股票单独选择，不能完全配对到这三天，因此 P0 只在这些新起点补训以获得同日期控制；近期八个 P0 则直接复用 Phase 3 已保存的预测。Phase 2/3 原始指标、预测文件和 checkpoint 均未覆盖。P0/P1/P2 的 27 个历史组合与 72 个近期组合均通过目标日期及真实 Open 完全匹配审计，详情见 `pool_audit/date_alignment_audit.json`。

## 近期同日期结果

每行明确给出模型，P0/P1/P2 单元格同时报告锁定 alpha，避免跨模型挑选后隐藏候选来源。

|股票|模型|Last Close NMSE|P0 NMSE (alpha)|P1 NMSE (alpha)|P2 NMSE (alpha)|
|---|---|---:|---:|---:|---:|
|sh.601088|DLinear|0.00038081|0.00038642 (1.00)|0.00037405 (0.50)|0.00037673 (0.25)|
|sh.601088|iTransformer|0.00038081|0.00048920 (0.50)|0.00050333 (0.50)|0.00038081 (0.00)|
|sh.601088|TimeXer|0.00038081|0.00040682 (0.50)|0.00038081 (0.00)|0.00038081 (0.00)|
|sh.601225|DLinear|0.00042185|0.00042185 (0.00)|0.00042185 (0.00)|0.00042185 (0.00)|
|sh.601225|iTransformer|0.00042185|0.00042185 (0.00)|0.00047604 (1.00)|0.00042185 (0.00)|
|sh.601225|TimeXer|0.00042185|0.00042185 (0.00)|0.00042185 (0.00)|0.00042185 (0.00)|
|sz.000983|DLinear|0.00068283|0.00068283 (0.00)|0.00068283 (0.00)|0.00068283 (0.00)|
|sz.000983|iTransformer|0.00068283|0.00068283 (0.00)|0.00068283 (0.00)|0.00068283 (0.00)|
|sz.000983|TimeXer|0.00068283|0.00068283 (0.00)|0.00062366 (0.25)|0.00058646 (0.50)|

逐模型、逐股票 alpha、窗口数、步长 NMSE、MAE/RMSE、Baseline 胜场、变化率相关性见 `per_stock_summary.csv`。Alpha 在开发集锁定并在近期固定；P0 alpha 直接继承 Phase 3 锁定值。

## 最终选择

未通过保守规则的股票保留 Phase 3 正式预测，不覆盖已验证提交 CSV。


## Training size and runtime

Full fit metadata and sample counts are in `pool_audit/training_fits.csv` and `training_window_counts.csv`.

|Pool|Model|Fits|Parameters|Mean seconds/fit|Total seconds|Peak GPU MiB|
|---|---|---:|---:|---:|---:|---:|
|P0|DLinear|9|610|1.39|12.52|20.4|
|P0|TimeXer|9|411141|2.78|25.03|90.8|
|P0|iTransformer|9|273669|1.65|14.84|73.7|
|P1|DLinear|5|610|2.83|14.13|21.5|
|P1|TimeXer|5|411141|5.36|26.78|91.9|
|P1|iTransformer|5|273669|4.71|23.56|74.6|
|P2|DLinear|5|610|12.51|62.57|25.5|
|P2|TimeXer|5|411141|45.00|224.99|96.0|
|P2|iTransformer|5|273669|25.16|125.80|78.3|

Window totals by stage and pool: `{"development/P0": 49857, "development/P1": 49857, "development/P2": 435240, "recent_40d/P1": 51534, "recent_40d/P2": 447810}`.

- sh.601088: `Phase3-Prediction` / `Phase3-selected model or Last Close as recorded in Phase 3` alpha=None — No joint candidate met >=5% mean improvement vs Baseline and P0, better median NMSE than both, and >=5/8 Baseline wins.
- sh.601225: `Phase3-Prediction` / `Phase3-selected model or Last Close as recorded in Phase 3` alpha=None — No joint candidate met >=5% mean improvement vs Baseline and P0, better median NMSE than both, and >=5/8 Baseline wins.
- sz.000983: `P2` / `TimeXer` alpha=0.5 — Met conservative joint selection rule.

三只股票最终方法为：神华 Last Close（NMSE 0.00038081）、陕西煤业 Last Close（0.00042185）、山西焦煤 P2 TimeXer alpha=0.5（0.00058646）。山西焦煤相对 Baseline 的近期 Mean NMSE 改善 14.1%，Median NMSE 从 0.00056086 降至 0.00050587，6/8 窗口胜出；变化率相关系数为 0.245。神华最佳联合候选只改善 1.8% 且仅 4/8 窗口胜出；陕西煤业没有改善。三股最终选择的等权平均 NMSE 为 0.00046304，Last Close 等权平均为 0.00049516。近期窗口已参与开发和选择，以上优势不代表独立测试结论。

## 最终预测与提交

入选的 P2 TimeXer 以 2026-10-30 为训练截止日，使用该时点的 28 只煤炭股、81,861 个股票内训练窗口训练 12 epoch；epoch 数为三个历史开发 P2 TimeXer 拟合最佳 epoch 的中位数，近期标签未用于决定训练轮数。批次为 512，GPU 训练耗时约 55 秒。最终 checkpoint 已随本分支上传到 `results/phase4_joint_training/checkpoints/final_P2_TimeXer_2026-10-30.pt`，SHA256 为 `83d646a1c54b6d2651eea6142ce8fbcf3375e574c5c16131048974799337c6b21`。

|股票|最终方法|2026-11-02|2026-11-03|2026-11-04|2026-11-05|2026-11-06|
|---|---|---:|---:|---:|---:|---:|
|中国神华|Last Close|51.990002|51.990002|51.990002|51.990002|51.990002|
|陕西煤业|Last Close|28.080000|28.080000|28.080000|28.080000|28.080000|
|山西焦煤|P2 TimeXer alpha=0.5|6.506514|6.524517|6.529971|6.547538|6.547826|

Phase 4 导出的三份 CSV 均含 `Date,Open` 两列、各 5 行，格式和日期验证均通过。文件位于 `final/submission/10235102458_prediction_601088_SH.csv`、`final/submission/10235102458_prediction_601225_SH.csv`、`final/submission/10235102458_prediction_000983_SZ.csv`。Phase 3 原提交文件仍保留在原目录。

## 数据、隔离和运行产物

股票池查询、原始/处理记录、停牌数、异常跳变、训练窗口量、训练参数量、耗时和 checkpoint 哈希分别见 `pool_audit/`、`development/` 与 `recent_40d/`。每个训练窗口的标签均严格截至各自统一日期边界；逐股票成交量/额 scaler 只在训练范围拟合。

Phase 3 正式预测与 Phase 2 历史结果保持原样。
