# Phase 3 Residual Forecasting：预测结果与实验说明

本分支基于 `phase2-results-data-docs` 创建，用于单独归档 Phase 3 的预测文件和说明。Phase 2 的正式预测文件和实验记录未覆盖。Phase 3 文件位于 `results/predictions/phase3/`。

## 实验设计

本轮检验将神经网络目标从绝对 Open 价格改为相对于预测起点前一交易日 Close 的变化率：`future_open / anchor - 1`，其中 `anchor = close[o - 1] > 0`。历史价格特征也以该 anchor 表示；成交量和成交额先做 `log1p`，再仅用当次训练范围拟合的统计量标准化。验证和预测沿用训练统计量。

比较三个版本：V0 为 Phase 2 已保存的绝对价格结果；V1 为新训练的残差模型；V2 为 V1 预测乘以在历史开发窗口锁定的 alpha。`alpha=0` 等于 Last Close Baseline。股票、架构、60 日输入、5 日预测和训练/验证边界沿用原设定。

第一层选用 `origin_02`、`origin_04`、`origin_06`，完成 27 次训练并按股票和模型锁定 alpha。第二层在此前保留区最后 40 个交易日上运行 8 个非重叠五日窗口，每 20 日重训练，共 18 次训练。近期回测用于快速选模，样本较少，不视为独立最终测试集。V0、V1、V2 在相同预测日期配对；日期、目标日期和真实 Open 的对齐审计均通过（开发窗口 27 组，近期窗口 72 组）。

正式 45 次拟合统一使用 batch size 128、固定随机种子、Adam、相同 epoch 上限和 early stopping。与早期 batch size 32 的同一开发段对照，前 9 次拟合总耗时由 72.20 秒降至 29.95 秒，约快 2.41 倍。GPU 利用率采样约 12%，表明单股票小模型仍受工作负载规模限制；本轮优化提升了吞吐，但没有改变预测任务或模型架构。

## 选模结论

保守规则要求近期平均 NMSE 至少改善 5%、中位数优于 Baseline 且 8 个窗口中至少 5 个胜出。没有神经网络候选同时满足这些条件，因此三只股票均选 Last Close Baseline。

- 中国神华 `sh.601088`：Baseline 平均 NMSE 0.000381。最佳神经网络 V1 DLinear 为 0.000386，较 Baseline 差约 1.47%。
- 陕西煤业 `sh.601225`：最佳神经网络 V1 DLinear 平均 NMSE 0.000388，相对 Baseline 改善约 7.98%，但仅在 8 个窗口中胜出 4 次且中位数更差，证据不足。
- 山西焦煤 `sz.000983`：最佳神经网络 V1 TimeXer 平均 NMSE 0.000521，相对 Baseline 改善约 23.69%，胜出 5/8 窗口，但中位数更差，未达到规则。

因此，当前数据不支持声称某个神经网络稳定优于 Last Close。V2 的收缩有时能改善对应 V1，但最终仍未改变选模结果。近期变化率与真实变化率的相关性及完整候选统计保存在候选汇总中，所有版本均保留，未筛除表现较差的窗口。

## 正式预测

训练数据截止日和最后输入日均为 2026-10-30。三只股票均选 Baseline，即未来五日 Open 等于最后已知 Close：

| 股票 | 2026-11-02 至 2026-11-06 的每日预测 Open |
| --- | ---: |
| 中国神华 `sh.601088` | 51.990002 |
| 陕西煤业 `sh.601225` | 28.080000 |
| 山西焦煤 `sz.000983` | 6.490000 |

三份 CSV 均为 `Date,Open` 表头、各 5 行、日期为 2026-11-02 至 2026-11-06，且均通过正式文件名和格式校验。预测数值与 Phase 2 相同；Phase 2 文件保持原样。

## 本目录文件

- `results/predictions/phase3/10235102458_prediction_601088_SH.csv`
- `results/predictions/phase3/10235102458_prediction_601225_SH.csv`
- `results/predictions/phase3/10235102458_prediction_000983_SZ.csv`
- `results/predictions/phase3/submission_validation.json`：逐文件校验结果与 SHA-256。
- `results/predictions/phase3/metadata.json`：预测方法、截止日期和源文件摘要。
- `results/predictions/phase3/prediction_details.csv`：逐股逐日预测值。
- `results/predictions/phase3/candidate_summary.csv`：V0/V1/V2 完整候选指标。
- `results/predictions/phase3/locked_alphas.json`：开发窗口锁定的 alpha。
- `results/predictions/phase3/alignment_audit.json`：配对数据对齐审计。
