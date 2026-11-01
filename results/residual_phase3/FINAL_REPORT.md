# Phase 3 Residual Forecasting 实验报告

## 实验状态与配对审计

- 27 次开发训练（3 股票 × 3 模型 × `origin_02/04/06`）和 18 次近期重训（3 × 3 × 2）均完成，退出码均为 0。近期生成 72 个 V1 窗口（每股票/模型 8 个）。另存每次 fit 的最优 checkpoint，共 45 个；checkpoint 留在本地和远端，不提交 GitHub。
- 本地取回的 45 个 checkpoint 均与训练 CSV 中记录的 SHA-256 一致。
- `alignment_audit.json` 确认 V0/V1/V2 的预测目标日期与真实 Open 完全配对：开发层 27 组，近期层 72 组。Phase 2 原结果文件未修改。
- 近期 40 个交易日来自 Phase 2 的 180 日保留区尾段，是快速选模集，不是独立最终测试集。
- 运行代码提交：开发阶段 `694086b`，近期与分析阶段 `0f86f04`。两个阶段均记录 batch 128、最多 50 epochs、patience 7、Adam、学习率 0.001、seed 2026。

## 调速记录

按用户要求停止了 batch 32 的首轮配置。该尝试完成 14/27 个 fit 后中断，记录在 `attempts/batch32_interrupted/`，不参与任何比较。随后全部 45 次正式 fit 统一使用 batch 128、训练/验证 pinned memory 与异步传输，并在 CUDA 上设置 `float32_matmul_precision=high`；模型、训练边界、epoch 上限、early stopping、优化器、学习率和种子保持不变。

在中国神华前 9 个同配置开发 fit 的计时日志中，batch 32 总运行时长为 72.20 秒，batch 128 为 29.95 秒，约快 2.41 倍。GPU 抽样利用率约 12%，显存使用约 0.4 GB；这些模型与 60×6 输入规模较小，GPU 大部分容量仍空闲。提高模型规模或输入长度会改变本轮固定条件，因此没有继续扩大计算规模。

## Alpha 锁定

Alpha 只在三个开发 origin 上按 mean NMSE 选择，近期阶段固定使用：

| 股票 | DLinear | iTransformer | TimeXer |
| --- | ---: | ---: | ---: |
| 中国神华 `sh.601088` | 1.00 | 0.50 | 0.50 |
| 陕西煤业 `sh.601225` | 0.00 | 0.00 | 0.00 |
| 山西焦煤 `sz.000983` | 0.00 | 0.00 | 0.00 |

`alpha=0` 会令预测 residual 全零，因此严格等于 Last Close Baseline。

## 近期 8 窗口结果

Mean/median NMSE、标准差、每模型胜出窗口数和相对改善完整表见 `candidate_summary.csv`。以下列出每股 Baseline 和近期 mean NMSE 最低的神经网络候选；相对改善按 `(Baseline mean - candidate mean) / Baseline mean` 计算。

| 股票 | 候选 | Mean NMSE | Median NMSE | 胜 Baseline 窗口 | 相对 Baseline 改善 |
| --- | --- | ---: | ---: | ---: | ---: |
| 中国神华 | Last Close | 0.000381 | 0.000368 | — | — |
| 中国神华 | V1 DLinear | 0.000386 | 0.000347 | 3/8 | -1.47% |
| 陕西煤业 | Last Close | 0.000422 | 0.000354 | — | — |
| 陕西煤业 | V1 DLinear | 0.000388 | 0.000383 | 4/8 | +7.98% |
| 山西焦煤 | Last Close | 0.000683 | 0.000561 | — | — |
| 山西焦煤 | V1 TimeXer | 0.000521 | 0.000577 | 5/8 | +23.69% |
| 山西焦煤 | V1 DLinear | 0.000657 | 0.000546 | 6/8 | +3.80% |

V1 相对 V0 的 mean NMSE 改善因股票和模型而异：DLinear 在神华、陕西、山西分别改善 44.39%、33.79%、7.51%；TimeXer 分别改善 0.28%、-9.88%、19.66%；iTransformer 分别为 -33.30%、-7.25%、1.46%。残差预测对部分模型/股票有帮助，但没有一致超过基线。

V2 收缩改善了神华的 iTransformer（mean NMSE 从 V1 的 0.000774 降至 0.000489）和 TimeXer（0.000546 降至 0.000407），仍高于该股 Last Close 的 0.000381。陕西煤业与山西焦煤开发集均锁定 alpha 0，V2 因而等于 Baseline。

## 是否采用神经网络

不采用。规则要求同时满足近期 mean NMSE 至少改善 5%、median 低于 Baseline、并在至少 5/8 个窗口胜过 Baseline：

- 神华没有神经网络候选优于 Baseline 的 mean NMSE。
- 陕西煤业 DLinear mean 改善 7.98%，但只赢 4/8 窗口，median 也高于 Baseline。
- 山西焦煤 TimeXer mean 改善 23.69% 且赢 5/8，但 median 0.000577 高于 Baseline 的 0.000561；DLinear 赢 6/8，但 mean 改善只有 3.80%。

因此最终三只股票全部选择 LastCloseBaseline。收益率相关系数见 `recent_return_correlations.csv`，逐预测步 NMSE 见 `recent_step_nmse.csv`。最佳 mean 候选的 V1 相关系数分别为神华 DLinear 0.077、陕西 DLinear 0.290、山西 TimeXer 0.395；40 个点（8 窗口 × 5 步）的相关性不足以推翻 NMSE 与一致性判断。alpha=0 的 V2 预测恒为零，相关系数无定义。

## 正式预测与校验

最终预测采用截至 2026-10-30 的最后收盘价，延续至 2026-11-02 至 2026-11-06：

| 日期 | 中国神华 Open | 陕西煤业 Open | 山西焦煤 Open |
| --- | ---: | ---: | ---: |
| 2026-11-02 | 51.990002 | 28.080000 | 6.490000 |
| 2026-11-03 | 51.990002 | 28.080000 | 6.490000 |
| 2026-11-04 | 51.990002 | 28.080000 | 6.490000 |
| 2026-11-05 | 51.990002 | 28.080000 | 6.490000 |
| 2026-11-06 | 51.990002 | 28.080000 | 6.490000 |

Phase 3 独立正式 CSV 位于 `final/submission/`，学号 `10235102458`。三份文件都通过格式校验：每份 `Date,Open` 表头、5 行日期顺序正确，且所有 Open 为有限正数；详见 `final/submission_validation.json`。这些结果与 Phase 2 最终 CSV 数值相同，Phase 2 文件保持原样。

## 产物导航

- `development_v1_windows.csv`、`development_alpha_scores.csv`、`development_alpha_selection.csv`、`locked_alphas.json`：开发残差训练和 alpha 搜索。
- `recent_v1_windows.csv`、`recent_training_runs.csv`：近期 72 个窗口与 18 次重训记录。
- `development_comparison.csv`、`recent_comparison.csv`：逐窗口 V0/V1/V2/Baseline 对照。
- `candidate_summary.csv`、`recent_step_nmse.csv`、`recent_return_correlations.csv`、`alignment_audit.json`：汇总指标、预测步长分析、相关性与配对审计。
- `logs/`、`experiment_config_*.json`、`gpu_environment.txt`：训练日志、阶段配置和远端环境。
- `checkpoints/`：45 个正式 fit 的最优权重；保存在本地和远端，不上传 GitHub。
- `final/`：最终方法、训练配置、预测明细、正式 CSV 和验证报告。
