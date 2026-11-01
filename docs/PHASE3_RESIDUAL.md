# Phase 3：Residual Forecasting 快速选模

本实验沿用 Phase 2 的三只股票、未复权预处理数据、日期截止、`seq_len=60`、`pred_len=5` 和官方模型实现。全部代码、配置、产物和模型 checkpoint 与 Phase 2 隔离在 `results/residual_phase3/`；不覆盖 Phase 2 的任何指标、预测文件或训练记录。

## 标签与输入

每个样本 origin `o` 的 anchor 固定为 `close[o-1]`，且必须大于 0。与 Phase 2 一样，CSV 数值以 float32 载入以严格复现其已保存真实价格精度。历史输入为 `[o-60,o)`；高、低、收、开价按 `price / anchor - 1` 转为相对价，特征顺序仍为 `high,low,close,volume,amount,open`。成交量和成交额分别先做 `log1p`，再用本次 fit 的训练区间行 `raw[:train_end]` 拟合的均值和标准差转换。验证和预测复用同一组统计量。标签为 `open[o:o+5] / anchor - 1`，损失是预测收益率与标签的 MSE。

恢复预测价格使用 `anchor * (1 + pred_return)`。因此 `mean((pred_return-target_return)^2)` 与恢复价格后以 anchor 为分母计算的课程 NMSE 数值严格相同。alpha=0 会将五个预测都恢复为 anchor，即 Last Close Baseline。不会把相对收益率再通过原 Open 的 scaler 逆变换。

DLinear 仍是官方结构，按变量独立分解/投影，不融合变量间信息；本实验不改结构。iTransformer 与 TimeXer 的官方 forecast 含 Non-stationary Transformer 序列归一化和反归一化，模型输出直接视为相对 Open 变化率。TimeXer 继续使用 `features="MS"`，open 位于最后一列。

## V0、V1、V2 与切分

- **V0**：直接读取 Phase 2 的已保存绝对价格预测，不重训。第一层取 `origin_02/04/06`；第二层取 Phase 2 180 日保留区末尾 8 个五日窗口。
- **V1**：三个官方模型改为预测上述相对收益率。
- **V2**：`alpha * V1_pred_return`，候选 alpha 固定为 `[0, 0.25, 0.5, 1]`，不额外训练网络。每个股票/模型的 alpha 只在三个开发起点上按平均 NMSE 选择并锁定；近期数据不参与搜参。

第一层为 3 股票 × 3 模型 × 3 起点，共 27 次训练。严格沿用 `data/splits/*.json` 中每个 origin 的 train、validation 和 test 边界；训练样本标签不会跨过 train_end，验证样本标签不会跨过 test origin。第二层取近期保留区最后 40 个交易日，origin 为这 40 天内每 5 天一个、共 8 个；每 20 个交易日重训一次，使用截至重训 origin 可见的数据和最近 126 个已知交易日验证，共 18 次训练。一个 20 日周期内权重固定，输入窗口随 origin 更新。

优化器、学习率、epoch 上限、early stopping 和随机种子沿用 Phase 2：Adam、0.001、最多 50 epochs、patience 7、seed 2026。为减少小批次 kernel launch 和数据传输开销，Phase 3 正式 GPU 运行统一用 batch 128；训练与验证均启用 pinned memory/异步传输，CUDA 上使用 PyTorch `float32_matmul_precision=high`。这不会改变模型结构、标签、切分、优化器或 early stopping；batch 较大意味着每 epoch 更新次数更少，故它作为明确记录的运行配置，并对全部 45 次训练统一使用。远端每次运行生成 `experiment_config.json`，记录代码提交、处理数据哈希和完整设置。

## 执行命令

本地先跑 CPU smoke 和离线测试：

```powershell
python -m unittest tests.test_residual_phase3 -v
python scripts/residual_phase3.py --mode smoke --device cpu
```

远程 GPU 按阶段依次运行，运行前记录 GPU 状态并确认训练进程占用符合预期：

```bash
python scripts/residual_phase3.py --mode development --device cuda --batch-size 128
python scripts/residual_phase3.py --mode recent --device cuda --batch-size 128
python scripts/analyze_residual_phase3.py
python scripts/residual_phase3.py --mode final --device cuda --student-id 10235102458
python scripts/export_submission.py --student-id 10235102458 \
  --input results/residual_phase3/final/prediction_details.csv \
  --metadata results/residual_phase3/final/metadata.json \
  --output-dir results/residual_phase3/final/submission
python scripts/validate_submission.py --student-id 10235102458 \
  --metadata results/residual_phase3/final/metadata.json \
  --output-dir results/residual_phase3/final/submission \
  --report results/residual_phase3/final/submission_validation.json
```

每次 fit 的 best epoch、train/validation loss、运行时长、设备、峰值 GPU 显存、checkpoint 路径和哈希写入 CSV；模型权重保留在 `results/residual_phase3/checkpoints/`，由 Git 忽略。阶段日志可单独重定向到同目录 `logs/`。

## 自动配对与选择标准

分析器要求 V0/V1/V2 对每个股票、模型、origin 的目标日期和真实 Open 完全相同，并输出 `alignment_audit.json`。报表包含 mean/median/std NMSE、8 个近期窗口中优于 baseline 的窗口数、相对 Last Close 与 V0 的改善比例、第 1—5 日 NMSE、预测收益率与真实收益率相关性。所有候选均保留，近期图表和汇总保存于本目录。

模型只有在近期 mean NMSE 至少改善 5%、median NMSE 低于 baseline，并在至少 5/8 窗口优于 baseline 时才允许入选；否则默认 Last Close Baseline，并在报告中指出样本量有限。近期 40 日是快速模型选择集，不是独立最终测试集。

预测导出/校验支持指定 metadata 和 report 路径，以便 Phase 3 生成文件留在本目录，不会改写 Phase 2 的正式 CSV 或校验报告。
