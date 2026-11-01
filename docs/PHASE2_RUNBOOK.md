# Phase 2 实验运行记录与命令

## 数据准备

```powershell
python scripts/prepare.py --as-of 2026-10-30
python scripts/experiment.py --mode smoke
```

预处理只从 `config/stocks.json` 读取股票信息；停牌行按 `tradestatus=0` 统计并排除在模型表外，原始 CSV 不修改。扩展窗口起点从至少 756 个训练交易日加 126 个验证交易日之后均匀选择 6 个，并在最近 180 个有效交易日之前结束。模型标准化在每个训练窗口内单独拟合。

## 远程 GPU 流程

远端连接必须由操作者提供可用的 SSH 服务和端口；密码只通过 SSH 交互提示输入，不放在命令、文件或日志中。登录后先记录 `nvidia-smi`、PyTorch/CUDA 版本、显存、内存和磁盘，再运行 CPU/GPU Smoke Test，然后分阶段运行：

```bash
python scripts/prepare.py --as-of 2026-10-30
python scripts/experiment.py --mode smoke --device cuda
python scripts/experiment.py --mode development --device cuda
python scripts/experiment.py --mode recent --device cuda
python scripts/predict_future.py --device cuda
python scripts/export_submission.py --student-id <真实学号>
python scripts/validate_submission.py --student-id <真实学号>
python scripts/analysis.py
```

Smoke 测试默认是 CPU 小模型；GPU smoke 可使用 `--device cuda --small --epochs 1 --batch-size 8`。`development` 只使用最近 180 日之前的数据选模；`recent` 评估保留区域且不改动开发阶段的选择结果；`final` 使用锁定的开发结果训练并推理。Checkpoint 位于 `results/checkpoints/` 并由 `.gitignore` 排除。

## 当前状态（2026-11-01）

Phase 2 远程 GPU 开发回测、近期保留区回测、选模、预测导出和格式校验均已完成。当前实验数据口径、逐项测试流程、指标和产物索引见 [`PHASE2_RESULTS_AND_DATA.md`](PHASE2_RESULTS_AND_DATA.md)。

- 三只股票均截至 2026-10-30；处理后有效行数及停牌审计见结果说明和 `results/metrics/preprocessing_audit.json`。
- CPU/GPU Smoke Test 已完成；Smoke 仅验证运行链路，不用于选模。
- 开发阶段共 54 次深度模型窗口训练。选模只使用六个开发窗口；三只股票均选择 LastCloseBaseline。
- 近期独立保留区包含每只股票 36 个不重叠五日窗口；深度模型共进行 81 次滚动重训。近期结果不反馈到选模。
- 正式预测文件使用学号 `10235102458` 生成，并通过提交格式校验；五个预测日期为 2026-11-02 至 2026-11-06。
- GPU 检查点保留在远端运行目录，不作为最终预测模型提交；最终选择的是 LastCloseBaseline。
