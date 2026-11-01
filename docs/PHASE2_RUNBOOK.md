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

- 三只股票原始数据均到 2026-10-30；SHA-256 见 `docs/DOWNLOAD_AUDIT.md`。
- 预处理已在本机运行，处理结果和六窗口 Manifest 已生成。
- 处理后行数：`sh.601088` 3,043（排除 74 条停牌）、`sh.601225` 3,099（无停牌）、`sz.000983` 3,107（排除 10 条停牌）；三者均截至 2026-10-30。
- 本地 CPU Smoke Test 已通过三个官方模型的前向、训练、反向、预测输出、checkpoint 保存/重载和 NMSE 计算。
- Smoke Test 使用中国神华第一开发窗口（起点 2017-11-17），各模型只训练 1 epoch；NMSE：DLinear 0.01398、iTransformer 0.00846、TimeXer 0.00663、Last Close Baseline 0.00044。它只验证链路，不能用于选模或正式预测。
- 远程 SSH 多次在握手阶段由服务器端重置连接，未进入密码认证；因此 GPU 检查、完整开发/近期回测和正式训练尚未执行。需要恢复远程实例或端口映射后重试。
- 尚无 `model_selection.json`、最终模型和正式预测结果；不要运行导出脚本或把 smoke 结果当作正式预测。
- 交易日期 2026-11-02 至 2026-11-06 已由 BaoStock `query_trade_dates` 返回为交易日。

## 学号

正式提交文件需要用户提供真实学号。未提供学号时，导出器只写入 draft 预览目录，不生成带占位学号的正式命名文件。
