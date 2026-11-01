# Phase 2 测试结果与数据说明

本页记录 2026-11-01 完成的 Phase 2 数据处理、模型验证、回测和预测产物。运行代码见 `scripts/`；远程运行归档见 `results/remote_gpu_run/`。

## 数据集与复用方法

本次使用 BaoStock 日线数据，复权口径为不复权（`adjustflag=3`），日期范围 2014-01-01 至 2026-10-30。股票代码、名称和文件名统一在 `config/stocks.json` 配置：

| 代码 | 股票 | 处理后行数 | 停牌行排除数 |
| --- | --- | ---: | ---: |
| `sh.601088` | 中国神华 | 3,043 | 74 |
| `sh.601225` | 陕西煤业 | 3,099 | 0 |
| `sz.000983` | 山西焦煤 | 3,107 | 10 |

三只处理后数据均截至 2026-10-30。预处理保留原始 CSV，不对缺失交易记录补值；停牌 (`tradestatus=0`) 行计入审计后从模型输入表排除。源数据、生成后数据及远端处理表的 SHA-256 可在 `docs/DOWNLOAD_AUDIT.md` 和 `results/remote_gpu_run/REMOTE_BUNDLE_MANIFEST.txt` 查到。Windows 与 Linux 导出的处理表换行符不同（CRLF/LF），内容的日期、行数和数值一致。

后续替换数据集时，先更新 `config/stocks.json` 的股票代码、名称和输出文件名，再将对应 BaoStock 原始 CSV 放入 `data/raw/`。执行 `python scripts/prepare.py --as-of YYYY-MM-DD --overwrite` 重新预处理；若预测日期变化，同时更新 `config/prediction_dates.json` 并用 BaoStock 交易日历核验。之后依次运行下方的 smoke、development、recent 与预测命令。窗口数、训练/验证长度等规则在 `scripts/prepare.py`、`scripts/experiment.py` 中定义；不同长度的数据集应先确认每个股票都满足最小训练与验证样本要求。

## 测试流程

1. **数据审计与预处理**：使用 `scripts/prepare.py` 生成模型表、窗口 Manifest 和停牌审计，核对截止日期与行数。
2. **CPU smoke test**：对中国神华首个开发窗口运行 DLinear、iTransformer、TimeXer，各训练 1 epoch；检查前向、反向、预测、checkpoint 保存/加载和 NMSE 计算。此测试只确认管线，不参与选模。
3. **GPU smoke test**：在 RTX 3080 Ti 上用相同三个官方模型执行 1 epoch，确认 CUDA 训练和 checkpoint 重载正常。
4. **开发集回测与选模**：三个股票分别使用六个扩展窗口；每个窗口运行三种深度模型并和 LastCloseBaseline 对照。共 54 次深度模型窗口训练。每只股票仅按六窗口 mean NMSE 决定最终预测器。
5. **近期保留区回测**：用最近 180 个交易日做独立近期评估，形成每股 36 个不重叠五日预测区间；三个深度模型各在九个滚动起点重训，共 81 次 GPU 重训。近期结果用于泛化诊断，不反馈到开发集选模。
6. **最终导出和校验**：使用 2026-10-30 截止数据生成 2026-11-02 至 2026-11-06 五个交易日预测；通过 `scripts/validate_submission.py --student-id 10235102458` 校验文件名、列、日期、行数、正数和有限值。

常用命令（远端需在可用 CUDA 环境中运行）：

```bash
python scripts/prepare.py --as-of 2026-10-30 --overwrite
python scripts/experiment.py --mode smoke --device cuda
python scripts/experiment.py --mode development --device cuda
python scripts/experiment.py --mode recent --device cuda
python scripts/predict_future.py --device cuda
python scripts/export_submission.py --student-id 10235102458
python scripts/validate_submission.py --student-id 10235102458
python scripts/analysis.py
```

## 结果摘要

所有远端阶段的退出码均为 `0`。远端环境为 Python 3.12.3、PyTorch 2.8.0+cu128、CUDA 可用、NVIDIA GeForce RTX 3080 Ti 12 GB；完整环境记录见 `results/remote_gpu_run/results/metrics/remote_environment.txt`。

开发集六窗口 mean NMSE：

| 股票 | DLinear | iTransformer | TimeXer | LastCloseBaseline（选中） |
| --- | ---: | ---: | ---: | ---: |
| 中国神华 | 0.001395 | 0.001342 | 0.001187 | 0.001181 |
| 陕西煤业 | 0.001729 | 0.000625 | 0.001367 | 0.000446 |
| 山西焦煤 | 0.001890 | 0.002948 | 0.001628 | 0.000568 |

三个股票均按开发集结果选择 LastCloseBaseline，因为其 mean NMSE 最低。近期 36 窗口评估中，LastCloseBaseline 的 mean NMSE 仍最低：神华 0.000716、陕西煤业 0.000981、山西焦煤 0.001121。该近期结果没有用于选择模型。

正式预测文件为 `results/predictions/10235102458_prediction_*.csv`，格式校验报告为 `results/predictions/submission_validation.json`。由于开发回测选中 LastCloseBaseline，正式预测以最后收盘价延续五个交易日：神华 51.990002、陕西煤业 28.080000、山西焦煤 6.490000。深度模型已完成回测/近期重训，但未作为最终预测器，因此没有最终深度模型 checkpoint。

## 结果文件索引

- `results/metrics/smoke_windows.csv`：本地 CPU smoke 数值。
- `results/remote_gpu_run/results/metrics/development_windows.csv`、`development_summary.csv`：开发窗口明细与汇总。
- `results/remote_gpu_run/results/metrics/recent_windows.csv`、`recent_summary.csv`、`recent_training_runs.csv`：近期回测与每次重训遥测。
- `results/remote_gpu_run/results/metrics/model_selection.json`、`model_comparison.csv`：选模决定与模型对照。
- `results/remote_gpu_run/results/analysis/`：汇总图表。
- `results/remote_gpu_run/logs/`：开发和近期流程日志及退出码。
- `results/predictions/`：最终预测明细、元数据、正式 CSV 和格式校验报告。
- `data/processed/`：本地平台预处理表；远端 Linux 处理副本另存于归档下的 `data/processed/`。

## 数据集更换时的可比性

更换股票、起止日期或供应商后，重新生成并保留新的 preprocessing audit 和 SHA-256 清单。新旧数据集的窗口日期、有效行数和模型选择不可直接混为同一实验；请将新一轮产物放在独立实验目录或新分支，并记录配置、代码提交、环境和截止日期。不要把 smoke test 指标当作正式回测，也不要用近期保留区结果反向调参或选模。
