# 股票预测课程作业

本项目用于探索中国神华（`sh.601088`）、陕西煤业（`sh.601225`）和山西焦煤（`sz.000983`）的时间序列预测。三只股票统一通过 BaoStock 获取日线行情，当前下载口径为不复权（`adjustflag=3`）。模型对照采用官方 Time-Series-Library 中的 DLinear、iTransformer、TimeXer。

## 目录

- `Time-Series-Library/`：官方上游 Git Submodule。
- `config/`：股票元数据、预测日期等配置。
- `data/raw/`：原始行情；原始数据按项目约定留在本地。
- `data/processed/`：预处理产物。
- `scripts/`：下载、预处理、训练、回测、分析和预测导出脚本。
- `tests/`：离线数据流程和实验单元测试。
- `docs/`：数据格式、实验流程和阶段说明。
- `results/`：回测指标、实验记录、预测结果；模型 checkpoint 默认不纳入 Git。

## 环境与下载

Windows PowerShell：

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-project.txt
python scripts/download.py
```

默认下载 2014-01-01 至 2026-10-30 的日线。可使用 `--start-date`、`--end-date` 指定日期；已有目标文件需显式使用 `--overwrite`。脚本从 `config/stocks.json` 读取股票信息，一次登录完成查询，原样保存所有请求字段，不补值、不删记录，并对 CSV 执行只读审计。

## 上游模型库

来源：[thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library)。重新克隆请使用：

```bash
git clone --recurse-submodules https://github.com/Prawn-not-edible/stock_preditcion.git
```

若已普通克隆：

```bash
git submodule update --init --recursive
```

上游版本见 `UPSTREAM_VERSION.md`；主仓库只记录 submodule commit 引用。

## 实验进度

### Phase 1：项目初始化与数据接入

已完成项目结构、BaoStock 下载脚本、CSV 只读审计和离线测试。数据下载与预处理说明见 `docs/`。

### Phase 2：绝对价格预测基线

已完成三只股票的预处理、扩展开发窗口和近期回测，使用 DLinear、iTransformer、TimeXer 预测未来 5 个交易日 Open。Phase 2 的数据划分、指标和结果归档于 `phase2-results-data-docs` 分支及 `results/remote_gpu_run/`；其正式预测文件在 Phase 3 中未覆盖。

### Phase 3：Residual Forecasting 快速选模

已完成 27 次历史开发训练和 18 次近期重训。**本阶段主要变化是训练目标**：由预测绝对 Open 价格改为预测相对预测起点前一日 Close 的变化率（residual return）。历史价格输入也以该 Close 为 anchor 转为相对价格；成交量和成交额使用 `log1p` 并仅根据当次训练数据拟合标准化统计量。预测时将变化率还原为 `anchor * (1 + pred_return)`。V0 使用 Phase 2 已保存预测，V1 为新训练的 residual 模型，V2 为 V1 预测乘以开发集锁定的收缩系数 alpha，其中 alpha=0 等于 Last Close Baseline。

为提升小模型 GPU 训练吞吐，正式训练统一采用 batch size 128、pinned memory/异步数据传输和 CUDA `float32_matmul_precision=high`。与 batch size 32 的早期试跑相比，神华前 9 个开发 fit 的总耗时由 72.20 秒降至 29.95 秒，约快 2.41 倍。优化没有改变模型架构、数据切分、优化器、学习率、epoch 上限、early stopping 或随机种子。早期 batch 32 试跑中断记录已归档，不计入实验结果。

近期结果未显示神经网络在所有保守选模条件下稳定优于 Baseline，因此三只股票最终均采用 Last Close Baseline。2026-11-02 至 2026-11-06 每日 Open 预测分别为中国神华 51.990002、陕西煤业 28.080000、山西焦煤 6.490000；三份提交 CSV 均通过格式校验。

从 `phase3-predictions-report` 分支可以取得完整训练代码、复现配置和详细结果：

- `scripts/residual_phase3.py`、`scripts/analyze_residual_phase3.py`：Residual 训练与分析实现。
- `tests/test_residual_phase3.py`：离线逻辑测试。
- `docs/PHASE3_RESIDUAL.md`：方法、运行命令和实验规则。
- `results/residual_phase3/`：阶段配置、训练日志、逐窗口结果、对齐审计和完整实验报告。
- `results/predictions/phase3/`：三份正式预测 CSV、验证报告、元数据和选模汇总。

Phase 3 的完整分析见 [`results/residual_phase3/FINAL_REPORT.md`](results/residual_phase3/FINAL_REPORT.md)，预测与实验说明见 [`docs/PHASE3_PREDICTION_RESULTS.md`](docs/PHASE3_PREDICTION_RESULTS.md)。

### Phase 4：煤炭行业多股票联合 Residual 训练

实验代码已在 `phase4-joint-training` 分支实现，复用 Phase 3 的 Residual 架构与目标，比较 P0 单股训练、P1 三目标股联合训练和 P2 按预测日期之前行业分类生成的煤炭股池联合训练。新增多股票 Dataset 保持每只股票序列独立，并采用逐股票训练期成交量/额标准化及均衡采样。近期窗口与 Phase 3 P0 完全配对，alpha 在历史开发窗口锁定。

煤炭池 BaoStock 查询、数据清洗与正式 GPU 训练正在进行；本分支 README 与 `results/phase4_joint_training/EXPERIMENT_REPORT.md` 将以实际完成的查询、训练和对齐审计更新进度。原始行情与 checkpoint 不推送，公开代码、行业/数据审计及评价结果归档于 Phase 4 分支。

## GitHub 同步约定

实验代码、数据审计、指标和文档通过阶段分支归档；提交时只暂存本轮相关文件。原始行情公开许可未确认前保留在本地，不推送。模型 checkpoint 不上传 GitHub。详见 `docs/GIT_WORKFLOW.md`。
