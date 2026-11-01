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

已在 `phase4-joint-training` 分支完成 BaoStock 煤炭池审计、CPU smoke/10 项离线测试、45 次历史开发拟合、12 次近期共享模型拟合和最终入选模型训练。实验比较 P0 单股训练、P1 三只目标股联合训练、P2 按预测日期之前分类构造的煤炭股池联合训练；近期窗口与 Phase 3 P0 严格按目标日期及实际 Open 配对。近期数据是快速模型比较集，不是独立测试集。

BaoStock 查询得到 32 只不同沪深 A 股煤炭/煤焦相关股票，32 只均有足够行情构造输入和标签，共 94,388 条有效行情记录、1,938 条停牌记录。P2 按历史分类时点分别纳入 26–28 只；最终预测训练使用 2026-10-30 时点的 28 只。明确披露 BaoStock 历史行业接口对已退市股票可能不完整，存在幸存者偏差。原始与处理行情、池成员快照和每股审计摘要均归档在 `data/coal_pool/`、`data/pool_manifest.csv` 与 `docs/COAL_POOL_AUDIT.md`。

为缩短煤炭池训练耗时，GPU 批次由 Phase 3 的 128 调为 512；所有 P0/P1/P2 共用该设置，模型架构、Residual 目标、Adam、学习率、随机种子、Early Stopping 规则和窗口长度不变。P2 TimeXer 在山西焦煤上的锁定 Alpha=0.5，近期 Mean NMSE 为 0.00058646，较 Last Close 的 0.00068283 改善 14.1%，Median NMSE 由 0.00056086 降至 0.00050587，8 个窗口中 6 个胜过 Baseline。它通过本轮保守规则；证据仍来自已用于开发的近期窗口。中国神华与陕西煤业没有联合候选达到门槛，继续采用 Phase 3 选择的 Last Close Baseline。

最终使用截至 2026-10-30 的 28 只煤炭股历史数据训练 P2 TimeXer 12 个 epoch，仅将通过规则的山西焦煤预测更新到 Phase 4 独立目录；中国神华和陕西煤业继续使用 Phase 3 Baseline。Phase 4 三份 CSV 均验证通过，Phase 3/Phase 2 原始结果未覆盖。训练代码、完整逐窗口结果、检查点哈希、配置、数据和审计见本分支 `results/phase4_joint_training/`；大型 checkpoint 保留在远端 GPU 环境，不提交 GitHub。

### Phase 5：按目标验证集选择共享训练与单股微调

在独立 `phase5-targetaware` 分支追加了 Phase 5 代码，没有替换 Phase 3/4 实现或结果。比较 P0 单股从头训练、P1 三只目标股票联合训练、P2 按时点可用的煤炭池联合训练，以及从同日期 P2 checkpoint 初始化的 P3 单股微调。保留 Residual 输入/目标、60 日输入、5 日预测和统一时间边界。按补丁说明先运行 DLinear、TimeXer；共完成 48 次历史开发拟合和 32 次近期拟合，全部训练记录、预测、配置、日志和模型选择归档于 `results/phase5_targetaware/`，checkpoint 留在远端 GPU。

近期 8 窗口只作为快速开发比较，不是独立测试集。Alpha 仅由三个历史开发起点选择。神华和陕煤没有候选同时满足近期均值、median 和至少 5/8 窗口胜出的规则，保留 Last Close；山西焦煤选择 P0 TimeXer、alpha=0.5，近期 mean NMSE 0.00054003，相对 Last Close 的 0.00068283 改善 20.9%，8 窗口胜 6。按开发期中位最佳 epoch 17 在截至 2026-10-30 的完整历史上重训；正式 CSV 已导出并校验，Phase 4 文件保持不变。完整方法、指标、最终预测和限制见 [`docs/PHASE5_TARGETAWARE.md`](docs/PHASE5_TARGETAWARE.md)。

### Phase 6：煤炭池共享模型的 Stock ID 静态修正消融

新增 `StockIDResidual`：在 Phase 5 P2 backbone 输出上加零初始化的 `Embedding(stock_id) → Linear(5)` 修正头，初始预测与无 ID backbone 完全相同。按补丁预设 `embedding_dim=8`、`id_l2=0.001`、batch 512、seed 2026，仅运行 TimeXer 与 DLinear。共完成 6 次历史开发和 4 次近期共享拟合。开发期 alpha 锁定，近期 8 窗口与 Phase 5 冻结 P2 预测逐日期和真实值配对；配置与输入数据哈希保存在 `results/phase6_stock_id/`。

没有 P2_ID 模型同时达到近期改善门槛并超过 Phase 5 已选方案，因此三只股票均保留 Phase 5 预测，没有生成新的最终模型或提交 CSV。近期数据已被前序实验使用，结果不构成独立测试。方法和完整比较见 [`docs/PHASE6_STOCK_ID.md`](docs/PHASE6_STOCK_ID.md)。

## GitHub 同步约定

实验代码、可公开的 BaoStock 行情、数据审计、指标和文档通过阶段分支归档；提交时只暂存本轮相关文件。凭据与模型 checkpoint 不上传 GitHub。详见 `docs/GIT_WORKFLOW.md`。
