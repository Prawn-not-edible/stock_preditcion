# Phase 4：煤炭行业多股票联合 Residual 训练

本阶段从 `phase3-predictions-report` 最新提交开始，Phase 3 近期 P0 结果直接复用；三个统一历史开发日期与 Phase 3 每只股票各自选取的开发日期不完全一致，因此只在这些日期补训 P0，以获得严格同日期的历史对照，不覆盖 Phase 3 指标、预测文件或 checkpoint。P1/P2 在单独的 `scripts/phase4_joint.py` 中共享官方模型参数。

## 唯一主要变量

保持 DLinear、iTransformer、TimeXer 架构，60 日输入、5 日 Open residual 标签、Phase 3 特征、MSE loss、Adam、学习率 0.001、seed 2026、50 epoch 上限与 patience 7 不变。P0 每只目标股票独训；P1 只用三只目标股构造均衡采样联合 Dataset；P2 使用按每个预测起点之前可查到的 BaoStock 煤炭行业分类形成的煤炭池。P1/P2 同一模型、同一预测起点各训练一个共享 checkpoint，并分别对三只目标股推理。

训练窗口保留每个证券各自的行情序列；只在 batch 维混合样本。成交量、成交额的 `log1p` 均值/标准差逐股票、逐次拟合，仅使用当次训练截止日前记录。WeightedRandomSampler 按样本所属股票的训练窗口数倒数加权，使长历史股票不会按样本量垄断训练。正式 GPU 运行使用 batch size 512（Phase 3 为 128）；RTX 3080 Ti 的 12 GB 显存实测峰值远低于容量，合成 batch-512 反向传播对三种模型均通过。所有 P0/P1/P2 使用同一批次配置；其余 optimizer、学习率、随机种子、Early Stopping 和 epoch 上限保持不变。

预测起点来自三只目标股的共同交易日及完整相同五日目标日期序列。每个训练池成员的训练标签都必须在共享训练截止日之前；验证标签严格早于目标预测日期；成员无须在目标日有行情，股票的训练资格也不依据目标日之后的数据完整度决定。无历史行业分类接口时，代码只将目标股用于历史 P2，不用当前行业标签倒推历史成员。

## 煤炭池来源和数据

`scripts/phase4_pool.py` 查询 BaoStock `query_stock_industry` 与 `query_stock_basic`，记录 API 实际字段和分类日期。仅保留沪深 A 股普通股票（排除指数、ETF、B 股等），筛选煤炭采掘类；BaoStock 混合石油/煤炭/燃料分类只接受公司名含煤、焦或炭的证券，并显式纳入三只目标股。历史分类支持时，对每个回测起点只采用不晚于该日的最近快照。行业和证券基础接口通常反映可观测成员；退市历史成员缺失所致的幸存者偏差会记入审计，不伪造历史行业成员。

行情为 2014-01-01 至 2026-10-30 日线、不复权（`adjustflag=3`）。每只股票独立过滤停牌/异常 OHLC 并保存摘要。20% 以上不复权价格跳变只记录事件日期，不自动删除或修复。至少有 60 个输入和 5 个标签交易记录的股票才可产生训练窗口；不足者记录排除原因。

## 运行

先查询股票池并下载独立数据：

```bash
python scripts/phase4_pool.py --start 2014-01-01 --end 2026-10-30
```

本地 smoke 与离线单元测试：

```bash
python scripts/phase4_joint.py --mode smoke --device cpu
python -m unittest tests.test_phase4_joint -v
```

GPU 训练按历史开发、近期滚动、分析的顺序运行：

```bash
python scripts/phase4_joint.py --mode development --device cuda --batch-size 512
python scripts/phase4_joint.py --mode recent --device cuda --batch-size 512
python scripts/analyze_phase4.py
```

近期八个窗口逐一核对 P0/P1/P2 目标日期和真实 Open。P0 Alpha 继承 Phase 3 锁定值；P1/P2 alpha 仅由三个历史开发日期选择，近期固定使用。同一 20 个交易日周期重训一次（连续 4 个五日窗口复用一个共享 checkpoint），每个窗口用当时最新 60 日输入重新推理。近期指标是快速模型比较集，不是新的独立测试集。只有相对 Last Close 与 P0 的 Mean NMSE 均改善至少 5%、Median NMSE 同时低于二者，并在至少 5/8 窗口战胜 Last Close 的候选才允许替换原正式预测；其余保留 Phase 3 文件。

## 产物

- `config/coal_stock_pool.json`：查询来源、池大小和代码列表。
- `data/coal_pool/raw/`、`data/coal_pool/processed/`、`data/coal_pool/manifests/`：独立股票池数据和日期化行业快照。
- `data/pool_manifest.csv`、`docs/COAL_POOL_AUDIT.md`：IPO/行情日期、有效记录、停牌、跳变事件和排除原因。
- `results/phase4_joint_training/development/`、`recent_40d/`：配对预测、训练运行和日期。
- `results/phase4_joint_training/checkpoints/`：共享 checkpoint；由 Git 忽略。
- `results/phase4_joint_training/`：alpha 锁定、逐窗口明细、同日比较、股票级汇总、最终选择和报告。

原始行情 CSV 和模型权重不提交 GitHub；代码、配置、审计摘要和可公开的评价结果提交。
