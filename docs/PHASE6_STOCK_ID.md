# Phase 6：P2 + Stock ID 静态修正

## 实验设计

本轮沿用 Phase 5 P2 的 residual 特征、标签、共享煤炭池、统一历史日期边界和三目标股等权验证 NMSE，只增加一个零初始化的股票 ID 修正头：

```text
pred_return = Phase5_backbone_residual(x) + Linear(Embedding(stock_id))
```

修正头输出 5 个预测日的静态偏置，未增加股票间动态注意力。股票 ID 词表从 `config/coal_stock_pool.json` 排序并保存在 checkpoint 中。预设参数为 `embedding_dim=8`、`id_l2=0.001`、batch size 512、seed 2026、学习率 0.001、最多 50 epoch、patience 7。开发期为 3 个共同历史起点；近期为 8 个不重叠五日窗口，每 20 个交易日训练一次。运行 TimeXer 与 DLinear，共 6 次开发拟合、4 次近期拟合。

P2 无 ID 对照直接读取 Phase 5 冻结预测，没有额外重训。Phase 6 检查 Phase 5 开发/近期配置的 batch、seed、学习率、epoch 上限、patience、验证长度、输入/预测窗口与模型集合。也核对三个目标股源 CSV 的 SHA256 与 Phase 5 最终元数据一致，并记录煤炭池所有输入数据、Phase 5 控制配置与预测文件的哈希。详细哈希在 `results/phase6_stock_id/pairing_audit.json` 及各阶段 `run_config.json`。

## 执行与审计

本地运行：

```powershell
python -m unittest discover -s tests -p 'test_phase6_stock_id.py' -v
python scripts/phase6_stock_id.py --mode smoke --device cpu
```

远端 RTX 4090 24 GB 上运行：

```bash
python scripts/phase6_stock_id.py --mode development --device cuda --models TimeXer,DLinear
python scripts/phase6_stock_id.py --mode recent --device cuda --models TimeXer,DLinear
python scripts/phase6_analyze.py
```

7 项离线测试通过；CPU smoke test 通过。另对两个真实官方模型执行了小型输入输出检查，输出均为 `[2, 5]` 且正常。GPU 阶段生成 72 条开发预测、192 条近期预测，开发和近期对照分别为 72/192 行，按目标股票、模型、起点和 alpha 完整配对；实际日期、目标日期、anchor 和真实 Open 均一致。近期拟合共耗时约 97.54 秒，开发拟合约 113.56 秒。DLinear 有 911 个参数，TimeXer 有 411,442 个参数。全部 checkpoint 保留在远端，不提交 GitHub。

## 近期对比

近期窗口重复用于前序 Phase 3/4/5 选模，因此这里只是模型诊断数据。Alpha 只用 3 个历史开发起点锁定。表中 baseline relative 表示相对 alpha=0 的 Phase 5 P2 Last Close NMSE 改善率；Phase 5 incumbent 是上一阶段每股最终选择的方案。保守选模还要求相对 Phase 5 incumbent 的 Mean NMSE 至少再改善 2%。

|股票|模型|Phase 5 P2 alpha|Phase 6 P2_ID alpha|P2_ID Mean NMSE|Median NMSE|对 Last Close 改善|胜/平/负|Phase 5 已选方案 Mean NMSE|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|中国神华|DLinear|0.00|0.25|0.00037654|0.00036607|1.1%|5/0/3|0.00038081|
|中国神华|TimeXer|0.00|0.00|0.00038081|0.00036833|0.0%|0/8/0|0.00038081|
|陕西煤业|DLinear|0.00|0.00|0.00042185|0.00035375|0.0%|0/8/0|0.00042185|
|陕西煤业|TimeXer|0.00|1.00|0.00047204|0.00057032|-11.9%|4/0/4|0.00042185|
|山西焦煤|DLinear|0.00|0.00|0.00068283|0.00056086|0.0%|0/8/0|0.00054003|
|山西焦煤|TimeXer|1.00|0.25|0.00064171|0.00058901|6.0%|5/0/3|0.00054003|

中国神华 DLinear 有 5/8 窗口胜出，但平均改善仅 1.1%，低于门槛。山西焦煤 TimeXer 相对 Last Close 改善 6.0%，但仍比 Phase 5 已选 P0 TimeXer 的 mean NMSE 高约 18.8%，所以没有超过 incumbent。陕西煤业两个 ID 模型均无改善。因此三只股票都保留 Phase 5 incumbent。

Phase 6 没有运行最终全历史重训，也没有生成新的提交 CSV。Phase 5 正式预测保持有效。近期结果不能证明泛化改善；有限的 3 个开发起点也可能导致 alpha 锁定不稳定。完整逐窗口预测、锁定 alpha、配对审计、模型选择和报告都在 `results/phase6_stock_id/`。
