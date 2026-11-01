# Phase 5：目标感知联合训练与微调

## 实验范围

本阶段在 Phase 4 基础上只增加训练池与微调策略比较，沿用 Phase 3 residual 表示、DLinear/TimeXer 官方模型实现、`seq_len=60`、`pred_len=5`、anchor 还原、统一日期边界和合法训练统计量。运行模型为 DLinear 与 TimeXer（按 Phase 5 补丁的推荐快速配置；未运行可选的 iTransformer）。

- P0：每只目标股票独立从头训练。
- P1：三只目标股票共享训练。
- P2：按历史时点可用行业成员构成的煤炭股票池共享训练。
- P3：从同日期 P2 checkpoint 初始化，再使用单只目标股票合法训练窗口微调。

开发集使用 2019-07-01、2022-07-01、2025-07-01 三个起点；近期评测使用 2026-08-28 至 2026-09-28 的 8 个非重叠五日窗口，分两个 20 交易日训练周期。P1/P2 checkpoint 依据三只目标股验证 NMSE 等权均值选择，P3 依据目标股验证 NMSE 选择，允许 epoch 0。所有方案 batch size 为 512、最多 50 epoch；P3 最多 10 epoch。训练池成员、样本数、起点、验证边界和参数均记录在训练运行表中。

## 运行与验证

本地执行：

```powershell
python -m unittest discover -s tests -p 'test_phase5_joint.py' -v
python scripts/phase5_joint.py --mode smoke --device cpu
```

远程 RTX 3080 Ti 12 GB 运行：

```bash
python -u scripts/phase5_joint.py --mode development --device cuda --models DLinear,TimeXer
python -u scripts/phase5_joint.py --mode recent --device cuda --models DLinear,TimeXer
python scripts/phase5_analyze.py
python scripts/phase5_final.py --device cuda --student-id 10235102458
python scripts/export_submission.py --student-id 10235102458 \
  --input results/phase5_targetaware/final/prediction_details.csv \
  --metadata results/phase5_targetaware/final/metadata.json \
  --output-dir results/phase5_targetaware/final/submission
python scripts/validate_submission.py --student-id 10235102458 \
  --metadata results/phase5_targetaware/final/metadata.json \
  --output-dir results/phase5_targetaware/final/submission \
  --report results/phase5_targetaware/final/submission_validation.json
```

离线测试 6 项通过，CPU smoke test 通过。GPU 开发阶段生成 288 条预测、48 条训练记录；近期阶段生成 768 条预测、32 条训练记录。训练记录含参数量、耗时、样本数、显存峰值字段、checkpoint SHA256；全部 checkpoint 保留在远程 GPU，不提交 GitHub。GPU 使用空闲时启动，训练时峰值记录约 20.45 MiB（PyTorch 记录口径）。

## 近期配对结果

指标均在相同股票、预测起点、目标日期及真实 Open 上计算。NMSE 为五个预测日的窗口均值；alpha 在三个历史开发窗口锁定。Last Close 对每只股票/日期相同，不随模型变化。

|股票|训练方式|模型|Alpha|Mean NMSE|Median NMSE|相对 Last Close|窗口胜/平/负|
|---|---|---|---:|---:|---:|---:|---:|
|中国神华|P1|DLinear|0.50|0.00037405|0.00039230|+1.8%|4/0/4|
|中国神华|其余候选|—|锁定为 0 或未达门槛|0.00038081|0.00036833|0.0%|0/8/0（alpha=0）|
|陕西煤业|P3|DLinear|0.25|0.00042000|0.00037649|+0.4%|3/0/5|
|陕西煤业|其余候选|—|锁定为 0 或未达门槛|0.00042185|0.00035375|0.0%|0/8/0（alpha=0）|
|山西焦煤|P0|TimeXer|0.50|0.00054003|0.00055873|+20.9%|6/0/2|
|山西焦煤|P1|TimeXer|0.25|0.00062373|0.00053016|+8.7%|4/0/4|
|山西焦煤|P2|TimeXer|1.00|0.00063043|0.00066070|+7.7%|6/0/2|
|山西焦煤|P3|TimeXer|1.00|0.00060039|0.00066687|+12.1%|5/0/3|

表中改善率定义为 `1 - candidate_mean_nmse / baseline_mean_nmse`。对个别 alpha=0 的行，胜平负为与 Last Close 同预测的逐窗平局。完整候选及均值、median、标准差、逐窗预测和锁定 alpha 见 `results/phase5_targetaware/comparison.csv`、`selected_recent_predictions.csv`、`locked_alphas.json`。

按预设门槛（相对 Last Close 均值至少改善 5%、median 更低且至少 5/8 窗口获胜），神华与陕煤均保留 Last Close。山西焦煤采用 P0 TimeXer、alpha=0.5。P1/P2/P3 在山西焦煤也有改善，但未超过 P0。近期数据曾用于 Phase 3/4 选模，不能视为独立测试；开发窗口只有三个，alpha 可能不稳定。

## 最终预测和提交文件

最终 TimeXer 使用 Phase 5 三个开发拟合中位最佳 epoch（17）训练至 2026-10-30。中国神华与陕西煤业使用 10 月 30 日 Close 基线。预测如下：

|股票|2026-11-02|2026-11-03|2026-11-04|2026-11-05|2026-11-06|方法|
|---|---:|---:|---:|---:|---:|---|
|中国神华|51.990002|51.990002|51.990002|51.990002|51.990002|Last Close|
|陕西煤业|28.080000|28.080000|28.080000|28.080000|28.080000|Last Close|
|山西焦煤|6.502748|6.502944|6.520289|6.513010|6.523802|P0 TimeXer, alpha=0.5|

三个正式 CSV 位于 `results/phase5_targetaware/final/submission/`，格式为 `Date,Open`，每份五行，均由 `validate_submission.py` 校验通过。Phase 3/4 正式文件没有覆盖。最终模型 checkpoint 的 SHA256 为 `3ad05b01d0438e20eae083405fe5ced1c22d1893dbf0d90f9b3e3edaaa665b34`，仅保留在远程 GPU 环境。

## 可复现产物

- `scripts/phase5_joint.py`：P0/P1/P2/P3 数据集与训练流程。
- `scripts/phase5_analyze.py`：历史 alpha 锁定、配对检查、近期规则选模。
- `scripts/phase5_final.py`：入选方案最终训练、推理和元数据。
- `tests/test_phase5_joint.py`：目标宏平均 checkpoint、Residual 还原、模型 IO 等离线测试。
- `results/phase5_targetaware/`：配置、训练记录、逐窗预测、对比表、日志、选择结果、正式预测与验证报告。

模型文件（`.pt`）没有纳入结果归档或 GitHub。`training_runs.csv` 的 checkpoint 路径指向当时的远程 GPU 运行目录，checkpoint SHA 可用于核对远端文件。
