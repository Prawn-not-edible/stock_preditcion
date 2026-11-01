# 股票预测课程作业

本项目用于探索中国神华（sh.601088）、陕西煤业（sh.601225）和山西焦煤（sz.000983）的时间序列预测。三只股票统一通过 BaoStock 获取日线行情，当前下载口径为不复权（`adjustflag=3`）。计划比较官方 Time-Series-Library 中的 DLinear、iTransformer、TimeXer。

## 目录

- `Time-Series-Library/`：官方上游 Git Submodule。
- `config/stocks.json`：股票元数据配置。
- `data/raw/`：原始行情（不自动清洗；当前数据仅本地保存）。
- `data/processed/`：后续处理产物。
- `scripts/`：下载脚本及后续流程占位脚本。
- `docs/`：数据格式、下载审计和 Git 流程。
- `results/`：分析、评测、预测结果和模型 checkpoint。

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

## GitHub 同步约定

使用 `main` 分支。每个有意义阶段完成并检查后，只暂存相关文件、提交明确描述的 commit，再推送 `origin main`。原始行情公开许可未确认前保留在本地，不推送。详见 `docs/GIT_WORKFLOW.md`。处理后的数据、正式预测 CSV、代码和文档均可纳入版本控制。

## 当前阶段

Phase 1 已实现项目结构、BaoStock 下载脚本、CSV 只读审计和离线测试。Phase 2 正在补充预处理、回测、训练和推理流程；其中完整训练与最终预测尚未完成，具体状态见下文阶段进度。

### Phase 2 实验进度

已实现基础预处理、六个扩展开发窗口 Manifest、官方 DLinear/iTransformer/TimeXer 外层训练器、近期滚动回测入口、正式推理/导出/校验入口。预处理和本地 CPU smoke 运行状态及 GPU 运行命令见 `docs/PHASE2_RUNBOOK.md`。完整 GPU 回测、模型选择和正式预测只有在远程服务器实际运行后才可标记完成；Smoke Test 数值不是正式预测。
