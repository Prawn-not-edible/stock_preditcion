# 原始数据格式

`data/raw/*.csv` 使用 UTF-8-SIG 编码，首行为列名，不含 DataFrame 行号。BaoStock 日线字段按 API 原样保存：`date, code, open, high, low, close, preclose, volume, amount, adjustflag, turn, tradestatus, pctChg, isST`。日期为 `YYYY-MM-DD`；价格单位为人民币元，成交量单位为股。请求口径为日频、不复权（`adjustflag=3`）。本阶段不填充空值、不补交易日、不变换价格、不删除记录。
