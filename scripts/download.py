#!/usr/bin/env python3
"""Download untouched daily BaoStock K-line data and perform read-only audits."""
from __future__ import annotations
import argparse, csv, json, os, tempfile
import pandas as pd
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST")
REQUIRED = FIELDS.split(",")
DEFAULT_START, DEFAULT_END, ADJUST = "2014-01-01", "2026-10-30", "3"


def load_stocks(path: Path | str = ROOT / "config" / "stocks.json") -> list[dict[str, str]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    stocks = data.get("stocks")
    if not isinstance(stocks, list) or not stocks:
        raise ValueError("配置必须包含非空 stocks 数组")
    required = {"name", "security_code", "exchange", "query_code", "output_file"}
    seen = set()
    for item in stocks:
        if not isinstance(item, dict) or not required <= item.keys():
            raise ValueError(f"股票配置缺少字段：{required}")
        if Path(item["output_file"]).name != item["output_file"] or not item["output_file"].endswith(".csv"):
            raise ValueError(f"不安全的输出文件名：{item['output_file']}")
        if item["query_code"] in seen:
            raise ValueError(f"重复股票代码：{item['query_code']}")
        seen.add(item["query_code"])
    return stocks


def validate_dates(start: str, end: str) -> None:
    try:
        a, b = date.fromisoformat(start), date.fromisoformat(end)
    except ValueError as exc:
        raise ValueError("日期必须使用 YYYY-MM-DD 格式") from exc
    if a.isoformat() != start or b.isoformat() != end or a > b:
        raise ValueError("日期必须为有效的 YYYY-MM-DD，且开始日期不晚于结束日期")


def query_rows(bs: Any, code: str, start: str, end: str) -> tuple[list[str], list[list[str]]]:
    rs = bs.query_history_k_data_plus(code, FIELDS, start_date=start, end_date=end, frequency="d", adjustflag=ADJUST)
    if rs.error_code != "0":
        raise RuntimeError(f"BaoStock 查询 {code} 失败 ({rs.error_code}): {rs.error_msg}")
    fields = list(rs.fields)
    if fields != REQUIRED:
        raise ValueError(f"{code} 返回字段不匹配；期望 {REQUIRED}，实际 {fields}")
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    if rs.error_code != "0":
        raise RuntimeError(f"BaoStock 读取 {code} 结果失败 ({rs.error_code}): {rs.error_msg}")
    if not rows:
        raise ValueError(f"{code} 在 {start} 至 {end} 没有返回记录")
    dataframe = pd.DataFrame(rows, columns=rs.fields)
    return fields, dataframe


def atomic_write_csv(path: Path, dataframe: pd.DataFrame, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"目标文件已存在；如需覆盖请传入 --overwrite：{path}")
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8-sig", newline="", dir=path.parent, suffix=".tmp", delete=False) as f:
            tmp = Path(f.name)
            dataframe.to_csv(f, index=False, encoding="utf-8-sig")
            f.flush(); os.fsync(f.fileno())
        if path.exists() and not overwrite:
            raise FileExistsError(f"目标文件已存在；如需覆盖请传入 --overwrite：{path}")
        os.replace(tmp, path)
    finally:
        if tmp and tmp.exists():
            tmp.unlink()


def audit_csv(path: Path | str, expected_code: str, expected_adjust: str = ADJUST) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"文件不存在或为空：{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        missing = [x for x in REQUIRED if x not in fields]
        if missing:
            raise ValueError(f"CSV 缺少字段：{missing}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV 无数据行：{path}")
    dates=[]; invalid_dates=[]; codes=set(); dup=[]; nulls={k:0 for k in fields}; suspended=0; nonpositive=[]; bounds=[]; wrong_adjust=0
    prev=None
    for i,row in enumerate(rows, start=2):
        raw=row.get("date", "")
        try: d=date.fromisoformat(raw)
        except (ValueError,TypeError): invalid_dates.append(i); d=None
        if d:
            dates.append(d.isoformat())
            if prev and d < prev: pass
            prev=d
        codes.add(row.get("code", ""))
        for k in fields:
            if row.get(k) in (None, ""): nulls[k]+=1
        if row.get("tradestatus")=="0": suspended+=1
        if row.get("adjustflag") != expected_adjust: wrong_adjust+=1
        vals={}
        for k in ("open","high","low","close"):
            try: vals[k]=float(row[k])
            except (ValueError,TypeError,KeyError): vals[k]=None
        if any(v is not None and v<=0 for v in vals.values()): nonpositive.append(i)
        hi,lo=vals["high"],vals["low"]
        if hi is not None and lo is not None and (hi < lo or any(v is not None and (v>hi or v<lo) for k,v in vals.items() if k not in ("high","low"))): bounds.append(i)
    valid_dates=[date.fromisoformat(x) for x in dates]
    return {"file":str(path),"size_bytes":path.stat().st_size,"rows":len(rows),"fields":fields,"missing_fields":missing,
      "codes":sorted(codes),"code_mismatch":codes != {expected_code},"invalid_date_rows":invalid_dates,
      "date_min":min(dates) if dates else None,"date_max":max(dates) if dates else None,
      "duplicate_dates":len(valid_dates)-len(set(valid_dates)),"ascending":valid_dates==sorted(valid_dates),
      "null_counts":{k:v for k,v in nulls.items() if v},"suspended_rows":suspended,
      "nonpositive_ohlc_rows":nonpositive,"ohlc_boundary_rows":bounds,"wrong_adjustflag_rows":wrong_adjust}


def render_audit(reports: list[dict[str, Any]], start: str, end: str) -> str:
    lines=["# BaoStock 下载审计", "", f"请求范围：{start} 至 {end}；频率：日线；adjustflag={ADJUST}（不复权）。", "", "本审计只读原始 CSV，不清洗、不补齐、不删除任何记录。异常仅报告。", ""]
    for r in reports:
        lines += [f"## {Path(r['file']).name}", "", f"- 文件：`{r['file']}`（{r['size_bytes']:,} bytes）", f"- 记录数：{r['rows']}", f"- 实际日期范围：{r['date_min']} 至 {r['date_max']}", f"- 返回代码：{', '.join(r['codes'])}；代码不匹配：{r['code_mismatch']}", f"- 日期：非法行 {len(r['invalid_date_rows'])}；重复日期 {r['duplicate_dates']}；升序：{r['ascending']}", f"- 空值：{json.dumps(r['null_counts'], ensure_ascii=False) if r['null_counts'] else '无'}", f"- 停牌记录（tradestatus=0）：{r['suspended_rows']}", f"- OHLC 非正值行：{len(r['nonpositive_ohlc_rows'])}；价格边界异常行：{len(r['ohlc_boundary_rows'])}", f"- adjustflag 与请求不一致行：{r['wrong_adjustflag_rows']}", ""]
    spans={(r['date_min'],r['date_max']) for r in reports}
    lines += ["## 跨股票比较", "", "各股票实际起止日期一致。" if len(spans)<=1 else "实际起止日期存在差异：" + "; ".join(f"{Path(r['file']).name}: {r['date_min']} 至 {r['date_max']}" for r in reports), "", "公开分发许可尚未核实，因此本地原始 CSV 不纳入本次公开推送。"]
    return "\n".join(lines)+"\n"


def download(start: str, end: str, overwrite: bool = False, config: Path = ROOT/"config/stocks.json", raw_dir: Path = ROOT/"data/raw") -> list[Path]:
    validate_dates(start,end)
    stocks=load_stocks(config)
    targets=[raw_dir/s["output_file"] for s in stocks]
    if not overwrite:
        existing=[str(p) for p in targets if p.exists()]
        if existing: raise FileExistsError("目标文件已存在；如需覆盖请传入 --overwrite："+", ".join(existing))
    import baostock as bs
    lg=bs.login()
    if lg.error_code != "0": raise RuntimeError(f"BaoStock 登录失败 ({lg.error_code}): {lg.error_msg}")
    outputs=[]
    try:
        for stock,path in zip(stocks,targets):
            fields,df=query_rows(bs,stock["query_code"],start,end)
            atomic_write_csv(path,df,overwrite)
            ds=df["date"].tolist()
            print(f"{stock['name']} {stock['query_code']}: {len(df)} 条，{min(ds)} 至 {max(ds)}，{path}")
            outputs.append(path)
    finally:
        bs.logout()
    return outputs


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start-date",default=DEFAULT_START); p.add_argument("--end-date",default=DEFAULT_END)
    p.add_argument("--overwrite",action="store_true"); p.add_argument("--config",type=Path,default=ROOT/"config/stocks.json")
    p.add_argument("--raw-dir",type=Path,default=ROOT/"data/raw")
    args=p.parse_args()
    outputs=download(args.start_date,args.end_date,args.overwrite,args.config,args.raw_dir)
    stocks=load_stocks(args.config)
    reports=[audit_csv(path,stock["query_code"]) for path,stock in zip(outputs,stocks)]
    report_path=ROOT/"docs/DOWNLOAD_AUDIT.md"; report_path.write_text(render_audit(reports,args.start_date,args.end_date),encoding="utf-8")
    print(f"只读审计已写入：{report_path}")

if __name__=="__main__": main()
