"""Build a dated BaoStock coal-equity pool and download audited raw K-lines."""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from download import FIELDS, query_rows, atomic_write_csv, audit_csv
from prepare_stocks import MODEL_COLS, REQUIRED_RAW

TARGETS = {"sh.601088", "sh.601225", "sz.000983"}
ASOF_DATES = []
COAL_WORDS = ("煤炭", "煤矿", "采掘")
A_SHARE = re.compile(r"^(sh\.(60|68)\d{4}|sz\.(00|30)\d{4})$")


def collect(rs) -> pd.DataFrame:
    if rs.error_code != "0":
        raise RuntimeError(f"BaoStock query failed ({rs.error_code}): {rs.error_msg}")
    fields, rows = list(rs.fields), []
    while rs.next():
        rows.append(rs.get_row_data())
    if rs.error_code != "0":
        raise RuntimeError(f"BaoStock result iteration failed ({rs.error_code}): {rs.error_msg}")
    return pd.DataFrame(rows, columns=fields)


def is_coal(industry: str, code_name: str = "") -> bool:
    s = str(industry or "").strip()
    name = str(code_name or "").strip()
    if "煤炭开采和洗选" in s or "煤矿" in s or "煤炭采选" in s:
        return True
    if "煤炭" in s and "石油" not in s and "其他燃料" not in s:
        return True
    # BaoStock's C25 label combines petroleum, coal and other fuels; admit only names
    # explicitly tied to coal/coking, not every petroleum/chemical company in that class.
    mixed_fuel = "石油" in s and ("煤炭" in s or "燃料" in s)
    return mixed_fuel and any(w in name for w in ("煤", "焦", "炭"))


def experiment_dates() -> list[str]:
    """The same deterministic, price-blind date rule used by phase4_joint.py."""
    target_files = [ROOT / "data/processed" / f"{s.replace('.', '_')}.csv" for s in sorted(TARGETS)]
    maps=[]
    for path in target_files:
        df=pd.read_csv(path,dtype={"date":str}); d=df.date.tolist()
        maps.append({tuple(d[o:o+5]):d[o] for o in range(60,len(d)-5+1)})
    seqs=set(maps[0]).intersection(*[set(m) for m in maps[1:]])
    shared=sorted(((maps[0][seq],seq) for seq in seqs),key=lambda x:x[0])
    selected=[]
    for year in (2019,2022,2025):
        opts=[x for x in shared if x[0].startswith(f"{year}-")]
        if not opts: raise ValueError(f"No common target sequence in {year}")
        selected.append(min(opts,key=lambda x:(abs(pd.Timestamp(x[0])-pd.Timestamp(f"{year}-07-01")),x[0]))[0])
    date_sets=[]
    for path in target_files: date_sets.append(set(pd.read_csv(path,dtype={"date":str}).date.tolist()))
    sessions=sorted(set.intersection(*date_sets))
    tail=[d for d in sessions if d<="2026-10-30"][-40:]
    if len(tail)<40: raise ValueError("fewer than 40 aligned recent target sessions")
    recent_starts=tail[::5]
    # One dated snapshot before the 8 recent origins is a legal, conservative recent pool.
    selected.append(recent_starts[0])
    selected.append("2026-10-30")
    return sorted(set(selected))


def build_pool(out: Path, start: str = "2014-01-01", end: str = "2026-10-30",
               reuse_snapshots: bool = False) -> pd.DataFrame:
    import baostock as bs
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"BaoStock login failed: {login.error_msg}")
    try:
        snapshot_path=out/"industry_membership.csv"
        if reuse_snapshots:
            if not snapshot_path.exists(): raise FileNotFoundError(snapshot_path)
            industry_history=pd.read_csv(snapshot_path,dtype=str).fillna("")
            industries=industry_history.copy()
            basic_cols=[c for c in ("code","code_name","ipoDate","outDate","type","status") if c in industry_history]
            basic=industry_history[basic_cols].drop_duplicates("code",keep="last")
            asof_dates=sorted(set(industry_history.classification_asof))
            historical_supported="current_only" not in asof_dates
        else:
            snapshots=[]; historical_supported=True; asof_dates=experiment_dates()
            for asof in asof_dates:
                print(f"query industry snapshot as-of {asof}",flush=True)
                try: response=bs.query_stock_industry(date=asof)
                except TypeError:
                    historical_supported=False; response=bs.query_stock_industry()
                df=collect(response)
                print(f"industry snapshot {asof}: {len(df)} rows; fields={list(df.columns)}",flush=True)
                df["classification_asof"]=asof if historical_supported else "current_only"
                snapshots.append(df)
                if not historical_supported: break
            industries=pd.concat(snapshots,ignore_index=True,sort=False)
        if not {"code", "industry"}.issubset(industries.columns):
            raise ValueError(f"query_stock_industry fields changed: {industries.columns.tolist()}")
        coal_codes={str(row.code).lower() for row in industries.itertuples()
                    if A_SHARE.fullmatch(str(row.code).lower()) and is_coal(row.industry,getattr(row,"code_name",""))}
        candidate_codes=sorted(coal_codes | TARGETS)
        if not reuse_snapshots:
            # Query basic records per candidate; avoid the slow exchange-wide endpoint.
            basic_rows=[]
            for i,code in enumerate(candidate_codes,1):
                response=bs.query_stock_basic(code=code)
                if response.error_code!="0" and "未登录" in str(response.error_msg):
                    bs.login(); response=bs.query_stock_basic(code=code)
                row=collect(response)
                if not row.empty: basic_rows.extend(row.to_dict("records"))
                if i%20==0: print(f"stock basic candidates {i}/{len(candidate_codes)}",flush=True)
            basic=pd.DataFrame(basic_rows)
        required_basic = {"code", "code_name", "ipoDate", "outDate", "type", "status"}
        if not required_basic.issubset(basic.columns):
            raise ValueError(f"query_stock_basic fields changed: {basic.columns.tolist()}")
        basic = basic.drop_duplicates("code", keep="last")
        industry_cols = [x for x in ("code", "code_name", "industry", "industryClassification", "industry_classification", "classification_asof") if x in industries]
        # Preserve every label (including non-coal snapshots) so as-of membership can reflect exits.
        industry_history=industries[industries.code.astype(str).str.lower().isin(candidate_codes)].copy()
        if not reuse_snapshots:
            basic_extra=basic.rename(columns={c:f"{c}_basic" for c in basic.columns if c!="code"})
            industry_history=industry_history.merge(basic_extra,on="code",how="left")
            for col in basic.columns:
                if col=="code": continue
                extra=f"{col}_basic"
                if extra in industry_history:
                    if col in industry_history: industry_history[col]=industry_history[col].fillna(industry_history[extra])
                    else: industry_history[col]=industry_history[extra]
        coal = industries[[is_coal(industry,name) for industry,name in zip(industries.industry,industries.code_name)]].copy()
        coal["pool_reason"] = "BaoStock industry label matched coal/mining keyword"
        target = basic[basic.code.isin(TARGETS)].copy()
        target = target.merge(industries.drop_duplicates("code", keep="last")[industry_cols], on=[c for c in ("code", "code_name") if c in target.columns and c in industry_cols], how="left") if "industry" not in target.columns else target
        target["pool_reason"] = "required target stock; included for P1/P2"
        merged = pd.concat([coal, target], ignore_index=True, sort=False)
        basic_extra=basic.rename(columns={c:f"{c}_basic" for c in basic.columns if c!="code"})
        merged=merged.merge(basic_extra,on="code",how="left")
        # Keep one observation per security and dated classification; history availability is not an eligibility rule.
        for col in basic.columns:
            if col=="code": continue
            basic_col=f"{col}_basic"
            if basic_col in merged:
                if col in merged: merged[col]=merged[col].fillna(merged[basic_col])
                else: merged[col]=merged[basic_col]
        merged["code"] = merged.code.astype(str).str.lower()
        merged["security_type"] = merged.type.astype(str)
        merged = merged[merged.code.map(lambda x: bool(A_SHARE.fullmatch(x))) & merged.security_type.eq("1")].copy()
        if not TARGETS.issubset(set(merged.code)):
            raise ValueError(f"Target stocks missing from coal pool: {sorted(TARGETS - set(merged.code))}")
        merged = merged.sort_values(["code", "classification_asof"]).drop_duplicates(["code", "classification_asof"], keep="last")
        out.mkdir(parents=True, exist_ok=True)
        industry_history.to_csv(out / "industry_membership.csv", index=False)
        (out / "query_metadata.json").write_text(json.dumps({
            "source": "BaoStock query_stock_basic and query_stock_industry",
            "queried_at": date.today().isoformat(), "history_range": [start, end],
            "industry_fields": industries.columns.tolist(), "basic_fields": basic.columns.tolist(),
            "historical_classification_supported": historical_supported,
            "classification_snapshots_requested": asof_dates,
            "selection_keywords": COAL_WORDS,
            "limitations": ["BaoStock may not expose delisted firms in current industry snapshots; surviving-current-classification bias remains.",
                            "No historical industry membership data was inferred when API did not support dated queries."]
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        unique = merged.sort_values(["code", "classification_asof"]).drop_duplicates("code", keep="last")
        config_path=ROOT/"config/coal_stock_pool.json"
        config_path.write_text(json.dumps({"source":"BaoStock","query_date":date.today().isoformat(),
            "start_date":start,"end_date":end,"pool_size":int(unique.code.nunique()),
            "security_codes":sorted(unique.code.unique().tolist()),
            "target_codes":sorted(TARGETS),"industry_classification_asofs":asof_dates,
            "membership_file":"data/coal_pool/manifests/industry_membership.csv",
            "limitations":["Current-industry endpoint may omit delisted historical firms.",
                            "Dated classification snapshots are applied only at or before each backtest origin."]},
            ensure_ascii=False,indent=2),encoding="utf-8")
        return merged
    finally:
        bs.logout()


def download_pool(pool: pd.DataFrame, raw_dir: Path, processed_dir: Path, manifest_path: Path,
                  start: str, end: str, overwrite: bool = False) -> list[dict]:
    import baostock as bs
    raw_dir.mkdir(parents=True, exist_ok=True); processed_dir.mkdir(parents=True, exist_ok=True)
    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"BaoStock login failed: {lg.error_msg}")
    rows = []
    try:
        for _, stock in pool.drop_duplicates("code").iterrows():
            code = stock.code
            raw_path = raw_dir / f"{code.replace('.', '_')}.csv"
            out_date=str(stock.get("outDate", "") or "")
            if out_date and out_date < start:
                rows.append({"code":code,"name":stock.get("code_name",""),"industry":stock.get("industry",""),
                    "classification_asof":stock.get("classification_asof",""),"pool_reason":stock.get("pool_reason",""),
                    "ipo_date":stock.get("ipoDate",""),"out_date":out_date,"raw_rows":0,"valid_trading_rows":0,
                    "suspended_rows":0,"potential_unadjusted_jump_count_ge_20pct":0,"potential_jump_dates":"",
                    "raw_file":"","processed_file":"","exclusion_reason":"delisted before requested range"})
                print(f"{code}: excluded; delisted {out_date} before {start}",flush=True)
                continue
            if raw_path.exists() and not overwrite:
                raw = pd.read_csv(raw_path, dtype=str, encoding="utf-8-sig", keep_default_na=False)
            else:
                try: _, raw = query_rows(bs, code, start, end)
                except ValueError as exc:
                    if "没有返回记录" not in str(exc): raise
                    rows.append({"code":code,"name":stock.get("code_name",""),"industry":stock.get("industry",""),
                        "classification_asof":stock.get("classification_asof",""),"pool_reason":stock.get("pool_reason",""),
                        "ipo_date":stock.get("ipoDate",""),"out_date":out_date,"raw_rows":0,"valid_trading_rows":0,
                        "suspended_rows":0,"potential_unadjusted_jump_count_ge_20pct":0,"potential_jump_dates":"",
                        "raw_file":"","processed_file":"","exclusion_reason":"BaoStock returned no rows in requested date range"})
                    print(f"{code}: excluded; no rows in requested date range",flush=True)
                    continue
                atomic_write_csv(raw_path, raw, overwrite=True)
            absent = sorted(set(REQUIRED_RAW) - set(raw.columns))
            if absent:
                raise ValueError(f"{code} missing raw fields: {absent}")
            if set(raw.code.astype(str).str.lower()) != {code}:
                raise ValueError(f"{code} raw code mismatch")
            if set(raw.adjustflag.astype(str)) != {"3"}:
                raise ValueError(f"{code} raw data is not consistently unadjusted (adjustflag=3)")
            if raw.date.duplicated().any(): raise ValueError(f"{code} duplicate raw dates")
            raw = raw.copy()
            raw["date"] = pd.to_datetime(raw.date, errors="coerce")
            invalid_date_rows=int(raw.date.isna().sum())
            quote_dates=raw.date.dropna()
            raw = raw[raw.date.notna() & (raw.date.dt.strftime("%Y-%m-%d") <= end)].copy()
            raw["tradestatus"] = raw.tradestatus.astype(str)
            suspended = int(raw.tradestatus.eq("0").sum())
            traded = raw[raw.tradestatus.eq("1")].copy()
            invalid_status = int((~raw.tradestatus.isin(["0", "1"])).sum())
            for c in MODEL_COLS[1:]:
                traded[c] = pd.to_numeric(traded[c], errors="coerce")
            numeric_ok=np.isfinite(traded[MODEL_COLS[1:]].to_numpy(dtype=float)).all(axis=1)
            traded=traded.loc[numeric_ok].copy()
            positive=(traded[["open","high","low","close"]]>0).all(axis=1)
            nonnegative=(traded[["volume","amount"]]>=0).all(axis=1)
            bounds=(traded.high>=traded[["open","close","low"]].max(axis=1)) & (traded.low<=traded[["open","close","high"]].min(axis=1))
            invalid_price=int((~positive).sum()); invalid_bounds=int((~bounds).sum())
            invalid_volume_amount=int((~nonnegative).sum())
            bad=~(positive & nonnegative & bounds)
            bad_dates=traded.loc[bad,"date"].astype(str).tolist()
            traded=traded.loc[~bad].sort_values("date")
            traded["date"] = traded.date.dt.strftime("%Y-%m-%d")
            processed_path = processed_dir / raw_path.name
            traded[MODEL_COLS].to_csv(processed_path, index=False)
            close = traded.close.to_numpy(dtype=float); op = traded.open.to_numpy(dtype=float)
            jump = np.maximum(np.abs(op[1:] / close[:-1] - 1), np.abs(close[1:] / close[:-1] - 1)) if len(close) > 1 else np.array([])
            events = np.flatnonzero(jump >= 0.20) + 1
            rows.append({"code": code, "name": stock.get("code_name", ""), "industry": stock.get("industry", ""),
                "classification_asof": stock.get("classification_asof", ""), "pool_reason": stock.get("pool_reason", ""),
                "ipo_date": stock.get("ipoDate", ""), "out_date": stock.get("outDate", ""),
                "first_quote_date": quote_dates.min().strftime("%Y-%m-%d") if len(quote_dates) else "",
                "last_quote_date": quote_dates.max().strftime("%Y-%m-%d") if len(quote_dates) else "",
                "first_valid_trade_date":str(traded.date.iloc[0]) if len(traded) else "",
                "last_valid_trade_date":str(traded.date.iloc[-1]) if len(traded) else "",
                "raw_rows": int(len(raw)), "valid_trading_rows": int(len(traded)), "suspended_rows": suspended,
                "invalid_date_rows":invalid_date_rows,
                "invalid_status_rows":invalid_status,"invalid_numeric_rows":int((~numeric_ok).sum()),
                "nonpositive_ohlc_rows":invalid_price,"ohlc_boundary_rows":invalid_bounds,
                "negative_volume_amount_rows":invalid_volume_amount,"excluded_anomaly_dates":";".join(bad_dates),
                "potential_unadjusted_jump_count_ge_20pct": int(len(events)),
                "potential_jump_dates": ";".join(traded.date.iloc[events].astype(str).tolist()),
                "raw_file": str(raw_path.relative_to(ROOT)), "processed_file": str(processed_path.relative_to(ROOT)),
                "exclusion_reason": "fewer than 65 valid traded rows for one legal window" if len(traded) < 65 else ""})
            print(f"{code}: {len(traded)} valid rows; {suspended} suspended; jumps {len(events)}", flush=True)
    finally:
        bs.logout()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    # Also publish the requested top-level summary path and a readable audit.
    pd.DataFrame(rows).to_csv(ROOT/"data/pool_manifest.csv",index=False)
    audit_lines=["# 煤炭股票池数据审计","",f"下载范围：{start} 至 {end}；日线；不复权（adjustflag=3）。",
        "行业标签来自 BaoStock `query_stock_industry`，证券属性来自 `query_stock_basic`。按每个预测起点之前最近可用的分类快照确定成员；目标股票显式保留。",
        "价格跳变只做标记，不删除或回溯修复。停牌记录不进入模型序列。","",
        f"股票数：{len(rows)}；可用于构造至少一个 60 日输入加 5 日标签的股票：{sum(not r['exclusion_reason'] for r in rows)}。","",
        "|代码|名称|行业|IPO|首条行情|末条行情|有效记录|停牌|潜在≥20%跳变|排除原因|",
        "|---|---|---|---|---|---|---:|---:|---:|---|"]
    for r in rows:
        audit_lines.append(f"|{r['code']}|{r['name']}|{r['industry']}|{r['ipo_date']}|{r['first_quote_date']}|{r['last_quote_date']}|{r['valid_trading_rows']}|{r['suspended_rows']}|{r['potential_unadjusted_jump_count_ge_20pct']}|{r['exclusion_reason']}|")
    audit_lines += ["","当前 BaoStock 行业分类可能缺少已退市公司的历史归属，存在幸存者偏差。若 API 提供历史日期分类，则回测只使用不晚于该次 origin 的快照；若不支持，历史 P2 不使用当前分类扩张训练池，以避免事后分类泄漏。",""]
    (ROOT/"docs/COAL_POOL_AUDIT.md").write_text("\n".join(audit_lines),encoding="utf-8")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", default="2014-01-01"); p.add_argument("--end", default="2026-10-30")
    p.add_argument("--pool-dir", type=Path, default=ROOT / "data/coal_pool/manifests")
    p.add_argument("--raw-dir", type=Path, default=ROOT / "data/coal_pool/raw")
    p.add_argument("--processed-dir", type=Path, default=ROOT / "data/coal_pool/processed")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--reuse-snapshots", action="store_true", help="reuse the saved dated industry query")
    a = p.parse_args()
    pool = build_pool(a.pool_dir, a.start, a.end, a.reuse_snapshots)
    rows = download_pool(pool, a.raw_dir, a.processed_dir, a.pool_dir / "pool_manifest.csv", a.start, a.end, a.overwrite)
    print(f"pool securities={pool.code.nunique()}, processed={sum(not r['exclusion_reason'] for r in rows)}")


if __name__ == "__main__": main()
