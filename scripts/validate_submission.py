"""Validate course prediction CSVs and their final-inference provenance."""
import argparse,csv,json,math
from datetime import date
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FILES={"sh.601088":"601088_SH","sh.601225":"601225_SH","sz.000983":"000983_SZ"}

def validate(path:Path,dates:list[str])->dict:
    raw=path.read_bytes()
    raw.decode("utf-8")  # reject BOM is allowed by UTF-8 technically; file must be UTF-8
    with path.open("r",encoding="utf-8",newline="") as f: rows=list(csv.reader(f))
    if not rows or rows[0]!=["Date","Open"]: raise ValueError(f"{path}: header must be Date,Open")
    if len(rows)!=6: raise ValueError(f"{path}: expected 5 data rows, found {len(rows)-1}")
    got=[r[0] for r in rows[1:]]
    for d in got: date.fromisoformat(d)
    if got!=dates or len(set(got))!=5: raise ValueError(f"{path}: dates mismatch, duplicate or unordered")
    vals=[]
    for row in rows[1:]:
        if len(row)!=2: raise ValueError(f"{path}: unexpected columns or index")
        v=float(row[1])
        if not math.isfinite(v) or v<=0: raise ValueError(f"{path}: Open must be finite and positive")
        vals.append(v)
    return {"file":str(path.resolve()),"rows":len(vals),"dates":got,"valid":True,"sha256":__import__('hashlib').sha256(raw).hexdigest()}

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--student-id"); p.add_argument("--output-dir",type=Path,default=ROOT/"results/predictions"); p.add_argument("--dates-file",type=Path,default=ROOT/"config/prediction_dates.json"); p.add_argument("--draft",action="store_true")
    args=p.parse_args(); cfg=json.loads(args.dates_file.read_text(encoding="utf-8")); dates=cfg["dates"]
    if cfg.get("trading_days_verified")!=dates: raise ValueError("五个日期尚未全部通过交易日历核验")
    meta=json.loads((ROOT/"results/predictions/metadata.json").read_text(encoding="utf-8"))
    if len(meta)!=3: raise ValueError("metadata.json must contain all three stocks")
    reports=[]
    for stock,suffix in FILES.items():
        m=next((x for x in meta if x.get("stock")==stock),None)
        if not m or m.get("training_cutoff")!="2026-10-30" or m.get("last_input_date")!="2026-10-30": raise ValueError(f"{stock}: provenance cutoff is not 2026-10-30")
        if m.get("prediction_source") not in ("trained_model","LastCloseBaseline"): raise ValueError(f"{stock}: prediction source is not a recognized executed pipeline")
        if args.draft or not args.student_id:
            path=args.output_dir/"draft"/f"{stock.replace('.','_')}_draft_preview.csv"
        else:
            path=args.output_dir/f"{args.student_id}_prediction_{suffix}.csv"
        reports.append(validate(path,dates))
    report={"all_valid":True,"formal_filenames":bool(args.student_id and not args.draft),"checked_on":date.today().isoformat(),"files":reports}
    out=ROOT/"results/predictions/submission_validation.json"; out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
