"""Export verified inference rows to course CSVs, or draft previews without an ID."""
import argparse, csv, json, math
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
FILES={"sh.601088":"601088_SH","sh.601225":"601225_SH","sz.000983":"000983_SZ"}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--student-id")
    p.add_argument("--output-dir",type=Path,default=ROOT/"results/predictions")
    p.add_argument("--dates-file",type=Path,default=ROOT/"config/prediction_dates.json")
    p.add_argument("--input",type=Path,default=ROOT/"results/predictions/prediction_details.csv")
    p.add_argument("--metadata",type=Path,default=ROOT/"results/predictions/metadata.json")
    args=p.parse_args()
    if not args.input.is_file(): raise FileNotFoundError(f"真实推理明细不存在：{args.input}；请先运行最终训练与推理")
    metadata=json.loads(args.metadata.read_text(encoding="utf-8"))
    dates=json.loads(args.dates_file.read_text(encoding="utf-8"))["dates"]
    df=pd.read_csv(args.input)
    if not {"stock","date","open"} <= set(df.columns): raise ValueError("推理明细必须含 stock,date,open")
    if not args.student_id: args.output_dir=args.output_dir/"draft"
    args.output_dir.mkdir(parents=True,exist_ok=True)
    written=[]
    for stock,suffix in FILES.items():
        part=df[df.stock==stock].copy()
        part["date"]=part["date"].astype(str)
        if part.date.tolist()!=dates or len(part)!=5: raise ValueError(f"{stock}: 推理日期必须与配置完全一致")
        vals=pd.to_numeric(part["open"],errors="coerce").to_numpy()
        if not all(math.isfinite(float(x)) and float(x)>0 for x in vals): raise ValueError(f"{stock}: Open 必须为有限正数")
        meta=next((m for m in metadata if m["stock"]==stock),None)
        if not meta or meta.get("training_cutoff")!="2026-10-30": raise ValueError(f"{stock}: 无法验证推理元数据或训练截止日期")
        prefix=args.student_id+"_" if args.student_id else ""
        name=f"{prefix}prediction_{suffix}.csv" if args.student_id else f"{stock.replace('.','_')}_draft_preview.csv"
        path=args.output_dir/name
        with path.open("w",encoding="utf-8",newline="") as f:
            w=csv.writer(f,lineterminator="\n"); w.writerow(["Date","Open"])
            w.writerows((d,format(float(v),".6f")) for d,v in zip(dates,vals))
        written.append(str(path.resolve()))
    print(json.dumps({"draft":not bool(args.student_id),"files":written},ensure_ascii=False,indent=2))

if __name__=="__main__": main()
