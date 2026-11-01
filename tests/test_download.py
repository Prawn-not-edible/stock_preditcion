import csv, tempfile, unittest
import pandas as pd
from pathlib import Path
from scripts.download import load_stocks, audit_csv, atomic_write_csv, REQUIRED

class DownloadOfflineTests(unittest.TestCase):
    def test_config(self):
        stocks=load_stocks(Path(__file__).parents[1]/"config/stocks.json")
        self.assertEqual([x["query_code"] for x in stocks],["sh.601088","sh.601225","sz.000983"])
    def test_audit_and_missing_field_error(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"ok.csv"
            row={k:"1" for k in REQUIRED}; row.update(date="2025-01-02",code="sh.601088",adjustflag="3",tradestatus="0",high="2",low="1",open="1.5",close="1.6")
            with p.open("w",encoding="utf-8-sig",newline="") as f:
                w=csv.DictWriter(f,fieldnames=REQUIRED); w.writeheader(); w.writerow(row)
            report=audit_csv(p,"sh.601088")
            self.assertEqual(report["rows"],1); self.assertEqual(report["suspended_rows"],1)
            p.write_text("date,code\n2025-01-01,sh.601088\n",encoding="utf-8-sig")
            with self.assertRaisesRegex(ValueError,"缺少字段"): audit_csv(p,"sh.601088")
    def test_atomic_write_refuses_existing(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"x.csv"; p.write_text("existing")
            with self.assertRaises(FileExistsError): atomic_write_csv(p,pd.DataFrame([dict.fromkeys(REQUIRED, "")], columns=REQUIRED))
            self.assertEqual(p.read_text(),"existing")

if __name__=="__main__": unittest.main()
