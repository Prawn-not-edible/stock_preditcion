"""Run from repository root: python -m unittest discover -s tests -p "test_phase5_joint.py" -v"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import phase4_joint as p4
import residual_phase3 as p3
import phase5_joint as p5
import phase5_analyze as p5a


class Phase5ObjectiveTests(unittest.TestCase):
    def test_only_three_target_stocks_decide_checkpoint(self):
        losses = {"sh.601088": [.001, .003], "sh.601225": [.004],
                  "sz.000983": [.006], "sh.600121": [1.0]}
        macro, pool, by_stock = p5.score_by_stock(losses, p4.TARGETS)
        self.assertAlmostEqual(macro, (.002 + .004 + .006) / 3)
        self.assertGreater(pool, macro)
        self.assertEqual(set(by_stock), set(losses))

    def test_missing_required_target_raises(self):
        with self.assertRaisesRegex(ValueError, "required target"):
            p5.score_by_stock({"sh.601088": [.001]}, p4.TARGETS)

    def test_exact_baseline_is_a_tie_not_a_win(self):
        rows = []
        for stock in p4.TARGETS:
            for pool in p5.POOLS:
                for i in range(8):
                    rows.append({"pool": pool, "model": "DLinear", "stock": stock,
                                 "alpha": 0.0, "origin_date": f"T{i}",
                                 "nmse": .0004 + i * .00001})
        s = p5a.compare_rows(pd.DataFrame(rows))
        self.assertTrue((s.wins == 0).all())
        self.assertTrue((s.ties == 8).all())
        self.assertTrue((s.losses == 0).all())

    def test_nonzero_locked_alpha_uses_separate_alpha_zero_baseline(self):
        selected = []
        baseline = []
        for stock in p4.TARGETS:
            for pool in p5.POOLS:
                for i in range(8):
                    common = {"pool": pool, "model": "DLinear", "stock": stock,
                              "origin_date": f"T{i}"}
                    baseline.append({**common, "alpha": 0.0, "nmse": 0.0005})
                    selected.append({**common, "alpha": 0.5, "nmse": 0.0004})
        metrics = p5a.compare_rows(pd.DataFrame(selected), pd.DataFrame(baseline))
        self.assertTrue(np.allclose(metrics.improvement_vs_baseline, 0.2))
        self.assertTrue((metrics.wins == 8).all())

    def test_alpha_locked_from_development_only(self):
        vals = []
        for pool in p5.POOLS:
            for stock in p4.TARGETS:
                for fold in range(3):
                    vals.append({"pool": pool, "model": "TimeXer", "stock": stock,
                                 "origin_date": f"D{fold}", "alpha": 0.0, "nmse": 0.5})
                    vals.append({"pool": pool, "model": "TimeXer", "stock": stock,
                                 "origin_date": f"D{fold}", "alpha": 1.0, "nmse": 0.3})
        alpha = p5a.choose_alphas(pd.DataFrame(vals))
        self.assertTrue(all(x == 1.0 for x in alpha.values()))


class Phase5EndToEndFixture(unittest.TestCase):
    def test_pretraining_then_finetune_uses_matching_origin_and_training_only_data(self):
        # Short synthetic time series for a CPU integration test (no BaoStock required).
        n = 260
        dates = pd.bdate_range("2024-01-02", periods=n).strftime("%Y-%m-%d").tolist()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            files = {}
            for stock_idx, stock in enumerate(p4.TARGETS):
                t = np.arange(n, dtype=float)
                close = 20 + stock_idx * 4 + .015 * t + .03 * np.sin(t / 9 + stock_idx)
                open_ = close + .03 * np.sin(t / 6)
                frame = pd.DataFrame({"date": dates,
                    "high": np.maximum(close, open_) + .5,
                    "low": np.minimum(close, open_) - .5,
                    "close": close,
                    "volume": 10000 + stock_idx * 2000 + t * 3,
                    "amount": (10000 + stock_idx * 2000 + t * 3) * close,
                    "open": open_})
                path = root / f"{stock.replace('.', '_')}.csv"
                frame.to_csv(path, index=False)
                files[stock] = path
            origin, val_start = dates[240], dates[200]
            opts = SimpleNamespace(device="cpu", batch_size=64, seed=2026,
                                   lr=.001, epochs=2, patience=2,
                                   finetune_lr=.0001, finetune_epochs=2,
                                   finetune_patience=2)
            p2ckpt = root / "P2.pt"
            p2 = p5.fit_model("DLinear", "P2", p4.TARGETS, files,
                              origin, val_start, opts, p2ckpt, list(p4.TARGETS))
            self.assertGreater(p2["training_samples"], 0)
            self.assertEqual(set(p2["val_by_stock"]), set(p4.TARGETS))
            self.assertTrue(np.isfinite(p2["target_val_nmse"]))
            p3ckpt = root / "P3.pt"
            p3out = p5.fit_model("DLinear", "P3", [p4.TARGETS[0]], files,
                                 origin, val_start, opts, p3ckpt, [p4.TARGETS[0]], pretrained=p2ckpt)
            self.assertGreaterEqual(p3out["best_epoch"], 0)
            self.assertLessEqual(p3out["best_epoch"], opts.finetune_epochs)
            saved = torch.load(p3ckpt, map_location="cpu", weights_only=False)
            self.assertEqual(saved["fine_tune_from"], str(p2ckpt))
            self.assertEqual(set(saved["volume_stats"]), {p4.TARGETS[0]})
            ret, target, anchor, target_dates = p5.predict_from_checkpoint(
                p3ckpt, files, p4.TARGETS[0], origin, "cpu")
            self.assertEqual(len(ret), 5)
            self.assertEqual(target_dates, tuple(dates[240:245]))
            self.assertTrue(np.isfinite(ret).all())
            self.assertAlmostEqual(p3.residual_metrics(np.zeros(5), target, anchor)["nmse"],
                np.mean(np.asarray(target) ** 2), places=12)


if __name__ == "__main__":
    unittest.main()
