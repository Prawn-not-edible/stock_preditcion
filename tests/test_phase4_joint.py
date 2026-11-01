import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import phase4_joint as p4
import residual_phase3 as p3


class MultiStockTimeIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, cls.raw = p4.read_processed(ROOT / "data/processed/sh_601088.csv")
        cls.path = ROOT / "data/processed/sh_601088.csv"

    def test_train_labels_end_before_common_cutoff(self):
        cutoff = self.df.date.iloc[260]
        ds = p4.MultiStockResidualDataset({"sh.601088": self.path}, ["sh.601088"], cutoff)
        for dates in ds.target_dates:
            self.assertLessEqual(dates[-1], ds.train_cutoffs["sh.601088"])
        self.assertEqual(ds.per_stock_stats["sh.601088"]["train_end_index"], 260)

    def test_validation_labels_strictly_precede_prediction_origin(self):
        val_start, origin = self.df.date.iloc[250], self.df.date.iloc[380]
        ds = p4.MultiStockResidualDataset({"sh.601088": self.path}, ["sh.601088"],
                                         val_start, val_start, origin)
        for dates in ds.target_dates:
            self.assertLess(dates[-1], origin)

    def test_volume_scaler_uses_only_train_rows(self):
        cutoff = self.df.date.iloc[260]
        ds = p4.MultiStockResidualDataset({"sh.601088": self.path}, ["sh.601088"], cutoff)
        mean, scale = p3.fit_volume_stats(self.raw, 260)
        np.testing.assert_allclose(ds.per_stock_stats["sh.601088"]["mean"], mean)
        np.testing.assert_allclose(ds.per_stock_stats["sh.601088"]["scale"], scale)

    def test_samples_keep_stock_identity_and_balanced_weights(self):
        cutoff = self.df.date.iloc[260]
        files = {"sh.601088": self.path, "sh.601225": self.path}
        ds = p4.MultiStockResidualDataset(files, list(files), cutoff)
        sampler = ds.balanced_sampler(2026)
        self.assertEqual(set(ds.stock_ids), set(files))
        self.assertAlmostEqual(float(sampler.weights[ds.stock_ids == "sh.601088"].sum()),
                               float(sampler.weights[ds.stock_ids == "sh.601225"].sum()))

    def test_three_official_models_have_finite_outputs_and_gradients(self):
        x = torch.randn(2, p3.SEQ_LEN, 6)
        for name in p3.MODELS:
            model = p3.make_model(name, small=True)
            pred = p3.output_return(model, x)
            self.assertEqual(tuple(pred.shape), (2, p3.PRED_LEN))
            loss = pred.square().mean()
            loss.backward()
            self.assertTrue(torch.isfinite(pred).all())
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_shared_historical_windows_are_exactly_paired(self):
        files = {s: ROOT / "data/processed" / f"{s.replace('.', '_')}.csv" for s in p4.TARGETS}
        dev = p4.choose_dev_dates(files)
        recent = p4.choose_recent_dates(files)
        self.assertEqual(len(dev), 3)
        self.assertEqual(len(recent), 8)
        self.assertTrue(all(len(set(dates)) == 5 for _, dates in dev + recent))

    def test_recent_common_targets_match_all_phase3_p0_pairs(self):
        windows = {";".join(days) for _, days in p4.choose_recent_dates(
            {s: ROOT / "data/processed" / f"{s.replace('.', '_')}.csv" for s in p4.TARGETS})}
        old = pd.read_csv(ROOT / "results/residual_phase3/recent_v1_windows.csv")
        for stock in p4.TARGETS:
            for model in p4.MODELS:
                found = set(old[(old.stock == stock) & (old.model == model)].target_dates)
                self.assertTrue(windows.issubset(found))


if __name__ == "__main__":
    unittest.main()
