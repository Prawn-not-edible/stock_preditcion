import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import residual_phase3 as rp


def synthetic_raw(n=260):
    t = np.arange(n, dtype=np.float64)
    close = 20.0 + 0.01 * t
    open_ = close + 0.02 * np.sin(t / 5)
    return np.column_stack((close + 0.3, close - 0.2, close,
                            1000 + 3 * t, 20000 + 7 * t, open_))


class ResidualTargetTests(unittest.TestCase):
    def test_anchor_uses_close_immediately_before_origin(self):
        raw = synthetic_raw()
        origin = 140
        mean, scale = rp.fit_volume_stats(raw, 100)
        x, anchor = rp.build_features(raw, origin, mean, scale)
        self.assertEqual(anchor, raw[origin - 1, 2])
        self.assertEqual(x.shape, (60, 6))
        self.assertAlmostEqual(float(x[-1, 2]), 0.0, places=7)
        changed = raw.copy()
        changed[origin:, 2] *= 5
        x_changed, anchor_changed = rp.build_features(changed, origin, mean, scale)
        np.testing.assert_array_equal(x, x_changed)
        self.assertEqual(anchor, anchor_changed)

    def test_target_is_exactly_next_five_open_returns(self):
        raw = synthetic_raw()
        origin = 140
        target, anchor = rp.build_target(raw, origin)
        np.testing.assert_allclose(target, raw[origin:origin + 5, 5] / raw[origin - 1, 2] - 1)
        self.assertEqual(anchor, raw[origin - 1, 2])
        self.assertEqual(target.shape, (5,))

    def test_zero_residual_restores_last_close_exactly(self):
        anchor = 51.9900016784668
        restored = rp.restore_open(anchor, np.zeros(5, dtype=np.float64))
        np.testing.assert_array_equal(restored, np.full(5, anchor, dtype=np.float64))

    def test_residual_mse_equals_restored_price_nmse(self):
        anchor = 28.08
        y = np.array([0.01, -0.02, 0.005, 0.0, 0.03])
        p = np.array([0.02, -0.01, 0.004, 0.001, 0.01])
        metrics = rp.residual_metrics(p, y, anchor)
        p_open, y_open = rp.restore_open(anchor, p), rp.restore_open(anchor, y)
        direct_nmse = np.mean(((p_open - y_open) / anchor) ** 2)
        self.assertAlmostEqual(metrics["residual_mse"], direct_nmse, places=14)
        self.assertAlmostEqual(metrics["nmse"], direct_nmse, places=14)

    def test_volume_scaler_uses_training_rows_only(self):
        raw = synthetic_raw()
        mean1, scale1 = rp.fit_volume_stats(raw, 100)
        changed = raw.copy(); changed[100:, 3:5] *= 100
        mean2, scale2 = rp.fit_volume_stats(changed, 100)
        np.testing.assert_array_equal(mean1, mean2)
        np.testing.assert_array_equal(scale1, scale2)


class ResidualModelTests(unittest.TestCase):
    def test_official_models_have_finite_gradients_and_five_outputs(self):
        x = torch.randn(2, 60, 6)
        target = torch.randn(2, 5)
        for name in rp.MODELS:
            with self.subTest(model=name):
                model = rp.make_model(name, small=True)
                pred = rp.output_return(model, x)
                self.assertEqual(tuple(pred.shape), (2, 5))
                loss = torch.nn.functional.mse_loss(pred, target)
                loss.backward()
                self.assertTrue(torch.isfinite(pred).all().item())
                self.assertTrue(torch.isfinite(loss).item())
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all().item()
                                    for p in model.parameters()))


class Phase2AlignmentTests(unittest.TestCase):
    def test_selected_development_and_recent_phase2_rows_match_exact_dates_and_truth(self):
        dev = rp.phase2_rows(ROOT / "results/remote_gpu_run/results/metrics/development_windows.csv")
        recent = rp.phase2_rows(ROOT / "results/remote_gpu_run/results/metrics/recent_windows.csv")
        for stock_cfg in rp.stock_configs():
            stock = stock_cfg["query_code"]
            df, raw = rp.matrix(ROOT / "data/processed" / stock_cfg["output_file"])
            manifest = rp.manifest_for(stock_cfg)
            for fold_id in rp.DEV_FOLDS:
                fold = next(f for f in manifest["splits"] if f["fold_id"] == fold_id)
                origin = int(fold["origin_index"])
                dates = ";".join(df.date.iloc[origin:origin + 5].astype(str))
                actual = raw[origin:origin + 5, 5]
                for model in rp.MODELS:
                    rp.assert_pair(stock, model, str(df.date.iloc[origin]), dates, actual,
                                   rp.load_v0_dev(stock, model, fold_id, dev))
            for origin in range(len(raw) - 40, len(raw) - 4, 5):
                dates = ";".join(df.date.iloc[origin:origin + 5].astype(str))
                actual = raw[origin:origin + 5, 5]
                for model in rp.MODELS:
                    rp.assert_pair(stock, model, str(df.date.iloc[origin]), dates, actual,
                                   rp.load_v0_recent(stock, model, dates, recent))


if __name__ == "__main__":
    unittest.main()
