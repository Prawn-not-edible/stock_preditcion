import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from prepare_stocks import get_folds, MODEL_COLS

class Phase2SplitTests(unittest.TestCase):
    def test_six_expanding_origins_reserve_recent_180(self):
        n=3043; folds=get_folds(n)
        self.assertEqual(len(folds),6)
        origins=[f["origin_index"] for f in folds]
        self.assertEqual(origins,sorted(set(origins)))
        self.assertGreaterEqual(folds[0]["segments"]["train"][1],756)
        self.assertEqual(folds[0]["segments"]["val"][1]-folds[0]["segments"]["val"][0],126)
        self.assertLessEqual(folds[-1]["segments"]["test"][1],n-180)
        self.assertEqual(MODEL_COLS[-1],"open")

    def test_insufficient_history_fails_clearly(self):
        with self.assertRaisesRegex(ValueError,"insufficient"):
            get_folds(900)

if __name__=="__main__": unittest.main()
