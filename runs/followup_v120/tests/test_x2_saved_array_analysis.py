import importlib.util
from pathlib import Path
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "x2_saved_array_analysis.py"
SPEC = importlib.util.spec_from_file_location("x2_saved_array_analysis", SCRIPT)
X2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(X2)


class QuantileSummaryTests(unittest.TestCase):
    def test_strict_threshold_counts_values_above_two_times_median(self):
        result = X2.quantile_summary([1, 1, 1, 1, 2, 2, 3, 4, 4, 5])
        self.assertEqual(result["median_ms"], 2)
        self.assertEqual(result["threshold_2x_median_ms"], 4)
        self.assertEqual(result["above_count"], 1)
        self.assertEqual(result["above_share"], .1)
        self.assertTrue(result["p99_above_threshold"])

    def test_p99_flag_can_be_false_when_tail_stops_at_threshold(self):
        result = X2.quantile_summary([1, 1, 1, 1, 2, 2, 3, 4, 4, 4])
        self.assertFalse(result["p99_above_threshold"])
        self.assertEqual(result["above_count"], 0)

    def test_invalid_vectors_fail_closed(self):
        for values in ([], [1, np.nan], [1, -1], [[1, 2]]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                X2.quantile_summary(values)


class SavedInventoryTests(unittest.TestCase):
    def test_profile_and_latency_inventories_have_frozen_counts(self):
        tidy, totals = X2.profile_rows()
        self.assertEqual(len(totals), 16)
        self.assertEqual(len(tidy), 160)
        primary, followup = X2.latency_rows()
        self.assertEqual(len(primary), 120)
        self.assertEqual(len(followup), 360)
        self.assertEqual(len(X2.matched_comparison(primary, followup)), 36)


if __name__ == "__main__":
    unittest.main()
