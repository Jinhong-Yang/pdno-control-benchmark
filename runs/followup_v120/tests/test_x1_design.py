from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REV = ROOT.parent / "revision_v2"
OLD = ROOT.parents[1] / "experiments" / "pdno_jevLite_20260927_v3"
sys.path.insert(0, str(REV / "src"))
sys.path.insert(0, str(OLD / "src"))
sys.path.insert(0, str(ROOT / "src"))

from x1.design import candidates, objective, summarize_indices
from x1.solvers import heat_advance, heat_forecast
from pdno.data.teacher_queries import feasible_candidates, restrict_field, _future_heat, _cost
from pdno.evaluation.closed_loop import _advance


class X1DesignTests(unittest.TestCase):
    def test_k10_matches_archived_candidate_generator(self):
        prev = np.array([0.21, -0.13], dtype=np.float32)
        b0 = np.array([0.31, -0.04], dtype=np.float32)
        for pde in ("burgers", "heat"):
            np.testing.assert_array_equal(candidates(pde, prev, b0, 10),
                                          feasible_candidates(pde, prev, b0))

    def test_k50_cardinality_and_design_indices(self):
        prev = np.array([0.2, 0.3], dtype=np.float32)
        b0 = np.array([0.25, 0.4], dtype=np.float32)
        got = candidates("heat", prev, b0, 50)
        self.assertEqual(got.shape, (50, 2))
        np.testing.assert_array_equal(got[48], prev)  # appended hold
        np.testing.assert_array_equal(got[49], b0)    # appended B0
        self.assertTrue(np.all(np.abs(got - prev) <= 0.1000001))

    def test_heat_horizon_solver_matches_frozen_reference(self):
        archive = OLD / "data/train_validation_v3/validation/heat/trajectories.npz"
        with np.load(archive, allow_pickle=False) as z:
            state = z["state_true"][0, 12].astype(np.float64)
            kappa = float(z["kappa"][0])
            decay = float(z["decay"][0])
        acts = np.array([[0.2, 0.4], [0.5, 0.3]], dtype=np.float32)
        got = heat_forecast(np.broadcast_to(state, (2, 256)), acts, kappa, decay, 8)
        expected = np.stack([[restrict_field(v, "heat") for v in _future_heat(state, kappa, decay, a)]
                             for a in acts])
        np.testing.assert_allclose(got, expected, rtol=0, atol=2e-7)

    def test_objective_and_choice_summaries(self):
        fields = np.zeros((2, 128), dtype=np.float32)
        cost = objective(fields, np.zeros(128), np.zeros(2), np.zeros(2), 1.2, "burgers")
        self.assertEqual(cost, 0.0)
        s = summarize_indices(np.array([4, 9, 4]), 10)
        self.assertAlmostEqual(s["hold_selection_rate"], 2 / 3)
        self.assertAlmostEqual(s["b0_selection_rate"], 1 / 3)

    def test_x1_scorer_matches_archived_fp32_cost_contract(self):
        rng = np.random.default_rng(20261003)
        for pde, qmax in (("burgers", 1.2), ("heat", 2.1322593092918396)):
            field = rng.normal(size=(8, 128)).astype(np.float32)
            goal = rng.normal(size=128).astype(np.float32) if pde == "heat" else np.zeros(128, np.float32)
            action = np.array([.17, .33], dtype=np.float32)
            prev = np.array([.21, .25], dtype=np.float32)
            expected = _cost(field, goal, action, prev, qmax, q_min=0.0 if pde == "heat" else None)[0]
            actual = objective(field, goal, action, prev, qmax, pde)
            self.assertEqual(actual, expected)

    def test_heat_plant_advance_keeps_full_grid(self):
        archive = OLD / "data/train_validation_v3/validation/heat/trajectories.npz"
        with np.load(archive, allow_pickle=False) as z:
            state = z["state_true"][0, 12].astype(np.float32)
            material = np.array([z["kappa"][0], z["decay"][0]], dtype=np.float32)
        action = np.array([0.2, 0.4], dtype=np.float32)
        got = heat_advance(state, action, float(material[0]), float(material[1]))
        ref = _advance(state, "heat", action, material, 0, "validation", None)
        self.assertEqual(got.shape, (256,))
        np.testing.assert_allclose(got, ref, rtol=0, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
