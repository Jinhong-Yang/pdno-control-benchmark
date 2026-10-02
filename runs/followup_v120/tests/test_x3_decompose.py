import importlib.util
from pathlib import Path
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "x3_decompose.py"
spec = importlib.util.spec_from_file_location("x3_decompose", SCRIPT)
x3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(x3)


class X3SolverEquivalence(unittest.TestCase):
    def test_burgers_vectorized_reference_matches_teacher(self):
        from pdno.data.teacher_queries import _future_burgers
        rng = np.random.default_rng(20261002)
        coeff = rng.normal(0, 0.03, size=(2, 33))
        actions = rng.uniform(-0.1, 0.1, size=(2, 2))
        nus = np.array([0.012, 0.021])
        got = x3._burgers_rollout(coeff, actions, nus)
        basis = np.stack([np.ones(256)] + [fn(k, np.arange(256) / 256)
                          for k in range(1, 17) for fn in
                          (lambda k, x: np.sin(2*np.pi*k*x), lambda k, x: np.cos(2*np.pi*k*x))])
        for i in range(2):
            initial = coeff[i] @ basis
            expected = _future_burgers(initial, nus[i], actions[i])
            np.testing.assert_allclose(got[i], expected, rtol=1e-6, atol=2e-7)

    def test_heat_vectorized_reference_matches_teacher(self):
        from pdno.data.teacher_queries import _future_heat
        rng = np.random.default_rng(20261003)
        coeff = rng.normal(0, 0.02, size=(2, 32))
        actions = rng.uniform(0.2, 0.7, size=(2, 2))
        kappas = np.array([0.004, 0.009]); decays = np.array([0.02, 0.04])
        got = x3._heat_rollout(coeff, actions, kappas, decays)
        x = (np.arange(256) + 1) / 257
        basis = np.stack([np.sin(j*np.pi*x) - x*np.sin(j*np.pi) for j in range(1, 33)])
        for i in range(2):
            initial = coeff[i] @ basis
            expected = _future_heat(initial, kappas[i], decays[i], actions[i])
            np.testing.assert_allclose(got[i], expected, rtol=1e-6, atol=2e-7)

    def test_error_vector_identity_and_ratio_are_separate(self):
        target = np.array([1.0, -2.0, 0.5])
        prop = np.array([0.2, -0.1, 0.3])
        op = np.array([-0.4, 0.3, 0.1])
        total = prop + op
        self.assertAlmostEqual(float(np.sum((total)**2)),
                               float(np.sum(prop**2) + np.sum(op**2) + 2*np.sum(prop*op)))
        ratio = np.linalg.norm(prop) / np.linalg.norm(total)
        self.assertFalse(np.isclose(ratio, np.sum(prop**2) / np.sum(total**2)))


if __name__ == "__main__":
    unittest.main()
