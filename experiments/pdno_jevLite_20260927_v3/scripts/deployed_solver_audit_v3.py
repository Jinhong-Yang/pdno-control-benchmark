"""Forced 200-tick solver checks at the deployed truth grid and action timing."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pdno.controllers.actions import project_box_slew
from pdno.data.generate import _dirichlet_actuators, _periodic_actuators
from pdno.physics.burgers import burgers_imex_split_step, periodic_grid
from pdno.physics.heat import heat_cn_step


def burgers(dt: float, initial: np.ndarray, nu: float, actions: np.ndarray, basis: np.ndarray) -> np.ndarray:
    state = initial.copy()
    substeps = round(0.02 / dt)
    assert np.isclose(substeps * dt, 0.02)
    for action in actions:
        forcing = action @ basis + 0.005
        for _ in range(substeps):
            state = burgers_imex_split_step(state, nu, dt, forcing)
    return state


def heat(dt: float, initial: np.ndarray, kappa: float, decay: float,
         actions: np.ndarray, basis: np.ndarray, dx: float) -> np.ndarray:
    state = initial.copy()
    substeps = round(0.02 / dt)
    assert np.isclose(substeps * dt, 0.02)
    for action in actions:
        forcing = action @ basis
        for _ in range(substeps):
            state = heat_cn_step(state, dx, dt, kappa, decay, forcing)
    return state


def rel_l2(actual: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(actual - reference) / max(np.linalg.norm(reference), 1e-12))


def main() -> int:
    rng = np.random.default_rng(20260927)
    n, ticks = 256, 200
    prev = np.zeros(2)
    proposed = np.column_stack((0.10 * np.sin(np.arange(ticks) / 19),
                                0.10 * np.cos(np.arange(ticks) / 23)))
    actions = np.stack([project_box_slew(a, prev, -1.0, 1.0, 0.15) for a in proposed])
    prev = np.full(2, 0.4)
    proposed_heat = 0.4 + np.column_stack((0.06 * np.sin(np.arange(ticks) / 19),
                                           0.06 * np.cos(np.arange(ticks) / 23)))
    heat_actions = np.stack([project_box_slew(a, prev, 0.0, 1.0, 0.1) for a in proposed_heat])

    x = periodic_grid(n)
    q0 = 0.15 * np.sin(2 * np.pi * x) + 0.05 * np.cos(4 * np.pi * x)
    nu = 0.02
    burgers_basis = _periodic_actuators(n)
    q_coarse = burgers(2.5e-4, q0, nu, actions, burgers_basis)
    q_fine = burgers(1.25e-4, q0, nu, actions, burgers_basis)
    expected_mean = np.mean(q0) + 200 * 0.02 * 0.005
    burgers_mean_error = float(abs(np.mean(q_coarse) - expected_mean))

    dx, kappa, decay = 1.0 / (n + 1), 0.01, 0.3
    xh = dx * np.arange(1, n + 1)
    u0 = 0.2 * np.sin(np.pi * xh) + 0.05 * np.sin(3 * np.pi * xh)
    heat_basis = _dirichlet_actuators(n)
    u_coarse = heat(0.02, u0, kappa, decay, heat_actions, heat_basis, dx)
    u_fine = heat(0.01, u0, kappa, decay, heat_actions, heat_basis, dx)
    heat_no_source = heat(0.02, u0, kappa, decay, np.zeros_like(heat_actions), heat_basis, dx)

    result = {
        "protocol": "v3 forced deployed-grid 200 tick audit; thresholds frozen in v3_preregistration.yaml before run",
        "test_opened": False,
        "burgers": {
            "grid": n, "outer_ticks": ticks, "outer_dt": 0.02, "nu": nu,
            "relative_l2_dt_0.00025_vs_0.000125": rel_l2(q_coarse, q_fine),
            "mean_error_vs_initial_plus_integrated_mean_source": burgers_mean_error,
            "threshold_relative_l2": 0.01, "threshold_mean_error": 1.0e-10,
            "finite": bool(np.isfinite(q_coarse).all() and np.isfinite(q_fine).all()),
        },
        "heat": {
            "grid": n, "outer_ticks": ticks, "outer_dt": 0.02, "kappa": kappa, "decay": decay,
            "relative_l2_dt_0.02_vs_0.01": rel_l2(u_coarse, u_fine),
            "max_abs_no_source_minus_forced": float(np.max(np.abs(heat_no_source - u_coarse))),
            "positive_forcing_increases_mean": bool(np.mean(u_coarse) > np.mean(heat_no_source)),
            "threshold_relative_l2": 0.05,
            "finite": bool(np.isfinite(u_coarse).all() and np.isfinite(u_fine).all()),
        },
        "passed": bool(
            np.isfinite(q_coarse).all() and np.isfinite(q_fine).all()
            and rel_l2(q_coarse, q_fine) <= 0.01 and burgers_mean_error <= 1.0e-10
            and np.isfinite(u_coarse).all() and np.isfinite(u_fine).all()
            and rel_l2(u_coarse, u_fine) <= 0.05 and np.mean(u_coarse) > np.mean(heat_no_source)
        ),
    }
    path = ROOT / "evidence" / "deployed_solver_audit_v3.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
