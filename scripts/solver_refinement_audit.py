from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pdno.physics.burgers import evolve_burgers, periodic_grid
from pdno.physics.finite_volume import evolve_burgers_fv
from pdno.physics.heat import heat_cn_step, heat_eigenmode


def _interp_periodic_cell_centers(values: np.ndarray, x_query: np.ndarray) -> np.ndarray:
    n = values.size
    x = (np.arange(n) + 0.5) / n
    xp = np.r_[x[-1] - 1.0, x, x[0] + 1.0]
    fp = np.r_[values[-1], values, values[0]]
    return np.interp(x_query, xp, fp)


def main() -> int:
    t_end, nu = 0.02, 0.02
    x_ref = periodic_grid(512)
    q0 = 0.15 * np.sin(2 * np.pi * x_ref) + 0.05 * np.cos(4 * np.pi * x_ref)
    spectral_ref = evolve_burgers(q0, nu, 2.5e-4, round(t_end / 2.5e-4))[-1]
    fv_errors = {}
    mass_errors = {}
    for n in (128, 256, 512):
        x_cell = (np.arange(n) + 0.5) / n
        initial = 0.15 * np.sin(2 * np.pi * x_cell) + 0.05 * np.cos(4 * np.pi * x_cell)
        stable_dt = min(2.5e-4, 0.1 / nu / n**2)
        steps = int(np.ceil(t_end / stable_dt))
        dt = t_end / steps
        trajectory = evolve_burgers_fv(initial, nu, 1 / n, dt, steps)
        predicted = _interp_periodic_cell_centers(trajectory[-1], x_ref)
        fv_errors[str(n)] = float(np.sqrt(np.mean((predicted - spectral_ref) ** 2)) / np.sqrt(np.mean(spectral_ref**2)))
        mass_errors[str(n)] = float(abs(np.mean(trajectory[-1]) - np.mean(initial)))

    n = 256
    dx = 1 / (n + 1)
    x = dx * np.arange(1, n + 1)
    t, kappa, decay, dt = 0.05, 0.01, 0.3, 2.5e-4
    u = np.sin(np.pi * x)
    for _ in range(round(t / dt)):
        u = heat_cn_step(u, dx, dt, kappa, decay)
    exact = heat_eigenmode(x, t, kappa, decay)
    heat_relative_error = float(np.linalg.norm(u - exact) / np.linalg.norm(exact))

    result = {
        "pde": "smooth forced-periodic Burgers without external forcing for cross-solver trajectory check",
        "burgers": {"nu": nu, "t_end": t_end, "spectral_reference_grid": 512,
                    "fv_relative_l2_vs_spectral_by_grid": fv_errors, "fv_absolute_mass_errors": mass_errors,
                    "passes_1pct_at_grid_256": fv_errors["256"] <= 0.01,
                    "grid_error_decreases_128_256_512": fv_errors["512"] < fv_errors["256"] < fv_errors["128"]},
        "heat": {"grid": n, "t": t, "kappa": kappa, "decay": decay, "relative_l2_vs_analytic": heat_relative_error,
                 "passes_1e-4": heat_relative_error <= 1e-4},
    }
    (ROOT / "evidence" / "solver_refinement_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if (result["burgers"]["grid_error_decreases_128_256_512"]
                 and result["burgers"]["passes_1pct_at_grid_256"] and result["heat"]["passes_1e-4"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
