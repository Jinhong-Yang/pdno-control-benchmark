"""Dirichlet heat equation reference formulas and discrete solver."""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded


def heat_eigenmode(x: np.ndarray, t: float, kappa: float, decay: float, mode: int = 1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if not (0.0 <= x).all() or not (x <= 1.0).all():
        raise ValueError("x must be in [0, 1]")
    if t < 0 or kappa <= 0 or decay < 0 or mode < 1:
        raise ValueError("invalid heat parameters")
    return np.sin(mode * np.pi * x) * np.exp(-(kappa * (mode * np.pi) ** 2 + decay) * t)


def heat_cn_step(state: np.ndarray, dx: float, dt: float, kappa: float, decay: float, forcing: np.ndarray | None = None) -> np.ndarray:
    """Crank–Nicolson step on cell-interior points with homogeneous Dirichlet ends."""
    u = np.asarray(state, dtype=np.float64)
    if u.ndim != 1 or u.size < 3 or dx <= 0 or dt <= 0 or kappa <= 0 or decay < 0:
        raise ValueError("invalid heat state or solver parameters")
    n = u.size
    r = kappa * dt / (2.0 * dx**2)
    c = decay * dt / 2.0
    lower_lhs = -r * np.ones(n - 1)
    diag_lhs = (1.0 + 2.0 * r + c) * np.ones(n)
    upper_lhs = -r * np.ones(n - 1)
    rhs = (1.0 - 2.0 * r - c) * u
    rhs[1:] += r * u[:-1]
    rhs[:-1] += r * u[1:]
    if forcing is not None:
        f = np.asarray(forcing, dtype=np.float64)
        if f.shape != u.shape:
            raise ValueError("forcing must have same shape as state")
        rhs += dt * f
    ab = np.zeros((3, n), dtype=np.float64)
    ab[0, 1:] = upper_lhs
    ab[1] = diag_lhs
    ab[2, :-1] = lower_lhs
    return solve_banded((1, 1), ab, rhs)
