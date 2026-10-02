"""Independent periodic finite-volume Burgers reference for solver cross-checks."""

from __future__ import annotations

import numpy as np


def rusanov_flux(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    f_left = 0.5 * left**2
    f_right = 0.5 * right**2
    speed = np.maximum(np.abs(left), np.abs(right))
    return 0.5 * (f_left + f_right) - 0.5 * speed * (right - left)


def burgers_fv_rhs(state: np.ndarray, dx: float, nu: float, forcing: np.ndarray | None = None) -> np.ndarray:
    """Periodic cell-centered FV RHS using Rusanov flux and centered diffusion."""
    q = np.asarray(state, dtype=np.float64)
    if q.ndim != 1 or q.size < 8 or dx <= 0 or nu <= 0:
        raise ValueError("invalid FV grid, state, or viscosity")
    right = np.roll(q, -1)
    left = np.roll(q, 1)
    face_right = rusanov_flux(q, right)
    face_left = np.roll(face_right, 1)
    result = -(face_right - face_left) / dx + nu * (right - 2 * q + left) / dx**2
    if forcing is not None:
        f = np.asarray(forcing, dtype=np.float64)
        if f.shape != q.shape:
            raise ValueError("forcing must have same shape as state")
        result += f
    return result


def evolve_burgers_fv(initial_state: np.ndarray, nu: float, dx: float, dt: float, steps: int, forcing: np.ndarray | None = None) -> np.ndarray:
    """RK4 finite-volume trajectory. Caller must choose a stable dt from CFL and diffusion limits."""
    q0 = np.asarray(initial_state, dtype=np.float64)
    if steps < 0 or dt <= 0:
        raise ValueError("steps must be nonnegative and dt positive")
    result = np.empty((steps + 1, q0.size), dtype=np.float64)
    result[0] = q0
    q = q0.copy()
    for i in range(steps):
        k1 = burgers_fv_rhs(q, dx, nu, forcing)
        k2 = burgers_fv_rhs(q + 0.5 * dt * k1, dx, nu, forcing)
        k3 = burgers_fv_rhs(q + 0.5 * dt * k2, dx, nu, forcing)
        k4 = burgers_fv_rhs(q + dt * k3, dx, nu, forcing)
        q = q + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        result[i + 1] = q
    return result
