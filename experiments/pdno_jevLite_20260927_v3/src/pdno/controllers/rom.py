"""Finite-candidate, 16-state reduced-order predictive controller (B1)."""

from __future__ import annotations

import numpy as np

from pdno.controllers.linear import _measurements
from pdno.data.generate import _dirichlet_actuators, _periodic_actuators


def _burgers_basis(n: int = 256, modes: int = 8) -> tuple[np.ndarray, np.ndarray]:
    x = np.arange(n, dtype=np.float64) / n
    basis, derivative = [], []
    for k in range(1, modes + 1):
        basis.extend((np.sin(2 * np.pi * k * x), np.cos(2 * np.pi * k * x)))
        derivative.extend((2 * np.pi * k * np.cos(2 * np.pi * k * x),
                           -2 * np.pi * k * np.sin(2 * np.pi * k * x)))
    return np.asarray(basis), np.asarray(derivative)


def _heat_basis(n: int = 256, modes: int = 16) -> tuple[np.ndarray, np.ndarray]:
    x = (np.arange(n, dtype=np.float64) + 1) / (n + 1)
    basis = np.asarray([np.sin(k * np.pi * x) for k in range(1, modes + 1)])
    return basis, (np.pi * np.arange(1, modes + 1)) ** 2


def _fit_modes(obs: dict, pde: str, basis: np.ndarray) -> np.ndarray:
    x, y = _measurements(obs, pde)
    grid = (np.arange(basis.shape[1]) / basis.shape[1] if pde == "burgers"
            else (np.arange(basis.shape[1]) + 1) / (basis.shape[1] + 1))
    if x.size == 0:
        return np.zeros(basis.shape[0], dtype=np.float64)
    design = np.stack([np.interp(x, grid, row, period=1.0) if pde == "burgers"
                       else np.interp(x, grid, row, left=0.0, right=0.0) for row in basis], axis=1)
    return np.linalg.solve(design.T @ design + 1e-3 * np.eye(basis.shape[0]), design.T @ y)


def _project_force(pde: str, basis: np.ndarray) -> np.ndarray:
    n = basis.shape[1]
    actuator = _periodic_actuators(n) if pde == "burgers" else _dirichlet_actuators(n)
    scale = 2.0 / n
    return scale * basis @ actuator.T


def _burgers_rollout(coeff: np.ndarray, nu: float, force_modes: np.ndarray,
                     action: np.ndarray, basis: np.ndarray, derivative: np.ndarray,
                     horizon: int, tick_dt: float = 0.02) -> np.ndarray:
    n = basis.shape[1]
    wave = np.repeat(2 * np.pi * np.arange(1, 9), 2)
    forcing = action @ force_modes.T

    def rhs(c):
        u = c @ basis
        ux = c @ derivative
        nonlinear = -(u * ux)
        return (2.0 / n) * (nonlinear @ basis.T) - nu * wave**2 * c + forcing

    dt = tick_dt / 8
    out = []
    for _ in range(horizon):
        for _ in range(8):
            k1 = rhs(coeff)
            k2 = rhs(coeff + 0.5 * dt * k1)
            k3 = rhs(coeff + 0.5 * dt * k2)
            k4 = rhs(coeff + dt * k3)
            coeff = coeff + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        out.append(coeff @ basis[:, ::2])
    return np.asarray(out)


def rom_candidate_costs(obs: dict, pde: str, candidates: np.ndarray, goal: np.ndarray,
                        q_max: float, horizon: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Predict each of K fixed feasible actions with a 16-state Galerkin ROM."""
    actions = np.asarray(candidates, dtype=np.float64)
    previous = np.asarray(obs["previous_applied_action"], dtype=np.float64)
    if actions.shape != (10, 2):
        raise ValueError("B1 requires the common K=10 candidate set")
    if pde == "burgers":
        basis, derivative = _burgers_basis()
        coeff = _fit_modes(obs, pde, basis)
        nu = float(np.asarray(obs["material_context"])[0])
        force_modes = _project_force(pde, basis)
        fields = np.stack([_burgers_rollout(coeff.copy(), nu, force_modes, action, basis,
                                            derivative, horizon) for action in actions])
    elif pde == "heat":
        basis, eigenvalues = _heat_basis()
        coeff = _fit_modes(obs, pde, basis)
        kappa, decay = (float(v) for v in np.asarray(obs["material_context"])[:2])
        rates = kappa * eigenvalues + decay
        force_modes = _project_force(pde, basis)
        fields = np.empty((len(actions), horizon, 128), dtype=np.float64)
        x_truth = (np.arange(basis.shape[1]) + 1) / (basis.shape[1] + 1)
        x_model = (np.arange(128) + 1) / 129
        for j, action in enumerate(actions):
            state = coeff.copy()
            forcing = action @ force_modes.T
            for h in range(horizon):
                decay_factor = np.exp(-rates * 0.02)
                state = decay_factor * state + np.divide((1.0 - decay_factor) * forcing, rates,
                                                          out=0.02 * forcing.copy(), where=rates > 1e-12)
                fields[j, h] = np.interp(x_model, x_truth, state @ basis)
    else:
        raise ValueError(f"unknown PDE {pde}")
    errors = fields - np.asarray(goal, dtype=np.float64)[None, None, :]
    tracking = np.mean(errors**2, axis=(1, 2))
    effort = 0.01 * np.sum(actions**2, axis=1)
    slew = 0.05 * np.sum((actions - previous)**2, axis=1)
    if pde == "heat":
        violation_field = np.maximum(-fields, 0.0) ** 2 + np.maximum(fields - q_max, 0.0) ** 2
    else:
        violation_field = np.maximum(np.abs(fields) - q_max, 0.0) ** 2
    violation = 10.0 * np.mean(violation_field, axis=(1, 2))
    return tracking + effort + slew + violation, fields


def rom_select_action(obs: dict, pde: str, candidates: np.ndarray, goal: np.ndarray,
                     q_max: float, horizon: int = 8) -> tuple[np.ndarray, np.ndarray]:
    costs, _ = rom_candidate_costs(obs, pde, candidates, goal, q_max, horizon)
    return np.asarray(candidates)[int(np.argmin(costs))], costs
