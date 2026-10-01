"""Matched-observation low-order LQR nominal controller and modal observer."""

from __future__ import annotations

import numpy as np
from functools import lru_cache
from scipy.linalg import solve_discrete_are

from pdno.controllers.actions import project_box_slew


def _measurements(obs: dict, pde: str) -> tuple[np.ndarray, np.ndarray]:
    n = obs["sensor_value"].shape[-1]
    valid = np.asarray(obs["sensor_mask"], dtype=bool)
    values = np.asarray(obs["sensor_value"], dtype=np.float64)
    if valid.ndim == 2:
        valid = valid[-1]
        values = values[-1]
    if pde == "burgers":
        x_s = np.arange(n) / n
        x_i = np.arange(64) / 64
    else:
        # The 16 measurements are interpolated at uniformly spaced truth-grid
        # indices 0..255; those samples represent cell centers (j+1)/257.
        # They are not a 16-node interior grid at j/17.
        x_s = np.linspace(1 / 257, 256 / 257, n)
        x_i = np.arange(1, 65) / 65
    positions = [x_s[valid]]
    observations = [values[valid]]
    if bool(obs["image_valid"]):
        image = np.asarray(obs["instrument_image"], dtype=np.float64).squeeze(0)
        mask = np.asarray(obs["image_mask"], dtype=bool).squeeze(0)
        col_valid = mask.mean(axis=0) >= 0.5
        ivals = np.zeros(image.shape[1], dtype=np.float64)
        for col in np.flatnonzero(col_valid):
            ivals[col] = np.median(image[mask[:, col], col])
        ivals = (ivals - 0.5) * (3.0 if pde == "burgers" else 2.0)
        positions.append(x_i[col_valid])
        observations.append(ivals[col_valid])
    x = np.concatenate(positions)
    y = np.concatenate(observations)
    return x, y


def _basis(pde: str, x: np.ndarray, modes: int = 7) -> np.ndarray:
    if pde == "burgers":
        cols = [np.ones_like(x)]
        for k in range(1, modes + 1):
            cols.extend((np.sin(2 * np.pi * k * x), np.cos(2 * np.pi * k * x)))
        return np.stack(cols, axis=1)
    return np.stack([np.sin(k * np.pi * x) for k in range(1, modes + 1)], axis=1)


def estimate_modes(obs: dict, pde: str, ridge: float = 1e-3, modes: int = 7) -> np.ndarray:
    x, y = _measurements(obs, pde)
    if x.size == 0:
        return np.zeros(1 + 2 * modes if pde == "burgers" else modes)
    design = _basis(pde, x, modes)
    gram = design.T @ design + ridge * np.eye(design.shape[1])
    return np.linalg.solve(gram, design.T @ y)


def _dlqr(a: np.ndarray, b: np.ndarray, q: np.ndarray | None = None, r: np.ndarray | None = None) -> np.ndarray:
    q = np.eye(a.shape[0]) if q is None else q
    r = 0.1 * np.eye(b.shape[1]) if r is None else r
    p = solve_discrete_are(a, b, q, r)
    return np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a)


@lru_cache(maxsize=8192)
def _burgers_gain(nu: float) -> np.ndarray:
    modes=7
    rates=np.repeat((2*np.pi*np.arange(1,modes+1))**2*nu,2)
    a=np.diag(np.exp(-rates*.02))
    samples=np.arange(256)/256
    actuator=np.empty((2,samples.size))
    for j,center in enumerate((.25,.75)):
        d=np.minimum(np.abs(samples-center),1-np.abs(samples-center))
        actuator[j]=np.exp(-.5*(d/.08)**2); actuator[j]-=actuator[j].mean(); actuator[j]/=np.max(np.abs(actuator[j]))
    modal=_basis("burgers",samples,modes)[:,1:]
    b_cont=2*modal.T@actuator.T/samples.size
    b=np.diag((1-np.exp(-rates*.02))/rates)@b_cont
    return _dlqr(a,b)


@lru_cache(maxsize=8192)
def _heat_gain(kappa: float, decay: float) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    modes=7
    rates=kappa*(np.pi*np.arange(1,modes+1))**2+decay
    a=np.diag(np.exp(-rates*.02))
    xgrid=(np.arange(256)+1)/257
    actuator=np.stack([np.exp(-.5*((xgrid-c)/.1)**2) for c in (.3,.7)],axis=1)
    modal=_basis("heat",xgrid,modes)
    b_cont=2*modal.T@actuator/xgrid.size
    b=np.diag((1-np.exp(-rates*.02))/rates)@b_cont
    return _dlqr(a,b),b_cont,rates


def nominal_lqr_action(obs: dict, pde: str, previous_applied: np.ndarray, *, ridge: float = 1e-3) -> np.ndarray:
    """Low-order discrete LQR action with the same causal sensor/image snapshot."""
    modes = 7
    coefficients = estimate_modes(obs, pde, ridge=ridge, modes=modes)
    if pde == "burgers":
        nu = float(np.asarray(obs["material_context"])[0])
        gain=_burgers_gain(nu)
        x = coefficients[1:]
        action = -gain @ x
        low, high, slew = -1.0, 1.0, 0.15
    else:
        kappa, decay = (float(v) for v in np.asarray(obs["material_context"])[:2])
        gain,b_cont,rates=_heat_gain(kappa,decay)
        xgrid=(np.arange(256)+1)/257
        modal=_basis(pde,xgrid,modes)
        goal = np.asarray(obs["goal_field"], dtype=np.float64)
        goal_coeff = np.linalg.lstsq(modal, goal, rcond=None)[0]
        steady_action = np.linalg.lstsq(b_cont, rates * goal_coeff, rcond=None)[0]
        error = coefficients - goal_coeff
        action = steady_action - gain @ error
        low, high, slew = 0.0, 1.0, 0.10
    return project_box_slew(action, np.asarray(previous_applied), low, high, slew)
