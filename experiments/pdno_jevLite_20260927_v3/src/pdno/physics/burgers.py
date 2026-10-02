"""Periodic forced Burgers reference solvers used in the benchmark."""

from __future__ import annotations

import numpy as np


def periodic_grid(n: int) -> np.ndarray:
    if n < 8:
        raise ValueError("periodic grid must contain at least 8 points")
    return np.arange(n, dtype=np.float64) / n


def spectral_derivative(values: np.ndarray, order: int = 1) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 8:
        raise ValueError("values must be a one-dimensional periodic field with n >= 8")
    if order not in (1, 2):
        raise ValueError("only first and second derivatives are supported")
    k = 2.0 * np.pi * np.fft.fftfreq(values.size, d=1.0 / values.size)
    multiplier = 1j * k if order == 1 else -(k**2)
    return np.fft.ifft(multiplier * np.fft.fft(values)).real


def burgers_rhs(
    state: np.ndarray,
    nu: float,
    forcing: np.ndarray | None = None,
    *,
    dealias: bool = True,
) -> np.ndarray:
    """Compute -d_x(q^2/2) + nu*d_xx(q), with optional 2/3 dealiasing."""
    q = np.asarray(state, dtype=np.float64)
    if q.ndim != 1 or q.size < 8:
        raise ValueError("state must be one-dimensional with n >= 8")
    if not np.isfinite(nu) or nu <= 0:
        raise ValueError("nu must be finite and positive")
    k = 2.0 * np.pi * np.fft.fftfreq(q.size, d=1.0 / q.size)
    flux_hat = np.fft.fft(0.5 * q**2)
    if dealias:
        mode = np.fft.fftfreq(q.size) * q.size
        flux_hat[np.abs(mode) > q.size / 3.0] = 0.0
    convective = np.fft.ifft(1j * k * flux_hat).real
    diffusive = np.fft.ifft(-(k**2) * np.fft.fft(q)).real
    result = -convective + nu * diffusive
    if forcing is not None:
        forcing = np.asarray(forcing, dtype=np.float64)
        if forcing.shape != q.shape:
            raise ValueError("forcing must have the same shape as state")
        result = result + forcing
    return result


def burgers_convection_rhs(state: np.ndarray, forcing: np.ndarray | None = None, *, dealias: bool = True) -> np.ndarray:
    """Explicit nonlinear/forcing part for diffusion-split time integration."""
    q = np.asarray(state, dtype=np.float64)
    if q.ndim != 1 or q.size < 8:
        raise ValueError("state must be one-dimensional with n >= 8")
    k = 2.0 * np.pi * np.fft.fftfreq(q.size, d=1.0 / q.size)
    flux_hat = np.fft.fft(0.5 * q**2)
    if dealias:
        mode = np.fft.fftfreq(q.size) * q.size
        flux_hat[np.abs(mode) > q.size / 3.0] = 0.0
    result = -np.fft.ifft(1j * k * flux_hat).real
    if forcing is not None:
        forcing = np.asarray(forcing, dtype=np.float64)
        if forcing.shape != q.shape:
            raise ValueError("forcing must have the same shape as state")
        result += forcing
    return result


def burgers_imex_split_step(state: np.ndarray, nu: float, dt: float, forcing: np.ndarray | None = None) -> np.ndarray:
    """Strang diffusion split: exact half-step diffusion + RK4 convection + exact half-step diffusion."""
    q = np.asarray(state, dtype=np.float64)
    if nu <= 0 or dt <= 0 or q.ndim != 1 or q.size < 8:
        raise ValueError("invalid state or IMEX parameters")
    k = 2.0 * np.pi * np.fft.fftfreq(q.size, d=1.0 / q.size)
    diffusion_multiplier = np.exp(-nu * k**2 * (dt / 2.0))
    q_half = np.fft.ifft(diffusion_multiplier * np.fft.fft(q)).real
    q_advected = rk4_step(q_half, dt, burgers_convection_rhs, forcing)
    return np.fft.ifft(diffusion_multiplier * np.fft.fft(q_advected)).real


def rk4_step(state: np.ndarray, dt: float, rhs, *rhs_args, **rhs_kwargs) -> np.ndarray:
    if dt <= 0 or not np.isfinite(dt):
        raise ValueError("dt must be finite and positive")
    k1 = rhs(state, *rhs_args, **rhs_kwargs)
    k2 = rhs(state + 0.5 * dt * k1, *rhs_args, **rhs_kwargs)
    k3 = rhs(state + 0.5 * dt * k2, *rhs_args, **rhs_kwargs)
    k4 = rhs(state + dt * k3, *rhs_args, **rhs_kwargs)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def evolve_burgers(
    initial_state: np.ndarray,
    nu: float,
    dt: float,
    steps: int,
    forcing: np.ndarray | None = None,
) -> np.ndarray:
    """Strang/IMEX trajectory; returns (steps+1, n), including t=0."""
    q0 = np.asarray(initial_state, dtype=np.float64)
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    trajectory = np.empty((steps + 1, q0.size), dtype=np.float64)
    trajectory[0] = q0
    q = q0.copy()
    for j in range(steps):
        q = burgers_imex_split_step(q, nu, dt, forcing)
        trajectory[j + 1] = q
    return trajectory


def energy_rate(state: np.ndarray, nu: float, forcing: np.ndarray | None = None) -> float:
    """Continuous periodic energy balance right-hand side under mean quadrature."""
    q = np.asarray(state, dtype=np.float64)
    f = 0.0 if forcing is None else np.asarray(forcing, dtype=np.float64)
    qx = spectral_derivative(q, 1)
    return float(-nu * np.mean(qx**2) + np.mean(q * f))
