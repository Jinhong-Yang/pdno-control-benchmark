from __future__ import annotations

import numpy as np
import torch
from scipy.linalg import solve_banded

from pdno.data.generate import _dirichlet_actuators, _periodic_actuators
from pdno.data.teacher_queries import restrict_field
from pdno.physics.heat import heat_cn_step


class BurgersHorizonCUDA:
    """FP64, graph-replayed copy of the frozen 256-point Burgers reference scheme."""

    def __init__(self, nu: np.ndarray, K: int, n: int = 256):
        self.B, self.K, self.n = len(nu), int(K), int(n)
        self.dt = 2.5e-4
        dev = "cuda"
        self.q = torch.zeros((self.B, self.K, n), device=dev, dtype=torch.float64)
        self.force = torch.zeros_like(self.q)
        k = 2 * np.pi * np.fft.fftfreq(n, d=1 / n)
        mode = np.fft.fftfreq(n) * n
        dk = 1j * k
        dk[np.abs(mode) > n / 3] = 0
        self.dk = torch.tensor(dk, device=dev, dtype=torch.complex128)
        self.mul = torch.exp(-torch.tensor(nu, device=dev, dtype=torch.float64)[:, None, None]
                             * torch.tensor(k * k, device=dev) * self.dt / 2)
        self.basis = torch.tensor(_periodic_actuators(n), device=dev, dtype=torch.float64)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self._outer_step()
            self._outer_step()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._outer_step()

    def rhs(self, q):
        return -torch.fft.ifft(self.dk * torch.fft.fft(0.5 * q * q, dim=-1), dim=-1).real + self.force

    def _outer_step(self):
        q = self.q
        for _ in range(80):
            q = torch.fft.ifft(self.mul * torch.fft.fft(q, dim=-1), dim=-1).real
            k1 = self.rhs(q)
            k2 = self.rhs(q + 0.5 * self.dt * k1)
            k3 = self.rhs(q + 0.5 * self.dt * k2)
            k4 = self.rhs(q + self.dt * k3)
            q = q + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            q = torch.fft.ifft(self.mul * torch.fft.fft(q, dim=-1), dim=-1).real
        self.q.copy_(q)

    @torch.no_grad()
    def forecast(self, initial: np.ndarray, actions: np.ndarray, H: int) -> np.ndarray:
        x = torch.as_tensor(initial, device="cuda", dtype=torch.float64)
        a = torch.as_tensor(actions, device="cuda", dtype=torch.float64)
        if x.shape != (self.B, self.n) or a.shape != (self.B, self.K, 2):
            raise ValueError("initial/action batch shape differs from fixed solver allocation")
        if not 1 <= H <= 16:
            raise ValueError("X1 horizon must be 1 through 16")
        self.q.copy_(x[:, None, :].expand(-1, self.K, -1))
        self.force.copy_(a @ self.basis)
        outputs = []
        for _ in range(H):
            self.graph.replay()
            outputs.append(self.q[:, :, ::2].float().clone())
        return torch.stack(outputs, dim=2).cpu().numpy()

    @torch.no_grad()
    def advance(self, initial: np.ndarray, actions: np.ndarray,
                extra_force: np.ndarray | None = None) -> np.ndarray:
        """Advance B independent plant states once (input actions shape [B,2])."""
        if self.K != 1:
            raise ValueError("plant advancement requires a K=1 solver")
        x = torch.as_tensor(initial, device="cuda", dtype=torch.float64)
        a = torch.as_tensor(actions, device="cuda", dtype=torch.float64)
        self.q.copy_(x[:, None, :])
        self.force.copy_(a[:, None, :] @ self.basis)
        if extra_force is not None:
            self.force.add_(torch.as_tensor(extra_force, device="cuda", dtype=torch.float64)[:, None, :])
        self.graph.replay()
        return self.q[:, 0].float().cpu().numpy()


def heat_forecast(initial: np.ndarray, actions: np.ndarray, kappa: float,
                  decay: float, H: int) -> np.ndarray:
    """Batched reference CN forecasts; input [K,256], [K,2], output [K,H,128]."""
    q = np.asarray(initial, dtype=np.float64).copy()
    a = np.asarray(actions, dtype=np.float64)
    if q.ndim != 2 or a.shape != (len(q), 2) or q.shape[1] != 256:
        raise ValueError("heat forecast expects [K,256] states and [K,2] actions")
    n = q.shape[1]
    dt = 0.02
    dx = 1 / (n + 1)
    r = kappa * dt / (2 * dx * dx)
    c = decay * dt / 2
    ab = np.zeros((3, n), dtype=np.float64)
    ab[0, 1:] = -r
    ab[1] = 1 + 2 * r + c
    ab[2, :-1] = -r
    forcing = a @ _dirichlet_actuators(n)
    outputs = []
    for _ in range(H):
        b = (1 - 2 * r - c) * q
        b[:, 1:] += r * q[:, :-1]
        b[:, :-1] += r * q[:, 1:]
        b += dt * forcing
        q = solve_banded((1, 1), ab, b.T).T
        outputs.append(np.stack([restrict_field(row, "heat") for row in q]))
    return np.stack(outputs, axis=1)


def heat_advance(state: np.ndarray, action: np.ndarray, kappa: float,
                 decay: float) -> np.ndarray:
    u = np.asarray(state, dtype=np.float64)
    forcing = np.asarray(action, dtype=np.float64) @ _dirichlet_actuators(u.size)
    return heat_cn_step(u, 1.0 / (u.size + 1), 0.02, kappa, decay, forcing).astype(np.float32)
