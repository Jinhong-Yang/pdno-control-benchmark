from __future__ import annotations

import math
import torch
from torch import nn

from pdno.models.encoders import ObservationEncoder


def spatial_basis(pde: str, x: torch.Tensor) -> torch.Tensor:
    """Return Q×N initial-field basis; boundary behavior matches the PDE contract."""
    if pde == "burgers":
        cols = [torch.ones_like(x)]
        for k in range(1, 17):
            cols.extend((torch.sin(2 * math.pi * k * x), torch.cos(2 * math.pi * k * x)))
        return torch.stack(cols, dim=0)
    if pde == "heat":
        # Correct only sin(k*pi)'s endpoint roundoff linearly. This keeps the
        # Dirichlet values exactly zero in FP32 while retaining the analytic
        # endpoint derivative; torch.where would incorrectly zero that flux.
        modes = []
        for k in range(1, 33):
            wave_number = x.new_tensor(k * math.pi)
            modes.append(torch.sin(wave_number * x) - x * torch.sin(wave_number))
        return torch.stack(modes, dim=0)
    raise ValueError(f"unknown PDE: {pde}")


def trunk_coordinates(pde: str, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    xx = x.reshape(1, -1).expand(tau.numel(), -1)
    tt = tau.reshape(-1, 1).expand(-1, x.numel())
    if pde == "burgers":
        return torch.stack((torch.sin(2 * math.pi * xx), torch.cos(2 * math.pi * xx), tt, tt**2), dim=-1)
    return torch.stack((xx, xx * (1 - xx), tt, tt**2), dim=-1)


def trunk_coordinates_points(pde: str, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    if x.shape != tau.shape:
        raise ValueError("pointwise x and tau must have matching shape")
    if pde == "burgers":
        return torch.stack((torch.sin(2 * math.pi * x), torch.cos(2 * math.pi * x), tau, tau**2), dim=-1)
    return torch.stack((x, x * (1 - x), tau, tau**2), dim=-1)


def polynomial_action_features(action: torch.Tensor, low: float, high: float) -> torch.Tensor:
    normalized = 2.0 * (action - low) / (high - low) - 1.0
    a1, a2 = normalized.unbind(dim=-1)
    return torch.stack((torch.ones_like(a1), a1, a2, a1.square(), a1 * a2, a2.square()), dim=-1)


class ActionFactorizedOperator(nn.Module):
    """Proposed operator. Encoder/response branch is evaluated once for all K candidates."""

    def __init__(self, pde: str, goal_dim: int, rank: int = 32, p_terms: int = 6):
        super().__init__()
        if p_terms != 6:
            raise ValueError("quadratic factorized basis uses six fixed polynomial terms")
        self.pde, self.rank, self.p_terms = pde, rank, p_terms
        self.encoder = ObservationEncoder(goal_dim)
        q_dim = 33 if pde == "burgers" else 32
        self.initial_head = nn.Linear(128, q_dim)
        self.response_branch = nn.Sequential(nn.Linear(128 + q_dim, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, p_terms * rank))
        self.trunk = nn.Sequential(nn.Linear(4, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, rank))
        self._last_candidate_batch_shape: tuple[int, ...] | None = None

    def predict_initial(self, obs: dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
        context = self.encoder(obs)
        q_coeff = self.initial_head(context)
        return torch.einsum("bq,qn->bn", q_coeff, spatial_basis(self.pde, x))

    def forward(self, obs: dict[str, torch.Tensor], actions: torch.Tensor, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        # actions [B,K,2]; one context and one response-branch invocation per request batch.
        context = self.encoder(obs)
        q_coeff = self.initial_head(context)
        branch_input = torch.cat((context, q_coeff), dim=-1)
        coeff = self.response_branch(branch_input).view(-1, self.p_terms, self.rank)
        trunk = self.trunk(trunk_coordinates(self.pde, x, tau))  # [H,N,R], cached at inference after freeze
        basis = spatial_basis(self.pde, x)
        q0 = torch.einsum("bq,qn->bn", q_coeff, basis)
        field_response = torch.einsum("bpr,hnr->bphn", coeff, trunk)
        low, high = (-1.0, 1.0) if self.pde == "burgers" else (0.0, 1.0)
        phi = polynomial_action_features(actions, low, high)
        delta = torch.einsum("bkp,bphn->bkhn", phi, field_response)
        if self.pde == "heat":
            delta = delta * (x * (1.0 - x)).reshape(1, 1, 1, -1)
        self._last_candidate_batch_shape = tuple(actions.shape)
        return q0[:, None, None, :] + tau.reshape(1, 1, -1, 1) * delta

    def evaluate_points(self, obs: dict[str, torch.Tensor], actions: torch.Tensor, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        context = self.encoder(obs)
        q_coeff = self.initial_head(context)
        coeff = self.response_branch(torch.cat((context, q_coeff), dim=-1)).view(-1, self.p_terms, self.rank)
        trunk = self.trunk(trunk_coordinates_points(self.pde, x, tau))
        basis = spatial_basis(self.pde, x)
        if x.ndim == 1:
            q0 = torch.einsum("bq,qn->bn", q_coeff, basis)
            response = torch.einsum("bpr,mr->bpm", coeff, trunk)
            x_mask = x.reshape(1, 1, -1)
        elif x.ndim == 2:
            q0 = torch.einsum("bq,qbm->bm", q_coeff, basis)
            response = torch.einsum("bpr,bmr->bpm", coeff, trunk)
            x_mask = x[:, None, :]
        else:
            raise ValueError("pointwise x/tau must have shape [M] or [B,M]")
        low, high = (-1.0, 1.0) if self.pde == "burgers" else (0.0, 1.0)
        phi = polynomial_action_features(actions, low, high)
        delta = torch.einsum("bkp,bpm->bkm", phi, response)
        if self.pde == "heat":
            delta = delta * (x_mask * (1.0 - x_mask))
        tau_mask = tau.reshape(1, 1, -1) if tau.ndim == 1 else tau[:, None, :]
        return q0[:, None, :] + tau_mask * delta


class CandidateConditionedOperator(nn.Module):
    """B4: shared observation encoder; all K actions are one candidate-batched branch call."""

    def __init__(self, pde: str, goal_dim: int, rank: int = 32, p_terms: int = 6):
        super().__init__()
        self.pde, self.rank, self.p_terms = pde, rank, p_terms
        self.encoder = ObservationEncoder(goal_dim)  # invoked once, not once per action
        q_dim = 33 if pde == "burgers" else 32
        self.initial_head = nn.Linear(128, q_dim)
        self.action_branch = nn.Sequential(nn.Linear(128 + q_dim + 2, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, p_terms * rank))
        self.trunk = nn.Sequential(nn.Linear(4, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, rank))
        self._last_branch_candidate_shape: tuple[int, ...] | None = None

    def predict_initial(self, obs: dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
        context = self.encoder(obs)
        q_coeff = self.initial_head(context)
        return torch.einsum("bq,qn->bn", q_coeff, spatial_basis(self.pde, x))

    def forward(self, obs: dict[str, torch.Tensor], actions: torch.Tensor, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        batch, candidates, _ = actions.shape
        context = self.encoder(obs)
        q_coeff = self.initial_head(context)
        repeated = torch.cat((context, q_coeff), dim=-1)[:, None, :].expand(-1, candidates, -1)
        branch_input = torch.cat((repeated, actions), dim=-1).reshape(batch * candidates, -1)
        # Single K-batched invocation makes B4 a strong throughput control.
        coeff = self.action_branch(branch_input).view(batch, candidates, self.p_terms, self.rank)
        trunk = self.trunk(trunk_coordinates(self.pde, x, tau))
        q0 = torch.einsum("bq,qn->bn", q_coeff, spatial_basis(self.pde, x))
        low, high = (-1.0, 1.0) if self.pde == "burgers" else (0.0, 1.0)
        phi = polynomial_action_features(actions, low, high)
        response = torch.einsum("bkpr,hnr->bkphn", coeff, trunk)
        delta = torch.einsum("bkp,bkphn->bkhn", phi, response)
        if self.pde == "heat":
            delta = delta * (x * (1.0 - x)).reshape(1, 1, 1, -1)
        self._last_branch_candidate_shape = tuple(actions.shape)
        return q0[:, None, None, :] + tau.reshape(1, 1, -1, 1) * delta

    def evaluate_points(self, obs: dict[str, torch.Tensor], actions: torch.Tensor, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        batch, candidates, _ = actions.shape
        context = self.encoder(obs)
        q_coeff = self.initial_head(context)
        repeated = torch.cat((context, q_coeff), dim=-1)[:, None, :].expand(-1, candidates, -1)
        branch_input = torch.cat((repeated, actions), dim=-1).reshape(batch * candidates, -1)
        coeff = self.action_branch(branch_input).view(batch, candidates, self.p_terms, self.rank)
        trunk = self.trunk(trunk_coordinates_points(self.pde, x, tau))
        basis = spatial_basis(self.pde, x)
        if x.ndim == 1:
            q0 = torch.einsum("bq,qn->bn", q_coeff, basis)
            response = torch.einsum("bkpr,mr->bkpm", coeff, trunk)
            x_mask = x.reshape(1, 1, -1)
        elif x.ndim == 2:
            q0 = torch.einsum("bq,qbm->bm", q_coeff, basis)
            response = torch.einsum("bkpr,bmr->bkpm", coeff, trunk)
            x_mask = x[:, None, :]
        else:
            raise ValueError("pointwise x/tau must have shape [M] or [B,M]")
        low, high = (-1.0, 1.0) if self.pde == "burgers" else (0.0, 1.0)
        phi = polynomial_action_features(actions, low, high)
        delta = torch.einsum("bkp,bkpm->bkm", phi, response)
        if self.pde == "heat":
            delta = delta * (x_mask * (1.0 - x_mask))
        tau_mask = tau.reshape(1, 1, -1) if tau.ndim == 1 else tau[:, None, :]
        return q0[:, None, :] + tau_mask * delta
