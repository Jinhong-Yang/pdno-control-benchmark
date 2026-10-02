from __future__ import annotations

import torch
from torch.nn import functional as F


def candidate_cost(predicted_field: torch.Tensor, goal: torch.Tensor, actions: torch.Tensor,
                   previous_applied: torch.Tensor, q_max: float, weight=None,
                   q_min: float | None = None) -> torch.Tensor:
    """Common explicit tracking/effort/slew/state-penalty cost, shape [B,K]."""
    if goal.ndim == 1:
        goal = goal.unsqueeze(0)
    error = predicted_field - goal[:, None, None, :]
    if weight is None:
        tracking = error.square().mean(dim=(-1, -2))
    else:
        tracking = (error.square() * weight.reshape(1, 1, 1, -1)).mean(dim=(-1, -2))
    effort = 0.01 * actions.square().sum(dim=-1)
    slew = 0.05 * (actions - previous_applied[:, None, :]).square().sum(dim=-1)
    if q_min is None:
        violation_field = F.relu(predicted_field.abs() - q_max).square()
    else:
        violation_field = (F.relu(q_min - predicted_field).square()
                           + F.relu(predicted_field - q_max).square())
    violation = 10.0 * violation_field.mean(dim=(-1, -2))
    return tracking + effort + slew + violation


def freeze_operator(operator) -> None:
    operator.eval()
    for parameter in operator.parameters():
        parameter.requires_grad_(False)


def b3_surrogate_gradient_step(policy, frozen_b4, obs: dict[str, torch.Tensor], x: torch.Tensor, tau: torch.Tensor,
                               goal: torch.Tensor, optimizer, q_max: float, bc_action: torch.Tensor | None = None,
                               bc_weight: float = 0.1, q_min: float | None = None) -> dict[str, float]:
    """One B3 update: direct policy gets gradients through a frozen candidate PI-DeepONet."""
    freeze_operator(frozen_b4)
    optimizer.zero_grad(set_to_none=True)
    action = policy(obs)
    predicted = frozen_b4(obs, action[:, None, :], x, tau)
    loss = candidate_cost(predicted, goal, action[:, None, :], obs["previous_applied_action"],
                          q_max, q_min=q_min).mean()
    if bc_action is not None:
        loss = loss + bc_weight * torch.nn.functional.mse_loss(action, bc_action)
    loss.backward()
    optimizer.step()
    return {"loss": float(loss.detach()), "action_grad_norm": float(policy.action_head[-1].weight.grad.detach().norm())}
