from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from pdno.controllers.actions import project_box_slew
from pdno.controllers.linear import nominal_lqr_action
from pdno.data.teacher_queries import feasible_candidates


def candidates(pde: str, previous: np.ndarray, b0_action: np.ndarray, K: int) -> np.ndarray:
    """Return the archived K=10 set or the sealed K=50 timing-sweep set."""
    if K == 10:
        return feasible_candidates(pde, previous, b0_action)
    if K != 50:
        raise ValueError("X1 only defines K=10 and K=50")
    low, high, delta = (-1.0, 1.0, 0.15) if pde == "burgers" else (0.0, 1.0, 0.10)
    n = math.ceil(math.sqrt(K - 2))
    axis = np.linspace(-delta, delta, n)
    lattice = np.array([[d1, d2] for d1 in axis for d2 in axis])
    indices = np.linspace(0, len(lattice) - 1, K - 2, dtype=int)
    expanded = [project_box_slew(previous + lattice[i], previous, low, high, delta)
                for i in indices]
    expanded.append(project_box_slew(previous, previous, low, high, delta))
    expanded.append(project_box_slew(b0_action, previous, low, high, delta))
    return np.asarray(expanded, dtype=np.float32)


def full_state_b0_action(state: np.ndarray, pde: str, material: np.ndarray,
                         goal: np.ndarray, previous: np.ndarray) -> np.ndarray:
    """Apply the same seven-mode B0 feedback law using a fully observed predicted state."""
    state = np.asarray(state, dtype=np.float32)
    obs = {
        "sensor_value": state[None, :],
        "sensor_mask": np.ones((1, state.size), dtype=bool),
        "image_valid": False,
        "material_context": np.asarray(material, dtype=np.float32),
        "goal_field": np.asarray(goal, dtype=np.float32),
    }
    return nominal_lqr_action(obs, pde, np.asarray(previous, dtype=np.float32))


def objective(forecast: np.ndarray, goal: np.ndarray, action: np.ndarray,
             previous: np.ndarray, q_max: float, pde: str) -> float:
    """Equation (3) on the supplied H endpoints, with per-decision action costs."""
    # Preserve the archived candidate scorer's FP32 arithmetic. Reference
    # trajectories are accumulated in FP64, then exposed to the decision rule
    # as FP32, matching the frozen CUDA scorer and _cost implementation.
    field = np.asarray(forecast, dtype=np.float32)
    target = np.asarray(goal, dtype=np.float32)
    if field.ndim != 2 or field.shape[0] == 0 or field.shape[1] != target.size:
        raise ValueError("forecast must be nonempty [H,N] and match scoring-grid goal")
    error = field - target[None, :]
    tracking = float(np.mean(error ** 2))
    action32 = np.asarray(action, dtype=np.float32)
    previous32 = np.asarray(previous, dtype=np.float32)
    effort = 0.01 * float(np.sum(action32 ** 2))
    slew = 0.05 * float(np.sum((action32 - previous32) ** 2))
    if pde == "burgers":
        violation = np.maximum(np.abs(field) - np.float32(q_max), 0.0) ** 2
    elif pde == "heat":
        violation = (np.maximum(-field, 0.0) ** 2
                     + np.maximum(field - np.float32(q_max), 0.0) ** 2)
    else:
        raise ValueError(f"unknown PDE {pde!r}")
    return tracking + effort + slew + 10.0 * float(np.mean(violation))


def score_fixed(state: np.ndarray, actions: np.ndarray, pde: str,
                material: np.ndarray, H: int, solver: Callable,
                goal_score: np.ndarray, previous: np.ndarray, q_max: float) -> np.ndarray:
    """Score fixed-action candidates using a solver returning [K,H,128]."""
    forecasts = solver(np.broadcast_to(state, (len(actions), len(state))).copy(), actions, H)
    return np.asarray([objective(forecasts[k], goal_score, actions[k], previous, q_max, pde)
                       for k in range(len(actions))], dtype=np.float64)


def score_with_feedback(state: np.ndarray, pde: str,
                         material: np.ndarray, goal_full: np.ndarray, H: int,
                         advance: Callable, goal_score: np.ndarray,
                         previous: np.ndarray, q_max: float,
                         b0_current: np.ndarray) -> tuple[float, np.ndarray]:
    """Roll one B0-feedback candidate and return its score and H projected actions."""
    future = np.asarray(state, dtype=np.float64).copy()
    rollout_actions = [np.asarray(b0_current, dtype=np.float32).copy()]
    pred = np.asarray(advance(future, rollout_actions[0]), dtype=np.float64)
    fields = [pred]
    last = rollout_actions[0]
    low, high, slew = (-1.0, 1.0, 0.15) if pde == "burgers" else (0.0, 1.0, 0.10)
    for _ in range(1, H):
        command = full_state_b0_action(pred, pde, material, goal_full, last)
        command = project_box_slew(command, last, low, high, slew)
        pred = np.asarray(advance(pred, command), dtype=np.float64)
        fields.append(pred)
        rollout_actions.append(np.asarray(command, dtype=np.float32))
        last = command
    scored = np.stack([v[::2] for v in fields]) if pde == "burgers" else np.stack(fields)
    score = objective(scored, goal_score, rollout_actions[0], previous, q_max, pde)
    return score, np.stack(rollout_actions)


def summarize_indices(choices: np.ndarray, K: int, feedback_rollout: bool = False) -> dict:
    indices = np.asarray(choices)
    hold = 4 if K == 10 else 48
    b0 = K - 1
    return {
        "hold_selection_rate": float(np.mean(indices == hold)),
        "b0_selection_rate": float(np.mean(indices == b0)),
        "feedback_rollout_selection_rate": (float(np.mean(indices == 10))
                                             if feedback_rollout else None),
    }
