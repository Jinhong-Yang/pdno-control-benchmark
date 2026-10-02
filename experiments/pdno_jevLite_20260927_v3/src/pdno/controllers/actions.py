"""Common action projection and explicit proposed/projected/applied tracking."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ActionRecord:
    proposed: np.ndarray
    projected: np.ndarray
    applied: np.ndarray


def project_box_slew(proposed: np.ndarray, previous_applied: np.ndarray, low: float, high: float, slew: float) -> np.ndarray:
    a = np.asarray(proposed, dtype=np.float64)
    prev = np.asarray(previous_applied, dtype=np.float64)
    if a.shape != prev.shape or a.ndim != 1:
        raise ValueError("action and previous applied action must be same-shape vectors")
    if low >= high or slew < 0:
        raise ValueError("invalid projection limits")
    lower = np.maximum(low, prev - slew)
    upper = np.minimum(high, prev + slew)
    return np.clip(a, lower, upper)


def account_action(proposed: np.ndarray, previous_applied: np.ndarray, low: float, high: float, slew: float) -> ActionRecord:
    p = np.asarray(proposed, dtype=np.float64).copy()
    projected = project_box_slew(p, previous_applied, low, high, slew)
    # Simulator and all subsequent state/observation transitions consume this exact value.
    applied = projected.copy()
    return ActionRecord(p, projected.copy(), applied)
