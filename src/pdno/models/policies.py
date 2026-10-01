from __future__ import annotations

import torch
from torch import nn

from pdno.models.encoders import ObservationEncoder


def bounded_slew_action(raw: torch.Tensor, previous: torch.Tensor, low: float, high: float, slew: float) -> torch.Tensor:
    lo = torch.maximum(torch.full_like(previous, low), previous - slew)
    hi = torch.minimum(torch.full_like(previous, high), previous + slew)
    return lo + (hi - lo) * torch.sigmoid(raw)


class DirectPolicy(nn.Module):
    """Small direct typed action policy used by B2 and B3."""

    def __init__(self, goal_dim: int, pde: str):
        super().__init__()
        self.pde = pde
        self.encoder = ObservationEncoder(goal_dim)
        self.action_head = nn.Sequential(nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 2))

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        raw = self.action_head(self.encoder(obs))
        low, high, slew = (-1.0, 1.0, 0.15) if self.pde == "burgers" else (0.0, 1.0, 0.10)
        return bounded_slew_action(raw, obs["previous_applied_action"], low, high, slew)
