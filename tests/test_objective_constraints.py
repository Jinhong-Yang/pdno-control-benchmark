import numpy as np
import torch

from pdno.controllers.objectives import candidate_cost
from pdno.data.teacher_queries import _cost


def test_heat_cost_uses_separate_undercooling_and_overtemperature_hinges():
    field = torch.tensor([[[[-0.2, 1.2]]]])
    goal = torch.zeros(1, 2)
    actions = torch.zeros(1, 1, 2)
    previous = torch.zeros(1, 2)
    signed = candidate_cost(field, goal, actions, previous, q_max=1.0, q_min=0.0)
    symmetric = candidate_cost(field, goal, actions, previous, q_max=1.0)
    assert signed.item() > symmetric.item()
    assert abs(signed.item() - 1.14) < 1e-6
    assert abs(symmetric.item() - 0.94) < 1e-6


def test_teacher_numpy_cost_matches_signed_heat_and_symmetric_burgers_contracts():
    future = np.asarray([[-0.2, 1.2]], dtype=np.float64)
    goal = np.zeros(2)
    action = np.zeros(2)
    previous = np.zeros(2)
    heat_cost, heat_peak = _cost(future, goal, action, previous, q_max=1.0, q_min=0.0)
    burgers_cost, burgers_peak = _cost(future, goal, action, previous, q_max=1.0)
    assert abs(heat_cost - 1.14) < 1e-12
    assert abs(burgers_cost - 0.94) < 1e-12
    assert heat_peak == 1.2
    assert burgers_peak == 1.2
