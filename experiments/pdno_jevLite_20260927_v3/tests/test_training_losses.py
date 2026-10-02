import pytest
import torch

from pdno.models.operators import ActionFactorizedOperator, CandidateConditionedOperator
from pdno.training.losses import observation_consistency_loss, pde_residual_loss


def _obs(pde: str, batch: int = 2):
    goal_dim = 33 if pde == "burgers" else 32
    material = ([0.02, -1, 1, 0.15, 0.02, 0.16, 1.2, 128] if pde == "burgers"
                else [0.1, 0.02, 0, 1, 0.1, 0.02, 0.16, 128])
    return {
        "sensor_value": torch.randn(batch, 8, 16),
        "sensor_mask": torch.ones(batch, 8, 16, dtype=torch.bool),
        "sensor_age": torch.zeros(batch, 8, 16),
        "instrument_image": torch.randn(batch, 1, 16, 64),
        "image_mask": torch.ones(batch, 1, 16, 64, dtype=torch.bool),
        "goal_coefficients": torch.zeros(batch, goal_dim),
        "material_context": torch.tensor(material, dtype=torch.float32).repeat(batch, 1),
        "previous_applied_action": torch.zeros(batch, 2),
        "applied_action_history": torch.zeros(batch, 8, 2),
        "image_age": torch.zeros(batch),
        "image_valid": torch.ones(batch, dtype=torch.bool),
    }


@pytest.mark.parametrize("pde", ["burgers", "heat"])
@pytest.mark.parametrize("model_type", ["P", "B4"])
def test_batched_pointwise_coordinates_match_separate_evaluation(pde, model_type):
    torch.manual_seed(7)
    model = (ActionFactorizedOperator(pde, 33 if pde == "burgers" else 32, rank=8)
             if model_type == "P" else CandidateConditionedOperator(
                 pde, 33 if pde == "burgers" else 32, rank=8)).eval()
    obs = _obs(pde)
    low, high = (-1.0, 1.0) if pde == "burgers" else (0.0, 1.0)
    actions = torch.rand(2, 1, 2) * (high - low) + low
    x = torch.rand(2, 12)
    tau = torch.rand(2, 12).clamp_min(0.02)
    batched = model.evaluate_points(obs, actions, x, tau)
    separate = torch.cat([
        model.evaluate_points({key: value[i:i+1] for key, value in obs.items()},
                              actions[i:i+1], x[i], tau[i])
        for i in range(2)
    ], dim=0)
    torch.testing.assert_close(batched, separate, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("pde", ["burgers", "heat"])
def test_complete_training_losses_are_finite_and_differentiable(pde):
    torch.manual_seed(11)
    model = ActionFactorizedOperator(pde, 33 if pde == "burgers" else 32, rank=8)
    obs = _obs(pde)
    q0 = torch.randn(2, 128)
    l_initial, l_sensor = observation_consistency_loss(model, obs, q0, q_scale=1.0)
    low, high = (-1.0, 1.0) if pde == "burgers" else (0.0, 1.0)
    actions = (torch.rand(2, 1, 2) * (high - low) + low)
    pde_loss, balance_loss, bc_loss = pde_residual_loss(
        model, obs, actions, torch.rand(2, 32), torch.rand(2, 32).clamp_min(0.02))
    total = l_initial + l_sensor + pde_loss + balance_loss + bc_loss
    assert all(torch.isfinite(term) for term in (l_initial, l_sensor, pde_loss, balance_loss, bc_loss))
    total.backward()
    grads = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert grads and all(grad is not None and torch.isfinite(grad).all() for grad in grads)
