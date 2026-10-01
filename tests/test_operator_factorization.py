import math
import torch

from pdno.models.operators import ActionFactorizedOperator, CandidateConditionedOperator
from pdno.models.policies import DirectPolicy
from pdno.controllers.objectives import b3_surrogate_gradient_step, freeze_operator


def _obs(batch=2, goal_dim=4):
    return {
        "sensor_value": torch.randn(batch, 8, 16),
        "sensor_mask": torch.ones(batch, 8, 16, dtype=torch.bool),
        "sensor_age": torch.zeros(batch, 8, 16),
        "instrument_image": torch.randn(batch, 1, 16, 64),
        "image_mask": torch.ones(batch, 1, 16, 64, dtype=torch.bool),
        "goal_coefficients": torch.randn(batch, goal_dim),
        "material_context": torch.randn(batch, 8),
        "previous_applied_action": torch.zeros(batch, 2),
        "applied_action_history": torch.zeros(batch, 8, 2),
        "image_age": torch.zeros(batch),
        "image_valid": torch.ones(batch, dtype=torch.bool),
    }


def test_factorized_operator_candidate_permutation_equivariance():
    torch.manual_seed(4)
    model = ActionFactorizedOperator("burgers", goal_dim=4, rank=8)
    obs = _obs()
    actions = torch.randn(2, 5, 2).clamp(-1, 1)
    x, tau = torch.arange(32, dtype=torch.float32) / 32, torch.linspace(1 / 8, 1, 8)
    out = model(obs, actions, x, tau)
    perm = torch.tensor([3, 0, 4, 1, 2])
    permuted = model(obs, actions[:, perm], x, tau)
    assert out.shape == (2, 5, 8, 32)
    torch.testing.assert_close(permuted, out[:, perm], atol=1e-6, rtol=1e-5)
    assert model._last_candidate_batch_shape == (2, 5, 2)


def test_b4_shares_encoder_once_and_batches_candidate_axis():
    model = CandidateConditionedOperator("heat", goal_dim=32, rank=8)
    calls = {"encoder": 0, "branch": 0}
    h1 = model.encoder.register_forward_hook(lambda *args: calls.__setitem__("encoder", calls["encoder"] + 1))
    h2 = model.action_branch.register_forward_hook(lambda *args: calls.__setitem__("branch", calls["branch"] + 1))
    out = model(_obs(goal_dim=32), torch.rand(2, 10, 2), torch.linspace(0, 1, 32), torch.linspace(1 / 8, 1, 8))
    h1.remove(); h2.remove()
    assert out.shape == (2, 10, 8, 32)
    assert calls == {"encoder": 1, "branch": 1}
    assert model._last_branch_candidate_shape == (2, 10, 2)


def test_heat_operator_enforces_homogeneous_dirichlet_boundary():
    for model in (ActionFactorizedOperator("heat", goal_dim=32, rank=8),
                  CandidateConditionedOperator("heat", goal_dim=32, rank=8)):
        x = torch.tensor([0.0, 0.2, 0.7, 1.0])
        out = model(_obs(goal_dim=32), torch.rand(2, 3, 2), x, torch.linspace(1 / 8, 1, 8))
        torch.testing.assert_close(out[..., 0], torch.zeros_like(out[..., 0]), atol=1e-7, rtol=0)
        torch.testing.assert_close(out[..., -1], torch.zeros_like(out[..., -1]), atol=1e-7, rtol=0)


def test_heat_initial_basis_preserves_analytic_boundary_flux_derivative():
    from pdno.models.operators import spatial_basis

    x = torch.tensor([0.0, 1.0], dtype=torch.float64, requires_grad=True)
    first_mode = spatial_basis("heat", x)[0]
    assert torch.max(torch.abs(first_mode)).item() < 1e-12
    derivative = torch.autograd.grad(first_mode.sum(), x)[0]
    torch.testing.assert_close(derivative, torch.tensor([math.pi, -math.pi], dtype=torch.float64),
                               rtol=1e-12, atol=1e-12)


def test_observation_encoder_uses_causal_applied_action_history():
    from pdno.models.encoders import ObservationEncoder

    encoder = ObservationEncoder(goal_dim=33).eval()
    obs = _obs(goal_dim=33)
    history_a = torch.zeros(2, 8, 2)
    history_b = history_a.clone()
    history_b[:, -1, 0] = 0.75
    with torch.no_grad():
        context_a = encoder({**obs, "applied_action_history": history_a})
        context_b = encoder({**obs, "applied_action_history": history_b})
    assert not torch.equal(context_a, context_b)


def test_direct_policy_obeys_action_box_and_slew():
    for pde, low, high, slew in (("burgers", -1, 1, 0.15), ("heat", 0, 1, 0.10)):
        policy = DirectPolicy(goal_dim=4, pde=pde)
        obs = _obs()
        obs["previous_applied_action"] = torch.full((2, 2), 0.8 if pde == "heat" else 0.0)
        action = policy(obs)
        prev = obs["previous_applied_action"]
        assert torch.all(action >= low) and torch.all(action <= high)
        assert torch.all(torch.abs(action - prev) <= slew + 1e-6)


def test_pointwise_operator_coordinate_gradient_matches_finite_difference():
    torch.manual_seed(13)
    model = ActionFactorizedOperator("burgers", goal_dim=33, rank=8)
    obs = _obs(goal_dim=33)
    # Use one independent observation so coordinates cannot interact through the batch axis.
    obs = {key: value[:1] for key, value in obs.items()}
    action = torch.tensor([[[0.1, -0.2]]])
    x = torch.tensor([0.231, 0.617], requires_grad=True)
    tau = torch.tensor([0.27, 0.73], requires_grad=True)
    y = model.evaluate_points(obs, action, x, tau)[0, 0]
    grad_x, grad_tau = torch.autograd.grad(y.sum(), (x, tau))
    eps = 1e-3
    with torch.no_grad():
        xp = model.evaluate_points(obs, action, x.detach() + eps, tau.detach())[0, 0]
        xm = model.evaluate_points(obs, action, x.detach() - eps, tau.detach())[0, 0]
        tp = model.evaluate_points(obs, action, x.detach(), tau.detach() + eps)[0, 0]
        tm = model.evaluate_points(obs, action, x.detach(), tau.detach() - eps)[0, 0]
    fd_x = (xp - xm) / (2 * eps)
    fd_tau = (tp - tm) / (2 * eps)
    torch.testing.assert_close(grad_x, fd_x, atol=3e-3, rtol=3e-2)
    torch.testing.assert_close(grad_tau, fd_tau, atol=3e-3, rtol=3e-2)


def test_b3_surrogate_gradient_flows_through_frozen_b4_to_policy_only():
    torch.manual_seed(9)
    policy = DirectPolicy(goal_dim=4, pde="burgers")
    b4 = CandidateConditionedOperator("burgers", goal_dim=4, rank=8)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    result = b3_surrogate_gradient_step(policy, b4, _obs(), torch.arange(32).float() / 32,
                                        torch.linspace(0.125, 1, 8), torch.zeros(32), optimizer, q_max=1.2)
    assert result["action_grad_norm"] > 0
    assert all(not parameter.requires_grad for parameter in b4.parameters())
    assert all(parameter.grad is None for parameter in b4.parameters())
