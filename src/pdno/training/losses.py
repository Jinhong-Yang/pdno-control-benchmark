from __future__ import annotations

import torch

from pdno.data.generate import _dirichlet_actuators, _periodic_actuators


def observation_consistency_loss(model, obs: dict[str, torch.Tensor],
                                initial_target: torch.Tensor, q_scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized initial-field and known-sensor consistency losses."""
    device = initial_target.device
    if model.pde == "burgers":
        sensor_x = torch.arange(16, device=device, dtype=torch.float32) / 16.0
    else:
        sensor_x = torch.linspace(1.0, 256.0, 16, device=device, dtype=torch.float32) / 257.0
    q0_sensors = model.predict_initial(obs, sensor_x)
    values = obs["sensor_value"][:, -1, :].float()
    mask = obs["sensor_mask"][:, -1, :].bool()
    residual = (q0_sensors - values).square() * mask
    sensor_loss = residual.sum() / mask.sum().clamp_min(1)
    field_loss = torch.nn.functional.mse_loss(model.predict_initial(obs, _model_grid(model.pde, device)),
                                              initial_target.float())
    scale2 = max(float(q_scale) ** 2, 1e-8)
    return field_loss / scale2, sensor_loss / scale2


def _model_grid(pde: str, device: torch.device) -> torch.Tensor:
    if pde == "burgers":
        return torch.arange(128, device=device, dtype=torch.float32) / 128.0
    return (torch.arange(128, device=device, dtype=torch.float32) + 1.0) / 129.0


def _actuator_values(pde: str, x: torch.Tensor) -> torch.Tensor:
    if pde == "burgers":
        table = torch.as_tensor(_periodic_actuators(256), device=x.device, dtype=x.dtype)
        position = x * 256.0
        lo0 = torch.floor(position)
        lo = lo0.long() % 256
        hi = (lo + 1) % 256
        frac = position - lo0
    else:
        table = torch.as_tensor(_dirichlet_actuators(256), device=x.device, dtype=x.dtype)
        position = x * 257.0 - 1.0
        lo = torch.floor(position).long().clamp(0, 254)
        frac = (position - lo.float()).clamp(0.0, 1.0)
        hi = (lo + 1).clamp(0, 255)
    return table[:, lo] * (1.0 - frac).unsqueeze(0) + table[:, hi] * frac.unsqueeze(0)


def _forcing(pde: str, x: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    actuators = _actuator_values(pde, x)
    return actions[:, 0, None] * actuators[0] + actions[:, 1, None] * actuators[1]


def _coordinate_derivatives(model, obs, actions, x, tau, *, second_derivative: bool = True):
    x = x.detach().float().requires_grad_(True)
    tau = tau.detach().float().requires_grad_(True)
    prediction = model.evaluate_points(obs, actions, x, tau)[:, 0, :]
    ones = torch.ones_like(prediction)
    q_x = torch.autograd.grad(prediction, x, ones, create_graph=True, retain_graph=True)[0]
    q_tau = torch.autograd.grad(prediction, tau, ones, create_graph=True, retain_graph=True)[0]
    q_xx = (torch.autograd.grad(q_x, x, ones, create_graph=True, retain_graph=True)[0]
            if second_derivative else None)
    return prediction, q_x, q_tau, q_xx, x, tau


def boundary_condition_diagnostic(model, obs: dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
    """Always-computed boundary diagnostic: periodic seam for Burgers, Dirichlet values for heat."""
    endpoints = torch.tensor([0.0, 1.0], device=action.device, dtype=torch.float32)
    boundary_tau = torch.ones(2, device=action.device, dtype=torch.float32)
    boundary = model.evaluate_points(obs, action, endpoints, boundary_tau)[:, 0, :]
    if model.pde == "burgers":
        return (boundary[:, 0] - boundary[:, 1]).square().mean()
    return boundary.square().mean()


def pde_residual_loss(model, obs: dict[str, torch.Tensor], action: torch.Tensor, x: torch.Tensor,
                      tau: torch.Tensor, physical_time_horizon: float = 0.16) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FP32 batched PDE residual, integrated mass/energy balance, and BC diagnostic.

    `x` and `tau` are per-sample collocation arrays [B,M]. The caller supplies one
    feasible random training action per sample in [B,1,2]. No autocast is used here.
    """
    batch = obs["material_context"].shape[0]
    if x.ndim != 2 or tau.shape != x.shape or x.shape[0] != batch or action.shape != (batch, 1, 2):
        raise ValueError("expected x/tau [B,M] and one sampled action [B,1,2] per observation")
    if torch.is_autocast_enabled(x.device.type):
        raise RuntimeError("PDE residual path requires autocast disabled")
    q, q_x, q_tau, q_xx, x_req, tau_req = _coordinate_derivatives(model, obs, action, x, tau)
    q_t = q_tau / float(physical_time_horizon)
    forcing = _forcing(model.pde, x_req, action[:, 0, :])
    material = obs["material_context"].float()
    if model.pde == "burgers":
        nu = material[:, 0:1]
        residual = q_t + q * q_x - nu * q_xx - forcing
    else:
        kappa, decay = material[:, 0:1], material[:, 1:2]
        residual = q_t - kappa * q_xx + decay * q - forcing
    pde_loss = residual.square().mean()

    # Independent quadrature check of the spatially integrated conservation law.
    quadrature_x = torch.linspace(0.0, 1.0, 65, device=x.device, dtype=torch.float32)
    x_mass = quadrature_x.unsqueeze(0).expand(batch, -1).clone()
    mass_tau = torch.rand((batch, 1), device=x.device, dtype=torch.float32).expand(-1, quadrature_x.numel()).clone()
    mass_q, mass_qx, mass_qtau, _, x_mass, mass_tau = _coordinate_derivatives(
        model, obs, action, x_mass, mass_tau, second_derivative=False)
    mass_qt = mass_qtau / float(physical_time_horizon)
    mass_force = _forcing(model.pde, x_mass, action[:, 0, :])
    integral_qt = torch.trapezoid(mass_qt, x_mass, dim=-1)
    integral_force = torch.trapezoid(mass_force, x_mass, dim=-1)
    if model.pde == "burgers":
        nu = material[:, 0]
        flux = 0.5 * (mass_q[:, -1].square() - mass_q[:, 0].square()) - nu * (mass_qx[:, -1] - mass_qx[:, 0])
        balance = integral_qt + flux - integral_force
    else:
        kappa, decay = material[:, 0], material[:, 1]
        diffusion_flux = kappa * (mass_qx[:, -1] - mass_qx[:, 0])
        integral_q = torch.trapezoid(mass_q, x_mass, dim=-1)
        balance = integral_qt - diffusion_flux + decay * integral_q - integral_force
    balance_loss = balance.square().mean()

    bc_loss = boundary_condition_diagnostic(model, obs, action)
    return pde_loss, balance_loss, bc_loss


def ranking_kl(predicted_cost: torch.Tensor, teacher_cost: torch.Tensor, temperature: float) -> torch.Tensor:
    target = torch.softmax(-teacher_cost / temperature, dim=-1)
    predicted_log_prob = torch.log_softmax(-predicted_cost / temperature, dim=-1)
    return torch.nn.functional.kl_div(predicted_log_prob, target, reduction="batchmean")
