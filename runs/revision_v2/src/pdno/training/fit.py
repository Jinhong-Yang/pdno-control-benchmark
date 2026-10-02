from __future__ import annotations

import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from pdno.controllers.objectives import b3_surrogate_gradient_step, candidate_cost
from pdno.models.operators import ActionFactorizedOperator, CandidateConditionedOperator
from pdno.models.policies import DirectPolicy
from pdno.training.losses import (boundary_condition_diagnostic, observation_consistency_loss,
                                  pde_residual_loss, ranking_kl)


def _load(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    with np.load(path, allow_pickle=False) as z:
        arrays = {key: z[key] for key in z.files}
    return {key: torch.as_tensor(value, device=device) for key, value in arrays.items() if key != "parent_id"}


def _obs(batch: dict[str, torch.Tensor], ix: torch.Tensor) -> dict[str, torch.Tensor]:
    keys = ("sensor_value", "sensor_mask", "sensor_age", "instrument_image", "image_mask", "goal_coefficients",
            "material_context", "previous_applied_action", "applied_action_history", "image_age", "image_valid")
    return {key: batch[key][ix] for key in keys}


def _grid(pde: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x = (torch.arange(128, device=device, dtype=torch.float32) / 128 if pde == "burgers"
         else (torch.arange(128, device=device, dtype=torch.float32) + 1) / 129)
    tau = torch.linspace(1 / 8, 1, 8, device=device, dtype=torch.float32)
    return x, tau


def _save_checkpoint(path: Path, model, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save({"state_dict": model.state_dict(), "metadata": meta}, temporary)
    os.replace(temporary, path)


def train_operator(query_path: Path, pde: str, method: str, seed: int, steps: int, batch_size: int,
                   out_path: Path, device_name: str = "cuda", rank: int = 32, lambda_phys: float = 0.1,
                   lambda_balance: float = 0.01, lr: float = 1e-3, timing_warmup: int = 10,
                   fused_optimizer: bool = False) -> dict:
    if method not in {"P", "B4", "B5"}:
        raise ValueError("operator method must be P, B4 or B5")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    train = _load(query_path, device)
    goal_dim = 33 if pde == "burgers" else 32
    model = (ActionFactorizedOperator(pde, goal_dim, rank=rank) if method in {"P", "B5"}
             else CandidateConditionedOperator(pde, goal_dim, rank=rank)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4,
                                  fused=fused_optimizer and device.type == "cuda")
    x_grid, tau_grid = _grid(pde, device)
    teacher_iqr = float(torch.quantile(train["teacher_cost"].float(), 0.75) - torch.quantile(train["teacher_cost"].float(), 0.25))
    temperature = max(0.1 * teacher_iqr, 1e-4)
    qmax = float(train["q_max"].float().max())
    q_scale = max(float(torch.sqrt(torch.mean(train["initial_field"].float().square()))), 1e-3)
    if pde == "burgers":
        nu_max = float(train["material_context"][:, 0].float().max())
        residual_scale = max(q_scale / 0.16, q_scale**2, nu_max * (32 * np.pi) ** 2 * q_scale, 1e-3)
    else:
        kappa_max = float(train["material_context"][:, 0].float().max())
        decay_max = float(train["material_context"][:, 1].float().max())
        residual_scale = max(q_scale / 0.16, kappa_max * (32 * np.pi) ** 2 * q_scale, decay_max * q_scale, 1e-3)
    timings, history = [], []
    model.train()
    for step in range(steps):
        ix = torch.randint(0, len(train["candidate_action"]), (batch_size,), device=device)
        obs = _obs(train, ix)
        actions = train["candidate_action"][ix].float()
        target = train["future_field"][ix].float()
        optimizer.zero_grad(set_to_none=True)
        t0 = time.perf_counter()
        prediction = model(obs, actions, x_grid, tau_grid)
        field_loss = F.mse_loss(prediction, target) / (q_scale**2)
        initial_loss, sensor_consistency_loss = observation_consistency_loss(
            model, obs, train["initial_field"][ix].float(), q_scale)
        pred_cost = candidate_cost(prediction, train["goal"][ix].float(), actions,
                                   obs["previous_applied_action"].float(), qmax,
                                   q_min=0.0 if pde == "heat" else None)
        rank_loss = ranking_kl(pred_cost, train["teacher_cost"][ix].float(), temperature)
        physics_loss = torch.zeros((), device=device)
        balance_loss = torch.zeros((), device=device)
        bc_loss = torch.zeros((), device=device)
        effective_phys = lambda_phys if method != "B5" else 0.0
        effective_balance = lambda_balance if method != "B5" else 0.0
        sample_index = torch.arange(len(ix), device=device)
        random_candidate_index = torch.randint(actions.shape[1], (len(ix),), device=device)
        sampled_actions = actions[sample_index, random_candidate_index][:, None, :]
        bc_loss = boundary_condition_diagnostic(model, obs, sampled_actions)
        if effective_phys or effective_balance:
            collocation_x = torch.rand((len(ix), 32), device=device)
            collocation_tau = torch.rand((len(ix), 32), device=device).clamp_min(0.02)
            physics_loss, balance_loss, bc_loss = pde_residual_loss(
                model, obs, sampled_actions, collocation_x, collocation_tau, physical_time_horizon=0.16)
            physics_loss = physics_loss / (residual_scale**2)
            balance_loss = balance_loss / (residual_scale**2)
        ramp = min(1.0, (step + 1) / max(1.0, 0.2 * steps))
        effective_phys *= ramp
        effective_balance *= ramp
        loss = (field_loss + initial_loss + sensor_consistency_loss + bc_loss + 0.1 * rank_loss
                + effective_phys * physics_loss + effective_balance * balance_loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        if step >= timing_warmup:
            timings.append(elapsed)
        if step == 0 or (step + 1) % max(1, steps // 10) == 0 or step + 1 == steps:
            history.append({"step": step + 1, "loss": float(loss.detach()), "field_loss": float(field_loss.detach()),
                            "initial_field_loss": float(initial_loss.detach()),
                            "sensor_consistency_loss": float(sensor_consistency_loss.detach()),
                            "boundary_diagnostic_loss": float(bc_loss.detach()),
                            "physics_residual_scale": residual_scale,
                            "rank_loss": float(rank_loss.detach()), "physics_loss": float(physics_loss.detach()),
                            "balance_loss": float(balance_loss.detach())})
    meta = {"method": method, "pde": pde, "seed": seed, "steps": steps, "batch_size": batch_size,
            "rank": rank, "temperature": temperature, "lambda_phys": lambda_phys if method != "B5" else 0,
            "lambda_balance": lambda_balance if method != "B5" else 0, "device": str(device),
            "physics_residual_scale": residual_scale, "fused_optimizer": fused_optimizer and device.type == "cuda",
            "parameter_count": sum(p.numel() for p in model.parameters()), "query_path": str(query_path), "history": history}
    _save_checkpoint(out_path, model, meta)
    summary = {**meta, "checkpoint": str(out_path), "step_seconds_median": float(np.median(timings)) if timings else None,
               "step_seconds_p90": float(np.quantile(timings, 0.9)) if timings else None,
               "step_seconds_count": len(timings), "elapsed_seconds": float(sum(timings)) if timings else None}
    out_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def train_bc(query_path: Path, pde: str, seed: int, steps: int, batch_size: int, out_path: Path,
             device_name: str = "cuda", lr: float = 1e-3, timing_warmup: int = 50) -> dict:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    data = _load(query_path, device)
    model = DirectPolicy(33 if pde == "burgers" else 32, pde).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    history, timings = [], []
    for step in range(steps):
        ix = torch.randint(0, len(data["candidate_action"]), (batch_size,), device=device)
        obs = _obs(data, ix)
        target_idx = data["teacher_best_index"][ix].long()
        target_action = data["candidate_action"][ix].gather(1, target_idx[:, None, None].expand(-1, 1, 2)).squeeze(1).float()
        optimizer.zero_grad(set_to_none=True)
        start = time.perf_counter()
        action = model(obs)
        loss = F.mse_loss(action, target_action)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if device.type == "cuda": torch.cuda.synchronize()
        if step >= timing_warmup: timings.append(time.perf_counter() - start)
        if step == 0 or (step + 1) % max(1, steps // 10) == 0 or step + 1 == steps:
            history.append({"step": step + 1, "bc_loss": float(loss.detach())})
    meta = {"method": "B2", "pde": pde, "seed": seed, "steps": steps, "batch_size": batch_size,
            "device": str(device), "parameter_count": sum(p.numel() for p in model.parameters()), "history": history}
    _save_checkpoint(out_path, model, meta)
    summary = {**meta, "checkpoint": str(out_path), "step_seconds_median": float(np.median(timings)) if timings else None,
               "step_seconds_p90": float(np.quantile(timings, 0.9)) if timings else None}
    out_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def train_b3(query_path: Path, pde: str, seed: int, steps: int, out_path: Path, b2_checkpoint: Path,
             b4_checkpoint: Path, device_name: str = "cuda", lr: float = 1e-3,
             timing_warmup: int = 50) -> dict:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    data = _load(query_path, device)
    goal_dim = 33 if pde == "burgers" else 32
    policy = DirectPolicy(goal_dim, pde).to(device)
    policy.load_state_dict(torch.load(b2_checkpoint, map_location=device, weights_only=False)["state_dict"])
    b4 = CandidateConditionedOperator(pde, goal_dim, rank=32).to(device)
    b4.load_state_dict(torch.load(b4_checkpoint, map_location=device, weights_only=False)["state_dict"])
    for parameter in b4.parameters(): parameter.requires_grad_(False)
    x, tau = _grid(pde, device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
    timings = []
    for step in range(steps):
        ix = torch.randint(0, len(data["candidate_action"]), (1,), device=device)
        obs = _obs(data, ix)
        start = time.perf_counter()
        summary = b3_surrogate_gradient_step(policy, b4, obs, x, tau, data["goal"][ix].float(), optimizer,
                                             float(data["q_max"][ix].max()),
                                             data["candidate_action"][ix].gather(1, data["teacher_best_index"][ix].long().reshape(1, 1, 1).expand(1, 1, 2)).squeeze(1).float(),
                                             q_min=0.0 if pde == "heat" else None)
        if device.type == "cuda": torch.cuda.synchronize()
        if step >= timing_warmup: timings.append(time.perf_counter() - start)
    meta = {"method": "B3", "pde": pde, "seed": seed, "steps": steps, "device": str(device),
            "frozen_surrogate": str(b4_checkpoint), "BC_initialization": str(b2_checkpoint),
            "parameter_count": sum(p.numel() for p in policy.parameters()), "last_step": summary}
    _save_checkpoint(out_path, policy, meta)
    report = {**meta, "checkpoint": str(out_path), "step_seconds_median": float(np.median(timings)) if timings else None,
              "step_seconds_p90": float(np.quantile(timings, 0.9)) if timings else None}
    out_path.with_suffix(".summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
