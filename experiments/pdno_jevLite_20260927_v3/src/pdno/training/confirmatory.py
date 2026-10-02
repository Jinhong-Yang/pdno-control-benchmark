"""Validation-selected staged fitting for the locked confirmatory model families."""

from __future__ import annotations

import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from pdno.controllers.objectives import b3_surrogate_gradient_step, candidate_cost
from pdno.models.encoders import ObservationEncoder
from pdno.models.operators import ActionFactorizedOperator, CandidateConditionedOperator, spatial_basis
from pdno.models.policies import DirectPolicy
from pdno.models.encoders import load_state_with_normalizer_defaults
from pdno.training.fit import _grid, _load, _obs, _save_checkpoint
from pdno.training.losses import (boundary_condition_diagnostic, observation_consistency_loss,
                                  pde_residual_loss, ranking_kl)


class _Observer(nn.Module):
    def __init__(self, pde: str):
        super().__init__()
        self.pde = pde
        self.encoder = ObservationEncoder(33 if pde == "burgers" else 32)
        self.initial_head = nn.Linear(128, 33 if pde == "burgers" else 32)

    def forward(self, obs, x):
        return torch.einsum("bq,qn->bn", self.initial_head(self.encoder(obs)), spatial_basis(self.pde, x))

    def predict_initial(self, obs, x):
        return self.forward(obs, x)


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _relative_nrmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.sqrt((pred - target).square().sum() / target.square().sum().clamp_min(1e-12)))


@torch.no_grad()
def _evaluate_observer(model: _Observer, data: dict[str, torch.Tensor], pde: str,
                       x: torch.Tensor, batch_size: int = 64) -> float:
    model.eval()
    numerator = torch.zeros((), device=x.device)
    denominator = torch.zeros((), device=x.device)
    for start in range(0, len(data["initial_field"]), batch_size):
        ids = torch.arange(start, min(start + batch_size, len(data["initial_field"])), device=x.device)
        pred = model(_obs(data, ids), x)
        target = data["initial_field"][ids].float()
        numerator += (pred - target).square().sum()
        denominator += target.square().sum()
    return float(torch.sqrt(numerator / denominator.clamp_min(1e-12)))


def train_observer(train_path: Path, validation_path: Path, pde: str, output_path: Path,
                   seed: int = 7, max_updates: int = 1000, eval_interval: int = 100,
                   batch_size: int = 64, device_name: str = "cuda") -> dict:
    """Warm up one fixed observer per PDE; every learned method loads this checkpoint."""
    _seed(seed)
    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    train, validation = _load(train_path, device), _load(validation_path, device)
    model = _Observer(pde).to(device)
    model.encoder.fit_input_normalizer(train["sensor_value"], train["sensor_mask"], train["material_context"])
    x, _ = _grid(pde, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_updates)
    qscale = max(float(torch.sqrt(train["initial_field"].float().square().mean())), 1e-3)
    history, best, best_step = [], float("inf"), 0
    started = time.perf_counter()
    for step in range(1, max_updates + 1):
        model.train()
        ids = torch.randint(0, len(train["initial_field"]), (batch_size,), device=device)
        target = train["initial_field"][ids].float()
        optimizer.zero_grad(set_to_none=True)
        obs = _obs(train, ids)
        field_obs_loss, sensor_obs_loss = observation_consistency_loss(model, obs, target, qscale)
        loss = field_obs_loss + sensor_obs_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); schedule.step()
        if step % eval_interval == 0 or step == max_updates:
            score = _evaluate_observer(model, validation, pde, x)
            history.append({"step": step, "validation_nrmse": score, "train_normalized_loss": float(loss.detach()),
                            "train_initial_field_loss": float(field_obs_loss.detach()),
                            "train_sensor_consistency_loss": float(sensor_obs_loss.detach())})
            if score < best:
                best, best_step = score, step
                _save_checkpoint(output_path, model, {"method": "observer", "pde": pde, "seed": seed,
                    "step": step, "validation_nrmse": score, "q_scale": qscale,
                    "input_normalizer_fit_split": "train",
                    "train_query_path": str(train_path), "validation_query_path": str(validation_path)})
    report = {"method": "observer", "pde": pde, "seed": seed, "updates": max_updates,
            "best_step": best_step, "best_validation_nrmse": best,
            "elapsed_seconds": time.perf_counter() - started, "history": history,
            "checkpoint": str(output_path)}
    output_path.with_suffix(".summary.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    return report


def _load_observer(model, path: Path, freeze: bool = True) -> None:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    state = saved["state_dict"]
    encoder_state = {key.removeprefix("encoder."): value for key, value in state.items()
                     if key.startswith("encoder.")}
    if not encoder_state:
        raise ValueError("observer checkpoint has no encoder.* weights")
    load_state_with_normalizer_defaults(model.encoder, encoder_state)
    if hasattr(model, "initial_head"):
        head_state = {key.removeprefix("initial_head."): value for key, value in state.items()
                      if key.startswith("initial_head.")}
        if not head_state:
            raise ValueError("observer checkpoint has no initial_head.* weights")
        model.initial_head.load_state_dict(head_state)
    if freeze:
        for parameter in model.encoder.parameters():
            parameter.requires_grad_(False)
        if hasattr(model, "initial_head"):
            for parameter in model.initial_head.parameters():
                parameter.requires_grad_(False)


@torch.no_grad()
def _evaluate_operator(model, validation: dict[str, torch.Tensor], train_cost_iqr: float,
                       pde: str, x: torch.Tensor, tau: torch.Tensor, qmax: float,
                       batch_size: int = 32) -> dict:
    model.eval()
    field_num = torch.zeros((), device=x.device)
    field_den = torch.zeros((), device=x.device)
    regrets, agreements = [], []
    for start in range(0, len(validation["candidate_action"]), batch_size):
        ids = torch.arange(start, min(start + batch_size, len(validation["candidate_action"])), device=x.device)
        obs = _obs(validation, ids)
        actions = validation["candidate_action"][ids].float()
        pred = model(obs, actions, x, tau)
        target = validation["future_field"][ids].float()
        field_num += (pred - target).square().sum()
        field_den += target.square().sum()
        cost = candidate_cost(pred, validation["goal"][ids].float(), actions,
                              obs["previous_applied_action"].float(), qmax,
                              q_min=0.0 if pde == "heat" else None)
        teacher = validation["teacher_cost"][ids].float()
        choice = cost.argmin(dim=1)
        chosen_teacher = teacher.gather(1, choice[:, None]).squeeze(1)
        regrets.extend(((chosen_teacher - teacher.min(dim=1).values) / max(train_cost_iqr, 1e-4)).cpu().tolist())
        agreements.extend((choice == teacher.argmin(dim=1)).float().cpu().tolist())
    nrmse = float(torch.sqrt(field_num / field_den.clamp_min(1e-12)))
    regret = float(np.mean(regrets))
    # Frozen composite for checkpoint selection: equal weight to relative field error and
    # train-IQR-normalized counterfactual teacher regret. Weights are common to P/B4/B5.
    score = 0.5 * nrmse + 0.5 * regret
    return {"validation_field_nrmse": nrmse, "validation_normalized_teacher_regret_mean": regret,
            "validation_teacher_best_agreement": float(np.mean(agreements)), "selection_score": score}


def train_operator_staged(train_path: Path, validation_path: Path, observer_path: Path, pde: str,
                          method: str, seed: int, output_path: Path, lambda_phys: float = 0.01,
                          lambda_balance: float = 0.01, lambda_obs: float = 1.0,
                          lambda_bc: float = 1.0, lambda_rank: float = 0.1, field_updates: int = 2000,
                          physics_updates: int = 3000, eval_interval: int = 100,
                          batch_size: int = 64, device_name: str = "cuda",
                          candidate_checkpoint_dir: Path | None = None) -> dict:
    if method not in {"P", "B4", "B5", "P-no-rank"}:
        raise ValueError("method must be P, B4, B5 or P-no-rank")
    _seed(seed)
    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    train, validation = _load(train_path, device), _load(validation_path, device)
    goal_dim = 33 if pde == "burgers" else 32
    model = (ActionFactorizedOperator(pde, goal_dim, rank=32) if method in {"P", "B5", "P-no-rank"}
             else CandidateConditionedOperator(pde, goal_dim, rank=32)).to(device)
    _load_observer(model, observer_path)
    x, tau = _grid(pde, device)
    train_iqr = float(torch.quantile(train["teacher_cost"].float().flatten(), .75) -
                      torch.quantile(train["teacher_cost"].float().flatten(), .25))
    temperature = max(0.1 * train_iqr, 1e-4)
    qscale = max(float(torch.sqrt(train["initial_field"].float().square().mean())), 1e-3)
    if pde == "burgers":
        residual_scale = max(qscale / .16, qscale**2,
                             float(train["material_context"][:, 0].float().max()) * (32 * np.pi)**2 * qscale, 1e-3)
    else:
        material = train["material_context"].float()
        residual_scale = max(qscale / .16, float(material[:, 0].max()) * (32 * np.pi)**2 * qscale,
                             float(material[:, 1].max()) * qscale, 1e-3)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=1e-4)
    total_updates = field_updates + physics_updates
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_updates)
    history, best, best_step, stale = [], float("inf"), 0, 0
    started = time.perf_counter()
    update = 0
    for phase, phase_updates in (("field", field_updates), ("physics_ranking", physics_updates)):
        stale = 0
        for local_step in range(1, phase_updates + 1):
            update += 1
            model.train()
            ids = torch.randint(0, len(train["candidate_action"]), (batch_size,), device=device)
            obs = _obs(train, ids)
            actions, target = train["candidate_action"][ids].float(), train["future_field"][ids].float()
            optimizer.zero_grad(set_to_none=True)
            pred = model(obs, actions, x, tau)
            field_loss = F.mse_loss(pred, target) / (qscale**2)
            initial_field_loss, sensor_consistency_loss = observation_consistency_loss(
                model, obs, train["initial_field"][ids].float(), qscale)
            l_obs = initial_field_loss + sensor_consistency_loss
            phys_loss = torch.zeros((), device=device)
            balance_loss = torch.zeros((), device=device)
            rank_loss = torch.zeros((), device=device)
            sample_index = torch.arange(len(ids), device=device)
            random_candidate_index = torch.randint(actions.shape[1], (len(ids),), device=device)
            sampled_actions = actions[sample_index, random_candidate_index][:, None, :]
            bc_loss = boundary_condition_diagnostic(model, obs, sampled_actions)
            if phase == "physics_ranking":
                qmax = float(train["q_max"].max())
                pred_cost = candidate_cost(pred, train["goal"][ids].float(), actions,
                                           obs["previous_applied_action"].float(), qmax,
                                           q_min=0.0 if pde == "heat" else None)
                rank_loss = ranking_kl(pred_cost, train["teacher_cost"][ids].float(), temperature)
                collocation_x = torch.rand((len(ids), 32), device=device)
                collocation_tau = torch.rand((len(ids), 32), device=device).clamp_min(.02)
                phys_loss, balance_loss, bc_loss = pde_residual_loss(
                    model, obs, sampled_actions, collocation_x, collocation_tau)
                phys_loss = phys_loss / residual_scale**2
                balance_loss = balance_loss / residual_scale**2
                ramp = min(1.0, local_step / max(1.0, .2 * physics_updates))
                lam_phys, lam_balance = ((0.0, 0.0) if method == "B5"
                                         else (lambda_phys, lambda_balance))
            else:
                ramp = 0.0
                lam_phys, lam_balance = (0.0, 0.0)
            method_rank_weight = 0.0 if method == "P-no-rank" else lambda_rank
            loss = (field_loss + lambda_obs * l_obs + lambda_bc * bc_loss
                    + method_rank_weight * rank_loss
                    + ramp * (lam_phys * phys_loss + lam_balance * balance_loss))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step(); scheduler.step()
            if update % eval_interval == 0 or update == total_updates:
                valid = _evaluate_operator(model, validation, train_iqr, pde, x, tau, float(train["q_max"].max()))
                row = {"update": update, "phase": phase, "train_loss": float(loss.detach()),
                       "train_field_loss": float(field_loss.detach()),
                       "train_initial_field_loss": float(initial_field_loss.detach()),
                       "train_sensor_consistency_loss": float(sensor_consistency_loss.detach()),
                       "train_bc_diagnostic_loss": float(bc_loss.detach()),
                       "train_physics_loss": float(phys_loss.detach()), "train_balance_loss": float(balance_loss.detach()),
                       "train_rank_loss": float(rank_loss.detach()), **valid}
                history.append(row)
                if candidate_checkpoint_dir is not None:
                    candidate_path = candidate_checkpoint_dir / f"update_{update:05d}.pt"
                    _save_checkpoint(candidate_path, model, {"method":method,"pde":pde,"seed":seed,
                        "steps":update,"rank":32,"observer_checkpoint":str(observer_path),
                        "train_query_path":str(train_path),"validation_query_path":str(validation_path),
                        "candidate_validation":valid,"candidate_checkpoint":True})
                significant_improvement = valid["selection_score"] < best * .995
                if valid["selection_score"] < best:
                    best, best_step = valid["selection_score"], update
                    _save_checkpoint(output_path, model, {"method":method,"pde":pde,"seed":seed,
                        "steps":update,"rank":32,"temperature":temperature,
                        "lambda_phys":0.0 if method=="B5" else lambda_phys,
                        "lambda_balance":0.0 if method=="B5" else lambda_balance,
                        "lambda_obs":lambda_obs,"lambda_bc":lambda_bc,"lambda_rank":method_rank_weight,
                        "observer_checkpoint":str(observer_path),
                        "train_query_path":str(train_path),"validation_query_path":str(validation_path),
                        "selection_score":best,"validation":valid,"residual_scale":residual_scale})
                if significant_improvement:
                    stale = 0
                else:
                    stale += 1
                if stale >= 5:
                    break
        if stale >= 5 and phase == "physics_ranking":
            break
    report = {"method":method,"pde":pde,"seed":seed,"best_step":best_step,"selection_score":best,
              "lambda_obs":lambda_obs,"lambda_bc":lambda_bc,
              "lambda_rank":0.0 if method=="P-no-rank" else lambda_rank,
              "max_updates":total_updates,"actual_updates":update,"elapsed_seconds":time.perf_counter()-started,
              "history":history,"checkpoint":str(output_path),"observer_checkpoint":str(observer_path)}
    output_path.with_suffix(".summary.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    return report


def train_direct_staged(train_path: Path, validation_path: Path, observer_path: Path, pde: str,
                        seed: int, output_path: Path, max_updates: int = 3000,
                        eval_interval: int = 100, batch_size: int = 64,
                        device_name: str = "cuda",
                        candidate_checkpoint_dir: Path | None = None) -> dict:
    _seed(seed)
    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    train, validation = _load(train_path, device), _load(validation_path, device)
    model = DirectPolicy(33 if pde == "burgers" else 32, pde).to(device)
    _load_observer(model, observer_path)
    for p in model.encoder.parameters(): p.requires_grad_(False)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_updates)
    best, best_step, stale, history = float("inf"), 0, 0, []
    started = time.perf_counter()
    for step in range(1, max_updates + 1):
        model.train()
        ids = torch.randint(0, len(train["candidate_action"]), (batch_size,), device=device)
        obs = _obs(train, ids)
        target_idx = train["teacher_best_index"][ids].long()
        target = train["candidate_action"][ids].gather(1, target_idx[:, None, None].expand(-1, 1, 2)).squeeze(1).float()
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(obs), target)
        loss.backward(); torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step(); scheduler.step()
        if step % eval_interval == 0 or step == max_updates:
            model.eval()
            errs = []
            with torch.no_grad():
                for start in range(0, len(validation["candidate_action"]), batch_size):
                    ids_v = torch.arange(start, min(start + batch_size, len(validation["candidate_action"])), device=device)
                    vobs = _obs(validation, ids_v)
                    vbest = validation["teacher_best_index"][ids_v].long()
                    vtarget = validation["candidate_action"][ids_v].gather(1, vbest[:, None, None].expand(-1, 1, 2)).squeeze(1).float()
                    errs.extend((model(vobs) - vtarget).square().mean(dim=-1).cpu().tolist())
            score = float(np.mean(errs))
            history.append({"step":step,"train_action_mse":float(loss.detach()),"validation_action_mse":score})
            if candidate_checkpoint_dir is not None:
                _save_checkpoint(candidate_checkpoint_dir / f"update_{step:05d}.pt", model,
                    {"method":"B2","pde":pde,"seed":seed,"steps":step,
                     "observer_checkpoint":str(observer_path),"validation_action_mse":score,
                     "candidate_checkpoint":True})
            significant_improvement = score < best * .995
            if score < best:
                best, best_step = score, step
                _save_checkpoint(output_path, model, {"method":"B2","pde":pde,"seed":seed,"steps":step,
                    "observer_checkpoint":str(observer_path),"train_query_path":str(train_path),
                    "validation_query_path":str(validation_path),"validation_action_mse":best})
            if significant_improvement: stale = 0
            else: stale += 1
            if stale >= 5: break
    report = {"method":"B2","pde":pde,"seed":seed,"best_step":best_step,"validation_action_mse":best,
              "actual_updates":step,"elapsed_seconds":time.perf_counter()-started,"history":history,"checkpoint":str(output_path)}
    output_path.with_suffix(".summary.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    return report


def train_b3_staged(train_path: Path, validation_path: Path, b2_path: Path, b4_path: Path,
                    pde: str, seed: int, output_path: Path, max_updates: int = 1000,
                    eval_interval: int = 100, device_name: str = "cuda",
                    candidate_checkpoint_dir: Path | None = None) -> dict:
    """B3 direct policy: B2 warm start and differentiable held-action cost through frozen B4."""
    _seed(seed)
    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    train, validation = _load(train_path, device), _load(validation_path, device)
    policy = DirectPolicy(33 if pde == "burgers" else 32, pde).to(device)
    policy.load_state_dict(torch.load(b2_path, map_location=device, weights_only=False)["state_dict"])
    surrogate = CandidateConditionedOperator(pde, 33 if pde == "burgers" else 32, rank=32).to(device)
    surrogate.load_state_dict(torch.load(b4_path, map_location=device, weights_only=False)["state_dict"])
    surrogate.eval()
    for parameter in surrogate.parameters(): parameter.requires_grad_(False)
    for parameter in policy.encoder.parameters(): parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_updates)
    x, tau = _grid(pde, device)
    qmax = float(train["q_max"].max())
    best, best_step, stale, history = float("inf"), 0, 0, []
    started = time.perf_counter()
    for step in range(1, max_updates + 1):
        policy.train()
        ids = torch.randint(0, len(train["candidate_action"]), (1,), device=device)
        obs = _obs(train, ids)
        ix = train["teacher_best_index"][ids].long()
        target = train["candidate_action"][ids].gather(1, ix.reshape(1,1,1).expand(1,1,2)).squeeze(1).float()
        optimizer.zero_grad(set_to_none=True)
        action = policy(obs)
        future = surrogate(obs, action[:, None, :], x, tau)
        objective = candidate_cost(future, train["goal"][ids].float(), action[:, None, :],
                                   obs["previous_applied_action"].float(), qmax,
                                   q_min=0.0 if pde == "heat" else None).mean()
        bc_loss = F.mse_loss(action, target)
        loss = objective + 0.1 * bc_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], 1.0)
        optimizer.step(); scheduler.step()
        if step % eval_interval == 0 or step == max_updates:
            policy.eval()
            action_costs, action_errors = [], []
            with torch.no_grad():
                for start in range(0, len(validation["candidate_action"]), 8):
                    ids_v = torch.arange(start, min(start + 8, len(validation["candidate_action"])), device=device)
                    vobs = _obs(validation, ids_v)
                    va = policy(vobs)
                    cand = va[:, None, :]
                    vf = surrogate(vobs, cand, x, tau)
                    vcost = candidate_cost(vf, validation["goal"][ids_v].float(), cand,
                                           vobs["previous_applied_action"].float(), qmax,
                                           q_min=0.0 if pde == "heat" else None).squeeze(1)
                    target_idx = validation["teacher_best_index"][ids_v].long()
                    target_action = validation["candidate_action"][ids_v].gather(
                        1, target_idx[:, None, None].expand(-1, 1, 2)).squeeze(1).float()
                    action_costs.extend(vcost.cpu().tolist())
                    action_errors.extend((va - target_action).square().mean(dim=-1).cpu().tolist())
            # B3 selects with its held-action surrogate cost and a fixed BC regularizer.
            score = float(np.mean(action_costs) + 0.1 * np.mean(action_errors))
            significant_improvement = score < best * .995
            history.append({"step":step,"train_objective":float(objective.detach()),"train_bc_loss":float(bc_loss.detach()),
                            "validation_surrogate_cost_mean":float(np.mean(action_costs)),
                            "validation_teacher_action_mse":float(np.mean(action_errors)),"selection_score":score})
            if candidate_checkpoint_dir is not None:
                _save_checkpoint(candidate_checkpoint_dir / f"update_{step:05d}.pt", policy,
                    {"method":"B3","pde":pde,"seed":seed,"steps":step,
                     "frozen_surrogate":str(b4_path),"BC_initialization":str(b2_path),
                     "validation_score":score,"candidate_checkpoint":True})
            if score < best:
                best, best_step = score, step
                _save_checkpoint(output_path, policy, {"method":"B3","pde":pde,"seed":seed,"steps":step,
                    "frozen_surrogate":str(b4_path),"BC_initialization":str(b2_path),
                    "observer_checkpoint":torch.load(b2_path,map_location="cpu",weights_only=False)["metadata"].get("observer_checkpoint"),
                    "validation_score":score,"train_query_path":str(train_path),"validation_query_path":str(validation_path)})
            if significant_improvement: stale = 0
            else: stale += 1
            if stale >= 5: break
    report = {"method":"B3","pde":pde,"seed":seed,"best_step":best_step,"validation_score":best,
              "actual_updates":step,"elapsed_seconds":time.perf_counter()-started,"history":history,
              "checkpoint":str(output_path),"frozen_surrogate":str(b4_path),"BC_initialization":str(b2_path)}
    output_path.with_suffix(".summary.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    return report
