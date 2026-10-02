"""Teacher counterfactual continuations for train/validation/calibration parents."""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from pdno.controllers.linear import nominal_lqr_action
from pdno.controllers.actions import project_box_slew
from pdno.data.generate import _dirichlet_actuators, _periodic_actuators
from pdno.physics.burgers import burgers_imex_split_step
from pdno.physics.heat import heat_cn_step


def restrict_field(field: np.ndarray, pde: str, nmodel: int = 128) -> np.ndarray:
    if pde == "burgers":
        if field.size != 2 * nmodel:
            raise ValueError("periodic truth field must be twice the model resolution")
        return field[::2].astype(np.float32)
    ntruth = field.size
    x_truth = np.arange(1, ntruth + 1) / (ntruth + 1)
    x_model = np.arange(1, nmodel + 1) / (nmodel + 1)
    return np.interp(x_model, np.r_[0.0, x_truth, 1.0], np.r_[0.0, field, 0.0]).astype(np.float32)


def feasible_candidates(pde: str, previous: np.ndarray, nominal: np.ndarray) -> np.ndarray:
    low, high, delta = (-1.0, 1.0, 0.15) if pde == "burgers" else (0.0, 1.0, 0.10)
    grid = [project_box_slew(previous + [d1, d2], previous, low, high, delta)
            for d1 in (-delta, 0.0, delta) for d2 in (-delta, 0.0, delta)]
    grid.append(project_box_slew(nominal, previous, low, high, delta))
    return np.asarray(grid, dtype=np.float32)


def _snapshot_observation(data: dict[str, np.ndarray], parent: int, tick: int, pde: str,
                          material: np.ndarray, goal_field: np.ndarray) -> tuple[dict, np.ndarray]:
    start = max(0, tick - 7)
    indices = np.arange(start, tick + 1)
    pad = 8 - indices.size
    if pde == "burgers":
        previous = np.zeros(2, dtype=np.float32) if tick == 0 else data["action_applied"][parent, tick - 1]
    else:
        previous = np.full(2, 0.4, dtype=np.float32) if tick == 0 else data["action_applied"][parent, tick - 1]
    applied_hist = np.zeros((8, 2), dtype=np.float32)
    action_start = max(0, tick - 8)
    action_indices = np.arange(action_start, tick)
    action_pad = 8 - action_indices.size
    if action_indices.size:
        applied_hist[action_pad:] = data["action_applied"][parent, action_indices]
    sensor_value = data["sensor_value"][parent, indices]
    sensor_mask = data["sensor_mask"][parent, indices]
    sensor_age = data["sensor_age"][parent, indices]
    obs = {
        "sensor_value": np.pad(sensor_value, ((pad, 0), (0, 0))),
        "sensor_mask": np.pad(sensor_mask, ((pad, 0), (0, 0))),
        "sensor_age": np.pad(sensor_age, ((pad, 0), (0, 0))),
        "sensor_capture_time": np.pad(data["sensor_capture_time"][parent, indices], ((pad, 0), (0, 0)), constant_values=-1),
        "sensor_receive_time": np.pad(data["sensor_receive_time"][parent, indices], ((pad, 0), (0, 0)), constant_values=-1),
        "instrument_image": data["instrument_image"][parent, tick],
        "image_mask": data["image_mask"][parent, tick],
        "image_age": np.asarray(data["image_age"][parent, tick], dtype=np.int16),
        "image_valid": np.asarray(data["image_valid"][parent, tick], dtype=bool),
        "image_capture_time": np.asarray(data["image_capture_time"][parent, tick], dtype=np.int32),
        "image_receive_time": np.asarray(data["image_receive_time"][parent, tick], dtype=np.int32),
        "previous_applied_action": previous.astype(np.float32),
        "applied_action_history": applied_hist,
        "material_context": material.astype(np.float32),
        "goal_field": goal_field.astype(np.float32),
        "goal_coefficients": np.zeros(33, dtype=np.float32) if pde == "burgers" else np.pad(
            np.linalg.lstsq(np.stack([np.sin(k * np.pi * (np.arange(1, len(goal_field) + 1) / (len(goal_field) + 1))) for k in range(1, 33)], axis=1), goal_field, rcond=None)[0], (0, 0)),
    }
    return obs, previous


def _cost(future: np.ndarray, goal: np.ndarray, action: np.ndarray, previous: np.ndarray,
          q_max: float, q_min: float | None = None) -> tuple[float, float]:
    error = future - goal[None, :]
    tracking = float(np.mean(error**2))
    effort = 0.01 * float(np.sum(action**2))
    slew = 0.05 * float(np.sum((action - previous) ** 2))
    if q_min is None:
        violation_field = np.maximum(np.abs(future) - q_max, 0.0) ** 2
        peak = float(np.max(np.abs(future)))
    else:
        violation_field = np.maximum(q_min - future, 0.0) ** 2 + np.maximum(future - q_max, 0.0) ** 2
        peak = float(np.max(future))
    violation = 10.0 * float(np.mean(violation_field))
    return tracking + effort + slew + violation, peak


def _future_burgers(initial: np.ndarray, nu: float, action: np.ndarray, ticks: int = 8) -> np.ndarray:
    q = initial.astype(np.float64).copy()
    basis = _periodic_actuators(q.size)
    forcing = action @ basis  # unknown future disturbance is zero by construction
    dt = 2.5e-4
    outputs = []
    for _ in range(ticks):
        for _ in range(80):
            q = burgers_imex_split_step(q, nu, dt, forcing)
        outputs.append(restrict_field(q, "burgers"))
    return np.asarray(outputs, dtype=np.float32)


def _future_heat(initial: np.ndarray, kappa: float, decay: float, action: np.ndarray, ticks: int = 8) -> np.ndarray:
    u = initial.astype(np.float64).copy()
    n = u.size
    dx = 1.0 / (n + 1)
    forcing = action @ _dirichlet_actuators(n)
    outputs = []
    for _ in range(ticks):
        u = heat_cn_step(u, dx, 0.02, kappa, decay, forcing)
        outputs.append(restrict_field(u, "heat"))
    return np.asarray(outputs, dtype=np.float32)


def build_teacher_queries(shard_path: Path, pde: str, split: str, output_path: Path,
                          snapshots_per_parent: int, parent_limit: int | None = None,
                          q_max_override: float | None = None) -> dict:
    if split not in ("train", "validation", "calibration"):
        raise ValueError("teacher targets may only be generated for train/validation/calibration; locked roles are sealed")
    with np.load(shard_path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    parent_count = data["state_true"].shape[0]
    parent_indices = range(parent_count if parent_limit is None else min(parent_count, parent_limit))
    time_indices = np.linspace(8, 120, snapshots_per_parent, dtype=int)
    heat_qmax = (1.25 * float(np.max(data["goal"])) if q_max_override is None else float(q_max_override)) if pde == "heat" else 1.2
    rows = []
    started = time.perf_counter()
    for parent in parent_indices:
        nu = float(data["nu"][parent]) if pde == "burgers" else None
        kappa = float(data["kappa"][parent]) if pde == "heat" else None
        decay = float(data["decay"][parent]) if pde == "heat" else None
        goal = np.zeros(128, dtype=np.float32) if pde == "burgers" else restrict_field(data["goal"][parent], "heat")
        material = (np.asarray([nu, -1, 1, 0.15, 0.02, 0.16, 1.2, 128], dtype=np.float32) if pde == "burgers"
                    else np.asarray([kappa, decay, 0, 1, 0.10, 0.02, 0.16, 128], dtype=np.float32))
        for tick in time_indices:
            obs, previous = _snapshot_observation(data, parent, int(tick), pde, material,
                                                  np.zeros(256, dtype=np.float32) if pde == "burgers" else data["goal"][parent])
            nominal = nominal_lqr_action(obs, pde, previous)
            candidates = feasible_candidates(pde, previous, nominal)
            initial = data["state_true"][parent, tick]
            future = np.stack([_future_burgers(initial, nu, action) if pde == "burgers"
                               else _future_heat(initial, kappa, decay, action) for action in candidates])
            costs_peaks = [_cost(future[k], goal, candidates[k], previous,
                                 heat_qmax if pde == "heat" else 1.2,
                                 q_min=0.0 if pde == "heat" else None)
                           for k in range(len(candidates))]
            costs = np.asarray([v[0] for v in costs_peaks], dtype=np.float32)
            peaks = np.asarray([v[1] for v in costs_peaks], dtype=np.float32)
            rows.append({"parent_id": str(data["parent_id"][parent]), "decision_tick": int(tick),
                         "observation": obs, "candidates": candidates, "future": future,
                         "initial_field": restrict_field(initial, pde),
                         "teacher_cost": costs, "teacher_peak": peaks, "teacher_best_index": int(np.argmin(costs)),
                         "goal": goal, "q_min": 0.0 if pde == "heat" else -1.2,
                         "q_max": heat_qmax if pde == "heat" else 1.2})
    elapsed = time.perf_counter() - started
    keys = ("sensor_value", "sensor_mask", "sensor_age", "sensor_capture_time", "sensor_receive_time",
            "instrument_image", "image_mask", "image_age", "image_valid", "image_capture_time", "image_receive_time",
            "previous_applied_action", "applied_action_history", "material_context", "goal_field", "goal_coefficients")
    arrays = {key: np.stack([row["observation"][key] for row in rows]) for key in keys}
    arrays.update({
        "parent_id": np.asarray([row["parent_id"] for row in rows]),
        "decision_tick": np.asarray([row["decision_tick"] for row in rows], dtype=np.int16),
        "candidate_action": np.stack([row["candidates"] for row in rows]),
        "future_field": np.stack([row["future"] for row in rows]),
        "initial_field": np.stack([row["initial_field"] for row in rows]),
        "teacher_cost": np.stack([row["teacher_cost"] for row in rows]),
        "teacher_peak": np.stack([row["teacher_peak"] for row in rows]),
        "teacher_best_index": np.asarray([row["teacher_best_index"] for row in rows], dtype=np.int8),
        "goal": np.stack([row["goal"] for row in rows]),
        "q_min": np.asarray([row["q_min"] for row in rows], dtype=np.float32),
        "q_max": np.asarray([row["q_max"] for row in rows], dtype=np.float32),
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    record = {"pde": pde, "split": split, "parent_count": len(set(arrays["parent_id"].tolist())),
              "query_count": len(rows), "snapshots_per_parent": snapshots_per_parent,
              "runtime_seconds": elapsed, "seconds_per_query": elapsed / len(rows),
              "generator_version": "v3-fresh-parent-split-signed-thermal-constraint",
              "path": str(output_path), "action_hold": "constant for horizon; first tick applied in closed loop",
              "future_disturbance": "zero in supervised counterfactuals", "candidate_count": 10,
              "state_constraint": "Burgers symmetric; heat lower=0 and train/validation-frozen upper",
              "targets_generated": True, "locked_test_opened": False}
    output_path.with_suffix(".record.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record
