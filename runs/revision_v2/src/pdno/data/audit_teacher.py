"""Leakage, candidate and causal audit for generated teacher-query shards."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np


def audit_teacher_shard(path: Path, pde: str, allowed_parent_ids: set[str]) -> dict:
    errors = []
    with np.load(path, allow_pickle=False) as f:
        a = {key: f[key] for key in f.files}
    required = {"parent_id", "decision_tick", "sensor_mask", "sensor_age", "sensor_capture_time", "sensor_receive_time",
                "image_valid", "image_age", "image_capture_time", "image_receive_time", "candidate_action", "previous_applied_action",
                "applied_action_history",
                "future_field", "teacher_cost", "teacher_peak", "teacher_best_index", "goal", "q_min", "q_max"}
    missing = sorted(required - a.keys())
    if missing:
        return {"passed": False, "errors": [f"missing keys: {missing}"]}
    ids = a["parent_id"].astype(str)
    unauthorized = sorted(set(ids.tolist()) - allowed_parent_ids)
    if unauthorized:
        errors.append(f"parent IDs outside declared {path.parent.parent.name} role: {unauthorized[:8]}")
    sm = a["sensor_mask"]
    capture, receive = a["sensor_capture_time"], a["sensor_receive_time"]
    # Sensor arrays are an eight-observation history. Each row's age is relative
    # to that historical observation tick, not the current teacher decision tick.
    history_ticks = (a["decision_tick"][:, None] - sm.shape[1] + 1
                     + np.arange(sm.shape[1])[None, :])[:, :, None]
    if np.any(sm & ((capture > history_ticks) | (receive > history_ticks) | (capture < 0) | (receive < capture))):
        errors.append("sensor event timestamp is after query decision")
    if np.any(sm & (a["sensor_age"] != history_ticks - capture)):
        errors.append("sensor age does not equal historical observation tick minus capture tick")
    iv = a["image_valid"]
    image_capture, image_receive = a["image_capture_time"], a["image_receive_time"]
    if np.any(iv & ((image_capture > a["decision_tick"]) | (image_receive > a["decision_tick"])
                    | (image_capture < 0) | (image_receive < image_capture))):
        errors.append("image event timestamp is after query decision")
    if np.any(iv & (a["image_age"] != a["decision_tick"] - image_capture)):
        errors.append("image age does not equal decision tick minus capture tick")
    history = a["applied_action_history"].astype(np.float32)
    previous = a["previous_applied_action"].astype(np.float32)
    if history.shape != (len(ids), 8, 2):
        errors.append(f"applied action history shape is {history.shape}, expected {(len(ids), 8, 2)}")
    elif not np.allclose(history[:, -1], previous, rtol=0.0, atol=1e-6):
        errors.append("last causal applied-action history item differs from previous_applied_action")
    if len(set(zip(ids.tolist(), a["decision_tick"].tolist()))) != len(ids):
        errors.append("duplicate parent_id/decision_tick query rows")
    actions = a["candidate_action"]
    low, high, slew = (-1.0, 1.0, 0.15) if pde == "burgers" else (0.0, 1.0, 0.10)
    if actions.shape[1:] != (10, 2):
        errors.append(f"candidate shape is {actions.shape[1:]}, expected (10,2)")
    if np.any(actions < low - 1e-6) or np.any(actions > high + 1e-6):
        errors.append("candidate violates action box")
    if np.any(np.abs(actions - previous[:, None]) > slew + 1e-6):
        errors.append("candidate violates slew from previous applied action")
    if not np.isfinite(a["future_field"]).all() or not np.isfinite(a["teacher_cost"]).all() or not np.isfinite(a["teacher_peak"]).all():
        errors.append("nonfinite teacher target/cost/peak")
    if np.any(~np.isfinite(a["q_min"])) or np.any(~np.isfinite(a["q_max"])) or np.any(a["q_min"] >= a["q_max"]):
        errors.append("invalid stored state-constraint bounds")
    if pde == "heat" and np.any(a["q_min"] != 0.0):
        errors.append("heat lower constraint is not the frozen normalized ambient bound 0")
    if pde == "burgers" and not np.allclose(a["q_min"], -a["q_max"], rtol=0.0, atol=1e-7):
        errors.append("Burgers stored bounds are not symmetric")
    future = a["future_field"].astype(np.float64)
    goal = a["goal"].astype(np.float64)[:, None, None, :]
    actions64 = actions.astype(np.float64)
    previous64 = previous.astype(np.float64)[:, None, :]
    tracking = np.mean((future - goal) ** 2, axis=(2, 3))
    effort = 0.01 * np.sum(actions64**2, axis=-1)
    slew = 0.05 * np.sum((actions64 - previous64) ** 2, axis=-1)
    if pde == "heat":
        violation = np.maximum(a["q_min"][:, None, None, None] - future, 0.0) ** 2
        violation += np.maximum(future - a["q_max"][:, None, None, None], 0.0) ** 2
        peak = np.max(future, axis=(2, 3))
    else:
        violation = np.maximum(np.abs(future) - a["q_max"][:, None, None, None], 0.0) ** 2
        peak = np.max(np.abs(future), axis=(2, 3))
    recomputed = tracking + effort + slew + 10.0 * np.mean(violation, axis=(2, 3))
    cost_delta = float(np.max(np.abs(recomputed - a["teacher_cost"])))
    peak_delta = float(np.max(np.abs(peak - a["teacher_peak"])))
    if cost_delta > 2e-5:
        errors.append(f"teacher cost does not reproduce stored signed constraint contract (max delta {cost_delta})")
    if peak_delta > 2e-5:
        errors.append(f"teacher peak does not reproduce stored PDE-specific endpoint (max delta {peak_delta})")
    if not np.array_equal(np.argmin(a["teacher_cost"], axis=1), a["teacher_best_index"]):
        errors.append("teacher best index does not match saved teacher costs")
    return {"passed": not errors, "errors": errors, "query_count": int(len(ids)),
            "parent_count": int(len(set(ids.tolist()))), "candidate_count": int(actions.shape[1]),
            "teacher_cost_recompute_max_abs_difference": cost_delta,
            "teacher_peak_recompute_max_abs_difference": peak_delta,
            "parent_id_sha256": __import__("hashlib").sha256("\n".join(sorted(set(ids.tolist()))).encode()).hexdigest()}


def audit_teacher_tree(root: Path, manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    parent_roles = {}
    for pde, group in manifest["roles"].items():
        for row in group["parents"]:
            parent_roles[row["parent_id"]] = row["split"]
    results, errors = [], []
    for path in sorted(root.glob("*/**/teacher_queries.npz")):
        split, pde = path.parent.parent.name, path.parent.name
        allowed = {parent for parent, role in parent_roles.items() if role == split}
        result = audit_teacher_shard(path, pde, allowed)
        if split not in {"train", "validation"}:
            result["errors"].append(f"disallowed teacher-query role {split}")
            result["passed"] = False
        results.append({"path": str(path), "split": split, "pde": pde, **result})
        errors.extend(f"{path}: {error}" for error in result["errors"])
    return {"passed": bool(results) and not errors, "errors": errors, "shards": results,
            "calibration_or_test_query_shards_found": any(row["split"] not in {"train", "validation"} for row in results)}
