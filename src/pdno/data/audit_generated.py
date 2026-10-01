"""Audit generated train/dev arrays for causal and action-accounting invariants."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import numpy as np


def audit_npz(path: Path, pde: str) -> dict:
    errors = []
    with np.load(path, allow_pickle=False) as f:
        arrays = {key: f[key] for key in f.files}
    required = {"parent_id", "state_true", "action_proposed", "action_projected", "action_applied",
                "sensor_value", "sensor_mask", "sensor_age", "sensor_capture_time", "sensor_receive_time",
                "instrument_image", "image_mask", "image_age", "image_valid", "image_capture_time", "image_receive_time"}
    missing = sorted(required - arrays.keys())
    if missing:
        errors.append(f"missing arrays: {missing}")
        return {"passed": False, "errors": errors}
    ids = arrays["parent_id"].astype(str)
    if len(ids) != len(set(ids.tolist())):
        errors.append("duplicate parent IDs in shard")
    truth = arrays["state_true"]
    n, time_count = truth.shape[:2]
    for key in required - {"parent_id"}:
        if arrays[key].shape[0] != n:
            errors.append(f"leading parent dimension differs for {key}")
    if not np.isfinite(truth).all():
        errors.append("nonfinite truth field")
    proposed, projected, applied = (arrays[key] for key in ("action_proposed", "action_projected", "action_applied"))
    action_count = max(time_count - 1, 0)
    if proposed.shape != (n, action_count, 2) or projected.shape != proposed.shape or applied.shape != proposed.shape:
        errors.append("action proposal/projection/applied arrays must share [parent,time_point-1,2] shape")
    if not np.isfinite(proposed).all() or not np.isfinite(projected).all() or not np.isfinite(applied).all():
        errors.append("nonfinite proposed/projected/applied action")
    decision = np.arange(time_count)[None, :, None]
    sm = arrays["sensor_mask"]
    sc, sr = arrays["sensor_capture_time"], arrays["sensor_receive_time"]
    if np.any(sm & ((sc > decision) | (sr > decision) | (sc < 0) | (sr < 0))):
        errors.append("sensor availability violates capture/receive <= decision")
    if np.any(sm & (arrays["sensor_age"] != decision - sc)):
        errors.append("sensor age differs from decision minus capture time")
    im = arrays["image_valid"]
    ic, ir = arrays["image_capture_time"], arrays["image_receive_time"]
    image_decision = np.arange(time_count)
    if np.any(im & ((ic > image_decision) | (ir > image_decision) | (ic < 0) | (ir < 0))):
        errors.append("image availability violates capture/receive <= decision")
    if np.any(im & (arrays["image_age"] != image_decision - ic)):
        errors.append("image age differs from decision minus capture time")
    if not np.array_equal(projected, applied):
        errors.append("stored simulator-applied actions differ from projection")
    low, high, slew = (-1.0, 1.0, 0.15) if pde == "burgers" else (0.0, 1.0, 0.10)
    if np.any(projected < low - 1e-6) or np.any(projected > high + 1e-6):
        errors.append("projected action outside action box")
    prior = np.concatenate([np.zeros((n, 1, 2)) if pde == "burgers" else np.full((n, 1, 2), 0.4), applied[:, :-1]], axis=1)
    if np.any(np.abs(applied - prior) > slew + 2e-6):
        errors.append("applied action violates slew limit relative to previous applied action")
    return {"passed": not errors, "errors": errors, "parent_count": int(n), "time_points": int(time_count),
            "parent_id_count": int(len(ids)),
            "parent_id_sha256": hashlib.sha256("\n".join(sorted(ids.tolist())).encode("utf-8")).hexdigest(),
            "action_count": int(action_count), "finite_truth": bool(np.isfinite(truth).all()),
            "all_applied_equal_projected": bool(np.array_equal(projected, applied))}


def audit_tree(root: Path) -> dict:
    outputs = []
    for path in sorted(root.glob("*/**/trajectories.npz")):
        pde = path.parent.name
        result = audit_npz(path, pde)
        outputs.append({"path": str(path), **result})
    errors = [f"{row['path']}: {e}" for row in outputs for e in row["errors"]]
    return {"passed": bool(outputs) and not errors, "errors": errors, "shards": outputs,
            "locked_role_outputs_found": any(part.startswith("locked_") or part == "calibration" for p in outputs for part in Path(p["path"]).parts)}
