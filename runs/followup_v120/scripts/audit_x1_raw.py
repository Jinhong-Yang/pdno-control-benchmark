from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = json.loads((ROOT / "config/X1_SPEC.json").read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(path: Path, K: int, feedback: bool, pde: str) -> dict:
    with np.load(path, allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    n, ticks = len(a["parent_id"]), a["action_applied"].shape[1]
    assert ticks == 200 and a["state_true"].shape == (n, 201, 256)
    assert a["goal"].shape == (n, 200, 256)
    assert a["selected_candidate_index"].shape == (n, 200)
    assert np.isfinite(a["episode_control_cost"]).all()
    assert np.isfinite(a["state_true"]).all() and np.isfinite(a["action_applied"]).all()
    assert a["selected_candidate_index"].min() >= 0
    assert a["selected_candidate_index"].max() < K + int(feedback)
    lo, hi, slew, qmax = (-1., 1., .15, 1.2) if pde == "burgers" else (0., 1., .10, 2.1322593092918396)
    prev = np.concatenate([np.full((n, 1, 2), 0. if pde == "burgers" else .4, dtype=np.float64),
                           a["action_applied"][:, :-1].astype(np.float64)], axis=1)
    act = a["action_applied"].astype(np.float64)
    assert np.all(act >= lo - 1e-7) and np.all(act <= hi + 1e-7)
    assert np.max(np.abs(act - prev)) <= slew + 1e-6
    error = a["state_true"][:, 1:].astype(np.float64) - a["goal"].astype(np.float64)
    if pde == "burgers":
        violation = np.maximum(np.abs(a["state_true"][:, 1:]) - qmax, 0) ** 2
    else:
        values = a["state_true"][:, 1:]
        violation = np.maximum(-values, 0) ** 2 + np.maximum(values - qmax, 0) ** 2
    reconstructed = (error ** 2).mean(axis=(1, 2)) * ticks
    reconstructed += (.01 * np.sum(act ** 2, axis=-1)).sum(axis=1)
    reconstructed += (.05 * np.sum((act - prev) ** 2, axis=-1)).sum(axis=1)
    reconstructed += (10 * violation.mean(axis=-1)).sum(axis=1)
    assert np.allclose(reconstructed, a["episode_control_cost"], rtol=1e-10, atol=2e-7)
    rmse = np.sqrt(np.sum(error ** 2, axis=(1, 2)) / (ticks * 256))
    assert np.allclose(rmse, a["tracking_rmse"], rtol=1e-10, atol=2e-7)
    changes = np.mean(np.abs(act - prev), axis=-1).mean(axis=1)
    assert np.allclose(changes, a["mean_abs_action_change"], rtol=0, atol=1e-8)
    return {"path": str(path), "sha256": sha(path), "n": n, "rows_verified": n,
            "mean_cost": float(a["episode_control_cost"].mean()),
            "min_choice": int(a["selected_candidate_index"].min()),
            "max_choice": int(a["selected_candidate_index"].max()),
            "status": "PASS_RAW_ARRAY_RECONSTRUCTION"}


def main():
    checks = []
    for pde, population in (("burgers", "burgers_nominal"), ("heat", "heat_nonnegative_nominal")):
        ids = None
        for cell in SPEC["cells"]:
            path = ROOT / "results/X1" / population / cell["id"] / "n128.npz"
            result = audit(path, cell["K"], cell["feedback_rollout"], pde)
            with np.load(path, allow_pickle=False) as z:
                current = z["parent_id"].astype(str)
            if ids is None:
                ids = current
            else:
                assert np.array_equal(ids, current), f"scenario cohort/order mismatch: {path}"
            checks.append(result)
    receipt = {"status": "PASS", "checked_arrays": len(checks), "checks": checks,
               "scope": "Raw state/action/cost/choice arithmetic and common first128 cohort only; not solver-source or independent physics validation."}
    out = ROOT / "results/X1_RAW_AUDIT.json"
    out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "checked_arrays": len(checks), "path": str(out)}, indent=2))


if __name__ == "__main__":
    main()
