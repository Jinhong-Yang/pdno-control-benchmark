"""Freeze nominal calibration-only empirical forecast margins for v3 predictors."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pdno.evaluation.closed_loop import load_controller, policy_action  # noqa: E402

PREDICTIVE = ("B1", "P", "P-no-rank", "B4", "B5")
SEEDS = (11, 23, 37)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    out = ROOT / "evidence" / "calibration_lock_v3.json"
    if out.exists():
        raise SystemExit("calibration lock already exists; refusing to overwrite")
    query_manifest = json.loads((ROOT / "evidence" / "v3_calibration_query_manifest.json").read_text(encoding="utf-8"))
    if query_manifest.get("status") != "V3_CALIBRATION_QUERIES_AUDITED_TEST_SEALED" or query_manifest.get("test_opened"):
        raise SystemExit("v3 calibration query manifest is not audited or test-sealed")
    checkpoint_manifest_path = ROOT / "evidence" / "confirmatory_checkpoint_manifest_v3.json"
    checkpoint_manifest = json.loads(checkpoint_manifest_path.read_text(encoding="utf-8"))
    if checkpoint_manifest.get("status") != "FROZEN_CONFIRMATORY_CHECKPOINTS" or checkpoint_manifest.get("test_opened"):
        raise SystemExit("v3 confirmatory checkpoint manifest is not frozen")
    checkpoint_map = {(row["pde"], row["method"], int(row["seed"])): ROOT / row["path"]
                      for row in checkpoint_manifest["checkpoints"] if row.get("method") != "observer"}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    groups = []
    for pde in ("burgers", "heat"):
        query = ROOT / "data" / "queries_calibration_v3" / "calibration" / pde / "teacher_queries.npz"
        with np.load(query, allow_pickle=False) as z:
            arrays = {key: z[key] for key in z.files}
        parent_ids = arrays["parent_id"].astype(str)
        unique_parents = sorted(set(parent_ids))
        if len(unique_parents) != 64 or len(parent_ids) != 64 * 8:
            raise SystemExit(f"expected 64x8 calibration queries for {pde}, got {len(unique_parents)} parents/{len(parent_ids)} rows")
        methods = [("B1", None)] + [(method, seed) for method in PREDICTIVE[1:] for seed in SEEDS]
        for method, seed in methods:
            checkpoint = None if method == "B1" else checkpoint_map[(pde, method, seed)]
            if checkpoint is not None and not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            model = None if method == "B1" else load_controller(method, pde, seed, checkpoint, device)
            parent_scores: dict[str, float] = {}
            invalid = 0
            for i, parent_id in enumerate(parent_ids):
                obs = {key: arrays[key][i] for key in (
                    "sensor_value", "sensor_mask", "sensor_age", "instrument_image", "image_mask",
                    "goal_coefficients", "material_context", "previous_applied_action",
                    "applied_action_history", "image_age", "image_valid", "goal_field")}
                proposed, forecast, choice, candidates = policy_action(
                    method, pde, obs, model, device, float(arrays["q_max"][i]))
                if forecast is None or choice is None or not np.isfinite(forecast).all():
                    invalid += 1
                    continue
                truth = arrays["future_field"][i, int(choice)]
                error = float(np.max(np.abs(forecast - truth)))
                if not math.isfinite(error):
                    invalid += 1
                    continue
                parent_scores[parent_id] = max(parent_scores.get(parent_id, 0.0), error)
            del model
            if invalid or set(parent_scores) != set(unique_parents):
                raise RuntimeError(f"invalid calibration forecasts for {pde}/{method}/{seed}: invalid={invalid}")
            values = np.asarray([parent_scores[pid] for pid in unique_parents], dtype=np.float64)
            order_index = math.ceil((len(values) + 1) * 0.95) - 1
            groups.append({"pde": pde, "method": method, "seed": seed,
                "parent_count": len(values), "query_count": len(parent_ids),
                "endpoint": "per-parent maximum absolute selected-action forecast error across 8 snapshots and 128 forecast grid",
                "finite_sample_order_index_zero_based": order_index,
                "margin": float(np.sort(values)[order_index]),
                "parent_scores": parent_scores, "action_selection_uses_margin": False,
                "checkpoint_sha256": sha(checkpoint) if checkpoint else None,
                "calibration_query_sha256": sha(query)})
    lock = {"protocol": "PDNO_JevLite_RTX5080_72H_frozen_spec_v3",
        "status": "CALIBRATION_LOCKED_TEST_SEALED", "test_opened": False,
        "quantile": 0.95, "exchangeability_scope": "nominal calibration parent episodes only",
        "calibration_parents_per_pde": 64, "snapshots_per_parent": 8,
        "groups": groups,
        "limitations": ["empirical nominal calibration margin only; not an OOD or uniform safety guarantee",
                        "residual-space or selected forecast coverage does not guarantee actual solution error"],
        "inputs_sha256": {"final_spec": sha(ROOT / "config" / "final_spec_v3.json"),
            "evaluation_addendum": sha(ROOT / "config" / "evaluation_addendum_v3.yaml"),
            "checkpoint_manifest": sha(checkpoint_manifest_path),
            "calibration_query_manifest": sha(ROOT / "evidence" / "v3_calibration_query_manifest.json"),
            "calibration_generation_receipt": sha(ROOT / "evidence" / "v3_calibration_generation_receipt.json")}}
    out.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": lock["status"], "groups": len(groups), "output": str(out),
                      "sha256": sha(out), "device": str(device), "test_opened": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
