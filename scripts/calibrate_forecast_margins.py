"""Freeze nominal parent-max forecast-error margins from calibration data only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pdno.controllers.objectives import candidate_cost  # noqa: E402
from pdno.controllers.rom import rom_candidate_costs  # noqa: E402
from pdno.models.operators import ActionFactorizedOperator, CandidateConditionedOperator  # noqa: E402
from pdno.training.fit import _grid, _load, _obs  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.no_grad()
def model_parent_scores(pde: str, method: str, checkpoint: Path | None, query: Path,
                        device: torch.device) -> dict[str, float]:
    with np.load(query, allow_pickle=False) as z:
        parent_ids = z["parent_id"].astype(str)
        raw = {k: z[k] for k in z.files}
    data = _load(query, device)
    if method != "B1":
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        rank = saved["metadata"].get("rank", 32)
        model = (ActionFactorizedOperator(pde, 33 if pde == "burgers" else 32, rank=rank)
                 if method in {"P", "B5"} else CandidateConditionedOperator(pde, 33 if pde == "burgers" else 32, rank=rank))
        model.load_state_dict(saved["state_dict"])
        model.to(device).eval()
        x, tau = _grid(pde, device)
    errors: dict[str, float] = {}
    for i, parent in enumerate(parent_ids):
        obs = {k: data[k][i:i+1] for k in ("sensor_value", "sensor_mask", "sensor_age", "instrument_image", "image_mask",
                "goal_coefficients", "material_context", "previous_applied_action", "applied_action_history", "image_age", "image_valid")}
        actions = data["candidate_action"][i:i+1].float()
        if method == "B1":
            obs_np = {k: v[i] for k, v in raw.items() if k in {"sensor_value", "sensor_mask", "instrument_image", "image_mask", "image_valid",
                      "material_context", "previous_applied_action", "applied_action_history", "goal_field"}}
            costs, predicted = rom_candidate_costs(obs_np, pde, raw["candidate_action"][i], raw["goal"][i], float(raw["q_max"][i]))
            chosen = int(np.argmin(costs))
            error = float(np.max(np.abs(predicted[chosen] - raw["future_field"][i, chosen])))
        else:
            predicted = model(obs, actions, x, tau)
            pred_cost = candidate_cost(predicted, data["goal"][i:i+1].float(), actions,
                obs["previous_applied_action"].float(), float(data["q_max"][i].item()))
            chosen = int(pred_cost.argmin(dim=1).item())
            error = float((predicted[0, chosen] - data["future_field"][i, chosen].float()).abs().max().item())
        errors[parent] = max(errors.get(parent, 0.0), error)
    return errors


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_manifest = json.loads((ROOT / "evidence/confirmatory_checkpoint_manifest.json").read_text(encoding="utf-8"))
    checkpoints = {(r["pde"], r["method"], int(r["seed"])): ROOT / r["path"]
                   for r in checkpoint_manifest["checkpoints"] if r.get("method") in {"P", "B4", "B5"}}
    groups = []
    for pde in ("burgers", "heat"):
        query = ROOT / "data/queries_calibration_v1/calibration" / pde / "teacher_queries.npz"
        with np.load(query, allow_pickle=False) as z:
            parent_count = len(set(z["parent_id"].astype(str).tolist()))
        if parent_count != 64:
            raise SystemExit(f"expected 64 nominal calibration parents for {pde}, found {parent_count}")
        methods = [("B1", None, None)] + [(method, seed, checkpoints[(pde, method, seed)])
                   for method in ("P", "B4", "B5") for seed in (11, 23, 37)]
        for method, seed, ckpt in methods:
            scores = model_parent_scores(pde, method, ckpt, query, device)
            values = sorted(scores.values())
            index = int(np.ceil((len(values) + 1) * 0.95) - 1)
            groups.append({"pde": pde, "method": method, "seed": seed, "parent_count": len(values),
                "endpoint": "max_abs_predicted_vs_truth_future_field_over_selected_candidate_horizon_grid_and_queries_per_parent",
                "finite_sample_order_index_zero_based": index, "margin": values[index],
                "parent_scores": scores, "action_selection_uses_margin": False})
    lock = {"protocol": "PDNO_JevLite_RTX5080_72H_frozen_v1+evaluation_addendum_v1",
        "status": "CALIBRATION_LOCKED_TEST_SEALED", "test_opened": False,
        "quantile": 0.95, "exchangeability_scope": "nominal calibration parent episodes only",
        "groups": groups,
        "inputs": {"frozen_config_sha256": sha(ROOT / "config/frozen_experiment.yaml"),
            "evaluation_addendum_sha256": sha(ROOT / "config/evaluation_addendum_v1.yaml"),
            "checkpoint_manifest_sha256": sha(ROOT / "evidence/confirmatory_checkpoint_manifest.json"),
            "calibration_query_manifest_sha256": sha(ROOT / "evidence/calibration_query_manifest.json")}}
    out = ROOT / "evidence/calibration_lock.json"
    out.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": lock["status"], "groups": len(groups), "output": str(out),
                      "sha256": sha(out), "device": str(device), "test_opened": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
