"""Choose the fixed H3 comparator from eligible required controllers on validation only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pdno.evaluation.closed_loop import load_controller, run_episode  # noqa: E402

ELIGIBLE = ("B0", "B1", "B2", "B3", "B4", "B5")
SEEDS = (11, 23, 37)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_root = ROOT / "runs" / "confirmatory_v3_training"
    rows: dict[str, dict] = {}
    qmax = 1.2
    with np.load(ROOT / "data" / "train_validation_v3" / "train" / "heat" / "trajectories.npz", allow_pickle=False) as z:
        qmax = max(qmax, 1.25 * float(z["goal"].max()))
    with np.load(ROOT / "data" / "train_validation_v3" / "validation" / "heat" / "trajectories.npz", allow_pickle=False) as z:
        qmax = max(qmax, 1.25 * float(z["goal"].max()))

    for pde in ("burgers", "heat"):
        validation_path = ROOT / "data" / "train_validation_v3" / "validation" / pde / "trajectories.npz"
        with np.load(validation_path, allow_pickle=False) as z:
            data = {key: z[key] for key in z.files}
        parent_count = len(data["parent_id"])
        if parent_count != 64:
            raise RuntimeError(f"expected 64 independent validation parents, got {parent_count} for {pde}")
        data["role"] = np.asarray(["validation"] * parent_count)
        data["disturbance_seed"] = np.asarray(
            [300000 + int(str(pid).rsplit(":", 1)[-1]) for pid in data["parent_id"]], dtype=np.int64)
        rows[pde] = {"parent_ids": data["parent_id"].astype(str).tolist(), "methods": {}}

        for method in ELIGIBLE:
            seeds = (None,) if method in {"B0", "B1"} else SEEDS
            seed_rows = []
            for seed in seeds:
                checkpoint = None if seed is None else model_root / pde / f"{method}_s{seed}.pt"
                if checkpoint is not None and not checkpoint.is_file():
                    raise FileNotFoundError(f"missing frozen required checkpoint: {checkpoint}")
                model = load_controller(method, pde, seed, checkpoint, device)
                episodes = [run_episode(data, i, pde, method, model, device,
                            1.2 if pde == "burgers" else qmax, ticks=200) for i in range(parent_count)]
                if any(row["fallback_count"] or row["causal_timestamp_violations"] for row in episodes):
                    raise RuntimeError(f"invalid validation episode for {pde}/{method}/{seed}")
                seed_rows.append({
                    "seed": seed,
                    "episode_cost_by_parent": [float(row["episode_control_cost"]) for row in episodes],
                    "state_violation_by_parent": [bool(row["truth_constraint_violation"]) for row in episodes],
                    "mean_episode_cost": float(np.mean([row["episode_control_cost"] for row in episodes])),
                    "violation_count": int(sum(row["truth_constraint_violation"] for row in episodes)),
                    "checkpoint_sha256": None if checkpoint is None else sha(checkpoint),
                })
                del model
            cost_matrix = np.asarray([row["episode_cost_by_parent"] for row in seed_rows], dtype=np.float64)
            violation_matrix = np.asarray([row["state_violation_by_parent"] for row in seed_rows], dtype=bool)
            rows[pde]["methods"][method] = {
                "seed_results": seed_rows,
                "mean_cost_by_parent_then_seed": cost_matrix.mean(axis=0).tolist(),
                "mean_episode_cost": float(cost_matrix.mean()),
                "mean_violation_rate_over_parent_then_seed": float(violation_matrix.mean()),
            }

    baseline_cost = {pde: float(rows[pde]["methods"]["B0"]["mean_episode_cost"])
                     for pde in ("burgers", "heat")}
    scores = {}
    for method in ELIGIBLE:
        scores[method] = float(np.mean([
            rows[pde]["methods"][method]["mean_episode_cost"] / max(baseline_cost[pde], 1e-12)
            for pde in ("burgers", "heat")]))
    selected = min(ELIGIBLE, key=lambda method: (scores[method], method != "B0", method))
    report = {
        "status": "VALIDATION_ONLY_H3_COMPARATOR_SELECTED",
        "test_opened": False,
        "device": str(device),
        "eligible_methods": list(ELIGIBLE),
        "excluded_proposed_methods": ["P", "P-no-rank"],
        "independent_parent_count_per_pde": 64,
        "seed_hierarchy": "average learned model seeds within each parent; parent remains the independent unit",
        "selection_rule": "lowest equal-weight mean across PDEs of method validation cost divided by validation B0 cost; B0 wins exact tie",
        "method_scores": scores,
        "selected_comparator": selected,
        "baseline_mean_cost": baseline_cost,
        "validation_data_sha256": {pde: sha(ROOT / "data" / "train_validation_v3" / "validation" / pde / "trajectories.npz")
                                    for pde in ("burgers", "heat")},
        "pdes": rows,
    }
    out = ROOT / "evidence" / "validation_closed_loop_comparator_v3.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "selected_comparator": selected,
                      "scores": scores, "output": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
