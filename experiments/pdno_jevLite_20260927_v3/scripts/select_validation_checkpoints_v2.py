"""Select train candidates by a fixed validation closed-loop subset only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pdno.evaluation.closed_loop import load_controller, run_episode  # noqa: E402

METHODS = ("P", "B4", "B5", "P-no-rank", "B2", "B3")
SEEDS = (11, 23, 37)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    temp = dst.with_name(dst.name + ".tmp")
    shutil.copyfile(src, temp)
    temp.replace(dst)


def main() -> int:
    config = json.loads((ROOT / "config" / "checkpoint_selection_v2.yaml").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidate_root = ROOT / "runs" / "confirmatory_v2_candidates"
    final_root = ROOT / "runs" / "confirmatory_v2"
    all_rows = []
    chosen_rows = []
    for pde in ("burgers", "heat"):
        data_path = ROOT / "data" / "validation" / pde / "trajectories.npz"
        with np.load(data_path, allow_pickle=False) as z:
            data = {key: z[key] for key in z.files}
        order = np.argsort(data["parent_id"].astype(str), kind="stable")[:int(config["validation_parents_per_pde"])]
        data = {key: value[order] if isinstance(value, np.ndarray) and value.ndim > 0 and len(value) == len(data["parent_id"]) else value
                for key, value in data.items()}
        parent_ids = data["parent_id"].astype(str).tolist()
        data["role"] = np.asarray(["validation"] * len(parent_ids))
        data["disturbance_seed"] = np.asarray(
            [300000 + int(pid.rsplit(":", 1)[-1]) for pid in parent_ids], dtype=np.int64)
        if pde == "burgers":
            q_max = 1.2
        else:
            with np.load(ROOT / "data" / "train" / "heat" / "trajectories.npz", allow_pickle=False) as z:
                train_goal_max = float(z["goal"].max())
            with np.load(ROOT / "data" / "validation" / "heat" / "trajectories.npz", allow_pickle=False) as z:
                validation_goal_max = float(z["goal"].max())
            q_max = max(1.2, 1.25 * max(train_goal_max, validation_goal_max))
        for method in METHODS:
            for seed in SEEDS:
                folder = candidate_root / pde / f"{method}_s{seed}"
                candidates = sorted(folder.glob("update_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
                if not candidates:
                    raise FileNotFoundError(f"no candidate checkpoints: {folder}")
                for path in candidates:
                    model = load_controller(method, pde, seed, path, device)
                    episodes = [run_episode(data, i, pde, method, model, device, q_max, ticks=200)
                                for i in range(len(parent_ids))]
                    latencies = np.concatenate([row["controller_latency_ms"] for row in episodes])
                    violations = sum(bool(row["truth_constraint_violation"]) for row in episodes)
                    fallback = sum(int(row["fallback_count"]) for row in episodes)
                    causal = sum(int(row["causal_timestamp_violations"]) for row in episodes)
                    costs = [float(row["episode_control_cost"]) for row in episodes]
                    finite = all(np.isfinite(costs).all() and np.isfinite(row["tracking_rmse"])
                                 and np.isfinite(row["action_proposed"]).all()
                                 and np.isfinite(row["action_projected"]).all()
                                 and np.isfinite(row["state_true"]).all() for row in episodes)
                    row = {"pde": pde, "method": method, "seed": seed,
                           "update": int(path.stem.split("_")[-1]), "candidate_path": str(path),
                           "candidate_sha256": sha(path), "mean_episode_cost": float(np.mean(costs)),
                           "p95_controller_latency_ms": float(np.quantile(latencies, .95)),
                           "controller_latency_measurements": int(len(latencies)),
                           "truth_constraint_violations": violations, "fallback_count": fallback,
                           "causal_timestamp_violations": causal,
                           "finite": finite, "eligible": finite and violations == 0 and fallback == 0 and causal == 0,
                           "episode_cost_by_parent": costs, "parent_ids": parent_ids}
                    all_rows.append(row)
                    del model
                group = [row for row in all_rows if row["pde"] == pde and row["method"] == method and row["seed"] == seed]
                eligible = [row for row in group if row["eligible"]]
                if eligible:
                    best_cost = min(row["mean_episode_cost"] for row in eligible)
                    tie_band = [row for row in eligible if row["mean_episode_cost"] <= best_cost * (1 + float(config["ranking"]["latency_tie_band"]))]
                    selected = min(tie_band, key=lambda row: (row["p95_controller_latency_ms"], row["update"]))
                    status = "VALIDATION_GATE_PASS"
                else:
                    selected = min(group, key=lambda row: (row["truth_constraint_violations"], row["fallback_count"],
                                                           row["causal_timestamp_violations"], row["mean_episode_cost"], row["update"]))
                    status = "NO_GO_NO_ZERO_VIOLATION_CANDIDATE_DIAGNOSTIC_ONLY"
                source = Path(selected["candidate_path"])
                destination = final_root / pde / f"{method}_s{seed}.pt"
                atomic_copy(source, destination)
                selected_record = {**selected, "selection_status": status,
                                   "official_checkpoint": str(destination),
                                   "official_checkpoint_sha256": sha(destination),
                                   "cost_tie_band_fraction": float(config["ranking"]["latency_tie_band"])}
                chosen_rows.append(selected_record)
                print(json.dumps({"pde": pde, "method": method, "seed": seed,
                                  "status": status, "update": selected["update"],
                                  "cost": selected["mean_episode_cost"],
                                  "latency_p95_ms": selected["p95_controller_latency_ms"]}), flush=True)
    parent_digest = hashlib.sha256("\n".join(
        f"{pde}:{pid}" for pde in ("burgers", "heat") for pid in sorted(
            next(row["parent_ids"] for row in chosen_rows if row["pde"] == pde))).encode()).hexdigest()
    report = {"protocol": config["protocol"], "status": "SELECTION_COMPLETE_WITH_DIAGNOSTIC_NOGO" if any(
        row["selection_status"].startswith("NO_GO") for row in chosen_rows) else "SELECTION_COMPLETE",
        "test_opened": False, "device": str(device), "config": config,
        "parent_ids_sha256": parent_digest, "selected": chosen_rows, "candidate_results": all_rows}
    out = ROOT / "evidence" / "validation_checkpoint_selection_v2.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
