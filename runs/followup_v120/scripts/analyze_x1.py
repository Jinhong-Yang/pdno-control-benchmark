from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = json.loads((ROOT / "config/X1_SPEC.json").read_text(encoding="utf-8"))
CELLS = SPEC["cells"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def match_ids(data: dict[str, np.ndarray], wanted: np.ndarray) -> dict[str, np.ndarray]:
    lookup = {str(v): i for i, v in enumerate(data["parent_id"].astype(str))}
    try:
        indices = np.asarray([lookup[str(v)] for v in wanted], dtype=int)
    except KeyError as exc:
        raise ValueError(f"missing paired scenario ID in baseline arrays: {exc}") from exc
    return {k: v[indices] if v.ndim and v.shape[0] == len(data["parent_id"]) else v for k, v in data.items()}


def paired_bootstrap(cost: np.ndarray, baseline: np.ndarray) -> dict:
    cost = np.asarray(cost, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    if cost.shape != baseline.shape or cost.ndim != 1:
        raise ValueError("cost and paired baseline must be aligned 1-D scenario vectors")
    rng = np.random.default_rng(2026100270)
    draws = rng.integers(0, len(cost), size=(10000, len(cost)))
    delta = cost - baseline
    mean_delta = np.mean(delta[draws], axis=1)
    denominator = max(float(np.mean(baseline)), 1e-12)
    fixed = mean_delta / denominator
    paired = mean_delta / np.maximum(np.mean(baseline[draws], axis=1), 1e-12)
    return {
        "relative_excess_fixed": float(np.mean(delta) / denominator),
        "fixed_ci_low": float(np.quantile(fixed, .025)),
        "fixed_ci_high": float(np.quantile(fixed, .975)),
        "relative_excess_paired": float(np.mean(delta) / denominator),
        "paired_ci_low": float(np.quantile(paired, .025)),
        "paired_ci_high": float(np.quantile(paired, .975)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--b0-burgers", type=Path, required=True)
    parser.add_argument("--b0-heat", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    input_hashes = {str(p): sha(p) for p in (args.b0_burgers, args.b0_heat)}
    reference_costs = {}
    for population, pde, b0_path in (
        ("burgers_nominal", "burgers", args.b0_burgers),
        ("heat_nonnegative_nominal", "heat", args.b0_heat),
    ):
        cells_data = {}
        for cell in CELLS:
            path = ROOT / "results/X1" / population / cell["id"] / "n128.npz"
            record = path.with_suffix(".record.json")
            if not path.exists() or not record.exists():
                raise FileNotFoundError(f"missing sealed cell output or record: {path}")
            data = load_npz(path)
            cells_data[cell["id"]] = (data, path, json.loads(record.read_text(encoding="utf-8")))
            input_hashes[str(path)] = sha(path)
            input_hashes[str(record)] = sha(record)
        ids = cells_data["K10_H8"][0]["parent_id"].astype(str)
        baseline = match_ids(load_npz(b0_path), ids)
        b0_cost = np.asarray(baseline["episode_control_cost"], dtype=np.float64)
        input_hashes[str(b0_path)] = sha(b0_path)
        id_hash = hashlib.sha256("\n".join(ids.tolist()).encode("utf-8")).hexdigest()
        reference_costs[pde] = {"ids": ids, "cost": b0_cost, "path": b0_path}
        for cell in CELLS:
            data, path, record = cells_data[cell["id"]]
            if not np.array_equal(data["parent_id"].astype(str), ids):
                raise ValueError(f"scenario order differs from sealed cohort in {path}")
            cost = np.asarray(data["episode_control_cost"], dtype=np.float64)
            choices = data["selected_candidate_index"]
            metrics = paired_bootstrap(cost, b0_cost)
            hold = 4 if cell["K"] == 10 else 48
            b0_index = int(cell["K"]) - 1
            applied = np.asarray(data["action_applied"], dtype=np.float32)
            initial_previous = (np.zeros(2, dtype=np.float32) if pde == "burgers"
                                else np.full(2, .4, dtype=np.float32))
            previous_actions = np.concatenate(
                [np.broadcast_to(initial_previous, (len(applied), 1, 2)), applied[:, :-1]], axis=1
            )
            applied_no_change = np.all(applied == previous_actions, axis=2)
            row = {
                "pde": pde,
                "population": population,
                "cell": cell["id"],
                "K": cell["K"],
                "H": cell["H"],
                "feedback_rollout": cell["feedback_rollout"],
                "n": len(ids),
                "mean_episode_cost": float(np.mean(cost)),
                "b0_mean_cost": float(np.mean(b0_cost)),
                **metrics,
                "mean_tracking_rmse": float(np.mean(data["tracking_rmse"])),
                "hold_selection_rate": float(np.mean(choices == hold)),
                "b0_selection_rate": float(np.mean(choices == b0_index)),
                "feedback_rollout_selection_rate": float(np.mean(choices == 10)) if cell["feedback_rollout"] else "",
                "applied_no_change_rate": float(np.mean(applied_no_change)),
                "mean_abs_action_change": float(np.mean(data["mean_abs_action_change"])),
                "scenario_ids_sha256": id_hash,
                "source_result_sha256": record["result_sha256"],
                "source_b0_sha256": sha(b0_path),
                "run_commit": record["git_commit"],
            }
            rows.append(row)
        if pde == "burgers":
            anchor_path = ROOT / "results/X1/burgers_nominal/K10_H8/n384.npz"
            anchor_record_path = anchor_path.with_suffix(".record.json")
            anchor = load_npz(anchor_path)
            anchor_record = json.loads(anchor_record_path.read_text(encoding="utf-8"))
            full_ids = anchor["parent_id"].astype(str)
            full_baseline = match_ids(load_npz(b0_path), full_ids)
            anchor_cost = anchor["episode_control_cost"].astype(np.float64)
            b0_full = full_baseline["episode_control_cost"].astype(np.float64)
            anchor_metrics = paired_bootstrap(anchor_cost, b0_full)
            anchor_row = {
                "pde": pde,
                "population": "burgers_anchor_full384",
                "cell": "K10_H8",
                "K": 10,
                "H": 8,
                "feedback_rollout": False,
                "n": len(full_ids),
                "mean_episode_cost": float(np.mean(anchor_cost)),
                "b0_mean_cost": float(np.mean(b0_full)),
                **anchor_metrics,
                "mean_tracking_rmse": float(np.mean(anchor["tracking_rmse"])),
                "hold_selection_rate": float(np.mean(anchor["selected_candidate_index"] == 4)),
                "b0_selection_rate": float(np.mean(anchor["selected_candidate_index"] == 9)),
                "feedback_rollout_selection_rate": "",
                "applied_no_change_rate": float(np.mean(np.all(
                    anchor["action_applied"] == np.concatenate([
                        np.broadcast_to(np.zeros(2, dtype=np.float32), (len(full_ids), 1, 2)),
                        anchor["action_applied"][:, :-1]], axis=1), axis=2))),
                "mean_abs_action_change": float(np.mean(anchor["mean_abs_action_change"])),
                "scenario_ids_sha256": hashlib.sha256("\n".join(full_ids.tolist()).encode("utf-8")).hexdigest(),
                "source_result_sha256": anchor_record["result_sha256"],
                "source_b0_sha256": sha(b0_path),
                "run_commit": anchor_record["git_commit"],
            }
            rows.append(anchor_row)
            input_hashes[str(anchor_path)] = sha(anchor_path)
            input_hashes[str(anchor_record_path)] = sha(anchor_record_path)
    burgers = {r["cell"]: r for r in rows
               if r["pde"] == "burgers" and r["population"] == "burgers_nominal"}
    base = burgers["K10_H8"]["relative_excess_fixed"]
    fixed = [burgers[c["id"]]["relative_excess_fixed"] for c in CELLS if not c["feedback_rollout"]]
    feedback = burgers["K10_H8_FB"]["relative_excess_fixed"]
    if any(v <= .5 * base for v in fixed):
        branch = "A"
    elif feedback <= .5 * base:
        branch = "C"
    else:
        branch = "B"
    out_csv = ROOT / "results/X1_ANALYSIS.csv"
    out_json = ROOT / "results/X1_ANALYSIS.json"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "status": "COMPLETE",
        "analysis": "paired B0 comparison on the same first128 lexical scenario IDs",
        "bootstrap_draws": 10000,
        "bootstrap_seed": 2026100270,
        "interval_scope": "pointwise percentile 95%; not familywise coverage",
        "selection_metric_semantics": {
            "hold_selection_rate": "selected explicit appended hold-candidate index (K10=4; K50=48); K50 retains a duplicate zero-offset lattice candidate, so this is candidate identity, not a count of no-change actions",
            "applied_no_change_rate": "fraction of all 200 ticks where both applied float32 action components exactly equal the preceding applied action; the initial previous action is [0,0] for Burgers and [0.4,0.4] for heat",
        },
        "decision_branch": branch,
        "burgers_baseline_excess": base,
        "fixed_cells_excess": {c["id"]: burgers[c["id"]]["relative_excess_fixed"] for c in CELLS if not c["feedback_rollout"]},
        "feedback_rollout_excess": feedback,
        "input_sha256": input_hashes,
        "csv_sha256": sha(out_csv),
        "rows": rows,
    }
    out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "branch": branch, "csv": str(out_csv), "sha256": result["csv_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
