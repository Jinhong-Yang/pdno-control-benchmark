"""Reconcile pre-test rollout workload with measured single-parent runtimes."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEARNED = ("P", "P-no-rank", "B4", "B5", "B2", "B3")
H3_LEARNED = ("B2", "B3", "B4", "B5")


def main() -> int:
    measured = json.loads((ROOT / "evidence" / "v2_validation_rollout_runtime.json").read_text(encoding="utf-8"))
    by_pde: dict[str, dict[str, float]] = {}
    for row in measured["rows"]:
        by_pde.setdefault(row["pde"], {})[row["method"]] = float(row["wall_seconds"])
    locked_parent_per_pde = 640  # 384 nominal + 128 coefficient OOD + 128 delay/dropout
    h3_parent_per_pde = 64
    locked_seconds = sum(locked_parent_per_pde * (
        runtimes["B0"] + runtimes["B1"] + 3 * sum(runtimes[method] for method in LEARNED))
        for runtimes in by_pde.values())
    h3_seconds = sum(h3_parent_per_pde * (
        runtimes["B0"] + runtimes["B1"] + 3 * sum(runtimes[method] for method in H3_LEARNED))
        for runtimes in by_pde.values())
    h3_episodes = h3_parent_per_pde * len(by_pde) * (2 + 3 * len(H3_LEARNED))
    candidate_counts = {"P": 10, "P-no-rank": 10, "B4": 10, "B5": 10, "B2": 6, "B3": 2}
    checkpoint_selection_seconds = sum(8 * sum(
        3 * candidate_counts[method] * runtimes[method] for method in LEARNED)
        for runtimes in by_pde.values())
    margin = 1.25
    estimate = {
        "status": "PRETEST_RUNTIME_BUDGET_ESTIMATE",
        "runtime_evidence": "evidence/v2_validation_rollout_runtime.json",
        "measurement_scope": "one validation parent per PDE/controller family; pilot seed 7; single GPU host; 200 ticks",
        "method_order": ["B0", "B1", *LEARNED],
        "measured_episode_seconds_by_pde_method": by_pde,
        "locked": {"parents_per_pde": locked_parent_per_pde, "method_variants_per_parent": 20,
                   "episodes": 25600, "point_estimate_hours": locked_seconds / 3600,
                   "planning_estimate_with_25pct_margin_hours": locked_seconds * margin / 3600,
                   "allocated_budget_hours": 20},
        "h3_validation": {"parents_per_pde": h3_parent_per_pde, "eligible_methods": ["B0", "B1", *H3_LEARNED],
                          "learned_seeds_per_method": 3, "episodes": h3_episodes,
                          "point_estimate_hours": h3_seconds / 3600},
        "checkpoint_selection": {"candidate_snapshots_per_seed": candidate_counts,
                                  "parents_per_candidate_per_pde": 8,
                                  "candidate_evaluations": 8 * 2 * 3 * sum(candidate_counts.values()),
                                  "point_estimate_hours": checkpoint_selection_seconds / 3600},
        "caveats": ["one parent per PDE/controller does not characterize runtime variance",
                    "estimates assume pilot seed rollout latency represents all mandatory model seeds",
                    "single-GPU jobs remain sequential; locked budget is feasible under the 25% planning margin, subject to supervised shard timing",
                    "test outcomes are not used to revise these estimates"],
        "locked_test_opened": False,
    }
    out = ROOT / "evidence" / "v2_evaluation_budget_reconciliation.json"
    out.write_text(json.dumps(estimate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"locked_point_hours": estimate["locked"]["point_estimate_hours"],
                      "locked_with_margin_hours": estimate["locked"]["planning_estimate_with_25pct_margin_hours"],
                      "h3_hours": estimate["h3_validation"]["point_estimate_hours"],
                      "checkpoint_selection_hours": estimate["checkpoint_selection"]["point_estimate_hours"],
                      "output": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
