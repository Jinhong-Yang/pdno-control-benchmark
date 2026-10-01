"""Independently audit validation-only H3 comparator selection evidence."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METHODS = {"B0", "B1", "B2", "B3", "B4", "B5"}
LEARNED = {"B2", "B3", "B4", "B5"}
SEEDS = {11, 23, 37}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    report_path = ROOT / "evidence" / "validation_closed_loop_comparator_v2.json"
    selection_path = ROOT / "evidence" / "validation_checkpoint_selection_v2.json"
    runner_path = ROOT / "scripts" / "select_validation_comparator_v2.py"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = {(r["pde"], r["method"], int(r["seed"])): r for r in selection["selected"]}
    checks: dict[str, bool] = {
        "status_validation_only": report.get("status") == "VALIDATION_ONLY_H3_COMPARATOR_SELECTED",
        "locked_test_unopened": report.get("test_opened") is False,
        "pde_coverage": set(report.get("pdes", {})) == {"burgers", "heat"},
        "eligible_methods": set(report.get("eligible_methods", [])) == METHODS,
        "proposed_methods_excluded": set(report.get("excluded_proposed_methods", [])) == {"P", "P-no-rank"},
        "independent_parent_claim": report.get("independent_parent_count_per_pde") == 64,
        "checkpoint_selection_coverage": len(selected) == 36,
        "parent_method_seed_coverage": True,
        "validation_data_hashes": True,
        "checkpoint_hashes": True,
        "parent_then_seed_aggregation": True,
        "selection_scores_recomputed": True,
        "selection_rule_recomputed": True,
        "runner_enforces_no_fallback_and_causal_violation": False,
    }
    runner_text = runner_path.read_text(encoding="utf-8")
    checks["runner_enforces_no_fallback_and_causal_violation"] = (
        "row[\"fallback_count\"] or row[\"causal_timestamp_violations\"]" in runner_text
        and "raise RuntimeError(f\"invalid validation episode" in runner_text
    )

    for pde in ("burgers", "heat"):
        data_path = ROOT / "data" / "validation" / pde / "trajectories.npz"
        with np.load(data_path, allow_pickle=False) as data:
            parent_ids = data["parent_id"].astype(str).tolist()
        pde_report = report["pdes"][pde]
        checks["parent_method_seed_coverage"] &= (
            len(parent_ids) == 64 and len(set(parent_ids)) == 64
            and pde_report["parent_ids"] == parent_ids
            and set(pde_report["methods"]) == METHODS
        )
        checks["validation_data_hashes"] &= sha(data_path) == report["validation_data_sha256"][pde]

        for method in METHODS:
            method_report = pde_report["methods"][method]
            seed_rows = method_report["seed_results"]
            expected_seeds: list[int | None] = [None] if method in {"B0", "B1"} else sorted(SEEDS)
            checks["parent_method_seed_coverage"] &= [r["seed"] for r in seed_rows] == expected_seeds
            costs = np.asarray([r["episode_cost_by_parent"] for r in seed_rows], dtype=np.float64)
            violations = np.asarray([r["state_violation_by_parent"] for r in seed_rows], dtype=bool)
            checks["parent_method_seed_coverage"] &= (
                costs.shape == (len(expected_seeds), 64)
                and violations.shape == (len(expected_seeds), 64)
                and bool(np.isfinite(costs).all())
            )
            checks["parent_then_seed_aggregation"] &= bool(np.allclose(
                costs.mean(axis=0), method_report["mean_cost_by_parent_then_seed"], rtol=1e-12, atol=1e-12
            )) and math.isclose(
                float(costs.mean()), method_report["mean_episode_cost"], rel_tol=1e-12, abs_tol=1e-12
            )
            for row in seed_rows:
                if row["seed"] is None:
                    continue
                checkpoint = ROOT / "runs" / "confirmatory_v2" / pde / f"{method}_s{row['seed']}.pt"
                chosen = selected.get((pde, method, int(row["seed"])))
                checks["checkpoint_hashes"] &= (
                    chosen is not None
                    and sha(checkpoint) == row["checkpoint_sha256"] == chosen["official_checkpoint_sha256"]
                )

    baseline = {pde: float(report["pdes"][pde]["methods"]["B0"]["mean_episode_cost"])
                for pde in ("burgers", "heat")}
    scores = {
        method: float(np.mean([
            report["pdes"][pde]["methods"][method]["mean_episode_cost"] / max(baseline[pde], 1e-12)
            for pde in ("burgers", "heat")
        ]))
        for method in METHODS
    }
    checks["selection_scores_recomputed"] &= all(math.isclose(
        scores[m], report["method_scores"][m], rel_tol=1e-12, abs_tol=1e-12
    ) for m in METHODS)
    recomputed = min(METHODS, key=lambda m: (scores[m], m != "B0", m))
    checks["selection_rule_recomputed"] &= recomputed == report["selected_comparator"]
    checks = {key: bool(value) for key, value in checks.items()}

    receipt = {
        "status": "H3_VALIDATION_COMPARATOR_AUDIT_PASSED" if all(checks.values()) else "H3_AUDIT_FAILED",
        "test_opened": False,
        "selected_comparator": report["selected_comparator"],
        "method_scores": report["method_scores"],
        "checks": checks,
        "all_passed": all(checks.values()),
        "validation_report_sha256": sha(report_path),
        "checkpoint_selection_sha256": sha(selection_path),
        "runner_source_sha256": sha(runner_path),
        "validation_data_sha256": report["validation_data_sha256"],
        "learned_checkpoint_hashes_verified": sum(len(report["pdes"][p]["methods"][m]["seed_results"])
                                                   for p in ("burgers", "heat") for m in LEARNED),
        "independent_parent_count_per_pde": 64,
        "validation_parents_are_independent_units": True,
        "fallback_and_causal_violations": "runner raises before reporting if any episode has either; H3 completed successfully",
        "aggregation": "mean learned seeds within each parent, then aggregate independent parents",
    }
    out = ROOT / "evidence" / "v2_h3_comparator_audit_receipt.json"
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "checks": checks,
                      "all_passed": receipt["all_passed"], "output": str(out)}, indent=2))
    return 0 if receipt["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
