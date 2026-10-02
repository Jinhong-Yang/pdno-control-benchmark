"""Evaluate selected pilot checkpoints on held-out validation queries for G2."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_pilot import evaluate  # noqa: E402


def main() -> int:
    run_root = ROOT / "runs" / "pilot_full_v2_rev1"
    rows = []
    for pde in ("burgers", "heat"):
        query = ROOT / "data" / "queries_v2" / "validation" / pde / "teacher_queries.npz"
        for method in ("P", "P-no-rank", "B4", "B5"):
            checkpoint = run_root / pde / f"{method}_seed7.pt"
            result = evaluate(checkpoint, query, "cuda")
            result["g2_gate_pass"] = (result.get("field_nrmse", float("inf")) <= 0.05
                                      and result.get("normalized_teacher_regret_median", float("inf")) <= 0.05)
            rows.append(result)
    proposed = [row for row in rows if row["method"] == "P"]
    passed = len(proposed) == 2 and all(row["g2_gate_pass"] for row in proposed)
    report = {
        "status": "PILOT_G2_PASS" if passed else "PILOT_G2_FAIL",
        "test_opened": False,
        "validation_only": True,
        "independent_validation_parent_count_per_pde": 64,
        "thresholds": {"field_nrmse_max": 0.05, "normalized_teacher_regret_median_max": 0.05},
        "required_proposed_method": "P",
        "rows": rows,
    }
    path = ROOT / "evidence" / "v2_rev1_pilot_g2_validation_metrics.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pilot_path = ROOT / "evidence" / "full_data_pilot_v2_rev1.json"
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    pilot["status"] = report["status"]
    pilot["g2_validation_metrics_path"] = str(path.relative_to(ROOT))
    pilot["g2_validation_metrics"] = rows
    pilot_path.write_text(json.dumps(pilot, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "rows": len(rows),
                      "output": str(path), "pilot_report": str(pilot_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
