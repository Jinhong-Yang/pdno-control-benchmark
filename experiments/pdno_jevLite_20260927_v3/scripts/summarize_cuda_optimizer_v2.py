"""Compare eager and fused AdamW timing with validation-only pilot diagnostics."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_pilot import evaluate  # noqa: E402


def main() -> int:
    run_root = ROOT / "runs" / "runtime_pilot_50_200_v2"
    runtime_rows = json.loads((ROOT / "evidence" / "v2_runtime_pilot_50_200.json").read_text(encoding="utf-8"))
    rows = []
    for method in ("P", "B4"):
        for pde in ("burgers", "heat"):
            query = ROOT / "data" / "queries_v2" / "validation" / pde / "teacher_queries.npz"
            variants = {}
            for variant, suffix in (("eager", ""), ("fused_adamw", "_fused_adamw")):
                stem = f"{method}_{pde}_s11{suffix}"
                timing = json.loads((run_root / f"{stem}.summary.json").read_text(encoding="utf-8"))
                runtime = next(row for row in runtime_rows if row["method"] == method and row["pde"] == pde
                               and bool(row.get("fused_optimizer", False)) == (variant == "fused_adamw"))
                validation = evaluate(run_root / f"{stem}.pt", query, "cuda")
                variants[variant] = {
                    "step_median_ms": timing["step_seconds_median"] * 1000,
                    "step_p90_ms": timing["step_seconds_p90"] * 1000,
                    "wall_seconds_including_startup": runtime["wall_seconds_including_startup"],
                    "cuda_peak_allocated_mib": runtime["cuda_peak_allocated_mib"],
                    "validation": validation,
                    "g2_gate_pass": (validation.get("field_nrmse", float("inf")) <= 0.05
                                     and validation.get("normalized_teacher_regret_median", float("inf")) <= 0.05),
                }
            eager, fused = variants["eager"], variants["fused_adamw"]
            variants["fused_adamw"]["median_step_speedup_percent"] = (
                100 * (eager["step_median_ms"] - fused["step_median_ms"]) / eager["step_median_ms"])
            rows.append({"method": method, "pde": pde, "steps": 250, "warmup_steps": 50,
                         "seed": 11, "variants": variants})
    all_gate_pass = all(v["g2_gate_pass"] for row in rows for v in row["variants"].values())
    result = {
        "status": "CUDA_SPEED_CANDIDATE_NOT_SELECTED_G2_SHORT_PILOT_FAIL" if not all_gate_pass else "CUDA_SPEED_CANDIDATE_ELIGIBLE_FOR_FROZEN_PILOT",
        "test_opened": False,
        "interpretation": "Fused AdamW changes floating-point update order. Keep as a candidate only; the 250-step pilot is too short and all field nRMSE values miss the predeclared G2 threshold.",
        "g2_thresholds": {"field_nrmse_max": 0.05, "normalized_teacher_regret_median_max": 0.05},
        "rows": rows,
    }
    out = ROOT / "evidence" / "v2_cuda_optimizer_comparison.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "rows": len(rows), "output": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
