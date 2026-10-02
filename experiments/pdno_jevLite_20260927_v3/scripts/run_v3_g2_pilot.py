"""One registered train/validation-only pilot per required learned mechanism."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from pdno.training.confirmatory import train_b3_staged, train_direct_staged, train_observer, train_operator_staged
from evaluate_pilot import evaluate


def main() -> int:
    data_root = ROOT / "data" / "queries_v3"
    output = ROOT / "runs" / "g2_pilot_v3"
    candidate_root = ROOT / "runs" / "g2_pilot_v3_candidates"
    progress_path = ROOT / "evidence" / "v3_g2_pilot_progress.json"
    result = {
        "status": "RUNNING", "pilot_seed": 7, "split": "train_validation_only",
        "test_opened": False, "calibration_opened": False, "rows": [],
    }
    started = time.perf_counter()
    for pde in ("burgers", "heat"):
        train = data_root / "train" / pde / "teacher_queries.npz"
        validation = data_root / "validation" / pde / "teacher_queries.npz"
        if not train.is_file() or not validation.is_file():
            raise FileNotFoundError(f"v3 teacher-query shard missing for {pde}")
        observer = output / "observer" / f"{pde}_s7.pt"
        obs_result = train_observer(train, validation, pde, observer, seed=7,
                                    max_updates=500, eval_interval=100,
                                    batch_size=64, device_name="cuda")
        result["rows"].append({"pde": pde, "method": "observer", **obs_result})
        operator_paths = {}
        for method in ("P", "P-no-rank", "B4", "B5"):
            checkpoint = output / pde / f"{method}_s7.pt"
            candidate_checkpoints = candidate_root / pde / f"{method}_s7"
            fit = train_operator_staged(
                train, validation, observer, pde, method, 7, checkpoint,
                lambda_phys=0.01, lambda_balance=0.01,
                field_updates=500, physics_updates=500, eval_interval=100,
                batch_size=64, device_name="cuda",
                candidate_checkpoint_dir=candidate_checkpoints,
            )
            operator_paths[method] = checkpoint
            metrics = evaluate(checkpoint, validation, device_name="cuda", batch_size=8)
            result["rows"].append({"pde": pde, "method": method, **fit, "validation_metrics": metrics})
            progress_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"pde": pde, "method": method,
                              "field_nrmse": metrics.get("field_nrmse"),
                              "normalized_teacher_regret_median": metrics.get("normalized_teacher_regret_median"),
                              "elapsed_seconds": time.perf_counter() - started}), flush=True)
        b2 = output / pde / "B2_s7.pt"
        b2_result = train_direct_staged(train, validation, observer, pde, 7, b2,
                                        max_updates=500, eval_interval=100,
                                        batch_size=64, device_name="cuda",
                                        candidate_checkpoint_dir=candidate_root / pde / "B2_s7")
        b2_metrics = evaluate(b2, validation, device_name="cuda", batch_size=8)
        result["rows"].append({"pde": pde, "method": "B2", **b2_result, "validation_metrics": b2_metrics})
        b3 = output / pde / "B3_s7.pt"
        b3_result = train_b3_staged(train, validation, b2, operator_paths["B4"], pde, 7, b3,
                                    max_updates=300, eval_interval=100,
                                    device_name="cuda",
                                    candidate_checkpoint_dir=candidate_root / pde / "B3_s7")
        b3_metrics = evaluate(b3, validation, device_name="cuda", batch_size=8)
        result["rows"].append({"pde": pde, "method": "B3", **b3_result, "validation_metrics": b3_metrics})
        progress_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"pde": pde, "method": "B2/B3",
                          "b2_teacher_action_mse": b2_metrics.get("teacher_action_mse"),
                          "b3_teacher_action_mse": b3_metrics.get("teacher_action_mse"),
                          "elapsed_seconds": time.perf_counter() - started}), flush=True)
    result.update(status="PILOT_COMPLETE", elapsed_seconds=time.perf_counter() - started)
    result["g2_field_gate_pass"] = all(
        row.get("validation_metrics", {}).get("field_nrmse", 0.0) <= 0.05
        for row in result["rows"] if row["method"] in {"P", "P-no-rank", "B4", "B5"}
    )
    progress_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "elapsed_seconds": result["elapsed_seconds"],
                      "g2_field_gate_pass": result["g2_field_gate_pass"],
                      "test_opened": False, "output": str(progress_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
