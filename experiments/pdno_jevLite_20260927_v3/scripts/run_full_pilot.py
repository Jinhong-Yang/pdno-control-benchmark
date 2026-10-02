"""Run seed-7 revision-1 pilot with train-only normalized shared observations."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pdno.training.confirmatory import (train_observer, train_operator_staged,
                                        train_direct_staged, train_b3_staged)


def _resume_or(path, run):
    summary = Path(path).with_suffix(".summary.json")
    if summary.exists():
        return json.loads(summary.read_text(encoding="utf-8"))
    return run()


def main():
    outputs = []
    base = ROOT / "data" / "queries_v2"
    run = ROOT / "runs" / "pilot_full_v2_rev1"
    observers = {}
    for pde in ("burgers", "heat"):
        tr, va = base / "train" / pde / "teacher_queries.npz", base / "validation" / pde / "teacher_queries.npz"
        observer = run / pde / "observer_seed7.pt"
        report = _resume_or(observer, lambda: train_observer(tr, va, pde, observer, seed=7, max_updates=1000,
                                eval_interval=100, batch_size=64, device_name="cuda"))
        observers[pde] = observer
        outputs.append(report)
        # Runtime profiling showed the optional multi-lambda screen is the first
        # selection work to remove. Keep the predeclared lambda=0.01 for both PDEs.
        for method in ("P", "P-no-rank", "B4", "B5"):
            path = run / pde / f"{method}_seed7.pt"
            report = _resume_or(path, lambda method=method, path=path:
                train_operator_staged(tr, va, observer, pde, method, 7, path,
                    lambda_phys=0.01, lambda_balance=0.01 if method != "B5" else 0.0,
                    field_updates=2000, physics_updates=3000, eval_interval=100,
                    batch_size=64, device_name="cuda"))
            outputs.append(report)
            print(json.dumps({"pilot": method, "pde":pde,"lambda_phys":0.01,
                              "best_step":report["best_step"],"selection_score":report["selection_score"],
                              "elapsed_seconds":report["elapsed_seconds"]}), flush=True)
        b2 = run / pde / "B2_seed7.pt"
        report = _resume_or(b2, lambda: train_direct_staged(tr, va, observer, pde, 7, b2, max_updates=3000,
                                     eval_interval=100, batch_size=64, device_name="cuda"))
        outputs.append(report)
        b3 = run / pde / "B3_seed7.pt"
        b4 = run / pde / "B4_seed7.pt"
        report = _resume_or(b3, lambda: train_b3_staged(tr, va, b2, b4, pde, 7, b3, max_updates=1000,
                                 eval_interval=100, device_name="cuda"))
        outputs.append(report)
        print(json.dumps({"pilot":"B2/B3","pde":pde,"B2_best_step":outputs[-2]["best_step"],
                          "B3_best_step":report["best_step"],"B3_validation_score":report["validation_score"]}), flush=True)
    path = ROOT / "evidence" / "full_data_pilot_v2_rev1.json"
    operators = [x for x in outputs if x.get("method") in {"P", "P-no-rank", "B4", "B5"}]
    passed = bool(operators) and all(x.get("validation", {}).get("validation_field_nrmse", float("inf")) <= 0.05
                                     and x.get("validation", {}).get("validation_normalized_teacher_regret_median", float("inf")) <= 0.05
                                     for x in operators if x.get("method") == "P")
    payload = {"status": "PILOT_G2_PASS" if passed else "PILOT_G2_FAIL_OR_INCOMPLETE",
               "test_opened": False, "selection_trials_removed": ["Burgers lambda_phys 0.1", "Burgers lambda_phys 1.0"],
               "required_controls_retained": ["B0", "B1", "B2", "B3", "B4", "B5", "P", "P-no-rank"],
               "runs": outputs}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
