"""Sequentially fit the mandatory v2 controller families on train/validation only."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pdno.training.confirmatory import (train_observer, train_operator_staged,
                                        train_direct_staged, train_b3_staged)


def _run(path: Path, fn):
    summary = path.with_suffix(".summary.json")
    if summary.exists():
        row = json.loads(summary.read_text(encoding="utf-8"))
        if row.get("checkpoint") == str(path):
            return row
        raise RuntimeError(f"existing summary points at a different checkpoint: {summary}")
    return fn()


def main() -> int:
    root = ROOT / "runs" / "confirmatory_v2_training"
    candidate_root = ROOT / "runs" / "confirmatory_v2_candidates"
    data = ROOT / "data" / "queries_v2"
    rows = []
    started = time.perf_counter()
    for pde in ("burgers", "heat"):
        train = data / "train" / pde / "teacher_queries.npz"
        validation = data / "validation" / pde / "teacher_queries.npz"
        observer = root / "observer" / f"{pde}_seed7.pt"
        row = _run(observer, lambda: train_observer(train, validation, pde, observer, seed=7,
                        max_updates=1000, eval_interval=100, batch_size=64, device_name="cuda"))
        rows.append(row)
        by_method_seed = {}
        for seed in (11, 23, 37):
            for method in ("P", "B4", "B5", "P-no-rank"):
                path = root / pde / f"{method}_s{seed}.pt"
                candidates = candidate_root / pde / f"{method}_s{seed}"
                row = _run(path, lambda method=method, path=path, seed=seed:
                    train_operator_staged(train, validation, observer, pde, method, seed, path,
                        lambda_phys=0.01, lambda_balance=0.01, field_updates=2000,
                        physics_updates=3000, eval_interval=500, batch_size=64, device_name="cuda",
                        candidate_checkpoint_dir=candidates))
                by_method_seed[(method, seed)] = path
                rows.append(row)
                print(json.dumps({"pde": pde, "method": method, "seed": seed,
                    "updates": row.get("actual_updates"), "selection_score": row.get("selection_score")}), flush=True)
            b2 = root / pde / f"B2_s{seed}.pt"
            b2_candidates = candidate_root / pde / f"B2_s{seed}"
            row = _run(b2, lambda: train_direct_staged(train, validation, observer, pde, seed, b2,
                max_updates=3000, eval_interval=500, batch_size=64, device_name="cuda",
                candidate_checkpoint_dir=b2_candidates))
            rows.append(row)
            print(json.dumps({"pde": pde, "method": "B2", "seed": seed,
                "updates": row.get("actual_updates"), "validation_action_mse": row.get("validation_action_mse")}), flush=True)
            b3 = root / pde / f"B3_s{seed}.pt"
            b3_candidates = candidate_root / pde / f"B3_s{seed}"
            b4 = by_method_seed[("B4", seed)]
            row = _run(b3, lambda: train_b3_staged(train, validation, b2, b4, pde, seed, b3,
                max_updates=1000, eval_interval=500, device_name="cuda",
                candidate_checkpoint_dir=b3_candidates))
            rows.append(row)
            print(json.dumps({"pde": pde, "method": "B3", "seed": seed,
                "updates": row.get("actual_updates"), "validation_score": row.get("validation_score")}), flush=True)
            manifest = ROOT / "evidence" / "v2_confirmatory_training_progress.json"
            manifest.write_text(json.dumps({"status": "RUNNING", "elapsed_seconds": time.perf_counter() - started,
                "results": rows}, indent=2) + "\n", encoding="utf-8")
    manifest = ROOT / "evidence" / "v2_confirmatory_training_progress.json"
    manifest.write_text(json.dumps({"status": "TRAINING_COMPLETE", "elapsed_seconds": time.perf_counter() - started,
        "results": rows}, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
