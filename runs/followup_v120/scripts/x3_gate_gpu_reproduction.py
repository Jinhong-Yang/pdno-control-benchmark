"""Sequential CUDA reproduction of frozen X3 validation E_total gate values."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
RUN = HERE.parents[1]
ROOT = RUN.parents[1]
sys.path.insert(0, str(RUN / "scripts"))
import x3_decompose as x3

SPEC = RUN / "config" / "X3_GATE_REPRO_SPEC.json"
FREEZE = RUN / "evidence" / "X3_FREEZE.json"
CPU_SUMMARY = RUN / "results" / "X3" / "X3_OPERATOR_SUMMARIES.json"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def cuda_gate(row: dict, device: torch.device) -> float:
    path = ROOT / row["queries"]
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    model = x3._model(row).to(device).eval()
    x, tau = x3._grid(row["pde"])
    x, tau = x.to(device), tau.to(device)
    field_num = torch.zeros((), device=device, dtype=torch.float32)
    field_den = torch.zeros((), device=device, dtype=torch.float32)
    with torch.inference_mode():
        for start in range(0, len(arrays["future_field"]), 32):
            end = min(start + 32, len(arrays["future_field"]))
            obs = {k: torch.as_tensor(arrays[k][start:end], device=device) for k in x3.OBS_KEYS}
            actions = torch.as_tensor(arrays["candidate_action"][start:end], dtype=torch.float32, device=device)
            target = torch.as_tensor(arrays["future_field"][start:end], dtype=torch.float32, device=device)
            pred = model(obs, actions, x, tau)
            field_num += (pred - target).square().sum()
            field_den += target.square().sum()
    value = float(torch.sqrt(field_num / field_den.clamp_min(1e-12)).cpu().item())
    del model
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", action="store_true", help="show frozen execution plan without touching CUDA")
    parser.add_argument("--run", action="store_true", help="execute sequential CUDA gate replay")
    args = parser.parse_args()
    if args.plan:
        freeze = x3.verify_frozen()
        cpu = json.loads(CPU_SUMMARY.read_text(encoding="utf-8"))
        print(json.dumps({"operators": len(freeze["roster"]), "device": "cuda (when --run)",
                          "queries_per_operator": 64, "candidate_count": 10, "horizon": 8,
                          "batch_size": 32, "inference_only": True,
                          "cpu_result_sha256": sha(CPU_SUMMARY), "cpu_operators": cpu["operator_count"],
                          "gate_acceptance": "four-decimal equality to unchanged saved gate"}, indent=2))
        return
    if not args.run:
        parser.error("Choose --plan or --run")
    x3.verify_frozen()
    if not SPEC.exists():
        raise RuntimeError("Missing committed X3 gate replay specification")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no fallback to CPU is permitted for this replay")
    device = torch.device("cuda")
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    cpu = json.loads(CPU_SUMMARY.read_text(encoding="utf-8"))
    cpu_rows = {(r["stage"], r["pde"], r["method"], int(r["seed"]), r["target_factor"]): r for r in cpu["rows"]}
    begin = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()
    results = []
    for i, row in enumerate(json.loads(FREEZE.read_text(encoding="utf-8"))["roster"], 1):
        key = (row["stage"], row["pde"], row["method"], int(row["seed"]), row["target_factor"])
        cpu_row = cpu_rows[key]
        cuda_value = cuda_gate(row, device)
        saved = float(row["gate_validation_field_nrmse"])
        cpu_value = float(cpu_row["E_total"])
        result = {
            "stage": row["stage"], "pde": row["pde"], "method": row["method"],
            "seed": row["seed"], "target_factor": row["target_factor"],
            "checkpoint": row["checkpoint"], "checkpoint_sha256": row["checkpoint_sha256"],
            "queries_sha256": row["queries_sha256"],
            "saved_gate_E_total": saved, "cpu_E_total": cpu_value, "cuda_E_total": cuda_value,
            "cuda_abs_difference_from_saved_gate": abs(cuda_value - saved),
            "cpu_abs_difference_from_saved_gate": abs(cpu_value - saved),
            "cuda_reproduces_gate_four_decimals": round(cuda_value, 4) == round(saved, 4),
        }
        results.append(result)
        print(json.dumps({"i": i, "n": 54, **{k: result[k] for k in ("stage", "pde", "method", "seed", "target_factor", "saved_gate_E_total", "cpu_E_total", "cuda_E_total", "cuda_reproduces_gate_four_decimals")}}), flush=True)
        torch.cuda.empty_cache()
    mismatches = [r for r in results if not r["cuda_reproduces_gate_four_decimals"]]
    end = datetime.now(timezone.utc).isoformat()
    record = {
        "status": "COMPLETE_GATE_REPRODUCED" if not mismatches else "COMPLETE_WITH_GATE_REPRODUCTION_MISMATCH",
        "spec_sha256": sha(SPEC), "script_sha256": sha(HERE), "freeze_sha256": sha(FREEZE),
        "cpu_summary_sha256": sha(CPU_SUMMARY), "device": torch.cuda.get_device_name(device),
        "start_utc": begin, "end_utc": end, "elapsed_wall_seconds": time.perf_counter() - t0,
        "operator_count": len(results), "four_decimal_gate_mismatch_count": len(mismatches),
        "max_cuda_abs_gate_difference": max(r["cuda_abs_difference_from_saved_gate"] for r in results),
        "rows": results,
        "limits": ["Inference-only selected-validation metric reproduction.",
                   "No weights, inputs, endpoints, gates, thresholds, or selection decisions were changed."],
    }
    out = RUN / "results" / "X3" / "X3_GATE_GPU_REPRODUCTION.json"
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    receipt = {"status": record["status"], "result_sha256": sha(out), "spec_sha256": sha(SPEC),
               "script_sha256": sha(HERE), "operator_count": len(results),
               "four_decimal_gate_mismatch_count": len(mismatches),
               "max_cuda_abs_gate_difference": record["max_cuda_abs_gate_difference"],
               "elapsed_wall_seconds": record["elapsed_wall_seconds"], "start_utc": begin, "end_utc": end}
    (RUN / "evidence" / "X3_GATE_GPU_REPRODUCTION.json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
