"""Measure real 200-tick validation rollout wall time before candidate selection."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pdno.evaluation.closed_loop import load_controller, run_episode  # noqa: E402


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    pilot_root = ROOT / "runs" / "pilot_full_v2_rev1"
    for pde in ("burgers", "heat"):
        path = ROOT / "data" / "validation" / pde / "trajectories.npz"
        with np.load(path, allow_pickle=False) as z:
            data = {key: z[key] for key in z.files}
        parent_id = data["parent_id"].astype(str)
        selected = int(np.argsort(parent_id, kind="stable")[0])
        data = {key: (value[[selected]] if isinstance(value, np.ndarray) and value.ndim and len(value) == len(parent_id) else value)
                for key, value in data.items()}
        pid = str(data["parent_id"][0])
        data["role"] = np.asarray(["validation"])
        data["disturbance_seed"] = np.asarray([300000 + int(pid.rsplit(":", 1)[-1])], dtype=np.int64)
        qmax = 1.2 if pde == "burgers" else max(1.2, 1.25 * float(data["goal"].max()))
        for method in ("B0", "B1", "P", "P-no-rank", "B4", "B5", "B2", "B3"):
            checkpoint = None
            model = None
            if method not in {"B0", "B1"}:
                options = sorted(pilot_root.glob(f"**/{pde}/{method}_s7.pt"))
                if not options:
                    options = sorted(pilot_root.glob(f"**/{pde}/{method}_seed7.pt"))
                if not options:
                    raise FileNotFoundError(f"pilot checkpoint missing for {pde}/{method}: {pilot_root}")
                checkpoint = options[0]
                model = load_controller(method, pde, 7, checkpoint, device)
            started = time.perf_counter()
            result = run_episode(data, 0, pde, method, model, device, qmax, ticks=200)
            elapsed = time.perf_counter() - started
            rows.append({"pde": pde, "method": method, "parent_id": pid,
                         "checkpoint": str(checkpoint) if checkpoint else None,
                         "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest() if checkpoint else None,
                         "device": str(device), "episode_ticks": 200, "wall_seconds": elapsed,
                         "ms_per_tick": elapsed * 5, "fallback_count": result["fallback_count"],
                         "causal_timestamp_violations": result["causal_timestamp_violations"],
                         "controller_latency_ms_p50": float(np.quantile(result["controller_latency_ms"], .5)),
                         "controller_latency_ms_p95": float(np.quantile(result["controller_latency_ms"], .95)),
                         "controller_latency_ms_p99": float(np.quantile(result["controller_latency_ms"], .99)),
                         "controller_latency_measurements": len(result["controller_latency_ms"])})
            print(json.dumps(rows[-1]), flush=True)
            del model
    report = {"status": "MEASURED_SINGLE_PARENT_VALIDATION_ROLLOUT_RUNTIME",
              "note": "one fixed parent per PDE and all eight required controller families; solver wall includes CPU simulation and all host work, CUDA models use actual runtime device; this pilot informs budget only",
              "device": str(device), "rows": rows}
    output = ROOT / "evidence" / "v2_validation_rollout_runtime.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "status": report["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
