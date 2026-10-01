"""Validation-only episode-time estimate for the frozen-size v3 evaluation."""
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
from pdno.evaluation.closed_loop import load_controller, run_episode


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = ROOT / "data" / "train_validation_v3" / "train"
    validation_data = ROOT / "data" / "train_validation_v3" / "validation"
    with np.load(train_data / "heat" / "trajectories.npz", allow_pickle=False) as z:
        heat_max = float(np.max(z["goal"]))
    with np.load(validation_data / "heat" / "trajectories.npz", allow_pickle=False) as z:
        heat_max = max(heat_max, float(np.max(z["goal"])))
    q_max = {"burgers": 1.2, "heat": 1.25 * heat_max}
    rows = []
    started = time.perf_counter()
    families = ("P", "P-no-rank", "B4", "B5", "B2", "B3")
    for pde in ("burgers", "heat"):
        path = validation_data / pde / "trajectories.npz"
        with np.load(path, allow_pickle=False) as archive:
            data = {key: archive[key][:1] for key in archive.files}
        methods = [("B0", None), ("B1", None)] + [(method, 7) for method in families]
        for method, seed in methods:
            checkpoint = None if seed is None else ROOT / "runs" / "g2_pilot_v3" / pde / f"{method}_s7.pt"
            model = load_controller(method, pde, seed, checkpoint, device)
            t0 = time.perf_counter()
            episode = run_episode(data, 0, pde, method, model, device, q_max[pde], query_tick=100, ticks=200)
            elapsed = time.perf_counter() - t0
            if episode["fallback_count"] or episode["causal_timestamp_violations"]:
                raise RuntimeError(f"validation rollout integrity failure: {pde}/{method}/{seed}: {episode}")
            policy_ms = np.asarray(episode["controller_latency_ms"], dtype=np.float64)
            rows.append({
                "pde": pde, "method": method, "pilot_seed": seed,
                "validation_parent_id": str(data["parent_id"][0]),
                "episode_wall_seconds": elapsed,
                "controller_only_ms_p50": float(np.quantile(policy_ms, 0.50)),
                "controller_only_ms_p95": float(np.quantile(policy_ms, 0.95)),
                "controller_only_ms_p99": float(np.quantile(policy_ms, 0.99)),
                "episode_control_cost": episode["episode_control_cost"],
                "fallback_count": episode["fallback_count"],
                "causal_timestamp_violations": episode["causal_timestamp_violations"],
            })
            del model

    estimated_seconds = 0.0
    for row in rows:
        multiplier = 640 if row["method"] in {"B0", "B1"} else 640 * 3
        estimated_seconds += row["episode_wall_seconds"] * multiplier
    result = {
        "status": "VALIDATION_ONLY_RUNTIME_PILOT_COMPLETE",
        "device": str(device),
        "pdes": ["burgers", "heat"],
        "validation_parent_count": 1,
        "ticks_per_episode": 200,
        "variants_per_pde": 8,
        "locked_test_expected_executions": 25600,
        "estimated_locked_evaluation_hours_unscaled": estimated_seconds / 3600.0,
        "estimated_locked_evaluation_hours_with_15pct_margin": estimated_seconds * 1.15 / 3600.0,
        "interpretation": "Episode wall time includes CPU truth solver and controller; per-controller p99 is not host-ready E2E tail latency.",
        "test_opened": False,
        "calibration_opened": False,
        "elapsed_seconds": time.perf_counter() - started,
        "validation_data_sha256": {
            pde: hashlib.sha256((validation_data / pde / "trajectories.npz").read_bytes()).hexdigest()
            for pde in ("burgers", "heat")
        },
        "rows": rows,
    }
    out = ROOT / "evidence" / "v3_closed_loop_runtime_pilot.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    result["output"] = str(out)
    print(json.dumps({key: result[key] for key in (
        "status", "estimated_locked_evaluation_hours_unscaled",
        "estimated_locked_evaluation_hours_with_15pct_margin", "elapsed_seconds",
        "test_opened", "calibration_opened", "output")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
