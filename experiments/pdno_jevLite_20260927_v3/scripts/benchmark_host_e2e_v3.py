"""Three-session batch-1 host-ready E2E tail latency from locked replay inputs."""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pdno.controllers.actions import project_box_slew  # noqa: E402
from pdno.evaluation.closed_loop import load_controller, policy_action  # noqa: E402

METHODS = ("B0", "B1", "P", "P-no-rank", "B4", "B5", "B2", "B3")
SEEDS = (11, 23, 37)
KEYS = ("sensor_value", "sensor_mask", "sensor_age", "instrument_image", "image_mask",
        "goal_coefficients", "material_context", "previous_applied_action",
        "applied_action_history", "image_age", "image_valid", "goal_field")
ROLES = ("locked_nominal", "locked_coefficient_ood", "locked_delay_dropout")
REQUESTS_PER_SESSION = 20000
SESSIONS = 3
WARMUP = 50


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _load_role_inputs(pde: str, method: str, seed: int) -> dict[str, dict[str, np.ndarray]]:
    raw_root = ROOT / "evidence" / "locked_test_raw_v3"
    output = {}
    for role in ROLES:
        path = raw_root / role / pde / f"{method}_s{'na' if seed < 0 else seed}.npz"
        with np.load(path, allow_pickle=False) as z:
            # Locked replay stores 20 snapshots per parent at ticks 0, 10, ..., 190.
            # Flatten parent × snapshot so each replay request is one causal observation.
            block = {key: z[f"latency_{key}"].reshape((-1,) + z[f"latency_{key}"].shape[2:])
                     for key in KEYS}
        n = len(block["sensor_value"])
        if n != {"locked_nominal": 384 * 20, "locked_coefficient_ood": 128 * 20,
                 "locked_delay_dropout": 128 * 20}[role]:
            raise RuntimeError(f"latency replay count mismatch for {pde}/{method}/{seed}/{role}: {n}")
        if any(len(value) != n for value in block.values()):
            raise RuntimeError(f"latency replay array count mismatch for {pde}/{method}/{seed}/{role}")
        output[role] = block
    return output


def _balanced_session_sample(inputs: dict[str, dict[str, np.ndarray]], pde: str,
                             seed: int, session: int) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    targets = {"locked_nominal": 12000, "locked_coefficient_ood": 4000, "locked_delay_dropout": 4000}
    rng = np.random.default_rng(20260926 + 100000 * session + (seed if seed >= 0 else 0)
                                + (0 if pde == "burgers" else 50000))
    sampled: dict[str, list[np.ndarray]] = {key: [] for key in KEYS}
    source_roles, source_indices = [], []
    role_index = {name: i for i, name in enumerate(ROLES)}
    for role in ROLES:
        pool = inputs[role]
        n = len(pool["sensor_value"])
        target = targets[role]
        first = rng.permutation(n)
        picked = first[:min(n, target)].tolist()
        if len(picked) < target:
            picked.extend(rng.choice(n, size=target-len(picked), replace=True).tolist())
        for key in KEYS:
            sampled[key].append(pool[key][picked])
        source_roles.extend([role_index[role]] * target)
        source_indices.extend(picked)
    combined = {key: np.concatenate(parts, axis=0) for key, parts in sampled.items()}
    order = rng.permutation(REQUESTS_PER_SESSION)
    return ({key: value[order] for key, value in combined.items()},
            np.asarray(source_roles, dtype=np.int8)[order],
            np.asarray(source_indices, dtype=np.int32)[order])


def _propose_and_project(pde: str, method: str, obs: dict, model,
                         device: torch.device, qmax: float) -> tuple[np.ndarray, float | None]:
    start_event = end_event = None
    if device.type == "cuda" and method not in {"B0", "B1"}:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    proposed, _, _, _ = policy_action(method, pde, obs, model, device, qmax)
    stream_ms = None
    if start_event is not None:
        end_event.record()
        torch.cuda.synchronize(device)
        stream_ms = float(start_event.elapsed_time(end_event))
    low, high, slew = (-1., 1., .15) if pde == "burgers" else (0., 1., .10)
    applied = project_box_slew(proposed, obs["previous_applied_action"], low, high, slew)
    if applied.shape != (2,) or not np.isfinite(applied).all():
        raise RuntimeError("controller returned an invalid projected action")
    return applied, stream_ms


def main() -> int:
    marker_path = ROOT / "state" / "LOCKED_TEST_ONCE_V3.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("status") != "LOCKED_TEST_EVALUATION_COMPLETE" or marker.get("test_opened") is not True:
        raise SystemExit("host latency replay is allowed only after the complete one-shot locked test")
    train_record_path = ROOT / "state" / "CONFIRMATORY_TRAINING_PROCESS_V3.json"
    if not train_record_path.is_file() or json.loads(train_record_path.read_text(encoding="utf-8")).get("status") != "EXITED_SUCCESS":
        raise SystemExit("confirmatory training supervisor did not exit successfully")
    destination = ROOT / "evidence" / "latency_raw_v3"
    if destination.exists():
        raise SystemExit("v3 latency raw output exists; refusing overwrite")
    checkpoint_path = ROOT / "evidence" / "confirmatory_checkpoint_manifest_v3.json"
    checkpoint_manifest = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if checkpoint_manifest.get("status") != "FROZEN_CONFIRMATORY_CHECKPOINTS":
        raise SystemExit("v3 checkpoint manifest is not frozen")
    checkpoint_map = {(row["pde"], row["method"], int(row["seed"])): ROOT / row["path"]
                      for row in checkpoint_manifest["checkpoints"]}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qmax = 1.2
    for role in ("train", "validation"):
        with np.load(ROOT / "data" / "train_validation_v3" / role / "heat" / "trajectories.npz", allow_pickle=False) as z:
            qmax = max(qmax, 1.25 * float(z["goal"].max()))
    variants = [(pde, method, seed) for pde in ("burgers", "heat") for method in METHODS
                for seed in ((-1,) if method in {"B0", "B1"} else SEEDS)]
    destination.mkdir(parents=True)
    summaries = []
    progress = destination.parent / "latency_progress_v3.json"
    start_all = time.perf_counter()
    completed = 0
    for session in range(1, SESSIONS + 1):
        order = list(variants)
        random.Random(8310 + session).shuffle(order)
        for pde, method, seed in order:
            inputs = _load_role_inputs(pde, method, seed)
            replay, source_roles, source_indices = _balanced_session_sample(inputs, pde, seed, session)
            checkpoint = None if seed < 0 else checkpoint_map[(pde, method, seed)]
            model = None if checkpoint is None else load_controller(method, pde, seed, checkpoint, device)
            condition_qmax = 1.2 if pde == "burgers" else qmax
            warm_rng = np.random.default_rng(900000 + session * 1000 + len(summaries))
            warm_indices = warm_rng.integers(0, REQUESTS_PER_SESSION, size=WARMUP)
            for index in warm_indices:
                obs = {key: replay[key][index] for key in KEYS}
                _propose_and_project(pde, method, obs, model, device, condition_qmax)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            times = np.empty(REQUESTS_PER_SESSION, dtype=np.float64)
            stream_times = np.full(REQUESTS_PER_SESSION, np.nan, dtype=np.float64)
            for index in range(REQUESTS_PER_SESSION):
                if device.type == "cuda" and method not in {"B0", "B1"}:
                    torch.cuda.synchronize(device)
                start_ns = time.perf_counter_ns()
                obs = {key: replay[key][index] for key in KEYS}
                _, stream_ms = _propose_and_project(pde, method, obs, model, device, condition_qmax)
                times[index] = (time.perf_counter_ns() - start_ns) / 1e6
                if stream_ms is not None:
                    stream_times[index] = stream_ms
            peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
            filename = f"session{session}_{pde}_{method}_s{'na' if seed < 0 else seed}.npz"
            raw_path = destination / filename
            temp_path = raw_path.with_name(raw_path.name + ".tmp")
            with temp_path.open("wb") as stream:
                np.savez_compressed(stream, latency_ms=times, cuda_stream_span_ms=stream_times,
                    source_role=source_roles, source_index=source_indices,
                    session=np.asarray(session), method=np.asarray(method),
                    seed=np.asarray(seed), pde=np.asarray(pde))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_path, raw_path)
            row = {"session": session, "pde": pde, "method": method,
                "seed": None if seed < 0 else seed, "sample_count": REQUESTS_PER_SESSION,
                "warmup_count": WARMUP, "source_condition_counts": {
                    "locked_nominal": 12000, "locked_coefficient_ood": 4000,
                    "locked_delay_dropout": 4000},
                "unique_source_inputs_in_pool": sum(len(inputs[role]["sensor_value"]) for role in ROLES),
                "p50_ms": float(np.quantile(times, .50)), "p95_ms": float(np.quantile(times, .95)),
                "p99_ms": float(np.quantile(times, .99)), "p999_ms": float(np.quantile(times, .999)),
                "execution_path": "CPU" if device.type == "cpu" or method in {"B0", "B1"} else "CUDA",
                "cuda_stream_span_p50_ms": float(np.nanquantile(stream_times, .50)) if np.isfinite(stream_times).any() else None,
                "cuda_stream_span_p99_ms": float(np.nanquantile(stream_times, .99)) if np.isfinite(stream_times).any() else None,
                "deadline_misses": {f"{deadline}ms": int(np.sum(times > deadline)) for deadline in (1, 2, 5, 10)},
                "deadline_miss_rate_5ms": float(np.mean(times > 5)),
                "peak_cuda_allocated_bytes": peak_memory,
                "raw_path": str(raw_path.relative_to(ROOT)), "raw_sha256": sha(raw_path)}
            summaries.append(row)
            completed += 1
            atomic_json(progress, {"status": "RUNNING_NO_FINAL_LATENCY_SUMMARY",
                "variants_complete": completed, "variants_expected": len(variants) * SESSIONS,
                "requests_complete": completed * REQUESTS_PER_SESSION,
                "elapsed_seconds": time.perf_counter() - start_all})
            del model, inputs, replay
    raw_files = sorted(destination.glob("*.npz"))
    if len(raw_files) != len(variants) * SESSIONS or completed != len(raw_files):
        raise RuntimeError(f"latency raw inventory mismatch: {len(raw_files)} files, {completed} variants")
    result = {"status": "HOST_READY_E2E_THREE_SESSION_COMPLETE", "test_opened": True,
        "boundary": "host replay arrays -> request dictionary preparation -> input transfer/model or ROM/candidate scoring -> return transfer -> common action projection -> finite typed action validation",
        "request_batch": 1, "sessions": SESSIONS, "requests_per_session_per_variant": REQUESTS_PER_SESSION,
        "warmup_requests_per_session_per_variant": WARMUP,
        "unique_source_replay_pool_per_variant": "all locked parents and their 20 stored causal snapshots at ticks 0,10,...,190; samples are replayed with replacement only when a session target exceeds one condition pool",
        "truth_solver_included": False, "single_process_sequential": True,
        "device": str(device), "target_p99_ms": 2, "hard_deadline_ms": 5,
        "test_marker_sha256": sha(marker_path), "checkpoint_manifest_sha256": sha(checkpoint_path),
        "aggregate_rows": summaries,
        "raw_files_sha256": {str(path.relative_to(ROOT)): sha(path) for path in raw_files},
        "elapsed_seconds": time.perf_counter() - start_all}
    output = ROOT / "evidence" / "latency_summary_v3.json"
    atomic_json(output, result)
    atomic_json(progress, {"status": "COMPLETE", "variants_complete": completed,
                           "requests_complete": completed * REQUESTS_PER_SESSION,
                           "elapsed_seconds": result["elapsed_seconds"]})
    print(json.dumps({"status": result["status"], "variants": completed,
                      "requests": completed * REQUESTS_PER_SESSION,
                      "elapsed_seconds": result["elapsed_seconds"],
                      "output": str(output), "sha256": sha(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
