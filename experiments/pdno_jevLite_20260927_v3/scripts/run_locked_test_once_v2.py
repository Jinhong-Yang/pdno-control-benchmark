"""One-shot supervised 200-tick locked test. A post-marker error is terminal."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pdno.evaluation.closed_loop import load_controller, run_episode  # noqa: E402

ROLES = {"locked_nominal": 384, "locked_coefficient_ood": 128, "locked_delay_dropout": 128}
METHODS = ("B0", "B1", "P", "P-no-rank", "B4", "B5", "B2", "B3")
SEEDS = (11, 23, 37)
TOTAL_RUNS = 25600


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def create_open_marker_once(path: Path, value: dict) -> None:
    """Reserve the sole test execution atomically across concurrent launchers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def save_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _summary_arrays(rows: list[dict]) -> dict[str, np.ndarray]:
    keys = ("parent_id", "episode_control_cost", "tracking_rmse", "truth_constraint_violation",
            "fallback_rate", "fallback_count", "field_nrmse_at_snapshot", "candidate_regret_at_snapshot",
            "causal_timestamp_violations", "snapshot_candidate_best_index", "action_proposed",
            "action_projected", "action_applied", "state_true", "goal")
    arrays = {}
    for key in keys:
        values = [r[key] for r in rows]
        if key == "parent_id":
            arrays[key] = np.asarray(values, dtype="U96")
        elif key == "snapshot_candidate_best_index":
            arrays[key] = np.asarray([-1 if x is None else x for x in values], dtype=np.int16)
        else:
            arrays[key] = np.asarray(values)
    arrays["fallback_errors"] = np.asarray([json.dumps(r["fallback_errors"]) for r in rows], dtype="U2048")
    for key in ("sensor_mask", "sensor_age", "sensor_capture_time", "sensor_receive_time",
                "image_valid", "image_age", "image_capture_time", "image_receive_time"):
        arrays[key] = np.stack([r[key] for r in rows])
    replay_keys = tuple(rows[0]["latency_replay"])
    for key in replay_keys:
        arrays[f"latency_{key}"] = np.stack([r["latency_replay"][key] for r in rows])
    return arrays


def _checkpoint_map(manifest_path: Path) -> dict[tuple[str, str, int], Path]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "FROZEN_CONFIRMATORY_CHECKPOINTS":
        raise RuntimeError("confirmatory checkpoint manifest is not frozen")
    expected = {(pde, method, seed) for pde in ("burgers", "heat")
                for method in METHODS[2:] for seed in SEEDS}
    result = {}
    for row in manifest.get("checkpoints", []):
        key = (row["pde"], row["method"], int(row["seed"]))
        if key in result:
            raise RuntimeError(f"duplicate checkpoint record: {key}")
        path = ROOT / row["path"]
        if not path.is_file() or sha(path) != row["sha256"]:
            raise RuntimeError(f"missing or changed frozen checkpoint: {path}")
        result[key] = path
    if result.keys() != expected:
        raise RuntimeError(f"checkpoint manifest coverage mismatch; missing={expected-result.keys()}, extra={result.keys()-expected}")
    return result


def main() -> int:
    marker = ROOT / "state" / "LOCKED_TEST_ONCE_V2.json"
    if marker.exists():
        raise SystemExit("v2 locked test already opened; runner refuses restart")
    cal = ROOT / "evidence" / "calibration_lock_v2.json"
    freeze_path = ROOT / "evidence" / "locked_evaluation_freeze_v2.json"
    checkpoint_manifest_path = ROOT / "evidence" / "confirmatory_checkpoint_manifest_v2.json"
    h3_path = ROOT / "evidence" / "validation_closed_loop_comparator_v2.json"
    selection_path = ROOT / "evidence" / "validation_checkpoint_selection_v2.json"
    raw_root = ROOT / "evidence" / "locked_test_raw_v2"
    if raw_root.exists():
        raise SystemExit("v2 raw output directory exists without a test marker; inspect manually")
    if not all(path.is_file() for path in (cal, freeze_path, checkpoint_manifest_path, h3_path, selection_path)):
        raise SystemExit("calibration lock, locked freeze, checkpoint manifest, H3 or selection evidence is missing")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "LOCKED_EVALUATION_INPUTS_FROZEN_TEST_SEALED" or freeze.get("test_opened") is not False:
        raise SystemExit("v2 locked inputs are not frozen in the sealed state")
    for relative, expected_hash in freeze.get("frozen_input_sha256", {}).items():
        frozen_path = ROOT / relative
        if not frozen_path.is_file() or sha(frozen_path) != expected_hash:
            raise SystemExit(f"frozen test source/config/evidence changed: {frozen_path}")
    for relative, expected_hash in freeze.get("checkpoint_sha256", {}).items():
        frozen_checkpoint = ROOT / relative
        if not frozen_checkpoint.is_file() or sha(frozen_checkpoint) != expected_hash:
            raise SystemExit(f"frozen checkpoint changed: {frozen_checkpoint}")
    h3 = json.loads(h3_path.read_text(encoding="utf-8"))
    selected = json.loads(selection_path.read_text(encoding="utf-8"))
    calibration = json.loads(cal.read_text(encoding="utf-8"))
    if h3.get("status") != "VALIDATION_ONLY_H3_COMPARATOR_SELECTED" or h3.get("test_opened") is not False:
        raise SystemExit("validation H3 comparator is missing or not validation-only")
    if selected.get("test_opened") is not False:
        raise SystemExit("validation checkpoint selection evidence is not test-sealed")
    if calibration.get("status") != "CALIBRATION_LOCKED_TEST_SEALED":
        raise SystemExit("calibration lock status is not sealed")
    checkpoint_map = _checkpoint_map(checkpoint_manifest_path)
    locked_root = ROOT / "data" / "locked_v2"
    input_files = sorted(locked_root.glob("*/**/trajectories.npz"))
    if len(input_files) != 6:
        raise SystemExit(f"expected six v2 locked parent shards; found {len(input_files)}")
    input_hashes = {str(path.relative_to(ROOT)): sha(path) for path in input_files}
    frozen_shard_hashes = {row["path"]: row["sha256"] for row in freeze.get("locked_parent_shards", [])}
    if input_hashes != frozen_shard_hashes:
        raise SystemExit("locked input shard inventory or hash differs from pre-open freeze")
    source_files = [ROOT / "config" / "evaluation_addendum_v2.yaml",
                    ROOT / "config" / "checkpoint_selection_v2.yaml",
                    ROOT / "config" / "research_version_2_preregistration.yaml",
                    ROOT / "config" / "frozen_experiment.yaml",
                    ROOT / "contracts" / "input_schema.json",
                    ROOT / "scripts" / "run_locked_test_once_v2.py",
                    ROOT / "scripts" / "aggregate_locked_test_v2.py",
                    ROOT / "scripts" / "start_locked_test_v2.ps1",
                    ROOT / "src" / "pdno" / "evaluation" / "closed_loop.py"]
    source_files.extend(sorted((ROOT / "src" / "pdno").rglob("*.py")))
    if not all(path.is_file() for path in source_files):
        raise SystemExit("one or more frozen evaluation source files are missing")
    model_manifest = json.loads(checkpoint_manifest_path.read_text(encoding="utf-8"))
    q_max = 1.2
    with np.load(ROOT / "data" / "train" / "heat" / "trajectories.npz", allow_pickle=False) as z:
        q_max = max(q_max, 1.25 * float(z["goal"].max()))
    with np.load(ROOT / "data" / "validation" / "heat" / "trajectories.npz", allow_pickle=False) as z:
        q_max = max(q_max, 1.25 * float(z["goal"].max()))
    manifest = {"status": "LOCKED_TEST_OPENED_ONCE_IN_PROGRESS", "test_opened": True,
        "opened_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "episode_ticks": 200,
        "query_tick": 100, "expected_episodes": TOTAL_RUNS,
        "method_inventory": list(METHODS), "learned_seeds": list(SEEDS),
        "calibration_lock_sha256": sha(cal), "locked_freeze_sha256": sha(freeze_path),
        "checkpoint_manifest_sha256": sha(checkpoint_manifest_path),
        "h3_comparator_sha256": sha(h3_path), "checkpoint_selection_sha256": sha(selection_path),
        "evaluation_source_sha256": {str(path.relative_to(ROOT)): sha(path) for path in source_files},
        "locked_parent_shard_sha256": input_hashes,
        "q_max_heat_train_validation_only": q_max,
        "policy": "failure_after_marker_is_terminal; no resume or rerun; raw score artifacts are not analyzed until complete"}
    create_open_marker_once(marker, manifest)
    raw_root.mkdir(parents=True)
    progress_path = raw_root / "progress.json"
    start = time.perf_counter()
    completed = 0
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for pde in ("burgers", "heat"):
            for role, expected_n in ROLES.items():
                shard = locked_root / role / pde / "trajectories.npz"
                with np.load(shard, allow_pickle=False) as z:
                    data = {key: z[key] for key in z.files}
                n = len(data["parent_id"])
                if n != expected_n or data["state_true"].shape[1:] != (201, 256):
                    raise RuntimeError(f"locked shard count or state length mismatch: {pde}/{role}")
                if "role" not in data:
                    data["role"] = np.asarray([role] * n)
                if "disturbance_seed" not in data:
                    data["disturbance_seed"] = np.asarray(
                        [300000 + int(str(pid).rsplit(":", 1)[-1]) for pid in data["parent_id"]], dtype=np.int64)
                for method in METHODS:
                    seeds = (None,) if method in {"B0", "B1"} else SEEDS
                    for seed in seeds:
                        checkpoint = checkpoint_map[(pde, method, seed)] if seed is not None else None
                        model = load_controller(method, pde, seed, checkpoint, device)
                        episodes = []
                        started = time.perf_counter()
                        qmax_use = 1.2 if pde == "burgers" else q_max
                        for i in range(n):
                            episode = run_episode(data, i, pde, method, model, device, qmax_use,
                                                  query_tick=100, ticks=200)
                            episodes.append(episode)
                        del model
                        arrays = _summary_arrays(episodes)
                        out = raw_root / role / pde / f"{method}_s{seed if seed is not None else 'na'}.npz"
                        save_npz_atomic(out, arrays)
                        completed += n
                        record = {"status": "SHARD_COMPLETE", "method": method, "seed": seed, "pde": pde,
                                  "role": role, "parent_count": n, "episode_ticks": 200,
                                  "elapsed_seconds": time.perf_counter() - started,
                                  "raw_sha256": sha(out), "test_opened": True,
                                  "paired_parent_unit": True,
                                  "q_max_heat_train_validation_only": q_max if pde == "heat" else None}
                        atomic_json(out.with_suffix(".record.json"), record)
                        atomic_json(progress_path, {"status": "RUNNING_NO_AGGREGATE_METRICS",
                            "completed_episodes": completed, "expected_episodes": TOTAL_RUNS,
                            "completed_shards": len(list(raw_root.glob("*/**/*.record.json"))),
                            "elapsed_seconds": time.perf_counter() - start})
        raw_files = sorted(raw_root.glob("*/**/*.npz"))
        if len(raw_files) != 120 or completed != TOTAL_RUNS:
            raise RuntimeError(f"raw shard inventory mismatch: {len(raw_files)} files, {completed} episodes")
        for path in raw_files:
            with np.load(path, allow_pickle=False) as z:
                if not np.array_equal(z["action_projected"], z["action_applied"]):
                    raise RuntimeError(f"projected/applied mismatch: {path}")
                if not np.isfinite(z["state_true"]).all() or not np.isfinite(z["action_proposed"]).all():
                    raise RuntimeError(f"nonfinite locked evidence: {path}")
        raw_artifacts = sorted([*raw_files, *raw_root.glob("*/**/*.record.json")])
        if len(raw_artifacts) != 240:
            raise RuntimeError(f"raw artifact inventory mismatch: expected 240, found {len(raw_artifacts)}")
        manifest.update({"status": "LOCKED_TEST_EVALUATION_COMPLETE", "episodes_complete": completed,
            "elapsed_seconds": time.perf_counter() - start,
            "raw_result_sha256": {str(path.relative_to(ROOT)): sha(path) for path in raw_files},
            "raw_artifact_sha256": {str(path.relative_to(ROOT)): sha(path) for path in raw_artifacts}})
        atomic_json(marker, manifest)
        atomic_json(progress_path, {"status": "COMPLETE", "completed_episodes": completed,
                                    "elapsed_seconds": time.perf_counter() - start})
    except Exception as exc:
        manifest.update({"status": "LOCKED_TEST_OPENED_INCONCLUSIVE_FAILURE",
                         "episodes_complete": completed,
                         "elapsed_seconds": time.perf_counter() - start,
                         "failure": f"{type(exc).__name__}: {exc}"})
        atomic_json(marker, manifest)
        atomic_json(progress_path, {"status": "TERMINAL_INCONCLUSIVE_FAILURE",
                                    "completed_episodes": completed, "failure": manifest["failure"]})
        raise
    print(json.dumps({"status": manifest["status"], "episodes": completed,
                      "elapsed_seconds": manifest["elapsed_seconds"], "raw_shards": len(raw_files)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
