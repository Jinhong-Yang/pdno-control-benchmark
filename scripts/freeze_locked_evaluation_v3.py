"""Freeze every v3 test input/hash while locked outcomes remain unopened."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    marker = ROOT / "state" / "LOCKED_TEST_ONCE_V3.json"
    process = ROOT / "state" / "LOCKED_TEST_PROCESS_V3.json"
    raw = ROOT / "evidence" / "locked_test_raw_v3"
    output = ROOT / "evidence" / "locked_evaluation_freeze_v3.json"
    if output.exists() or marker.exists() or process.exists() or raw.exists():
        raise SystemExit("freeze is too late or an execution/output marker exists; inspect without reopening")
    cal_path = ROOT / "evidence" / "calibration_lock_v3.json"
    cal_audit_path = ROOT / "evidence" / "v3_calibration_lock_audit_receipt.json"
    checkpoint_path = ROOT / "evidence" / "confirmatory_checkpoint_manifest_v3.json"
    h3_path = ROOT / "evidence" / "validation_closed_loop_comparator_v3.json"
    selection_path = ROOT / "evidence" / "validation_checkpoint_selection_v3.json"
    audit_path = ROOT / "evidence" / "locked_200tick_data_audit_v3.json"
    generation_path = ROOT / "evidence" / "v3_locked_generation_receipt.json"
    pretraining_path = ROOT / "evidence" / "pretraining_spec_freeze_v3.json"
    cal, checkpoints, h3, selection, audit, generation, pretraining = [load(path) for path in (
        cal_path, checkpoint_path, h3_path, selection_path, audit_path, generation_path, pretraining_path)]
    cal_audit = load(cal_audit_path)
    if cal.get("status") != "CALIBRATION_LOCKED_TEST_SEALED" or cal.get("test_opened") is not False:
        raise SystemExit("calibration lock invalid or test-opened")
    if cal_audit.get("status") != "CALIBRATION_LOCK_AUDIT_PASSED" or cal_audit.get("test_opened") is not False:
        raise SystemExit("calibration lock audit failed or test-opened")
    if checkpoints.get("status") != "FROZEN_CONFIRMATORY_CHECKPOINTS" or checkpoints.get("test_opened"):
        raise SystemExit("confirmatory checkpoints are not frozen/test-sealed")
    if h3.get("status") != "VALIDATION_ONLY_H3_COMPARATOR_SELECTED" or h3.get("test_opened") is not False:
        raise SystemExit("H3 selection was not made solely on validation")
    if selection.get("test_opened") is not False:
        raise SystemExit("checkpoint selection evidence is not test-sealed")
    if (audit.get("status") != "V3_LOCKED_PARENT_DATA_AUDITED_TEST_SEALED" or not audit.get("passed")
            or audit.get("test_opened") is not False or audit.get("outer_ticks") != 200):
        raise SystemExit("locked data audit failed or opened test")
    if generation.get("status") != "V3_LOCKED_PARENT_DATA_GENERATED_TEST_SEALED" or generation.get("test_opened"):
        raise SystemExit("locked generation receipt invalid")
    if pretraining.get("status") != "PRETRAINING_SPEC_FROZEN_TEST_SEALED" or pretraining.get("locked_test_opened"):
        raise SystemExit("pretraining freeze invalid")
    if len(checkpoints.get("checkpoints", [])) != 36 or len(checkpoints.get("observers", [])) != 2:
        raise SystemExit("checkpoint inventory is incomplete")
    for row in [*checkpoints["checkpoints"], *checkpoints["observers"]]:
        path = ROOT / row["path"]
        if not path.is_file() or sha(path) != row["sha256"]:
            raise SystemExit(f"checkpoint hash mismatch: {path}")
    shards = sorted((ROOT / "data" / "locked_v3").glob("*/**/trajectories.npz"))
    if len(shards) != 6:
        raise SystemExit(f"expected six v3 locked data shards; got {len(shards)}")
    shard_rows = []
    expected_count = {"locked_nominal": 384, "locked_coefficient_ood": 128, "locked_delay_dropout": 128}
    for path in shards:
        role, pde = path.parent.parent.name, path.parent.name
        if role not in expected_count or pde not in {"burgers", "heat"}:
            raise SystemExit(f"unexpected locked shard: {path}")
        with np.load(path, allow_pickle=False) as z:
            if len(z["parent_id"]) != expected_count[role] or z["state_true"].shape != (expected_count[role], 201, 256):
                raise SystemExit(f"locked shard shape/count mismatch: {path}")
            parent_digest = hashlib.sha256("\n".join(sorted(z["parent_id"].astype(str))).encode()).hexdigest()
        shard_rows.append({"path": str(path.relative_to(ROOT)), "sha256": sha(path),
                           "bytes": path.stat().st_size, "role": role, "pde": pde,
                           "parent_count": expected_count[role], "sorted_parent_ids_sha256": parent_digest})
    if len(audit.get("shards", [])) != 6 or {row["sha256"] for row in shard_rows} != {row["sha256"] for row in audit["shards"]}:
        raise SystemExit("locked audit shard hashes do not match current inputs")
    source_files = [ROOT / "config" / name for name in (
        "final_spec_v3.json", "evaluation_addendum_v3.yaml", "checkpoint_selection_v3.yaml",
        "frozen_experiment.yaml", "v3_preregistration.yaml")]
    source_files += [ROOT / "scripts" / name for name in (
        "run_locked_test_once_v3.py", "aggregate_locked_test_v3.py", "start_locked_test_v3.ps1",
        "generate_locked_data_v3.py", "audit_locked_roles_v3.py", "freeze_locked_evaluation_v3.py",
        "generate_calibration_data_v3.py", "generate_calibration_queries_v3.py", "calibrate_forecast_margins_v3.py",
        "select_validation_comparator_v3.py",
        "audit_validation_comparator_v3.py",
        "audit_calibration_queries_v3.py",
        "audit_calibration_lock_v3.py",
        "freeze_checkpoint_manifest_v3.py", "benchmark_host_e2e_v3.py", "analyze_latency_v3.py")]
    source_files += sorted((ROOT / "src" / "pdno").rglob("*.py"))
    evidence_files = [cal_path, checkpoint_path, h3_path, selection_path, audit_path, generation_path,
                      ROOT / "evidence" / "v3_calibration_generation_receipt.json",
                      ROOT / "evidence" / "v3_calibration_data_audit_receipt.json",
                      ROOT / "evidence" / "v3_calibration_query_manifest.json",
                      ROOT / "evidence" / "v3_calibration_query_audit_receipt.json",
                      cal_audit_path,
                      ROOT / "evidence" / "v3_h3_comparator_audit_receipt.json",
                      ROOT / "evidence" / "pretraining_spec_freeze_v3.json",
                      ROOT / "evidence" / "v3_closed_loop_runtime_pilot.json"]
    all_files = sorted(set(source_files + evidence_files))
    missing = [path for path in all_files if not path.is_file()]
    if missing:
        raise SystemExit("missing locked freeze inputs: " + ", ".join(str(path) for path in missing))
    freeze = {"status": "LOCKED_EVALUATION_INPUTS_FROZEN_TEST_SEALED", "test_opened": False,
        "protocol": "PDNO_JevLite_RTX5080_72H_frozen_spec_v3", "episode_ticks": 200,
        "query_tick": 100, "parent_method_seed_episodes": 25600,
        "h3_selected_comparator": h3["selected_comparator"],
        "calibration_lock_sha256": sha(cal_path), "checkpoint_manifest_sha256": sha(checkpoint_path),
        "calibration_lock_audit_sha256": sha(cal_audit_path),
        "validation_h3_sha256": sha(h3_path), "validation_checkpoint_selection_sha256": sha(selection_path),
        "pretraining_spec_freeze_sha256": sha(pretraining_path),
        "locked_data_audit_sha256": sha(audit_path), "locked_parent_shards": shard_rows,
        "frozen_input_sha256": {str(path.relative_to(ROOT)): sha(path) for path in all_files},
        "checkpoint_sha256": {row["path"]: row["sha256"] for row in [*checkpoints["checkpoints"], *checkpoints["observers"]]},
        "test_tuning": False}
    tmp = output.with_name(output.name + ".tmp")
    tmp.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(output)
    print(json.dumps({"status": freeze["status"], "shards": len(shard_rows),
                      "inputs": len(all_files), "output": str(output), "sha256": sha(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
