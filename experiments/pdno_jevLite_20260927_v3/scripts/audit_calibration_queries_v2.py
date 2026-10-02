"""Audit calibration-only teacher query coverage, parent roles, and artifact hashes."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TICKS = np.linspace(8, 120, 8, dtype=int).tolist()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    query_manifest_path = ROOT / "evidence" / "v2_calibration_query_manifest.json"
    data_audit_path = ROOT / "evidence" / "v2_calibration_data_audit_receipt.json"
    generation_path = ROOT / "evidence" / "v2_calibration_generation_receipt.json"
    role_manifest_path = ROOT / "data" / "manifests" / "parent_roles_metadata_only.json"
    query_manifest = json.loads(query_manifest_path.read_text(encoding="utf-8"))
    data_audit = json.loads(data_audit_path.read_text(encoding="utf-8"))
    roles = json.loads(role_manifest_path.read_text(encoding="utf-8"))
    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    if query_manifest.get("status") != "V2_CALIBRATION_QUERIES_GENERATED_TEST_SEALED" or query_manifest.get("test_opened") is not False:
        raise SystemExit("calibration query manifest missing or not test-sealed")
    if data_audit.get("status") != "CALIBRATION_TRAJECTORY_AUDIT_PASSED" or not data_audit.get("all_checks_passed"):
        raise SystemExit("calibration parent trajectory audit missing or failed")
    if roles.get("metadata_only") is not True or roles.get("test_opened") is not False:
        raise SystemExit("metadata-only parent manifest invalid or test opened")
    if generation.get("test_opened") is not False:
        raise SystemExit("calibration trajectory generation receipt is not test-sealed")

    records = query_manifest.get("records", [])
    if len(records) != 2 or {r.get("pde") for r in records} != {"burgers", "heat"}:
        raise SystemExit("expected one query shard record per PDE")
    shard_rows = []
    for record in records:
        pde = record["pde"]
        path = Path(record["path"]).resolve()
        sidecar = path.with_suffix(".record.json")
        if record.get("split") != "calibration" or record.get("locked_test_opened") is not False:
            raise SystemExit(f"query shard is not calibration-only/test-sealed: {pde}")
        if record.get("parent_count") != 64 or record.get("query_count") != 512 or record.get("snapshots_per_parent") != 8:
            raise SystemExit(f"query count mismatch for {pde}: {record}")
        if record.get("candidate_count") != 10 or not path.is_file() or not sidecar.is_file():
            raise SystemExit(f"query artifact/sidecar missing or candidate count changed: {pde}")
        with np.load(path, allow_pickle=False) as z:
            arrays = {key: z[key] for key in z.files}
        required = {"parent_id", "decision_tick", "candidate_action", "future_field", "teacher_cost",
                    "teacher_peak", "teacher_best_index", "sensor_value", "sensor_mask",
                    "previous_applied_action", "applied_action_history"}
        if not required.issubset(arrays):
            raise SystemExit(f"query schema missing keys for {pde}: {sorted(required-set(arrays))}")
        parent_ids = arrays["parent_id"].astype(str).tolist()
        counts = Counter(parent_ids)
        expected_ids = {r["parent_id"] for r in roles["roles"][pde]["parents"] if r["split"] == "calibration"}
        if set(counts) != expected_ids or len(counts) != 64 or set(counts.values()) != {8}:
            raise SystemExit(f"query parent role/count mismatch for {pde}")
        ticks_by_parent = {pid: sorted(arrays["decision_tick"][i].astype(int).tolist())
                           for pid in counts
                           for i in [np.flatnonzero(arrays["parent_id"].astype(str) == pid)]}
        if any(ticks != EXPECTED_TICKS for ticks in ticks_by_parent.values()):
            raise SystemExit(f"snapshot tick coverage mismatch for {pde}")
        if arrays["candidate_action"].shape != (512, 10, 2) or arrays["future_field"].shape != (512, 10, 8, 128):
            raise SystemExit(f"candidate/future shape mismatch for {pde}")
        if arrays["teacher_cost"].shape != (512, 10) or arrays["teacher_peak"].shape != (512, 10):
            raise SystemExit(f"teacher label shape mismatch for {pde}")
        if not all(bool(np.isfinite(arrays[k]).all()) for k in ("candidate_action", "future_field", "teacher_cost", "teacher_peak")):
            raise SystemExit(f"non-finite candidate/teacher data in {pde}")
        if np.any(arrays["teacher_best_index"] < 0) or np.any(arrays["teacher_best_index"] >= 10):
            raise SystemExit(f"teacher best index outside candidate batch for {pde}")
        shard_rows.append({"pde": pde, "query_path": str(path.relative_to(ROOT)), "sidecar_path": str(sidecar.relative_to(ROOT)),
            "parent_count": len(counts), "queries": len(parent_ids), "snapshots_per_parent": 8,
            "decision_ticks": EXPECTED_TICKS, "candidate_action_shape": [512, 10, 2],
            "future_field_shape": [512, 10, 8, 128], "query_sha256": sha(path), "sidecar_sha256": sha(sidecar),
            "matches_metadata_calibration_role": True, "test_opened": False,
            "runtime_seconds": record["runtime_seconds"]})

    receipt = {"status": "CALIBRATION_QUERY_AUDIT_PASSED", "test_opened": False,
        "pde_count": 2, "parents_per_pde": 64, "snapshots_per_parent": 8,
        "queries_per_pde": 512, "candidate_count": 10, "decision_ticks": EXPECTED_TICKS,
        "role_manifest_sha256": sha(role_manifest_path),
        "calibration_data_audit_sha256": sha(data_audit_path),
        "generation_receipt_sha256": sha(generation_path),
        "query_manifest_sha256": sha(query_manifest_path), "shards": shard_rows,
        "all_checks_passed": True}
    out = ROOT / "evidence" / "v2_calibration_query_audit_receipt.json"
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "shards": shard_rows, "output": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
