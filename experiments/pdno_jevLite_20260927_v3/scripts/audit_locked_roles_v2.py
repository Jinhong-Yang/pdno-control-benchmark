"""Audit v2 locked trajectory count, isolation, numerics, and causal/action contracts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pdno.data.audit_generated import audit_npz  # noqa: E402

COUNTS = {"locked_nominal": 384, "locked_coefficient_ood": 128, "locked_delay_dropout": 128}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    if (ROOT / "state" / "LOCKED_TEST_ONCE_V2.json").exists():
        raise SystemExit("locked test marker exists; this audit is pre-open only")
    receipt_path = ROOT / "evidence" / "v2_locked_generation_receipt.json"
    if not receipt_path.is_file():
        raise SystemExit("locked data generation receipt missing")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "V2_LOCKED_PARENT_DATA_GENERATED_TEST_SEALED" or receipt.get("test_opened"):
        raise SystemExit("locked generation receipt invalid or test-opened")
    role_manifest_path = ROOT / "data" / "manifests" / "parent_roles_metadata_only.json"
    role_manifest = json.loads(role_manifest_path.read_text(encoding="utf-8"))
    locked_ids = set()
    expected_by_role_pde = {}
    role_ids_by_pde = {}
    for block in role_manifest["roles"].values():
        pde = block["parents"][0]["pde"]
        role_ids_by_pde[pde] = {}
        for split in block["role_counts"]:
            ids = {row["parent_id"] for row in block["parents"] if row["split"] == split}
            role_ids_by_pde[pde][split] = ids
            if split.startswith("locked_"):
                expected_by_role_pde[(split, pde)] = ids
                locked_ids.update(ids)
        splits = list(role_ids_by_pde[pde])
        for i, left in enumerate(splits):
            for right in splits[i + 1:]:
                if role_ids_by_pde[pde][left] & role_ids_by_pde[pde][right]:
                    raise RuntimeError(f"metadata parent roles overlap for {pde}: {left}/{right}")
    pretest_ids = set()
    for role in ("train", "validation"):
        for pde in ("burgers", "heat"):
            with np.load(ROOT / "data" / role / pde / "trajectories.npz", allow_pickle=False) as z:
                pretest_ids.update(z["parent_id"].astype(str).tolist())
    if locked_ids & pretest_ids:
        raise RuntimeError("locked parent IDs overlap train/validation IDs")
    rows = []
    observed_locked_ids = set()
    for role, expected in COUNTS.items():
        for pde in ("burgers", "heat"):
            path = ROOT / "data" / "locked_v2" / role / pde / "trajectories.npz"
            result = audit_npz(path, pde)
            if not result["passed"] or result["parent_count"] != expected or result["time_points"] != 201:
                raise RuntimeError(f"locked data audit failed for {role}/{pde}: {result}")
            with np.load(path, allow_pickle=False) as z:
                ids = z["parent_id"].astype(str).tolist()
                shard_ids = set(ids)
                if len(ids) != len(shard_ids) or shard_ids & pretest_ids:
                    raise RuntimeError(f"duplicate or non-independent parent IDs: {role}/{pde}")
                if shard_ids != expected_by_role_pde[(role, pde)]:
                    raise RuntimeError(f"parent IDs do not match metadata role assignment: {role}/{pde}")
                if shard_ids & observed_locked_ids:
                    raise RuntimeError(f"locked parent IDs overlap across shards: {role}/{pde}")
                observed_locked_ids.update(shard_ids)
                if not np.all(z["action_applied"] == z["action_projected"]):
                    raise RuntimeError(f"applied/projected action mismatch: {role}/{pde}")
                if role == "locked_nominal" and pde == "heat":
                    step_count = int(np.sum(z["goal_step_tick"] >= 0))
                    if step_count != 192:
                        raise RuntimeError(f"heat setpoint step count must be exactly 192, got {step_count}")
            rows.append({"role": role, "pde": pde, "parent_count": expected,
                         "audit": result, "sha256": sha(path), "bytes": path.stat().st_size,
                         "matches_metadata_role_ids": True, "pairwise_unique_locked_ids": True})
    if observed_locked_ids != locked_ids or len(observed_locked_ids) != sum(COUNTS.values()) * 2:
        raise RuntimeError("observed locked shard IDs do not match the complete unique metadata locked pool")
    report = {"status": "V2_LOCKED_DATA_AUDIT_PASS_TEST_SEALED", "passed": True,
              "test_opened": False, "outer_ticks": 200,
              "locked_parent_total": sum(COUNTS.values()) * 2,
              "train_validation_overlap_count": 0,
              "locked_ids_match_metadata_roles": True,
              "locked_cohorts_pairwise_disjoint": True,
              "role_manifest_sha256": sha(role_manifest_path),
              "calibration_lock_sha256": sha(ROOT / "evidence" / "calibration_lock_v2.json"),
              "shards": rows}
    out = ROOT / "evidence" / "locked_200tick_data_audit_v2.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "shards": len(rows), "output": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
