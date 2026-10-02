"""Generate fresh 200-tick v2 locked parents only after calibration is frozen."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pdno.data.generate import generate_roles  # noqa: E402

ROLES = ("locked_nominal", "locked_coefficient_ood", "locked_delay_dropout")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    lock_path = ROOT / "evidence" / "calibration_lock_v2.json"
    checkpoint_manifest_path = ROOT / "evidence" / "confirmatory_checkpoint_manifest_v2.json"
    h3_path = ROOT / "evidence" / "validation_closed_loop_comparator_v2.json"
    if not all(path.is_file() for path in (lock_path, checkpoint_manifest_path, h3_path)):
        raise SystemExit("calibration/checkpoint/H3 freeze prerequisite missing")
    calibration = json.loads(lock_path.read_text(encoding="utf-8"))
    checkpoints = json.loads(checkpoint_manifest_path.read_text(encoding="utf-8"))
    h3 = json.loads(h3_path.read_text(encoding="utf-8"))
    if calibration.get("status") != "CALIBRATION_LOCKED_TEST_SEALED" or calibration.get("test_opened") is not False:
        raise SystemExit("calibration must be frozen with test sealed")
    if checkpoints.get("status") != "FROZEN_CONFIRMATORY_CHECKPOINTS" or h3.get("test_opened") is not False:
        raise SystemExit("final confirmatory checkpoints and H3 must remain frozen")
    out_root = ROOT / "data" / "locked_v2"
    if out_root.exists():
        raise SystemExit("locked v2 output already exists; do not regenerate")
    manifest_path = ROOT / "data" / "manifests" / "parent_roles_metadata_only.json"
    rows = generate_roles(manifest_path, out_root, roles=ROLES, allow_locked=True, outer_ticks=200)
    if len(rows) != 6 or sum(row["parent_count"] for row in rows) != 1280:
        raise RuntimeError(f"locked v2 role generation count mismatch: {rows}")
    receipt = {"status": "V2_LOCKED_PARENT_DATA_GENERATED_TEST_SEALED",
               "test_opened": False, "outer_ticks": 200, "parent_total": 1280,
               "calibration_lock_sha256": sha(lock_path), "checkpoint_manifest_sha256": sha(checkpoint_manifest_path),
               "roles": rows}
    target = ROOT / "evidence" / "v2_locked_generation_receipt.json"
    target.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "parent_total": receipt["parent_total"],
                      "receipt": str(target)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
