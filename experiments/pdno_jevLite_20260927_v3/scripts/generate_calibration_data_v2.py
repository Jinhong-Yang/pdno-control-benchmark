"""Generate the frozen v2 calibration parent trajectories only after model selection."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pdno.data.generate import generate_roles  # noqa: E402
from pdno.data.manifests import audit_role_manifest  # noqa: E402


def main() -> int:
    spec = json.loads((ROOT / "config" / "final_spec_v2.json").read_text(encoding="utf-8"))
    freeze = json.loads((ROOT / "evidence" / "confirmatory_checkpoint_manifest_v2.json").read_text(encoding="utf-8"))
    h3 = json.loads((ROOT / "evidence" / "validation_closed_loop_comparator_v2.json").read_text(encoding="utf-8"))
    if freeze.get("status") != "FROZEN_CONFIRMATORY_CHECKPOINTS" or h3.get("test_opened") is not False:
        raise SystemExit("final model checkpoints and validation H3 must be frozen before calibration generation")
    if (ROOT / "evidence" / "calibration_lock_v2.json").exists():
        raise SystemExit("v2 calibration lock already exists; refusing to overwrite")
    manifest_path = ROOT / "data" / "manifests" / "parent_roles_metadata_only.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = audit_role_manifest(manifest)
    if not audit["passed"] or audit["test_opened"]:
        raise SystemExit(f"metadata-only role manifest failed: {audit}")
    out_root = ROOT / "data" / "calibration_v2"
    if out_root.exists():
        raise SystemExit("calibration output path already exists; inspect previous attempt")
    rows = generate_roles(manifest_path, out_root, roles=("calibration",), outer_ticks=200)
    if len(rows) != 2 or any(row["parent_count"] != 64 for row in rows):
        raise RuntimeError(f"expected 64 calibration parents/PDE, got {rows}")
    receipt = {"status": "V2_CALIBRATION_TRAJECTORIES_GENERATED_TEST_SEALED",
               "test_opened": False, "outer_ticks": 200, "roles": rows,
               "parent_manifest_sha256": __import__("hashlib").sha256(manifest_path.read_bytes()).hexdigest()}
    path = ROOT / "evidence" / "v2_calibration_generation_receipt.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "rows": rows, "receipt": str(path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
