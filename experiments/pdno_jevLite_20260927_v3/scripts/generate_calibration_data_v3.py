"""Generate only fresh v3 calibration parents after model/H3 freeze."""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from pdno.data.generate import generate_roles
from pdno.data.manifests import audit_role_manifest

def main() -> int:
    manifest_path=ROOT/"data/manifests/parent_roles_metadata_only.json"
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    if not audit_role_manifest(manifest)["passed"] or manifest.get("test_opened"):
        raise SystemExit("v3 parent role manifest failed audit or is already open")
    required=(ROOT/"config/final_spec_v3.json",ROOT/"evidence/confirmatory_checkpoint_manifest_v3.json",
              ROOT/"evidence/validation_closed_loop_comparator_v3.json")
    if not all(p.is_file() for p in required): raise SystemExit("frozen v3 spec, checkpoints, or H3 evidence missing")
    if (ROOT/"evidence/calibration_lock_v3.json").exists(): raise SystemExit("calibration lock exists; refusing to overwrite")
    spec=json.loads(required[0].read_text(encoding="utf-8"))
    checkpoints=json.loads(required[1].read_text(encoding="utf-8"))
    h3=json.loads(required[2].read_text(encoding="utf-8"))
    if spec.get("status")!="FROZEN_FAIL_INCLUSIVE_PRETRAINING_SPEC" or checkpoints.get("test_opened") is not False or h3.get("test_opened") is not False:
        raise SystemExit("v3 pre-calibration prerequisites are not frozen and sealed")
    out=ROOT/"data/calibration_v3"
    if out.exists(): raise SystemExit("v3 calibration target directory exists; inspect instead of overwriting")
    rows=generate_roles(manifest_path,out,roles=("calibration",),outer_ticks=200)
    if len(rows)!=2 or any(r["parent_count"]!=64 for r in rows): raise RuntimeError(f"calibration role count mismatch: {rows}")
    receipt={"status":"V3_CALIBRATION_TRAJECTORIES_GENERATED_TEST_SEALED","test_opened":False,"outer_ticks":200,
             "roles":rows,"parent_manifest_sha256":hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
    target=ROOT/"evidence/v3_calibration_generation_receipt.json"
    target.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({"status":receipt["status"],"rows":rows,"receipt":str(target)},indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
