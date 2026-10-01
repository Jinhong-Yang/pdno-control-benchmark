"""Generate the one eligible fresh 200-tick v3 locked dataset after calibration lock."""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from pdno.data.generate import generate_roles
ROLES=("locked_nominal","locked_coefficient_ood","locked_delay_dropout")
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def main()->int:
    marker=ROOT/"state/LOCKED_TEST_ONCE_V3.json"
    if marker.exists(): raise SystemExit("v3 locked-test marker exists; do not regenerate or rerun")
    lock=ROOT/"evidence/calibration_lock_v3.json"
    ck=ROOT/"evidence/confirmatory_checkpoint_manifest_v3.json"
    h3=ROOT/"evidence/validation_closed_loop_comparator_v3.json"
    freeze=ROOT/"evidence/pretraining_spec_freeze_v3.json"
    if not all(p.is_file() for p in (lock,ck,h3,freeze)): raise SystemExit("v3 calibration/checkpoint/H3/pretraining freeze missing")
    cal=json.loads(lock.read_text(encoding="utf-8")); checkpoints=json.loads(ck.read_text(encoding="utf-8")); comparator=json.loads(h3.read_text(encoding="utf-8"))
    if cal.get("status")!="CALIBRATION_LOCKED_TEST_SEALED" or cal.get("test_opened") is not False: raise SystemExit("v3 calibration is not frozen and test sealed")
    if checkpoints.get("status")!="FROZEN_CONFIRMATORY_CHECKPOINTS" or checkpoints.get("test_opened") is not False or comparator.get("test_opened") is not False: raise SystemExit("v3 model/H3 freeze invalid")
    out=ROOT/"data/locked_v3"
    if out.exists(): raise SystemExit("v3 locked target directory exists; inspect instead of overwriting")
    manifest=ROOT/"data/manifests/parent_roles_metadata_only.json"
    rows=generate_roles(manifest,out,roles=ROLES,allow_locked=True,outer_ticks=200)
    if len(rows)!=6 or sum(r["parent_count"] for r in rows)!=1280: raise RuntimeError(f"v3 locked parent inventory mismatch: {rows}")
    receipt={"status":"V3_LOCKED_PARENT_DATA_GENERATED_TEST_SEALED","test_opened":False,"outer_ticks":200,
        "parent_total":1280,"calibration_lock_sha256":sha(lock),"checkpoint_manifest_sha256":sha(ck),"roles":rows}
    target=ROOT/"evidence/v3_locked_generation_receipt.json"
    target.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({"status":receipt["status"],"parent_total":1280,"receipt":str(target)},indent=2)); return 0
if __name__=="__main__":raise SystemExit(main())
