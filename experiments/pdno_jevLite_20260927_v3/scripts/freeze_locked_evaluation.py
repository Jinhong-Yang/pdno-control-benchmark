"""Pre-open integrity review and one-time locked-evaluation input freeze."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from pdno.data.audit_generated import audit_npz  # noqa:E402

def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    if (ROOT/"state/LOCKED_TEST_ONCE.json").exists(): raise SystemExit("test once-marker already exists; freeze step is too late")
    if (ROOT/"runs/locked_eval_work").exists() or (ROOT/"evidence/locked_test_raw").exists(): raise SystemExit("locked evaluation output directory already exists")
    calibration=json.loads((ROOT/"evidence/calibration_lock.json").read_text(encoding="utf-8"))
    if calibration.get("status")!="CALIBRATION_LOCKED_TEST_SEALED" or calibration.get("test_opened") is not False:
        raise SystemExit("calibration is not in sealed state")
    audit=json.loads((ROOT/"evidence/locked_200tick_data_audit.json").read_text(encoding="utf-8"))
    if not audit.get("passed") or audit.get("outer_ticks")!=200 or audit.get("test_opened") is not False: raise SystemExit("200-tick parent data audit failed")
    comp=json.loads((ROOT/"evidence/validation_closed_loop_comparator.json").read_text(encoding="utf-8"))
    if comp.get("status")!="VALIDATION_ONLY_COMPARATOR_SELECTED" or comp.get("selected_comparator") not in {"B0","B1"}: raise SystemExit("validation comparator selection missing")
    checkpoint=json.loads((ROOT/"evidence/confirmatory_checkpoint_manifest.json").read_text(encoding="utf-8"))
    if checkpoint.get("status")!="CHECKPOINTS_FROZEN" or checkpoint.get("test_opened") is not False: raise SystemExit("frozen checkpoint manifest invalid")
    for row in checkpoint["checkpoints"]:
        p=ROOT/row["path"]
        if not p.is_file() or sha(p)!=row["sha256"]: raise SystemExit(f"checkpoint hash mismatch: {p}")
    shards=sorted((ROOT/"data/locked_v2").glob("*/**/trajectories.npz"))
    if len(shards)!=6: raise SystemExit("expected exactly six 200-tick locked parent shards")
    role_meta=json.loads((ROOT/"data/manifests/parent_roles_metadata_only.json").read_text(encoding="utf-8"))
    checks=[]
    for p in shards:
        role,pde=p.parent.parent.name,p.parent.name
        a=audit_npz(p,pde)
        if not a["passed"]: raise SystemExit(f"causal/action audit failed: {p}: {a['errors']}")
        expected=next(x["parent_count"] for x in audit["shards"] if x["pde"]==pde and x["role"]==role)
        if a["parent_count"]!=expected or a["time_points"]!=201: raise SystemExit(f"parent count/length mismatch: {p}")
        checks.append({"path":str(p.relative_to(ROOT)),"sha256":sha(p),"bytes":p.stat().st_size,"parent_count":a["parent_count"]})
    trained={r["method"] for r in checkpoint["checkpoints"]}
    required={f"{m}_{p}_s{s}" for m in ("P","B4","B5","B2","B3") for p in ("burgers","heat") for s in (11,23,37)}
    actual={f"{r['method']}_{r['pde']}_s{r['seed']}" for r in checkpoint["checkpoints"] if r.get("method") in {"P","B4","B5","B2","B3"}}
    if actual!=required: raise SystemExit("required learned checkpoint coverage is incomplete")
    inputs=["config/frozen_experiment.yaml","config/evaluation_addendum_v1.yaml","config/evaluation_addendum_v2.yaml",
       "evidence/calibration_lock.json","evidence/confirmatory_checkpoint_manifest.json","evidence/locked_200tick_data_audit.json",
       "evidence/validation_closed_loop_comparator.json","scripts/run_locked_test_once.py","scripts/freeze_locked_evaluation.py","scripts/select_validation_comparator.py","src/pdno/evaluation/closed_loop.py",
       "src/pdno/controllers/linear.py","src/pdno/data/generate.py","data/manifests/parent_roles_metadata_only.json",
       "evidence/closed_loop_runtime_pilot.json","scripts/pilot_closed_loop_runtime.py","scripts/audit_locked_roles.py","tests/test_closed_loop.py"]
    freeze={"status":"LOCKED_EVALUATION_INPUTS_FROZEN_TEST_SEALED","test_opened":False,
      "episode_ticks":200,"snapshot_tick":100,"h3_selected_comparator":comp["selected_comparator"],
      "calibration_lock_sha256":sha(ROOT/"evidence/calibration_lock.json"),"parent_shards":checks,
      "files_sha256":{name:sha(ROOT/name) for name in inputs},
      "checkpoint_files_sha256":{r["path"]:r["sha256"] for r in checkpoint["checkpoints"]}}
    out=ROOT/"evidence/locked_evaluation_freeze.json"
    out.write_text(json.dumps(freeze,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({"status":freeze["status"],"selected_comparator":freeze["h3_selected_comparator"],
      "shards":len(checks),"checkpoint_count":len(checkpoint["checkpoints"]),"output":str(out),"sha256":sha(out)},indent=2))

if __name__=="__main__":main()
