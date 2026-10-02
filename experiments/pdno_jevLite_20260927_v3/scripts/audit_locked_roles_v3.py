"""Audit fresh 200-tick v3 locked parent shards; never evaluates a controller."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from pdno.data.audit_generated import audit_npz
ROLES={"locked_nominal":384,"locked_coefficient_ood":128,"locked_delay_dropout":128}
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def main()->int:
    ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,default=ROOT/"data/locked_v3");args=ap.parse_args()
    if (ROOT/"state/LOCKED_TEST_ONCE_V3.json").exists():raise SystemExit("v3 locked test already opened; parent audit step is late")
    receipt_path=ROOT/"evidence/v3_locked_generation_receipt.json"
    if not receipt_path.is_file():raise SystemExit("v3 locked generation receipt missing")
    receipt=json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status")!="V3_LOCKED_PARENT_DATA_GENERATED_TEST_SEALED" or receipt.get("test_opened") is not False:raise SystemExit("v3 locked data receipt is invalid/open")
    manifest=json.loads((ROOT/"data/manifests/parent_roles_metadata_only.json").read_text(encoding="utf-8"));rows=[]
    for pde in ("burgers","heat"):
        for role,n in ROLES.items():
            path=args.root/role/pde/"trajectories.npz"
            base=audit_npz(path,pde)
            with np.load(path,allow_pickle=False) as z:
                ids=set(z["parent_id"].astype(str));expected={r["parent_id"] for r in manifest["roles"][pde]["parents"] if r["split"]==role}
                errors=list(base.get("errors",[]))
                if ids!=expected or len(ids)!=n:errors.append("parent IDs differ from frozen v3 role manifest")
                if z["state_true"].shape[1:]!=(201,256):errors.append("expected 200-tick 256-cell truth trajectory")
                if not np.array_equal(z["action_projected"],z["action_applied"]):errors.append("projected/applied action accounting mismatch")
                if role=="locked_nominal" and pde=="heat":
                    st=z["goal_step_tick"]
                    if int(np.sum(st==100))!=192 or int(np.sum(st<0))!=192:errors.append("heat step/fixed allocation is not balanced at tick 100")
                if role=="locked_coefficient_ood":
                    if pde=="burgers":
                        ix=z["role_index"]
                        if not np.all((z["nu"][ix%2==0]>=.005)&(z["nu"][ix%2==0]<=.008)):errors.append("Burgers alternating low-nu OOD group is outside [0.005,0.008]")
                        if not np.all((z["nu"][ix%2==1]>=.035)&(z["nu"][ix%2==1]<=.045)):errors.append("Burgers alternating high-nu OOD group is outside [0.035,0.045]")
                    if pde=="heat" and not np.all((z["kappa"]>=.024)&(z["kappa"]<=.032)):errors.append("heat kappa outside frozen OOD range")
                rows.append({"pde":pde,"role":role,"parent_count":len(ids),"errors":errors,"passed":not errors,
                    "audit":base,"sha256":sha(path),"bytes":path.stat().st_size})
    passed=all(r["passed"] for r in rows)
    report={"status":"V3_LOCKED_PARENT_DATA_AUDITED_TEST_SEALED" if passed else "AUDIT_FAILED",
        "passed":passed,"test_opened":False,"outer_ticks":200,"shards":rows,"generation_receipt_sha256":sha(receipt_path)}
    out=ROOT/"evidence/locked_200tick_data_audit_v3.json";out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({"status":report["status"],"shards":len(rows),"output":str(out)},indent=2));return 0 if passed else 1
if __name__=="__main__":raise SystemExit(main())
