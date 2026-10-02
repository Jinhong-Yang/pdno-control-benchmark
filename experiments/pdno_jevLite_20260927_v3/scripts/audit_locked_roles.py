"""Audit locked parent shards against the manifest; this never scores controllers."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from pdno.data.audit_generated import audit_npz  # noqa:E402

ap=argparse.ArgumentParser(); ap.add_argument("--root",type=Path,required=True); ap.add_argument("--outer-ticks",type=int,required=True)
args=ap.parse_args(); manifest=json.loads((ROOT/"data/manifests/parent_roles_metadata_only.json").read_text(encoding="utf-8"))
report={"root":str(args.root),"outer_ticks":args.outer_ticks,"test_opened":False,"shards":[]}
for pde in ("burgers","heat"):
 for role,n in (("locked_nominal",384),("locked_coefficient_ood",128),("locked_delay_dropout",128)):
  path=args.root/role/pde/"trajectories.npz"
  with np.load(path,allow_pickle=False) as z: arrays={k:z[k] for k in z.files}
  row=audit_npz(path,pde); role_rows=[r for r in manifest["roles"][pde]["parents"] if r["split"]==role]
  allowed=[r["parent_id"] for r in role_rows]
  errors=list(row["errors"])
  if set(arrays["parent_id"].astype(str))!=set(allowed): errors.append("parent IDs differ from frozen role manifest")
  if arrays["state_true"].shape[1]!=args.outer_ticks+1: errors.append("episode length differs from requested tick count")
  if role=="locked_coefficient_ood":
   if pde=="burgers":
    nu=arrays["nu"]
    low_idx=np.arange(n)%2==0; high_idx=~low_idx
    if not np.all((nu[low_idx]>=.005)&(nu[low_idx]<=.008)): errors.append("low-nu role-order half is outside OOD range")
    if not np.all((nu[high_idx]>=.035)&(nu[high_idx]<=.045)): errors.append("high-nu role-order half is outside OOD range")
   elif not np.all((arrays["kappa"]>=.024)&(arrays["kappa"]<=.032)): errors.append("heat OOD kappa outside frozen range")
  if role=="locked_nominal" and pde=="heat":
   step=arrays["goal_step_tick"]
   expected=args.outer_ticks//2
   if int(np.sum(step==expected))!=192 or int(np.sum(step<0))!=192: errors.append("heat step/fixed goal allocation is not balanced at episode midpoint")
  errors=[e for e in errors if e]
  report["shards"].append({"pde":pde,"role":role,"parent_count":len(arrays["parent_id"]),"errors":errors,
   "data_audit":row,"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"bytes":path.stat().st_size})
report["passed"]=all(not r["errors"] for r in report["shards"])
report["status"]="LOCKED_PARENT_DATA_AUDITED" if report["passed"] else "AUDIT_FAILED"
out=ROOT/"evidence"/(f"locked_{args.outer_ticks}tick_data_audit.json")
out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
print(json.dumps({"status":report["status"],"output":str(out),"shards":len(report["shards"]),"test_opened":False},indent=2))
raise SystemExit(0 if report["passed"] else 1)
