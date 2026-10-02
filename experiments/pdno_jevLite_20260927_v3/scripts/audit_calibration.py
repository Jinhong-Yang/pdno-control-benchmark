"""Audit calibration-only trajectory and teacher-query shards and freeze input hashes."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pdno.data.audit_generated import audit_npz  # noqa: E402
from pdno.data.audit_teacher import audit_teacher_shard  # noqa: E402

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

manifest = json.loads((ROOT / "data/manifests/parent_roles_metadata_only.json").read_text(encoding="utf-8"))
reports=[]
for pde in ("burgers", "heat"):
    allowed={r["parent_id"] for r in manifest["roles"][pde]["parents"] if r["split"]=="calibration"}
    traj=ROOT/"data/evaluation_v1/calibration"/pde/"trajectories.npz"
    query=ROOT/"data/queries_calibration_v1/calibration"/pde/"teacher_queries.npz"
    ta=audit_npz(traj,pde)
    ids=set(ta.get("parent_ids",[]))
    if ids != allowed or ta.get("parent_count") != 64:
        ta["passed"]=False; ta.setdefault("errors",[]).append("trajectory parents differ from frozen calibration role pool")
    qa=audit_teacher_shard(query,pde,allowed)
    with np.load(query,allow_pickle=False) as z:
        qids=z["parent_id"].astype(str); counts={p:int(np.sum(qids==p)) for p in allowed}
    if set(qids.tolist()) != allowed or any(n!=8 for n in counts.values()):
        qa["passed"]=False; qa.setdefault("errors",[]).append("calibration queries must cover each of 64 parents exactly 8 times")
    reports.append({"pde":pde,"trajectory_audit":ta,"query_audit":qa,
        "trajectory_sha256":sha(traj),"query_sha256":sha(query),"trajectory_bytes":traj.stat().st_size,"query_bytes":query.stat().st_size})
failed=any(not r["trajectory_audit"]["passed"] or not r["query_audit"]["passed"] for r in reports)
record={"status":"CALIBRATION_DATA_AUDITED" if not failed else "AUDIT_FAILED","test_opened":False,
    "parent_role_manifest_sha256":sha(ROOT/"data/manifests/parent_roles_metadata_only.json"),"shards":reports}
out=ROOT/"evidence/calibration_query_manifest.json"
out.write_text(json.dumps(record,indent=2,sort_keys=True)+"\n",encoding="utf-8")
print(json.dumps({"status":record["status"],"output":str(out),"sha256":sha(out),
    "parents_per_pde":64,"queries_per_parent":8},indent=2))
raise SystemExit(1 if failed else 0)
