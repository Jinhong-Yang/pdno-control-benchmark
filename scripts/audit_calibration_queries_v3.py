"""Audit fresh v3 calibration trajectories and queries without opening locked roles."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from pdno.data.audit_generated import audit_npz
from pdno.data.audit_teacher import audit_teacher_shard

def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def main()->int:
    manifest_path=ROOT/"data/manifests/parent_roles_metadata_only.json"
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    data_receipt=ROOT/"evidence/v3_calibration_generation_receipt.json"
    query_receipt=ROOT/"evidence/v3_calibration_query_generation_receipt.json"
    if not all(p.is_file() for p in (data_receipt,query_receipt)): raise SystemExit("v3 calibration generation receipt missing")
    if json.loads(data_receipt.read_text(encoding="utf-8")).get("test_opened") is not False or json.loads(query_receipt.read_text(encoding="utf-8")).get("test_opened") is not False:
        raise SystemExit("calibration generation receipt does not keep test sealed")
    rows=[]
    for pde in ("burgers","heat"):
        allowed={r["parent_id"] for r in manifest["roles"][pde]["parents"] if r["split"]=="calibration"}
        traj=ROOT/f"data/calibration_v3/calibration/{pde}/trajectories.npz"
        query=ROOT/f"data/queries_calibration_v3/calibration/{pde}/teacher_queries.npz"
        ta=audit_npz(traj,pde)
        with np.load(traj,allow_pickle=False) as z:
            trajectory_ids=set(z["parent_id"].astype(str)); ticks=z["state_true"].shape[1]
            action_ok=np.array_equal(z["action_projected"],z["action_applied"])
        if trajectory_ids!=allowed or len(allowed)!=64 or ticks!=201 or not action_ok:
            raise SystemExit(f"v3 calibration trajectory identity/shape/action mismatch: {pde}")
        qa=audit_teacher_shard(query,pde,allowed)
        with np.load(query,allow_pickle=False) as z:
            parent_ids=z["parent_id"].astype(str); decision_ticks=z["decision_tick"]
            counts={pid:int(np.sum(parent_ids==pid)) for pid in allowed}
            if len(parent_ids)!=512 or set(parent_ids)!=allowed or any(v!=8 for v in counts.values()): raise SystemExit(f"v3 calibration query coverage mismatch: {pde}")
            expected=np.linspace(8,120,8,dtype=int)
            if not np.array_equal(np.unique(decision_ticks),expected): raise SystemExit(f"v3 calibration query tick grid mismatch: {pde}")
            qmax=float(z["q_max"].max()) if pde=="heat" else 1.2
        if not ta["passed"] or not qa["passed"]: raise SystemExit(f"v3 calibration data/query science audit failed: {pde}")
        rows.append({"pde":pde,"parent_count":64,"trajectory_ticks":ticks,"query_count":512,
            "queries_per_parent":8,"decision_ticks":expected.tolist(),"trajectory_sha256":sha(traj),
            "query_sha256":sha(query),"q_max":qmax,"trajectory_audit":ta,"query_audit":qa})
    report={"status":"V3_CALIBRATION_QUERIES_AUDITED_TEST_SEALED","test_opened":False,
        "role_manifest_sha256":sha(manifest_path),"data_generation_receipt_sha256":sha(data_receipt),
        "query_generation_receipt_sha256":sha(query_receipt),"shards":rows,"all_checks_passed":True}
    out=ROOT/"evidence/v3_calibration_query_manifest.json"
    out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    data_audit={"status":"CALIBRATION_TRAJECTORY_AUDIT_PASSED","all_checks_passed":True,"test_opened":False,
        "role_manifest_sha256":sha(manifest_path),"generation_receipt_sha256":sha(data_receipt),
        "shards":[{"pde":r["pde"],"trajectory_sha256":r["trajectory_sha256"],"parent_count":r["parent_count"],"ticks":r["trajectory_ticks"]} for r in rows]}
    (ROOT/"evidence/v3_calibration_data_audit_receipt.json").write_text(json.dumps(data_audit,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    query_audit={"status":"CALIBRATION_QUERY_AUDIT_PASSED","all_checks_passed":True,"test_opened":False,
        "role_manifest_sha256":sha(manifest_path),"calibration_data_audit_sha256":sha(ROOT/"evidence/v3_calibration_data_audit_receipt.json"),
        "generation_receipt_sha256":sha(query_receipt),"query_manifest_sha256":sha(out),"shards":[
            {"pde":r["pde"],"query_sha256":r["query_sha256"],"parent_count":r["parent_count"],"queries":r["query_count"],"snapshots_per_parent":r["queries_per_parent"]} for r in rows]}
    (ROOT/"evidence/v3_calibration_query_audit_receipt.json").write_text(json.dumps(query_audit,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({"status":report["status"],"parents_per_pde":64,"queries_per_pde":512,"output":str(out)},indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
