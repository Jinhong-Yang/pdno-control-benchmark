"""Generate v3 calibration-only teacher continuations after pre-calibration freeze."""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from pdno.data.teacher_queries import build_teacher_queries

def main() -> int:
    receipt=ROOT/"evidence/v3_calibration_generation_receipt.json"
    if not receipt.is_file() or json.loads(receipt.read_text(encoding="utf-8")).get("test_opened") is not False:
        raise SystemExit("v3 calibration trajectory receipt missing or unsealed")
    out=ROOT/"data/queries_calibration_v3"
    if out.exists(): raise SystemExit("v3 calibration query output exists; refusing overwrite")
    maxima=[]
    for role in ("train","validation"):
        with np.load(ROOT/f"data/train_validation_v3/{role}/heat/trajectories.npz",allow_pickle=False) as z:
            maxima.append(float(np.max(z["goal"])))
    qmax=1.25*max(maxima)
    rows=[]
    for pde in ("burgers","heat"):
        rows.append(build_teacher_queries(ROOT/f"data/calibration_v3/calibration/{pde}/trajectories.npz",pde,"calibration",
            out/f"calibration/{pde}/teacher_queries.npz",8,None,qmax if pde=="heat" else None))
    result={"status":"V3_CALIBRATION_QUERIES_GENERATED_TEST_SEALED","test_opened":False,
            "parents_per_pde":64,"snapshots_per_parent":8,"heat_qmax_train_validation_only":qmax,
            "rows":rows,"trajectory_receipt_sha256":hashlib.sha256(receipt.read_bytes()).hexdigest()}
    target=ROOT/"evidence/v3_calibration_query_generation_receipt.json"
    target.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({"status":result["status"],"rows":rows,"receipt":str(target)},indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
