"""Freeze required v3 best-validation checkpoints and shared observers by hash."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
METHODS=("P","P-no-rank","B4","B5","B2","B3");PDES=("burgers","heat");SEEDS=(11,23,37)
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def main()->int:
    out=ROOT/"evidence/confirmatory_checkpoint_manifest_v3.json"
    if out.exists():raise SystemExit("v3 checkpoint manifest exists; refusing overwrite")
    progress=ROOT/"evidence/v3_confirmatory_training_progress.json"
    g2=ROOT/"evidence/v3_g2_pilot_progress.json"
    if not progress.is_file() or not g2.is_file():raise SystemExit("v3 training progress or G2 result missing")
    pr=json.loads(progress.read_text(encoding="utf-8"));gp=json.loads(g2.read_text(encoding="utf-8"))
    if pr.get("status")!="TRAINING_COMPLETE" or gp.get("g2_field_gate_pass") is not False:raise SystemExit("training incomplete or G2 failure not explicitly recorded")
    rows=[];observers=[]
    for pde in PDES:
        op=ROOT/f"runs/confirmatory_v3_training/observer/{pde}_seed7.pt"
        if not op.is_file():raise SystemExit(f"shared observer missing: {op}")
        oi=torch.load(op,map_location="cpu",weights_only=False);om=oi.get("metadata",{})
        if om.get("method")!="observer" or om.get("pde")!=pde:raise SystemExit(f"observer metadata mismatch: {op}")
        observers.append({"path":str(op.relative_to(ROOT)),"sha256":sha(op),"bytes":op.stat().st_size,"method":"observer","pde":pde,"seed":7,"metadata":om})
        for method in METHODS:
            for seed in SEEDS:
                cp=ROOT/f"runs/confirmatory_v3_training/{pde}/{method}_s{seed}.pt"
                summary=cp.with_suffix(".summary.json")
                if not cp.is_file() or not summary.is_file():raise SystemExit(f"checkpoint/summary missing: {cp}")
                item=torch.load(cp,map_location="cpu",weights_only=False);meta=item.get("metadata",{});sm=json.loads(summary.read_text(encoding="utf-8"))
                if (meta.get("method"),meta.get("pde"),int(meta.get("seed",-1)))!=(method,pde,seed):raise SystemExit(f"checkpoint identity mismatch: {cp}")
                if sm.get("checkpoint")!=str(cp) or int(meta.get("steps",-1))!=int(sm.get("best_step",-2)):raise SystemExit(f"checkpoint does not match its best-validation summary: {cp}")
                rows.append({"path":str(cp.relative_to(ROOT)),"sha256":sha(cp),"bytes":cp.stat().st_size,"method":method,"pde":pde,"seed":seed,
                    "selection_status":"G2_FAIL_RETAINED_DIAGNOSTIC" if method in ("P","P-no-rank","B4","B5") else "BEST_VALIDATION_CHECKPOINT",
                    "selected_update":sm["best_step"],"validation_summary":sm,"metadata":meta})
    if len(rows)!=36 or len(observers)!=2:raise SystemExit(f"expected 36 learned checkpoints and 2 observers; got {len(rows)}/{len(observers)}")
    result={"protocol":"PDNO_JevLite_RTX5080_72H_frozen_spec_v3","status":"FROZEN_CONFIRMATORY_CHECKPOINTS","test_opened":False,
        "learned_checkpoint_count":36,"observer_count":2,"g2_field_gate_pass":False,"g2_failure_retained":True,
        "training_progress_sha256":sha(progress),"g2_pilot_sha256":sha(g2),"checkpoints":rows,"observers":observers}
    out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    selected=[{"pde":r["pde"],"method":r["method"],"seed":r["seed"],"official_checkpoint_sha256":r["sha256"],
               "selection_status":r["selection_status"],"update":r["selected_update"],"checkpoint":r["path"]} for r in rows]
    selection={"protocol":"validation_checkpoint_selection_v3","status":"VALIDATION_SELECTED_G2_FAIL_RETAINED","test_opened":False,
        "selection_source":"per-model train/validation best checkpoint summary; no calibration/test data used",
        "selected":selected,"g2_field_gate_pass":False,"g2_failure_retained":True}
    selection_path=ROOT/"evidence/validation_checkpoint_selection_v3.json"
    selection_path.write_text(json.dumps(selection,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({"status":result["status"],"learned_checkpoints":len(rows),"observers":len(observers),"g2_field_gate_pass":False,"output":str(out)},indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
