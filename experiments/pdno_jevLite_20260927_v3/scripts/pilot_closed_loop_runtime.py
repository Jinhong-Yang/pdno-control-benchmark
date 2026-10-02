"""Measure 200-tick rollout runtime on one validation parent per required variant."""
from __future__ import annotations
import hashlib,json,time
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from pdno.evaluation.closed_loop import load_controller,run_episode  # noqa:E402

def main():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qmax=1.2
    with np.load(ROOT/"data/v2/train/heat/trajectories.npz",allow_pickle=False) as z:qmax=max(qmax,1.25*float(z["goal"].max()))
    with np.load(ROOT/"data/v2/validation/heat/trajectories.npz",allow_pickle=False) as z:qmax=max(qmax,1.25*float(z["goal"].max()))
    rows=[]; total_start=time.perf_counter()
    for pde in ("burgers","heat"):
        with np.load(ROOT/"data/v2/validation"/pde/"trajectories.npz",allow_pickle=False) as z:d={k:z[k][:1] for k in z.files}
        d["role"]=np.asarray(["validation"]); d["disturbance_seed"]=np.asarray([300000+int(str(d["parent_id"][0]).rsplit(":",1)[-1])])
        methods=[("B0",None),("B1",None)]+[(m,s) for m in ("P","B4","B5","B2","B3") for s in (11,23,37)]
        for method,seed in methods:
            ckpt=None if seed is None else ROOT/"runs/confirmatory_frozen_v1"/pde/f"{method}_s{seed}.pt"
            model=load_controller(method,pde,seed,ckpt,device)
            t0=time.perf_counter(); result=run_episode(d,0,pde,method,model,device,1.2 if pde=="burgers" else qmax,ticks=200)
            if result["fallback_count"] or result["causal_timestamp_violations"]: raise RuntimeError(f"smoke failure for {pde}/{method}/{seed}")
            rows.append({"pde":pde,"method":method,"seed":seed,"elapsed_seconds":time.perf_counter()-t0,
                "runtime_seconds_per_episode":time.perf_counter()-t0,"fallback_count":result["fallback_count"],
                "episode_control_cost":result["episode_control_cost"],"test_opened":False})
            del model
    estimated=sum(r["runtime_seconds_per_episode"]*640*1.15 for r in rows)
    out={"status":"VALIDATION_ONLY_CLOSED_LOOP_RUNTIME_PILOT_COMPLETE","test_opened":False,"device":str(device),
        "sample":"one 200-tick validation parent per PDE and every controller/learned seed; includes tick100 exact regret query",
        "elapsed_seconds":time.perf_counter()-total_start,"measured_variant_count":len(rows),
        "estimated_locked_episodes":2*640*17,"extrapolation_factor_for_other_parents_and_roles":1.15,
        "estimated_locked_evaluation_seconds":estimated,"estimated_locked_evaluation_hours":estimated/3600,"rows":rows,
        "validation_comparator_sha256":hashlib.sha256((ROOT/"evidence/validation_closed_loop_comparator.json").read_bytes()).hexdigest()}
    dest=ROOT/"evidence/closed_loop_runtime_pilot.json";dest.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({"status":out["status"],"elapsed_seconds":out["elapsed_seconds"],"estimated_locked_evaluation_hours":out["estimated_locked_evaluation_hours"],"output":str(dest)},indent=2))
if __name__=="__main__":main()
