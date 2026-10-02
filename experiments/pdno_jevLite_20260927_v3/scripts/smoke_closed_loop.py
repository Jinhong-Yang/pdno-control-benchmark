"""Two-tick validation-only smoke across all frozen controller implementations."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from pdno.evaluation.closed_loop import load_controller,run_episode  # noqa:E402

def main():
 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
 pde="heat"; path=ROOT/"data/evaluation_v1/calibration/heat/trajectories.npz"
 with np.load(path,allow_pickle=False) as z: d={k:z[k][:1] for k in z.files}
 d["role"]=np.asarray(["train"]); d["disturbance_seed"]=np.asarray([300000],dtype=np.int64)
 d["initial_condition_seed"]=np.asarray([100000],dtype=np.int64); d["physical_parameter_seed"]=np.asarray([200000],dtype=np.int64)
 rows=[]
 for method in ("B0","B1","P","B4","B5","B2","B3"):
  ckpt=None if method in {"B0","B1"} else ROOT/"runs/confirmatory_frozen_v1/heat"/f"{method}_s11.pt"
  model=load_controller(method,pde,11,ckpt,device)
  result=run_episode(d,0,pde,method,model,device,1.0,query_tick=100,ticks=2)
  if result["action_applied"].shape!=(2,2) or not np.isfinite(result["episode_control_cost"]):
   raise RuntimeError(f"invalid closed-loop smoke output for {method}")
  rows.append({"method":method,"finite":True,"action_shape":list(result["action_applied"].shape),"fallback_count":result["fallback_count"],"fallback_errors":result["fallback_errors"]})
 snapshot=[]
 for method in ("B1","P","B4","B5"):
  ckpt=None if method=="B1" else ROOT/"runs/confirmatory_frozen_v1/heat"/f"{method}_s11.pt"
  model=load_controller(method,pde,11,ckpt,device)
  result=run_episode(d,0,pde,method,model,device,1.0,query_tick=100,ticks=101)
  if not np.isfinite(result["candidate_regret_at_snapshot"]) or not np.isfinite(result["field_nrmse_at_snapshot"]):
   raise RuntimeError(f"snapshot diagnostic missing for {method}")
  snapshot.append({"method":method,"candidate_regret":result["candidate_regret_at_snapshot"],"field_nrmse":result["field_nrmse_at_snapshot"],"fallback_count":result["fallback_count"]})
 out={"status":"VALIDATION_ONLY_CLOSED_LOOP_SMOKE_PASS","test_opened":False,"device":str(device),"results":rows,"tick100_snapshot":snapshot}
 dest=ROOT/"evidence/closed_loop_smoke.json"; dest.write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")
 print(json.dumps(out,indent=2))
if __name__=="__main__": main()
