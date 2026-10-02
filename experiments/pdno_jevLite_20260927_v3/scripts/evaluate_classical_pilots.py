"""Validation-only comparison of B0 LQR and B1 finite-candidate ROM choices."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from pdno.controllers.linear import nominal_lqr_action
from pdno.controllers.rom import rom_select_action

def _obs(data,i):
    return {k:data[k][i] for k in ("sensor_value","sensor_mask","sensor_age","instrument_image","image_mask",
            "image_valid","material_context","goal_field","previous_applied_action")}

def run(pde):
    path=ROOT/"data"/"queries_v2"/"validation"/pde/"teacher_queries.npz"
    with np.load(path,allow_pickle=False) as f: d={k:f[k] for k in f.files}
    selected={"B0":[],"B1":[]}; mismatch=[]; b1_runtime=[]
    import time
    for i in range(len(d["candidate_action"])):
        obs=_obs(d,i); actions=d["candidate_action"][i]; true_cost=d["teacher_cost"][i]
        b0=nominal_lqr_action(obs,pde,obs["previous_applied_action"])
        b0idx=int(np.argmin(np.sum((actions-b0[None,:])**2,axis=1)))
        mismatch.append(float(np.linalg.norm(actions[b0idx]-b0)))
        t=time.perf_counter()
        _, b1cost=rom_select_action(obs,pde,actions,d["goal"][i],float(d["q_max"][i]))
        b1_runtime.append(time.perf_counter()-t)
        b1idx=int(np.argmin(b1cost))
        selected["B0"].append(float(true_cost[b0idx]-np.min(true_cost)))
        selected["B1"].append(float(true_cost[b1idx]-np.min(true_cost)))
    return {"pde":pde,"split":"validation","parents":len(set(d["parent_id"].astype(str))),"queries":len(d["parent_id"]),
            "B0_nominal_candidate_mismatch_max":max(mismatch),
            "B0_teacher_regret_mean":float(np.mean(selected["B0"])),"B0_teacher_regret_median":float(np.median(selected["B0"])),
            "B1_teacher_regret_mean":float(np.mean(selected["B1"])),"B1_teacher_regret_median":float(np.median(selected["B1"])),
            "B1_cpu_ms_median":1000*float(np.median(b1_runtime)),"B1_cpu_ms_p90":1000*float(np.quantile(b1_runtime,.9)),
            "test_opened":False}

def main():
    results=[run(pde) for pde in ("burgers","heat")]
    path=ROOT/"evidence"/"classical_validation_pilot_v2.json"
    path.write_text(json.dumps(results,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(results,indent=2))

if __name__=="__main__": main()
