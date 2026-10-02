"""Select the H3 fixed non-generative comparator using validation closed loops only."""
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
    out={"status":"VALIDATION_ONLY_COMPARATOR_SELECTED","test_opened":False,"selection_rule":"equal-weight mean of PDE-specific normalized mean episode costs; B0 wins exact tie","pdes":{}}
    qmax=1.2
    with np.load(ROOT/"data/v2/train/heat/trajectories.npz",allow_pickle=False) as z:qmax=max(qmax,1.25*float(z["goal"].max()))
    with np.load(ROOT/"data/v2/validation/heat/trajectories.npz",allow_pickle=False) as z:qmax=max(qmax,1.25*float(z["goal"].max()))
    for pde in ("burgers","heat"):
        with np.load(ROOT/"data/v2/validation"/pde/"trajectories.npz",allow_pickle=False) as z:d={k:z[k] for k in z.files}
        n=len(d["parent_id"]); d["role"]=np.asarray(["validation"]*n)
        d["disturbance_seed"]=np.asarray([300000+int(str(x).rsplit(":",1)[-1]) for x in d["parent_id"]],dtype=np.int64)
        costs={}
        for method in ("B0","B1"):
            model=load_controller(method,pde,None,None,device)
            rows=[run_episode(d,i,pde,method,model,device,1.2 if pde=="burgers" else qmax,ticks=200) for i in range(n)]
            if any(r["fallback_count"] for r in rows): raise RuntimeError(f"validation fallback seen in {pde}/{method}")
            costs[method]=np.asarray([r["episode_control_cost"] for r in rows])
        out["pdes"][pde]={"parent_count":n,"parent_ids":d["parent_id"].astype(str).tolist(),
            "mean_control_cost":{k:float(v.mean()) for k,v in costs.items()},
            "mean_cost_ratio_to_B0":{k:float(v.mean()/max(costs["B0"].mean(),1e-12)) for k,v in costs.items()},
            "paired_B1_minus_B0_mean":float(np.mean(costs["B1"]-costs["B0"]))}
    score={m:float(np.mean([out["pdes"][p]["mean_cost_ratio_to_B0"][m] for p in ("burgers","heat")])) for m in ("B0","B1")}
    out["normalized_selection_scores"]=score; out["selected_comparator"]="B1" if score["B1"]<score["B0"] else "B0"
    out["validation_data_sha256"]={p:__import__("hashlib").sha256((ROOT/"data/v2/validation"/p/"trajectories.npz").read_bytes()).hexdigest() for p in ("burgers","heat")}
    path=ROOT/"evidence/validation_closed_loop_comparator.json"
    path.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({"status":out["status"],"selected_comparator":out["selected_comparator"],"scores":score,"path":str(path),"test_opened":False},indent=2))
if __name__=="__main__":main()
