"""Sequential, resumable confirmatory training using only frozen train/validation shards."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from pdno.training.confirmatory import (train_observer,train_operator_staged,
        train_direct_staged,train_b3_staged)


def _run(path,fn):
    summary=Path(path).with_suffix(".summary.json")
    if summary.exists(): return json.loads(summary.read_text(encoding="utf-8"))
    return fn()


def main():
    root=ROOT/"runs"/"confirmatory_frozen_v1"
    data=ROOT/"data"/"queries_v2"
    results=[]
    started=time.perf_counter()
    for pde in ("burgers","heat"):
        tr=data/"train"/pde/"teacher_queries.npz"
        va=data/"validation"/pde/"teacher_queries.npz"
        observer=root/"observer"/f"{pde}_seed7.pt"
        row=_run(observer,lambda:train_observer(tr,va,pde,observer,seed=7,max_updates=1000,
                    eval_interval=100,batch_size=64,device_name="cuda"))
        results.append(row)
        for seed in (11,23,37):
            p=root/pde/f"P_s{seed}.pt"
            row=_run(p,lambda:train_operator_staged(tr,va,observer,pde,"P",seed,p,
                    lambda_phys=0.01,lambda_balance=0.01,field_updates=2000,
                    physics_updates=3000,eval_interval=100,batch_size=64,device_name="cuda"))
            results.append(row); print(json.dumps({"pde":pde,"seed":seed,"method":"P","updates":row.get("actual_updates"),"selection_score":row.get("selection_score")}),flush=True)
            b4=root/pde/f"B4_s{seed}.pt"
            row4=_run(b4,lambda:train_operator_staged(tr,va,observer,pde,"B4",seed,b4,
                    lambda_phys=0.01,lambda_balance=0.01,field_updates=2000,
                    physics_updates=3000,eval_interval=100,batch_size=64,device_name="cuda"))
            results.append(row4); print(json.dumps({"pde":pde,"seed":seed,"method":"B4","updates":row4.get("actual_updates"),"selection_score":row4.get("selection_score")}),flush=True)
            b5=root/pde/f"B5_s{seed}.pt"
            row5=_run(b5,lambda:train_operator_staged(tr,va,observer,pde,"B5",seed,b5,
                    lambda_phys=0.0,lambda_balance=0.0,field_updates=2000,
                    physics_updates=3000,eval_interval=100,batch_size=64,device_name="cuda"))
            results.append(row5); print(json.dumps({"pde":pde,"seed":seed,"method":"B5","updates":row5.get("actual_updates"),"selection_score":row5.get("selection_score")}),flush=True)
            b2=root/pde/f"B2_s{seed}.pt"
            row2=_run(b2,lambda:train_direct_staged(tr,va,observer,pde,seed,b2,max_updates=3000,
                    eval_interval=100,batch_size=64,device_name="cuda"))
            results.append(row2); print(json.dumps({"pde":pde,"seed":seed,"method":"B2","updates":row2.get("actual_updates"),"val_mse":row2.get("validation_action_mse")}),flush=True)
            b3=root/pde/f"B3_s{seed}.pt"
            row3=_run(b3,lambda:train_b3_staged(tr,va,b2,b4,pde,seed,b3,max_updates=1000,
                    eval_interval=100,device_name="cuda"))
            results.append(row3); print(json.dumps({"pde":pde,"seed":seed,"method":"B3","updates":row3.get("actual_updates"),"validation_score":row3.get("validation_score")}),flush=True)
            manifest=ROOT/"evidence"/"confirmatory_training_progress.json"
            manifest.write_text(json.dumps({"status":"RUNNING","elapsed_seconds":time.perf_counter()-started,
                                            "results":results},indent=2)+"\n",encoding="utf-8")
    manifest=ROOT/"evidence"/"confirmatory_training_progress.json"
    manifest.write_text(json.dumps({"status":"TRAINING_COMPLETE","elapsed_seconds":time.perf_counter()-started,
                                    "results":results},indent=2)+"\n",encoding="utf-8")


if __name__=="__main__": main()
