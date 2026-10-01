"""Aggregate frozen train/validation model outcomes without accessing locked roles."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
p=json.loads((ROOT/"evidence"/"confirmatory_training_progress.json").read_text(encoding="utf-8"))
if p.get("status")!="TRAINING_COMPLETE": raise SystemExit("confirmatory fitting incomplete")
rows=[]
for pde in ("burgers","heat"):
    for method in ("P","B4","B5","B2","B3"):
        values=[]; updates=[]; elapsed=[]
        for r in p["results"]:
            if r.get("pde")!=pde or r.get("method")!=method: continue
            updates.append(r.get("actual_updates")); elapsed.append(r.get("elapsed_seconds"))
            if method in ("P","B4","B5"):
                best=min(r["history"],key=lambda x:x["selection_score"])
                values.append({"seed":r["seed"],"selected_update":r["best_step"],
                    "field_nrmse":best["validation_field_nrmse"],
                    "normalized_teacher_regret_mean":best["validation_normalized_teacher_regret_mean"],
                    "teacher_best_agreement":best["validation_teacher_best_agreement"],
                    "selection_score":r["selection_score"]})
            elif method=="B2":
                values.append({"seed":r["seed"],"selected_update":r["best_step"],
                    "validation_teacher_action_mse":r["validation_action_mse"]})
            else:
                values.append({"seed":r["seed"],"selected_update":r["best_step"],
                    "validation_selection_score":r["validation_score"]})
        if len(values)!=3: raise SystemExit(f"expected three seed results for {pde}/{method}, found {len(values)}")
        aggregate={"pde":pde,"method":method,"seed_results":values,"actual_updates":updates,
                   "elapsed_seconds":elapsed}
        metric=next(k for k in values[0] if k not in ("seed","selected_update"))
        vs=[v[metric] for v in values]
        aggregate["metric_name"]=metric
        aggregate["metric_mean"]=float(np.mean(vs))
        aggregate["metric_sd_across_three_seeds_descriptive_only"]=float(np.std(vs,ddof=1))
        rows.append(aggregate)
out={"status":"FROZEN_CONFIRMATORY_TRAIN_VALIDATION_SUMMARY","test_opened":False,
     "training_progress_sha256":__import__("hashlib").sha256((ROOT/"evidence"/"confirmatory_training_progress.json").read_bytes()).hexdigest(),
     "groups":rows}
path=ROOT/"evidence"/"confirmatory_summary.json"
path.write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")
print(json.dumps({"groups":len(rows),"summary":str(path),"test_opened":False},indent=2))
