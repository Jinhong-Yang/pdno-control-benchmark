"""Aggregate completed locked parent results with paired parent bootstrap."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
MARKER=ROOT/"state/LOCKED_TEST_ONCE_V2.json"
mark=json.loads(MARKER.read_text(encoding="utf-8"))
if mark.get("status")!="LOCKED_TEST_EVALUATION_COMPLETE" or mark.get("test_opened") is not True:
    raise SystemExit("locked evaluation is not complete; no partial analysis")
raw=ROOT/"evidence/locked_test_raw_v2"
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
expected={(role,pde,method,seed) for role in ("locked_nominal","locked_coefficient_ood","locked_delay_dropout")
          for pde in ("burgers","heat") for method in ("B0","B1","P","P-no-rank","B4","B5","B2","B3")
          for seed in ((None,) if method in {"B0","B1"} else (11,23,37))}
loaded={}
for p in sorted(raw.glob("*/**/*.npz")):
    role,pde=p.parent.parent.name,p.parent.name
    stem=p.stem
    if stem.endswith("_sna"): method,seed=stem[:-4],None
    else:
        method,s=stem.rsplit("_s",1); seed=int(s)
    with np.load(p,allow_pickle=False) as z:d={k:z[k] for k in z.files}
    projected=d["action_projected"].astype(np.float64)
    applied=d["action_applied"].astype(np.float64)
    proposed=d["action_proposed"].astype(np.float64)
    states=d["state_true"].astype(np.float64)
    if projected.shape!=(len(d["parent_id"]),200,2) or not np.isfinite(projected).all() or not np.isfinite(proposed).all():
        raise SystemExit(f"invalid action evidence shape/finiteness: {p}")
    if not np.array_equal(projected,applied):raise SystemExit(f"applied/projected action mismatch: {p}")
    low,high,slew=(-1.0,1.0,.15) if pde=="burgers" else (0.0,1.0,.10)
    initial=np.zeros((len(projected),1,2)) if pde=="burgers" else np.full((len(projected),1,2),.4)
    previous=np.concatenate((initial,applied[:,:-1]),axis=1)
    if np.any(applied<low-1e-7) or np.any(applied>high+1e-7) or np.any(np.abs(applied-previous)>slew+1e-7):
        raise SystemExit(f"box/slew action constraint violation: {p}")
    if states.shape!=(len(projected),201,256) or not np.isfinite(states).all():raise SystemExit(f"invalid state evidence: {p}")
    ticks=np.arange(200)[None,:]
    sm=d["sensor_mask"].astype(bool); sc=d["sensor_capture_time"]; sr=d["sensor_receive_time"]
    if np.any(sm & ((sc<0)|(sr<0)|(sc>ticks[:,:,None])|(sr>ticks[:,:,None])|(sr<sc))):
        raise SystemExit(f"noncausal sensor timestamp evidence: {p}")
    im=d["image_valid"].astype(bool); ic=d["image_capture_time"]; ir=d["image_receive_time"]
    if np.any(im & ((ic<0)|(ir<0)|(ic>ticks)|(ir>ticks)|(ir<ic))):
        raise SystemExit(f"noncausal image timestamp evidence: {p}")
    key=(role,pde,method,seed)
    if key in loaded: raise SystemExit(f"duplicate raw shard {key}")
    if len(d["parent_id"]) != {"locked_nominal":384,"locked_coefficient_ood":128,"locked_delay_dropout":128}[role]:
        raise SystemExit(f"raw shard parent count mismatch: {p}")
    rel=str(p.relative_to(ROOT))
    if mark.get("raw_result_sha256",{}).get(rel)!=sha(p): raise SystemExit(f"raw shard hash mismatch: {p}")
    loaded[key]=d
if loaded.keys()!=expected:
    raise SystemExit(f"raw shard inventory mismatch; missing={expected-loaded.keys()}, extra={loaded.keys()-expected}")
records=sorted(raw.glob("*/**/*.record.json"))
if len(records)!=120: raise SystemExit(f"expected 120 raw shard records; found {len(records)}")
for record in records:
    rel=str(record.relative_to(ROOT))
    if mark.get("raw_artifact_sha256",{}).get(rel)!=sha(record): raise SystemExit(f"raw record hash mismatch: {record}")

def wilson_upper(success,n,z=1.959963984540054):
    if n==0:return None
    p=success/n; den=1+z*z/n
    center=(p+z*z/(2*n))/den
    radius=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return float(min(1.0,center+radius))

def paired_boot(a,b,rng,reps=10000):
    n=len(a); ix=rng.integers(0,n,size=(reps,n))
    v=(a-b)[ix].mean(axis=1)
    return {"mean":float(np.mean(a-b)),"ci95_percentile":[float(np.quantile(v,.025)),float(np.quantile(v,.975))],"replicates":reps}

comparator=json.loads((ROOT/"evidence/validation_closed_loop_comparator_v2.json").read_text(encoding="utf-8"))["selected_comparator"]
rng=np.random.default_rng(20260926)
summary={"status":"LOCKED_TEST_AGGREGATED","test_opened":True,"comparison_unit":"parent_episode",
 "training_seed_handling":"average within parent for overall estimate; per-seed rows reported, seeds are not counted as independent environments",
 "paired_parent_bootstrap_replicates":10000,"bootstrap_rng_seed":20260926,"h3_validation_selected_comparator":comparator,"groups":[]}
for role in ("locked_nominal","locked_coefficient_ood","locked_delay_dropout"):
 for pde in ("burgers","heat"):
    comparator_seeds=(None,) if comparator in {"B0","B1"} else (11,23,37)
    baseline_rows=[loaded[(role,pde,comparator,seed)] for seed in comparator_seeds]
    baseline=baseline_rows[0]
    base_ids=baseline["parent_id"].astype(str)
    if any(not np.array_equal(row["parent_id"].astype(str),base_ids) for row in baseline_rows):
        raise SystemExit(f"H3 comparator parent order mismatch: {role}/{pde}")
    bcost_by_parent=np.mean([row["episode_control_cost"] for row in baseline_rows],axis=0).astype(np.float64)
    bviol_by_parent=np.mean([row["truth_constraint_violation"].astype(float) for row in baseline_rows],axis=0)
    group={"role":role,"pde":pde,"parent_count":len(base_ids),"controller_metrics":[],"paired_vs_h3_comparator":{}}
    for method in ("B0","B1","P","P-no-rank","B4","B5","B2","B3"):
        seeds=(None,) if method in {"B0","B1"} else (11,23,37)
        by_seed=[]
        for seed in seeds:
            d=loaded[(role,pde,method,seed)]
            ids=d["parent_id"].astype(str)
            if not np.array_equal(ids,base_ids): raise SystemExit(f"parent order mismatch for {role}/{pde}/{method}/{seed}")
            viol=d["truth_constraint_violation"].astype(bool)
            by_seed.append({"seed":seed,"mean_control_cost":float(d["episode_control_cost"].mean()),
                "mean_tracking_rmse":float(d["tracking_rmse"].mean()),"violation_count":int(viol.sum()),
                "violation_rate":float(viol.mean()),"violation_wilson_upper95":wilson_upper(int(viol.sum()),len(viol)),
                "mean_fallback_rate":float(d["fallback_rate"].mean()),"fallback_episode_count":int(np.sum(d["fallback_count"]>0)),
                "mean_forecast_nrmse_at_tick100":float(np.nanmean(d["field_nrmse_at_snapshot"])) if np.isfinite(d["field_nrmse_at_snapshot"]).any() else None,
                "mean_candidate_cost_gap_at_tick100":float(np.nanmean(d["candidate_regret_at_snapshot"])) if np.isfinite(d["candidate_regret_at_snapshot"]).any() else None,
                "causal_timestamp_violation_count":int(d["causal_timestamp_violations"].sum()),
                "fallback_errors":int(sum(bool(x and x!="[]") for x in d["fallback_errors"].astype(str)))})
        cost_by_parent=np.mean([loaded[(role,pde,method,s)]["episode_control_cost"] for s in seeds],axis=0)
        violation_by_parent=np.mean([loaded[(role,pde,method,s)]["truth_constraint_violation"].astype(float) for s in seeds],axis=0)
        fallback_by_parent=np.mean([loaded[(role,pde,method,s)]["fallback_rate"] for s in seeds],axis=0)
        field_rows=[loaded[(role,pde,method,s)]["field_nrmse_at_snapshot"] for s in seeds]
        field_by_parent=np.nanmean(field_rows,axis=0) if method in {"B1","P","P-no-rank","B4","B5"} else None
        group["controller_metrics"].append({"method":method,"seed_results":by_seed,
            "mean_control_cost_over_parent_and_seed":float(np.mean(cost_by_parent)),
            "mean_tracking_rmse_over_parent_and_seed":float(np.mean([loaded[(role,pde,method,s)]["tracking_rmse"].mean() for s in seeds])),
            "mean_truth_violation_rate_over_parent_and_seed":float(np.mean(violation_by_parent)),
            "mean_fallback_rate_over_parent_and_seed":float(np.mean(fallback_by_parent)),
            "mean_field_nrmse_at_tick100":float(np.nanmean(field_by_parent)) if field_by_parent is not None and np.isfinite(field_by_parent).any() else None})
        cost_ratio=(cost_by_parent-bcost_by_parent)/max(float(bcost_by_parent.mean()),1e-12)
        diff=paired_boot(cost_ratio,np.zeros_like(cost_ratio),rng)
        vdiff=paired_boot(violation_by_parent-bviol_by_parent,np.zeros(len(base_ids)),rng)
        compv=bviol_by_parent>0
        methodv=violation_by_parent>0
        group["paired_vs_h3_comparator"][method]={"normalized_cost_difference_mean":diff["mean"],
            "normalized_cost_difference_ci95":diff["ci95_percentile"],"violation_rate_difference_mean":vdiff["mean"],
            "violation_rate_difference_ci95":vdiff["ci95_percentile"],"paired_both_violate":int(np.sum(methodv&compv)),
            "method_only_violation":int(np.sum(methodv&~compv)),"comparator_only_violation":int(np.sum(~methodv&compv)),
            "paired_neither_violate":int(np.sum(~methodv&~compv)),"parent_count":len(base_ids)}
    group["B1_vs_B0_cost_gap"]={"paired_parent_bootstrap":paired_boot(
        np.mean([loaded[(role,pde,"B1",None)]["episode_control_cost"]],axis=0),
        np.mean([loaded[(role,pde,"B0",None)]["episode_control_cost"]],axis=0),rng)}
    group["P_vs_B5_cost_gap"]={"paired_parent_bootstrap":paired_boot(
        np.mean([loaded[(role,pde,"P",s)]["episode_control_cost"] for s in (11,23,37)],axis=0),
        np.mean([loaded[(role,pde,"B5",s)]["episode_control_cost"] for s in (11,23,37)],axis=0),rng)}
    summary["groups"].append(group)
summary["raw_manifest_sha256"]=hashlib.sha256(MARKER.read_bytes()).hexdigest()
out=ROOT/"evidence/locked_test_analysis_v2.json"
out.write_text(json.dumps(summary,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
print(json.dumps({"status":summary["status"],"groups":len(summary["groups"]),"output":str(out),"sha256":hashlib.sha256(out.read_bytes()).hexdigest()},indent=2))
