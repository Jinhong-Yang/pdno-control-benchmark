"""Independent CPU arithmetic check of the completed E1 original-parent oracles.

Does not rerun controllers, alter results, or certify the whole revision.
Recorded fallback/causality counters are checked, not reconstructed timestamps.
"""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import time
import numpy as np

R = Path(__file__).resolve().parents[1]
OLD = R.parents[1]/"experiments/pdno_jevLite_20260927_v3"
start = time.perf_counter()
checks, records, hashes = [], [], {}
ROLES = {"locked_nominal": 384, "locked_coefficient_ood": 128, "locked_delay_dropout": 128}
METHODS = {"O-cand-state", "O-cand-obs"}

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4*1024*1024), b""):
            h.update(block)
    return h.hexdigest()

def check(name, ok, detail=None):
    checks.append(dict(name=name, passed=bool(ok), detail=detail))
    if not ok:
        (R/"results/E1_INDEPENDENT_RAW_AUDIT_FAILED.json").write_text(json.dumps(dict(checks=checks), indent=2))
        raise AssertionError((name, detail))

for filename in ["E1_heat_COMPLETE.json", "E1_burgers_cuda_COMPLETE.json"]:
    path = R/"results"/filename
    receipt = json.loads(path.read_text())
    check(filename, receipt["status"] == "COMPLETE" and receipt["executions"] == 1280)
    hashes[str(path.relative_to(R))] = sha(path)
hashes["scripts/audit_completed_oracles.py"] = sha(Path(__file__))
hashes["scripts/run_e1_burgers_cuda.py"] = sha(R/"scripts/run_e1_burgers_cuda.py")
hashes["results/REVISION_COST_SUMMARY.csv"] = sha(R/"results/REVISION_COST_SUMMARY.csv")
with (R/"results/REVISION_COST_SUMMARY.csv").open(newline="") as f:
    summary = [r for r in csv.DictReader(f) if r["stage"] == "E1"]
paths = sorted((R/"results/E1_raw").glob("*/*/*.npz"))
expected = {(role, pde, method) for role in ROLES for pde in ["burgers", "heat"] for method in METHODS}
observed = {(p.parent.parent.name, p.parent.name, p.stem) for p in paths}
check("full E1 factorial", observed == expected and len(paths) == len(expected) == 12)
unique_parents = set()
for path in paths:
    role, pde, method = path.parent.parent.name, path.parent.name, path.stem
    key = str(path.relative_to(R))
    meta = json.loads(path.with_suffix(".record.json").read_text())
    hashes[str(path.with_suffix(".record.json").relative_to(R))] = sha(path.with_suffix(".record.json"))
    digest = sha(path); hashes[key] = digest
    check(key+"/hash", digest == meta["sha256"])
    with np.load(path, allow_pickle=False) as z:
        ids, states, actions = z["parent_id"], z["state_true"], z["action_applied"]
        # Burgers has a fixed zero target and its CUDA oracle archive omits it.
        # Heat has nonzero/time-varying goals and must supply the recorded field.
        check(key+"/goal_contract", "goal" in z or pde == "burgers")
        goal = z["goal"] if "goal" in z else np.zeros((1,1,256), dtype=np.float64)
        costs, rmse = z["episode_control_cost"], z["tracking_rmse"]
        events, indices = z["truth_constraint_violation"], z["selected_candidate_index"]
        fallback, causal = z["fallback_count"], z["causal_timestamp_violations"]
    n = ROLES[role]
    check(key+"/record_identity", meta["status"] == "COMPLETE" and meta["parents"] == n
          and meta["role"] == role and meta["method"] == method and meta.get("pde",pde) == pde)
    check(key+"/shapes", states.shape == (n,201,256) and actions.shape == (n,200,2)
          and indices.shape == (n,200) and len(ids) == len(set(ids)) == n)
    check(key+"/finite", all(np.isfinite(x).all() for x in [states, actions, goal, costs, rmse]))
    check(key+"/recorded_runtime_counters", not np.any(fallback) and not np.any(causal))
    check(key+"/indices", np.issubdtype(indices.dtype, np.integer) and np.all((indices >= 0) & (indices < 10)))
    baseline = OLD/f"evidence/locked_test_raw_v3/{role}/{pde}/B0_sna.npz"
    baseline_key = str(baseline.relative_to(R.parents[1]))
    if baseline_key not in hashes:
        hashes[baseline_key] = sha(baseline)
    with np.load(baseline, allow_pickle=False) as z:
        baseline_ids = z["parent_id"]
        baseline_cost = z["episode_control_cost"].astype(np.float64)
    check(key+"/parent_alignment", np.array_equal(ids, baseline_ids))
    unique_parents.update(str(x) for x in ids)
    heat = pde == "heat"
    low, high, slew, qmax, initial = (0,1,.1,2.1322593092918396,.4) if heat else (-1,1,.15,1.2,0)
    a = actions.astype(np.float64)
    previous = np.concatenate([np.full((n,1,2), initial), a[:,:-1]], axis=1)
    max_slew = float(np.max(np.abs(a-previous)))
    check(key+"/action_constraints", np.all(a >= low-1e-7) and np.all(a <= high+1e-7) and max_slew <= slew+1e-7)
    u = states[:,1:].astype(np.float64)
    error = u-goal.astype(np.float64)
    violation = np.maximum(-u,0)+np.maximum(u-qmax,0) if heat else np.maximum(np.abs(u)-qmax,0)
    tracking = np.mean(error**2, axis=2).sum(axis=1)
    effort = .01*np.sum(a**2, axis=(1,2))
    action_change = .05*np.sum((a-previous)**2, axis=(1,2))
    penalty = 10*np.mean(violation**2, axis=2).sum(axis=1)
    reconstructed = tracking+effort+action_change+penalty
    max_error = float(np.max(np.abs(reconstructed-costs)))
    check(key+"/cost", np.allclose(reconstructed,costs,atol=1e-5,rtol=1e-6), max_error)
    check(key+"/rmse", np.allclose(np.sqrt(np.mean(error**2,axis=(1,2))),rmse,atol=1e-7,rtol=1e-6))
    check(key+"/strict_event_mask", np.array_equal(np.max(violation,axis=(1,2)) > 0, events))
    rows = [r for r in summary if (r["role"],r["pde"],r["method"]) == (role,pde,method)]
    check(key+"/summary_key", len(rows) == 1)
    row = rows[0]
    values = {"mean_cost": float(np.mean(costs.astype(np.float64))),
              "B0_mean_cost": float(baseline_cost.mean()),
              "tracking_rmse_mean": float(np.mean(rmse.astype(np.float64))),
              "any_violation_rate": float(np.mean(events))}
    values["relative_excess"] = (values["mean_cost"]-values["B0_mean_cost"])/values["B0_mean_cost"]
    for field,value in values.items():
        check(key+"/summary/"+field, np.isclose(value,float(row[field]),rtol=1e-12,atol=1e-12))
    records.append(dict(role=role,pde=pde,method=method,parents=n,cost_max_abs_reconstruction_error=max_error,
                        max_slew=max_slew,sha256=digest))
    print("Audited",role,pde,method,flush=True)
check("parent and execution totals", len(unique_parents) == 1280 and sum(r["parents"] for r in records) == 2560)
result = dict(status="PASS_E1_INDEPENDENT_RAW_ARITHMETIC",time_utc=datetime.now(timezone.utc).isoformat(),
              checks=checks,check_count=len(checks),arrays=len(paths),unique_parents=len(unique_parents),
              executions=sum(r["parents"] for r in records),records=records,input_sha256=hashes,
              elapsed_seconds=time.perf_counter()-start,
              scope="All12 E1 arrays: hashes, factorial/parent alignment, finite state/action/goal/cost, stored runtime counters, index range, action constraints, float64 cost/RMSE reconstruction, strict event masks and source-summary arithmetic.",
              limits="No counterfactual candidate forecasts or complete timestamp arrays are stored here; this does not independently reconstruct argmin optimality or observation causality. The Burgers CUDA writer fills zero fallback/causality counters and separately asserts received sensor timestamps at runtime; those stored zeros are not independent timestamp evidence. Burgers zero goal is specified by the hashed runner; heat requires its recorded goal. Action bounds use1e-7 floating-point tolerance. Confidence intervals, other revision experiments, full original preservation and release still require separate audits.")
(R/"results/E1_INDEPENDENT_RAW_AUDIT.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({k:result[k] for k in ["status","check_count","arrays","unique_parents","executions","elapsed_seconds"]}))
