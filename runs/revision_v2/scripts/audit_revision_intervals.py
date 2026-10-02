"""Independent count-weighted reconstruction of paired-parent bootstrap intervals.

Default requires the complete revision raw audit. --stage E1 or E5 permits a
scoped numerical check of a completed stage while the experiment queue runs.
The E5 scoped check does not replace full state/action cost reconstruction.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np

R=Path(__file__).resolve().parents[1]
OLD=R.parents[1]/"experiments/pdno_jevLite_20260927_v3"
ROLES=("locked_nominal","locked_coefficient_ood","locked_delay_dropout")
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def bootstrap(cost,base):
    # Convert index draws to parent multiplicities. This avoids copying the
    # production expression that indexes and averages per-draw differences.
    n=len(base); rng=np.random.default_rng(2026100270)
    fixed=[];joint=[]
    base=base.astype(np.float64); delta=cost.astype(np.float64)-base
    denominator=max(float(np.sum(base)/n),1e-12)
    for start in range(0,10000,125):
        size=min(125,10000-start)
        indices=rng.integers(0,n,size=(size,n))
        weights=np.zeros((size,n),dtype=np.int32)
        np.add.at(weights,(np.repeat(np.arange(size),n),indices.ravel()),1)
        assert np.all(weights.sum(axis=1)==n)
        numerator=(weights@delta)/n
        fixed.extend(numerator/denominator)
        joint.extend(numerator/np.maximum((weights@base)/n,1e-12))
    return dict(zip(("relative_excess","fixed_ci_low","fixed_ci_high","joint_ci_low","joint_ci_high"),
                    (float(np.sum(delta)/n/denominator),*np.quantile(fixed,[.025,.975]),
                     *np.quantile(joint,[.025,.975]))))

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=["E1","E5"])
    args=parser.parse_args()
    stages=[args.stage] if args.stage else ["E1","E3","E4","E5"]
    hashes={};records=[]
    if args.stage=="E1":
        for name in ("E1_heat_COMPLETE.json","E1_burgers_cuda_COMPLETE.json"):
            path=R/"results"/name
            assert json.loads(path.read_text())["status"]=="COMPLETE"
            hashes[str(path.relative_to(R))]=sha(path)
    elif args.stage=="E5":
        from revision_audit_contract import validate_outcome_inventory
        path=R/"results/E5_EVALUATION.json"
        receipt=json.loads(path.read_text())
        assert receipt["status"]=="COMPLETE" and receipt["conditions"]==20
        hashes[str(path.relative_to(R))]=sha(path)
        inventory=R/"results/E5_COMPLETION_INVENTORY_AUDIT.json"
        audit=json.loads(inventory.read_text())
        assert audit["status"]=="PASS_E5_COMPLETION_INVENTORY_HASHES_AND_PARENT_ALIGNMENT"
        assert audit["evaluation_receipt_sha256"]==sha(path)
        actual=[]
        for item in audit["rows"]:
            raw=R/item["path"]
            assert sha(raw)==item["sha256"],("changed E5 raw",str(raw))
            actual.append((raw.parent.parent.name,raw.parent.name,raw.stem))
        validate_outcome_inventory("E5",actual)
        hashes[str(inventory.relative_to(R))]=sha(inventory)
    else:
        path=R/"results/REVISION_RAW_AUDIT.json"
        assert json.loads(path.read_text())["status"]=="PASS"
        hashes[str(path.relative_to(R))]=sha(path)
    summary=R/"results/REVISION_COST_SUMMARY.csv"
    with summary.open(newline="",encoding="utf-8-sig") as f: rows=[r for r in csv.DictReader(f) if r["stage"] in stages]
    keys=[(r["stage"],r["role"],r["pde"],r["method"]) for r in rows]
    assert len(set(keys))==len(keys),"duplicate summary row"
    hashes[str(summary.relative_to(R))]=sha(summary)
    expected={}
    for stage in stages:
        for role in ROLES:
            for pde in ("burgers","heat"):
                paths=sorted((R/f"results/{stage}_raw/{role}/{pde}").glob("*.npz"))
                for path in paths:
                    label=path.stem.rsplit("_s",1)[0] if stage in ("E3","E5") else path.stem
                    expected.setdefault((stage,role,pde,label),[]).append(path)
                if stage=="E4":
                    expected[stage,role,pde,f"{pde}_phys0.01_rank0.1_s11"]=[OLD/f"evidence/locked_test_raw_v3/{role}/{pde}/P_s11.npz"]
    assert set(keys)==set(expected),"missing or extra summary conditions"
    if args.stage:assert len(rows)=={"E1":12,"E5":24}[args.stage]
    def read(path):
        hashes[str(path.relative_to(R.parents[1]))]=sha(path)
        with np.load(path,allow_pickle=False) as z:return z["parent_id"].copy(),z["episode_control_cost"].astype(np.float64)
    for row,key in zip(rows,keys):
        stage,role,pde,label=key
        baseline=(R/f"results/E5_raw/{role}/{pde}/B0_sna.npz") if stage=="E5" else OLD/f"evidence/locked_test_raw_v3/{role}/{pde}/B0_sna.npz"
        parent,base=read(baseline); costs=[]
        assert int(row["parents"])==len(parent)
        for path in expected[key]:
            ids,cost=read(path);assert np.array_equal(ids,parent)
            costs.append(cost)
        # Each training seed remains paired within the same physical parent.
        averaged=np.sum(costs,axis=0)/len(costs)
        result=bootstrap(averaged,base)
        errors={name:abs(value-float(row[name])) for name,value in result.items()}
        assert max(errors.values())<1e-9,(key,errors)
        records.append({"key":list(key),"parents":len(parent),"model_arrays":len(costs),
                        "recomputed":result,"absolute_errors":errors})
    output={"status":f"PASS_SCOPED_{args.stage}_INTERVAL_AUDIT" if args.stage else "PASS_ALL_REVISION_INTERVAL_AUDIT",
            "scope":stages,"summary_rows":len(records),"verified_values":5*len(records),
            "draws":10000,"seed":2026100270,"paired_unit":"parent; seed averaging within parent",
            "method":"Count-weighted parent multiplicities; fixed and resampled denominator on identical draws",
            "records":records,"source_sha256":hashes,"auditor_sha256":sha(Path(__file__)),
            "limits":"Numerical reconstruction under the recorded pointwise paired-parent bootstrap; not simultaneous coverage, more independent seeds, or an iid/OOD guarantee. Scoped checks do not replace full state/action cost reconstruction or the final complete-campaign audit."}
    out=R/"results"/(f"{args.stage}_INTERVAL_AUDIT.json" if args.stage else "REVISION_INTERVAL_AUDIT.json")
    out.write_text(json.dumps(output,indent=2)+"\n")
    print(json.dumps({k:output[k] for k in ("status","summary_rows","verified_values")}))

if __name__=="__main__":main()
