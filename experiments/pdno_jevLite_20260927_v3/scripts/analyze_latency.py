"""Summarize tail latency and seed-paired H1 p99 ratios from frozen raw sessions."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
summary_path=ROOT/"evidence/latency_summary.json"
summary=json.loads(summary_path.read_text(encoding="utf-8"))
if summary.get("status")!="HOST_READY_E2E_THREE_SESSION_COMPLETE":raise SystemExit("three-session latency runs are incomplete")
raw_root=ROOT/"evidence/latency_raw"
raw={}
for p in raw_root.glob("*.npz"):
    with np.load(p,allow_pickle=False) as z:
        raw[(str(z["pde"].item()),str(z["method"].item()),int(z["seed"].item()),int(z["session"].item()))]=z["latency_ms"].astype(np.float64)
summary_groups=[]
for pde in ("burgers","heat"):
    for method in ("B0","B1","P","B4","B5","B2","B3"):
        seeds=(-1,) if method in {"B0","B1"} else (11,23,37)
        for seed in seeds:
            sessions=[raw[(pde,method,seed,s)] for s in (1,2,3)]
            values=np.concatenate(sessions)
            summary_groups.append({"pde":pde,"method":method,"seed":None if seed==-1 else seed,"sample_count":len(values),
                "p50_ms":float(np.quantile(values,.5)),"p95_ms":float(np.quantile(values,.95)),
                "p99_ms":float(np.quantile(values,.99)),"p999_ms":float(np.quantile(values,.999)),
                "deadline_misses":{"1ms":int(np.sum(values>1)),"2ms":int(np.sum(values>2)),"5ms":int(np.sum(values>5)),"10ms":int(np.sum(values>10))},
                "deadline_miss_rate_5ms":float(np.mean(values>5))})

def _block_sample(sessions,rng,block=64,n=3500):
    out=[]
    chosen=rng.integers(0,3,size=3)
    for s in chosen:
        arr=sessions[s]; pieces=[]
        while sum(len(x) for x in pieces)<n:
            start=int(rng.integers(0,max(1,len(arr)-block+1)))
            pieces.append(arr[start:start+block])
        out.append(np.concatenate(pieces)[:n])
    return np.concatenate(out)

rng=np.random.default_rng(20260926); h1=[]
for pde in ("burgers","heat"):
    point=[]; boot=[]
    for seed in (11,23,37):
        p=[raw[(pde,"P",seed,s)] for s in (1,2,3)]
        b=[raw[(pde,"B4",seed,s)] for s in (1,2,3)]
        point.append(float(np.quantile(np.concatenate(p),.99)/max(np.quantile(np.concatenate(b),.99),1e-12)))
    for _ in range(5000):
        selected=rng.choice((11,23,37),size=3,replace=True); ratios=[]
        for seed in selected:
            ps=[raw[(pde,"P",int(seed),s)] for s in (1,2,3)]
            bs=[raw[(pde,"B4",int(seed),s)] for s in (1,2,3)]
            vp=_block_sample(ps,rng); vb=_block_sample(bs,rng)
            ratios.append(float(np.quantile(vp,.99)/max(np.quantile(vb,.99),1e-12)))
        boot.append(float(np.mean(ratios)))
    h1.append({"pde":pde,"comparison":"P/B4 p99 host-ready latency ratio; <1 favors P",
        "per_seed_p99_ratio":dict(zip((11,23,37),point)),"three_seed_mean_ratio_descriptive":float(np.mean(point)),
        "hierarchical_session_block_bootstrap_ci95":[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],
        "bootstrap_replicates":5000,"request_block_length":64,"seed_count":3})
out={"status":"LOCKED_LATENCY_ANALYSIS_COMPLETE","locked_test_opened":True,
    "target_p99_ms":2,"hard_deadline_ms":5,"three_independent_sessions":True,
    "latency_definition":summary["boundary"],"seed_aggregation_caveat":"Only three model seeds; describe per seed and do not treat as environment repetitions.",
    "controller_groups":summary_groups,"H1_p99_ratio_analysis":h1,
    "latency_summary_sha256":hashlib.sha256(summary_path.read_bytes()).hexdigest()}
path=ROOT/"evidence/latency_analysis.json";path.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n",encoding="utf-8")
print(json.dumps({"status":out["status"],"groups":len(summary_groups),"H1":h1,"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()},indent=2))
