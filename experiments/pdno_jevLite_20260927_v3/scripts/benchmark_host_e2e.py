"""Three-session batch-1 host-ready E2E/tail latency on unique locked replay inputs."""
from __future__ import annotations
import hashlib,json,random,time
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from pdno.controllers.actions import project_box_slew  # noqa:E402
from pdno.evaluation.closed_loop import load_controller,policy_action  # noqa:E402

METHODS=("B0","B1","P","B4","B5","B2","B3")
SEEDS=(None,) if False else None
INPUT_KEYS=("sensor_value","sensor_mask","sensor_age","instrument_image","image_mask","goal_coefficients","material_context","previous_applied_action","applied_action_history","image_age","image_valid","goal_field")

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def _load_replay(pde,method,seed):
    by_role={}
    for role in ("locked_nominal","locked_coefficient_ood","locked_delay_dropout"):
        path=ROOT/"evidence/locked_test_raw"/role/pde/f"{method}_s{seed if seed is not None else 'na'}.npz"
        with np.load(path,allow_pickle=False) as z:
            by_role[role]={k:z[f"latency_{k}"] for k in INPUT_KEYS}
    counts={k:len(by_role["locked_nominal"][k]) for k in INPUT_KEYS}
    if any(v!=7680 for v in counts.values()) or len(by_role["locked_coefficient_ood"]["sensor_value"])!=2560 or len(by_role["locked_delay_dropout"]["sensor_value"])!=2560:
        raise RuntimeError(f"locked replay input counts differ: {pde}/{method}/{seed}: {counts}")
    # A 3:1:1 round-robin gives the declared 60/20/20 condition weights and avoids
    # running one condition only at the beginning/end of a session.
    mixed={k:[] for k in INPUT_KEYS}; nper=2130
    for i in range(nper):
        for role in ("locked_nominal","locked_coefficient_ood","locked_delay_dropout","locked_nominal","locked_nominal"):
            for k in INPUT_KEYS:mixed[k].append(by_role[role][k][i])
    return {k:np.stack(v) for k,v in mixed.items()}

def _one_request(pde,method,obs_np,model,device,qmax):
    proposed,_,_,_=policy_action(method,pde,obs_np,model,device,qmax)
    low,high,slew=(-1.,1.,.15) if pde=="burgers" else (0.,1.,.10)
    applied=project_box_slew(proposed,obs_np["previous_applied_action"],low,high,slew)
    if applied.shape!=(2,) or not np.isfinite(applied).all():raise RuntimeError("invalid typed host action")
    return applied

def main():
    marker=json.loads((ROOT/"state/LOCKED_TEST_ONCE.json").read_text(encoding="utf-8"))
    if marker.get("status")!="LOCKED_TEST_EVALUATION_COMPLETE" or marker.get("test_opened") is not True:
        raise SystemExit("latency replay is only allowed after completed locked evaluation")
    dest=ROOT/"evidence/latency_raw"
    if dest.exists():raise SystemExit("latency evidence exists; refusing overwrite")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qmax=1.2
    with np.load(ROOT/"data/v2/train/heat/trajectories.npz",allow_pickle=False) as z:qmax=max(qmax,1.25*float(z["goal"].max()))
    with np.load(ROOT/"data/v2/validation/heat/trajectories.npz",allow_pickle=False) as z:qmax=max(qmax,1.25*float(z["goal"].max()))
    ck=json.loads((ROOT/"evidence/confirmatory_checkpoint_manifest.json").read_text(encoding="utf-8"))
    checkpoints={(r["pde"],r["method"],int(r["seed"])):ROOT/r["path"] for r in ck["checkpoints"] if r.get("method") in {"P","B4","B5","B2","B3"}}
    variants=[(p,m,s) for p in ("burgers","heat") for m in METHODS for s in ((None,) if m in {"B0","B1"} else (11,23,37))]
    all_summaries=[]; progress=dest.parent/"latency_progress.json"; start=time.perf_counter()
    dest.mkdir(parents=True)
    for session in range(3):
        order=list(variants); random.Random(8310+session).shuffle(order)
        for pde,method,seed in order:
            replay=_load_replay(pde,method,seed)
            if len(replay["sensor_value"])!=10650:raise RuntimeError("unique latency replay sample count differs")
            checkpoint=None if seed is None else checkpoints[(pde,method,seed)]
            model=load_controller(method,pde,seed,checkpoint,device)
            measure_stream=device.type=="cuda" and method not in {"B0","B1"}
            warm_ids=np.arange(session*50,(session+1)*50)
            for ix in warm_ids:
                obs={k:replay[k][ix] for k in INPUT_KEYS};_one_request(pde,method,obs,model,device,1.2 if pde=="burgers" else qmax)
            if measure_stream:torch.cuda.synchronize()
            ids=np.arange(150+session*3500,150+(session+1)*3500)
            timings=np.empty(len(ids),np.float64)
            # CUDA events expose the device-stream span of model/ROM and scoring.
            # The outer wall clock remains the host-ready action latency and includes
            # Python/input preparation, transfers, projection, and response validation.
            stream_timings=np.full(len(ids),np.nan,np.float64)
            peak_before=torch.cuda.max_memory_allocated(device) if device.type=="cuda" else 0
            if measure_stream:
                event_start=torch.cuda.Event(enable_timing=True);event_end=torch.cuda.Event(enable_timing=True)
            for j,ix in enumerate(ids):
                obs={k:replay[k][ix] for k in INPUT_KEYS}
                if measure_stream:torch.cuda.synchronize()
                if measure_stream:
                    event_start.record()
                t0=time.perf_counter_ns();_one_request(pde,method,obs,model,device,1.2 if pde=="burgers" else qmax)
                if measure_stream:event_end.record()
                if measure_stream:torch.cuda.synchronize()
                timings[j]=(time.perf_counter_ns()-t0)/1e6
                if measure_stream:stream_timings[j]=event_start.elapsed_time(event_end)
            np.savez_compressed(dest/f"session{session+1}_{pde}_{method}_s{seed if seed is not None else 'na'}.npz",
                latency_ms=timings,cuda_stream_span_ms=stream_timings,
                replay_indices=ids.astype(np.int32),session=np.asarray(session+1),
                method=np.asarray(method),seed=np.asarray(-1 if seed is None else seed),pde=np.asarray(pde))
            row={"session":session+1,"pde":pde,"method":method,"seed":seed,"sample_count":len(timings),
                "unique_replay_inputs":len(np.unique(ids)),"p50_ms":float(np.quantile(timings,.50)),
                "p95_ms":float(np.quantile(timings,.95)),"p99_ms":float(np.quantile(timings,.99)),
                "p999_ms":float(np.quantile(timings,.999)),
                "execution_path":"CPU" if device.type=="cpu" or method in {"B0","B1"} else "CUDA",
                "cuda_stream_span_p50_ms":float(np.nanquantile(stream_timings,.50)) if np.isfinite(stream_timings).any() else None,
                "cuda_stream_span_p99_ms":float(np.nanquantile(stream_timings,.99)) if np.isfinite(stream_timings).any() else None,
                "host_wall_minus_cuda_stream_p50_ms":float(np.nanquantile(timings-stream_timings,.50)) if np.isfinite(stream_timings).any() else None,
                "deadline_misses":{"1ms":int(np.sum(timings>1)),"2ms":int(np.sum(timings>2)),"5ms":int(np.sum(timings>5)),"10ms":int(np.sum(timings>10))},
                "deadline_miss_rate_5ms":float(np.mean(timings>5)),"peak_cuda_allocated_bytes_session":max(peak_before,int(torch.cuda.max_memory_allocated(device))) if device.type=="cuda" else None}
            all_summaries.append(row)
            progress.write_text(json.dumps({"status":"RUNNING_NO_FINAL_LATENCY_SUMMARY","variants_complete":len(all_summaries),
                "variants_total":len(variants)*3,"elapsed_seconds":time.perf_counter()-start},indent=2)+"\n",encoding="utf-8")
            del model,replay
    raw=sorted(dest.glob("*.npz"))
    manifest={"status":"HOST_READY_E2E_THREE_SESSION_COMPLETE","locked_test_opened":True,
        "boundary":"host arrays ready -> typed proposed action -> common box/slew projection -> host action returned; includes input dictionary prep, host-to-device, model/ROM, candidate cost, device-to-host, projection and finite-action validation",
        "request_batch":1,"sessions":3,"timed_samples_per_session":3500,"warmup_requests_per_session":50,
        "unique_timed_inputs_per_variant":10500,"no_training_or_large_solver_ran_during_measurement":True,
        "test_marker_sha256":sha(ROOT/"state/LOCKED_TEST_ONCE.json"),"checkpoint_manifest_sha256":sha(ROOT/"evidence/confirmatory_checkpoint_manifest.json"),
        "aggregate_rows":all_summaries,"raw_files_sha256":{str(p.relative_to(ROOT)):sha(p) for p in raw}}
    out=ROOT/"evidence/latency_summary.json";out.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    progress.write_text(json.dumps({"status":"COMPLETE","variants_complete":len(all_summaries),"elapsed_seconds":time.perf_counter()-start},indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":manifest["status"],"variants":len(all_summaries),"samples_per_variant":10500,
        "elapsed_seconds":time.perf_counter()-start,"output":str(out),"sha256":sha(out)},indent=2))

if __name__=="__main__":main()
