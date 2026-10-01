"""One-shot 200-tick parent-paired locked closed-loop evaluation.

The script writes a durable opened marker before its first score-capable target read.
A failure after that marker is an opened, inconclusive test and must not be rerun.
"""
from __future__ import annotations
import hashlib, json, time
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from pdno.evaluation.closed_loop import load_controller,run_episode  # noqa:E402

def sha(path:Path)->str: return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    marker=ROOT/"state/LOCKED_TEST_ONCE.json"
    if marker.exists(): raise SystemExit("locked test already opened; one-shot runner refuses a second run")
    cal=ROOT/"evidence/calibration_lock.json"
    if not cal.is_file(): raise SystemExit("calibration lock is missing")
    evaluation_freeze=ROOT/"evidence/locked_evaluation_freeze.json"
    if not evaluation_freeze.is_file(): raise SystemExit("pre-open locked evaluation freeze is missing")
    frozen=json.loads(evaluation_freeze.read_text(encoding="utf-8"))
    if frozen.get("status")!="LOCKED_EVALUATION_INPUTS_FROZEN_TEST_SEALED" or frozen.get("test_opened") is not False:
        raise SystemExit("locked evaluation inputs are not frozen in the sealed state")
    locked_config=ROOT/"config/evaluation_addendum_v2.yaml"
    for p in (locked_config,ROOT/"scripts/run_locked_test_once.py",ROOT/"src/pdno/evaluation/closed_loop.py"):
        if not p.is_file(): raise SystemExit(f"locked evaluation freeze input is missing: {p}")
    cp_manifest=json.loads((ROOT/"evidence/confirmatory_checkpoint_manifest.json").read_text(encoding="utf-8"))
    checkpoint_map={(r["pde"],r["method"],int(r["seed"])):ROOT/r["path"] for r in cp_manifest["checkpoints"] if r.get("method") in {"P","B4","B5","B2","B3"}}
    locked_root=ROOT/"data/locked_v2"; raw_root=ROOT/"evidence/locked_test_raw"
    input_files=sorted(locked_root.glob("*/**/trajectories.npz"))
    if len(input_files)!=6: raise SystemExit(f"expected 6 locked_v2 parent shards; found {len(input_files)}")
    work=raw_root
    if work.exists(): raise SystemExit("locked raw evidence directory already exists; refusing to overwrite")
    inputs={str(p.relative_to(ROOT)):sha(p) for p in input_files}
    freeze={"status":"TEST_OPENED_ONCE_IN_PROGRESS","test_opened":True,"opened_at_local":time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "episode_ticks":200,"query_tick":100,"calibration_lock_sha256":sha(cal),
        "locked_evaluation_freeze_sha256":sha(evaluation_freeze),
        "evaluation_addendum_v2_sha256":sha(locked_config),"runner_sha256":sha(ROOT/"scripts/run_locked_test_once.py"),
        "closed_loop_source_sha256":sha(ROOT/"src/pdno/evaluation/closed_loop.py"),
        "checkpoint_manifest_sha256":sha(ROOT/"evidence/confirmatory_checkpoint_manifest.json"),"locked_parent_shards_sha256":inputs}
    marker.write_text(json.dumps(freeze,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    work.mkdir(parents=True)
    progress=work/"progress.json"; start=time.perf_counter(); done=0
    try:
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        qmax=1.2
        with np.load(ROOT/"data/v2/train/heat/trajectories.npz",allow_pickle=False) as z: qmax=max(qmax,1.25*float(z["goal"].max()))
        with np.load(ROOT/"data/v2/validation/heat/trajectories.npz",allow_pickle=False) as z: qmax=max(qmax,1.25*float(z["goal"].max()))
        for pde in ("burgers","heat"):
            for role in ("locked_nominal","locked_coefficient_ood","locked_delay_dropout"):
                path=locked_root/role/pde/"trajectories.npz"
                with np.load(path,allow_pickle=False) as z: d={k:z[k] for k in z.files}
                n=len(d["parent_id"])
                if d["state_true"].shape[1]!=201 or n!={"locked_nominal":384,"locked_coefficient_ood":128,"locked_delay_dropout":128}[role]:
                    raise RuntimeError(f"locked parent shard count/length mismatch: {pde}/{role}")
                if "role" not in d: d["role"]=np.asarray([role]*n)
                if "disturbance_seed" not in d:
                    d["disturbance_seed"]=np.asarray([300000+int(str(x).rsplit(":",1)[-1]) for x in d["parent_id"]],dtype=np.int64)
                methods=[("B0",None),("B1",None)]+[(m,s) for m in ("P","B4","B5","B2","B3") for s in (11,23,37)]
                for method,seed in methods:
                    ckpt=checkpoint_map[(pde,method,seed)] if seed is not None else None
                    model=load_controller(method,pde,seed,ckpt,device)
                    rows=[]; t0=time.perf_counter()
                    for i in range(n):
                        result=run_episode(d,i,pde,method,model,device,1.2 if pde=="burgers" else qmax,query_tick=100,ticks=200)
                        rows.append(result); done+=1
                        if done%16==0:
                            progress.write_text(json.dumps({"status":"RUNNING_NO_METRICS_EMITTED","episodes_complete":done,
                                "episodes_total":2*640*17,"elapsed_seconds":time.perf_counter()-start},indent=2)+"\n",encoding="utf-8")
                    arrays={"parent_id":np.asarray([r["parent_id"] for r in rows]),
                        "episode_control_cost":np.asarray([r["episode_control_cost"] for r in rows]),
                        "tracking_rmse":np.asarray([r["tracking_rmse"] for r in rows]),
                        "truth_constraint_violation":np.asarray([r["truth_constraint_violation"] for r in rows],bool),
                        "fallback_rate":np.asarray([r["fallback_rate"] for r in rows]),
                        "fallback_count":np.asarray([r["fallback_count"] for r in rows]),
                        "fallback_errors":np.asarray([json.dumps(r["fallback_errors"]) for r in rows]),
                        "field_nrmse_at_snapshot":np.asarray([r["field_nrmse_at_snapshot"] for r in rows]),
                        "candidate_regret_at_snapshot":np.asarray([r["candidate_regret_at_snapshot"] for r in rows]),
                        "causal_timestamp_violations":np.asarray([r["causal_timestamp_violations"] for r in rows],dtype=np.int32),
                        "sensor_mask":np.stack([r["sensor_mask"] for r in rows]),"sensor_age":np.stack([r["sensor_age"] for r in rows]),
                        "sensor_capture_time":np.stack([r["sensor_capture_time"] for r in rows]),"sensor_receive_time":np.stack([r["sensor_receive_time"] for r in rows]),
                        "image_valid":np.stack([r["image_valid"] for r in rows]),"image_age":np.stack([r["image_age"] for r in rows]),
                        "image_capture_time":np.stack([r["image_capture_time"] for r in rows]),"image_receive_time":np.stack([r["image_receive_time"] for r in rows]),
                        "snapshot_candidate_best_index":np.asarray([-1 if r["snapshot_candidate_best_index"] is None else r["snapshot_candidate_best_index"] for r in rows],dtype=np.int16),
                        "action_proposed":np.stack([r["action_proposed"] for r in rows]),
                        "action_projected":np.stack([r["action_projected"] for r in rows]),
                        "action_applied":np.stack([r["action_applied"] for r in rows]),
                        "state_true":np.stack([r["state_true"] for r in rows]),"goal":np.stack([r["goal"] for r in rows])}
                    for key in rows[0]["latency_replay"]:
                        arrays[f"latency_{key}"]=np.stack([r["latency_replay"][key] for r in rows])
                    target=work/role/pde/f"{method}_s{seed if seed is not None else 'na'}.npz"
                    target.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(target,**arrays)
                    (target.with_suffix(".record.json")).write_text(json.dumps({"method":method,"seed":seed,"pde":pde,"role":role,
                        "parent_count":n,"episode_ticks":200,"elapsed_seconds":time.perf_counter()-t0,
                        "test_opened":True,"paired_parent_unit":True,"qmax_heat_train_validation_only":qmax if pde=="heat" else None},indent=2)+"\n",encoding="utf-8")
                    del model
        # Validate action accounting before publishing any summary.
        raw_files=sorted(work.glob("*/**/*.npz"))
        for path in raw_files:
            with np.load(path,allow_pickle=False) as z:
                if not np.array_equal(z["action_projected"],z["action_applied"]): raise RuntimeError(f"applied/projected mismatch: {path}")
                if not np.isfinite(z["state_true"]).all(): raise RuntimeError(f"nonfinite rollout: {path}")
        raw_root.parent.mkdir(parents=True,exist_ok=True)
        # Preserve the in-progress files in place; final status is written only after every shard audit.
        final_hashes={str(p.relative_to(ROOT)):sha(p) for p in raw_files}
        freeze.update({"status":"LOCKED_TEST_EVALUATION_COMPLETE","episodes_complete":done,"episodes_total":done,
            "elapsed_seconds":time.perf_counter()-start,"raw_result_sha256":final_hashes})
        marker.write_text(json.dumps(freeze,indent=2,sort_keys=True)+"\n",encoding="utf-8")
        progress.write_text(json.dumps({"status":"COMPLETE","episodes_complete":done,"elapsed_seconds":time.perf_counter()-start},indent=2)+"\n",encoding="utf-8")
    except Exception as exc:
        freeze.update({"status":"LOCKED_TEST_OPENED_INCONCLUSIVE_FAILURE","episodes_complete":done,
            "elapsed_seconds":time.perf_counter()-start,"failure":f"{type(exc).__name__}: {exc}"})
        marker.write_text(json.dumps(freeze,indent=2,sort_keys=True)+"\n",encoding="utf-8")
        raise
    print(json.dumps({"status":freeze["status"],"episodes":done,"elapsed_seconds":freeze["elapsed_seconds"],"raw_files":len(final_hashes)},indent=2))

if __name__=="__main__": main()
