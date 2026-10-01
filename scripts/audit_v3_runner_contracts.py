"""Static path/method/one-shot contract audit for every versioned v3 research runner."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def main()->int:
    names=("run_confirmatory_models_v3.py","freeze_checkpoint_manifest_v3.py","select_validation_comparator_v3.py",
        "audit_validation_comparator_v3.py","generate_calibration_data_v3.py","generate_calibration_queries_v3.py",
        "audit_calibration_queries_v3.py","calibrate_forecast_margins_v3.py","audit_calibration_lock_v3.py",
        "generate_locked_data_v3.py","audit_locked_roles_v3.py","freeze_locked_evaluation_v3.py",
        "run_locked_test_once_v3.py","aggregate_locked_test_v3.py","start_locked_test_v3.ps1",
        "benchmark_host_e2e_v3.py","analyze_latency_v3.py","freeze_pretraining_spec_v3.py")
    files={n:ROOT/"scripts"/n for n in names};missing=[n for n,p in files.items() if not p.is_file()]
    if missing:raise SystemExit("missing version-specific runner(s): "+", ".join(missing))
    text={n:p.read_text(encoding="utf-8") for n,p in files.items() if p.suffix==".py"}
    checks={
      "confirmatory_covers_all_learned_families": all(x in text["run_confirmatory_models_v3.py"] for x in ("P-no-rank","B4","B3","(11, 23, 37)")),
      "checkpoint_freeze_covers_36_models_and_g2_failure": all(x in text["freeze_checkpoint_manifest_v3.py"] for x in ("len(rows)!=36","P-no-rank","G2_FAIL_RETAINED_DIAGNOSTIC")),
      "calibration_generation_requires_frozen_models": all(x in text["generate_calibration_data_v3.py"] for x in ("confirmatory_checkpoint_manifest_v3.json","validation_closed_loop_comparator_v3.json","calibration_lock_v3.json")),
      "calibration_queries_use_v3_data_and_signed_cost": all(x in text["generate_calibration_queries_v3.py"] for x in ("train_validation_v3","1.25*max(maxima)","queries_calibration_v3")),
      "calibration_query_audit_checks_64x8_and_role": all(x in text["audit_calibration_queries_v3.py"] for x in ("len(allowed)!=64","len(parent_ids)!=512","audit_teacher_shard")),
      "locked_generation_requires_calibration_and_200_ticks": all(x in text["generate_locked_data_v3.py"] for x in ("calibration_lock_v3.json","allow_locked=True","outer_ticks=200","LOCKED_TEST_ONCE_V3.json")),
      "locked_data_audit_checks_v3_parent_manifest_and_heat_step": all(x in text["audit_locked_roles_v3.py"] for x in ("parent_roles_metadata_only.json","goal_step_tick","==100","locked_200tick_data_audit_v3.json")),
      "locked_preopen_freeze_requires_audits": all(x in text["freeze_locked_evaluation_v3.py"] for x in ("CALIBRATION_LOCK_AUDIT_PASSED","V3_LOCKED_PARENT_DATA_AUDITED_TEST_SEALED","LOCKED_EVALUATION_INPUTS_FROZEN_TEST_SEALED")),
      "locked_runner_is_atomic_single_open_and_covers_inventory": all(x in text["run_locked_test_once_v3.py"] for x in ("os.O_EXCL","LOCKED_TEST_ONCE_V3.json","P-no-rank","TOTAL_RUNS = 25600","no resume or rerun")),
      "locked_aggregator_covers_every_mandatory_method": all(x in text["aggregate_locked_test_v3.py"] for x in ("P-no-rank","B3","LOCKED_TEST_ONCE_V3.json")),
      "host_latency_only_after_complete_locked_test": all(x in text["benchmark_host_e2e_v3.py"] for x in ("LOCKED_TEST_EVALUATION_COMPLETE","LOCKED_TEST_ONCE_V3.json")),
      "no_precalibration_runner_reads_locked_payloads": "locked_v3" not in text["run_confirmatory_models_v3.py"] and "locked_v3" not in text["select_validation_comparator_v3.py"],
    }
    stale=[]
    for name,body in text.items():
        if name=="audit_v3_runner_contracts.py":continue
        if "_V2.json" in body or "locked_v2" in body or "confirmatory_v2" in body or "data/v2" in body or "data\\v2" in body:
            stale.append(name)
    checks["v3_runners_have_no_v2_data_or_marker_paths"]=not stale
    checks={k:bool(v) for k,v in checks.items()}
    result={"status":"V3_RUNNER_CONTRACT_AUDIT_PASSED" if all(checks.values()) else "V3_RUNNER_CONTRACT_AUDIT_FAILED",
        "passed":all(checks.values()),"test_opened":False,"calibration_generated":False,"checks":checks,
        "stale_path_files":stale,"runner_sha256":{name:sha(path) for name,path in files.items()}}
    out=ROOT/"evidence/v3_runner_contract_audit.json";out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({"status":result["status"],"checks":checks,"stale_path_files":stale,"output":str(out)},indent=2));return 0 if result["passed"] else 1
if __name__=="__main__":raise SystemExit(main())
