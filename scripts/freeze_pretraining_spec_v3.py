"""Hash the fail-inclusive v3 model/data/evaluator contract before seed training."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def main()->int:
    out=ROOT/"evidence/pretraining_spec_freeze_v3.json"
    if out.exists():raise SystemExit("v3 pretraining freeze exists; use an explicit amended freeze after documenting a code change")
    locked=ROOT/"data/locked_v3";cal=ROOT/"data/calibration_v3"
    if (locked.exists() and any(locked.rglob("*.npz"))) or (cal.exists() and any(cal.rglob("*.npz"))):raise SystemExit("calibration/locked targets must not exist before pretraining freeze")
    required=[ROOT/"config/final_spec_v3.json",ROOT/"config/evaluation_addendum_v3.yaml",ROOT/"config/checkpoint_selection_v3.yaml",
        ROOT/"config/v3_preregistration.yaml",ROOT/"data/manifests/parent_roles_metadata_only.json",
        ROOT/"evidence/preflight_v3.json",ROOT/"evidence/full_teacher_query_audit_v3.json",
        ROOT/"evidence/parent_lineage_audit_v3.json",ROOT/"evidence/v3_g2_pilot_progress.json",
        ROOT/"evidence/v3_closed_loop_runtime_pilot.json",ROOT/"evidence/runtime_cuda_optimizer_assessment_v3.json",
        ROOT/"scripts/run_confirmatory_models_v3.py",ROOT/"scripts/freeze_checkpoint_manifest_v3.py",
        ROOT/"scripts/freeze_pretraining_spec_v3.py",
        ROOT/"scripts/select_validation_comparator_v3.py",ROOT/"scripts/run_v3_g2_pilot.py",
        ROOT/"scripts/pilot_closed_loop_runtime_v3.py",ROOT/"src/pdno/training/confirmatory.py",
        ROOT/"scripts/audit_v3_runner_contracts.py",ROOT/"evidence/v3_runner_contract_audit.json",
        ROOT/"scripts/generate_calibration_data_v3.py",ROOT/"scripts/generate_calibration_queries_v3.py",
        ROOT/"scripts/audit_calibration_queries_v3.py",ROOT/"scripts/calibrate_forecast_margins_v3.py",
        ROOT/"scripts/audit_calibration_lock_v3.py",ROOT/"scripts/generate_locked_data_v3.py",
        ROOT/"scripts/audit_locked_roles_v3.py",ROOT/"scripts/freeze_locked_evaluation_v3.py",
        ROOT/"scripts/run_locked_test_once_v3.py",ROOT/"scripts/aggregate_locked_test_v3.py",
        ROOT/"scripts/benchmark_host_e2e_v3.py",ROOT/"scripts/analyze_latency_v3.py",
        ROOT/"scripts/audit_validation_comparator_v3.py",ROOT/"scripts/start_locked_test_v3.ps1",
        ROOT/"src/pdno/evaluation/closed_loop.py",ROOT/"src/pdno/models/operators.py",ROOT/"src/pdno/controllers/objectives.py",
        ROOT/"evidence/failures_and_modifications_v3.jsonl",ROOT/"reports/FEATURE_USAGE_AND_RUNTIME_BOUNDARY_v3.md"]
    required.extend(sorted((ROOT/"src/pdno").rglob("*.py")))
    required.extend(sorted((ROOT/"tests").rglob("*.py")))
    required.extend(sorted(p for p in (ROOT/"data/train_validation_v3").rglob("*") if p.is_file()))
    required.extend(sorted(p for p in (ROOT/"data/queries_v3").rglob("*") if p.is_file()))
    missing=[p for p in required if not p.is_file()]
    if missing:raise SystemExit("missing v3 freeze input(s): "+", ".join(str(p) for p in missing))
    spec_path=ROOT/"config/final_spec_v3.json";spec=json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("decision_context",{}).get("g2_pilot_status")!="G2_FAIL_RETAINED":raise SystemExit("final spec must preserve the registered G2 failure")
    spec["status"]="FROZEN_FAIL_INCLUSIVE_PRETRAINING_SPEC"
    spec_path.write_text(json.dumps(spec,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    inputs={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(required))}
    result={"status":"PRETRAINING_SPEC_FROZEN_TEST_SEALED","protocol":"PDNO_JevLite_RTX5080_72H_frozen_spec_v3",
        "spec_path":"config/final_spec_v3.json","spec_sha256":sha(spec_path),"inputs_sha256":inputs,
        "train_validation_target_data_only":True,"calibration_generated":False,"locked_test_generated":False,"locked_test_opened":False,
        "g2_pilot_disposition":"G2_FAIL_RETAINED_NO_THRESHOLD_CHANGE","feature_use_limitation":"B0/B1 share the same raw causal observation object but explicitly use fewer temporal features than learned controllers; disclose in analysis",
        "author_input_required":True}
    out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({"status":result["status"],"input_count":len(inputs),"spec_sha256":result["spec_sha256"],"output":str(out)},indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
