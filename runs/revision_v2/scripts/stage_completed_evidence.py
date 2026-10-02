"""Copy completed, small evidence into the portable manuscript; keep input hashes."""
from pathlib import Path
import json,hashlib,shutil
R=Path(__file__).resolve().parents[1];P=R.parents[1]/'publications/ieee_access_pdno_revision_v2_20261002';D=P/'overleaf/data/revision';D.mkdir(parents=True,exist_ok=True)
names=['P0_1_gradient_audit.json','P0_1_gradient_trace.csv','P0_2_runtime_repairs.json','P0_3_registration_audit.json','E1_candidate_frequency.csv','E1_candidate_frequency_record.json','E6_original_heat_cost_decomposition.csv','E7_joint_denominator_bootstrap.csv','E9_B0_tuning_audit.json','REVISION_COST_SUMMARY.csv','REVISION_SUMMARY_RECEIPT.json','E1_heat_COMPLETE.json','E1_burgers_cuda_COMPLETE.json','BATCH_ROLLOUT_TESTS.json']
assert json.loads((R/'results/E1_heat_COMPLETE.json').read_text())['status']=='COMPLETE'
assert json.loads((R/'results/E1_burgers_cuda_COMPLETE.json').read_text())['status']=='COMPLETE'
records=[]
optional_groups={
 'E1_INDEPENDENT_RAW_AUDIT.json':['E1_INDEPENDENT_RAW_AUDIT.json'],
 'E1_INTERVAL_AUDIT.json':['E1_INTERVAL_AUDIT.json'],
 'REVISION_INTERVAL_AUDIT.json':['REVISION_INTERVAL_AUDIT.json'],
 'E2_ANALYSIS.json':['E2_ANALYSIS.json','E2_per_session.csv','E2_per_seed.csv','E2_pooled.csv','E2_ratio_bootstrap.csv','E2_stage_profile.csv','E2_latency_histograms.csv'],
 'REVISION_TRAINING_EXPORT.json':['REVISION_TRAINING_EXPORT.json','REVISION_TRAINING_SELECTION.csv','REVISION_TRAINING_CURVES.csv'],
 'E4_gradient_record.json':['E4_gradient_record.json','E4_selected_gradient_diagnostics.csv','E4_ACTION_AGREEMENT.csv'],
 'REVISION_RAW_AUDIT.json':['REVISION_RAW_AUDIT.json','REVISION_RAW_OUTCOME_AUDIT.csv','E6_new_heat_cost_decomposition.csv'],
}
for marker,group in optional_groups.items():
    path=R/'results'/marker
    if path.exists():
        receipt=json.loads(path.read_text())
        allowed={
            'E1_INDEPENDENT_RAW_AUDIT.json':['PASS_E1_INDEPENDENT_RAW_ARITHMETIC'],
            'E1_INTERVAL_AUDIT.json':['PASS_SCOPED_E1_INTERVAL_AUDIT'],
            'REVISION_INTERVAL_AUDIT.json':['PASS_ALL_REVISION_INTERVAL_AUDIT'],
        }.get(marker,['COMPLETE','PASS'])
        assert receipt['status'] in allowed, (marker,receipt['status'])
        names.extend(group)
for name in names:
    src=R/'results'/name;out=D/name;shutil.copy2(src,out);records.append({'source':str(src.relative_to(R)),'portable_path':str(out.relative_to(P/'overleaf')),'sha256':hashlib.sha256(out.read_bytes()).hexdigest()})
for name in ['PHASE1_SPEC.json','E2_COUNT_CORRECTION.json','E3_OBSERVER_SCOPE_AMENDMENT.json']:
    src=R/'config'/name;shutil.copy2(src,D/name);records.append({'source':str(src.relative_to(R)),'portable_path':str((D/name).relative_to(P/'overleaf')),'sha256':hashlib.sha256(src.read_bytes()).hexdigest()})
for name in ['E2_ENVIRONMENT_MIDRUN.json','E2_ENVIRONMENT_LATE_RUN.json','E2_CODE_PATH_AUDIT.json']:
    src=R/'evidence'/name
    if src.exists():
        shutil.copy2(src,D/name);records.append({'source':str(src.relative_to(R)),'portable_path':str((D/name).relative_to(P/'overleaf')),'sha256':hashlib.sha256(src.read_bytes()).hexdigest()})
(D/'STAGED_EVIDENCE.json').write_text(json.dumps({'status':'COMPLETED_EVIDENCE_ONLY_NOT_FINAL_PACKAGE','items':records},indent=2))
print('Staged',len(records),'completed evidence/configuration items')
