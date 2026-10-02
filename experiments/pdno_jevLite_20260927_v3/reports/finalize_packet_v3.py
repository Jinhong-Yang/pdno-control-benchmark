"""Preserve historical admin state and seal the final, analysis-only deliverables."""
import argparse
import hashlib
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

R = Path(__file__).resolve().parents[1]
ROOT = R.parents[1]
O = R/'reports'

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def write(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')

args = argparse.ArgumentParser()
args.add_argument('stage', choices=['state', 'seal'])
stage = args.parse_args().stage
qa = read(O/'FINAL_PACKET_VALIDATION.json')
if qa['status'] != 'PASS':
    raise SystemExit('Final packet QA must pass before administration or sealing.')
now = datetime.now(timezone.utc)
kst = now.astimezone(timezone(timedelta(hours=9))).isoformat(timespec='seconds')
runtime = read(O/'RUNTIME_METADATA.json')
start = datetime.fromisoformat(runtime['started_at_utc'].replace('Z','+00:00'))
elapsed = (now-start).total_seconds()
task = runtime['task_id']
outcome = 'NO_GO_V3_PROPOSED_CLAIM'

if stage == 'state':
    dest = O/'administrative_update_before'
    if dest.exists():
        raise SystemExit('Administrative snapshot exists; inspect instead of repeating updates.')
    dest.mkdir()
    paths = [(R/'state'/f, 'v3_'+f) for f in ('STATE.md','HANDOFF.md','RUN_LEDGER.md','RUN_STATE.json')]
    paths += [(ROOT/'state'/f, 'root_'+f) for f in ('STATE.md','HANDOFF.md','RUN_LEDGER.md')]
    before = {}
    for p, name in paths:
        before[str(p)] = sha(p) if p.exists() else None
        if p.exists():
            (dest/name).write_bytes(p.read_bytes())

    status = f'''## Final v3 independent analysis completed — {kst}

This dated addendum supersedes older current-status statements below; historical text is retained.
Scientific outcome: **{outcome}** for the implemented frozen manufacturing PDE benchmark.
The English manuscript and all five requested report files are complete for human review;
authors remain AUTHOR_INPUT_REQUIRED and no submission or external upload was performed.
Actual current task: `{task}`, runtime-confirmed **gpt-6-astra / xhigh**.
This task independently analyzed frozen v3 evidence and authored the documents; it did not
run training, solver/inference, calibration generation, a new test or a CUDA benchmark.
Earlier v2 audit and earlier experiment executor identity are separate; the latter remains unknown.

- Evidence scope: 1,280 parent episodes, 25,600 controller executions, 120 test shards,
  120 timing rows and 2,400,000 requests. The one-shot experiment is terminal.
- Independent arithmetic audit: 1,694 passing checks. Final packet QA: {qa['checks']} passing
  checks at this update, 183 claim rows, all 16 corrected CUDA groups matched raw arrays.
- G2: all 24 selected predictors fail 0.05 nRMSE. Nominal P cost exceeds B0 by 28.04%
  (Burgers) and 13.70% (heat). H1 mean seed p99 ratios 0.98591/0.99296 miss the 25% target.
- H3 refers to the implemented normalization; its mismatch with the original parent-wise
  floored estimand is disclosed. H2 control benefit is unsupported; H4 was not evaluated.
- Heat binary violation events already occur at the first advanced step. Passive calibration
  is not an acceptance gate; conditional harm is unavailable, not zero.
- CUDA percentile correction is independently verified. Event-span/host ratios do not
  establish compute occupancy. No optimization speedup or scheduled real-time result exists.
- Entry hash audit: 1,072 entries, with only pre-existing administrative RUN_STATE/RUN_LEDGER
  differences. This final administrative update also changes STATE/HANDOFF where present;
  before/after hashes are recorded separately. Raw arrays, sources, models and freeze manifests
  remain unchanged. Final raw-freeze SHA256:
  `2fc83da6f7c577d09d1c75ce44e276726a6705383e074974f94d9f321a3eea41`.
- Read `reports/REPORT_KO.md`, `reports/DECISION_PACKET.json`, `reports/claim_evidence.csv`,
  `reports/failures_and_limitations.md`, `reports/REPRODUCE.md`, and `manuscript_v3.md`.
  Final receipt and hashes are `reports/COMPLETION_RECEIPT.json` and
  `reports/DELIVERABLE_HASHES.json`; administrative update is separate from raw freeze.

Status axes: data_access=COMPLETE_FROZEN_V3_REVIEW;
physics_contract=LIMITED_FIXTURES_PASS_ENDPOINT_LIMITS;
implementation=COMPLETE_IMPLEMENTED_SCOPE_PROTOCOL_GAPS_DISCLOSED;
optimization=G2_FAIL_SUFFICIENCY_UNESTABLISHED;
scientific_effect=NEGATIVE_IMPLEMENTED_CLAIM_RISK_INCONCLUSIVE;
reproducibility=HASH_AND_ARITHMETIC_AUDITED_HISTORICAL_GPU_HOURS_UNKNOWN.

Next action: human review of the bounded negative-result manuscript. Do not reopen v3 test,
retune, or edit frozen evidence. Any new science or optimization needs a separate prospectively
frozen version. Historical progress forecasts/budgets below are not current remaining budgets.

'''
    for name in ('STATE.md','HANDOFF.md'):
        p = R/'state'/name
        old = p.read_text(encoding='utf-8-sig')
        p.write_text(status + '---\n\n' + old, encoding='utf-8')
    ledger = R/'state/RUN_LEDGER.md'
    with ledger.open('a',encoding='utf-8') as f:
        f.write('\n\n' + status)
    p = R/'state/RUN_STATE.json'
    d = read(p)
    historical_keys = ['stage','analysis_model_status','next_executable_step','blocking_items',
        'cuda_optimization_status','estimated_astra_wall_hours_since_delegation',
        'elapsed_wall_hours_since_original_start','remaining_wall_hours',
        'luna_evidence_hours_remaining_until_62h','locked_test_data_generated','astra_final_task_status']
    d['pre_final_analysis_status_snapshot'] = {k:d.get(k) for k in historical_keys}
    d.update(stage='V3_FINAL_ANALYSIS_COMPLETE',
        analysis_model_status='Runtime verified gpt-6-astra xhigh; independent v3 audit and requested manuscript/report package complete for human review.',
        astra_final_task_status='COMPLETE_VERIFIED_DRAFT_AUTHOR_INPUT_REQUIRED_NO_SUBMISSION',
        next_executable_step='Human review only. v3 is terminal; no test reopening or tuning. Separate prospective version required for new science/runtime optimization.',
        blocking_items=['Author metadata/declarations and human review required before publication; not blockers to completed analytical deliverables.'],
        cuda_optimization_status='No new optimization benchmark. Corrected p99 fields verified against frozen raw. Event span includes possible CPU/launch/synchronization gaps and does not establish GPU compute occupancy or compute-bound execution.',
        locked_test_data_generated=True,
        estimated_astra_wall_hours_since_delegation=elapsed/3600,
        actual_analysis_task_elapsed_seconds_at_state_update=elapsed,
        elapsed_wall_hours_since_original_start=None,remaining_wall_hours=None,
        luna_evidence_hours_remaining_until_62h=None,
        budget_status='Historical snapshots preserved separately. Current analysis task wall time measured from local turn_context; original total elapsed/remaining/GPU-active budget not independently reconciled.',
        post_freeze_state_updated_at=kst)
    d['final_analysis'] = {'task_id':task,'actual_model':runtime['actual_model'],
        'actual_reasoning_effort':runtime['actual_reasoning_effort'],'outcome':outcome,
        'status_axes':read(O/'DECISION_PACKET.json')['status_dimensions'],
        'arithmetic_checks':1694,'packet_qa_checks_at_state_update':qa['checks'],
        'claim_rows':183,'new_gpu_experiment_executions':0,
        'completion_receipt':'reports/COMPLETION_RECEIPT.json','deliverable_hashes':'reports/DELIVERABLE_HASHES.json'}
    write(p,d)

    root_note = f'''## Latest completed analysis — {kst}: manufacturing PDE v3

The separate PDNO–JevLite v3 task `{task}` completed independent frozen-evidence analysis
and the requested manuscript/report package using actual `gpt-6-astra` / `xhigh`.
Scientific decision: **{outcome}**; this does not change any earlier Darcy/AIP result.
See [v3 Korean report](../experiments/pdno_jevLite_20260927_v3/reports/REPORT_KO.md),
[v3 manuscript](../experiments/pdno_jevLite_20260927_v3/manuscript_v3.md), and
[v3 handoff](../experiments/pdno_jevLite_20260927_v3/state/HANDOFF.md).
All 1,694 arithmetic checks passed; 183 claims link to evidence. Required B3/B4 and all
negative results retained. Author input is unresolved; no submission. New GPU/solver/model
experiments in this final task: zero. v3 is terminal and must not be rerun or retuned.
Status axes and full provenance are in the v3 decision packet and completion receipt.
This addendum supersedes older current-status wording only for this PDNO task.

'''
    for name in ('STATE.md','HANDOFF.md'):
        p = ROOT/'state'/name
        p.write_text(root_note+'---\n\n'+p.read_text(encoding='utf-8-sig'),encoding='utf-8')
    p = ROOT/'state/RUN_LEDGER.md'
    if not p.exists():
        p.write_text('# Root task ledger\n\nCreated for this final administrative update; earlier per-experiment ledgers remain authoritative and are not reconstructed here.\n\n',encoding='utf-8')
    with p.open('a',encoding='utf-8') as f:
        f.write('\n'+root_note)
    write(O/'FINAL_ANALYSIS_STATE_UPDATE.json', {
        'at_kst':kst,'administrative_only':True,'historical_text_preserved':True,
        'state_backup_directory':str(dest.relative_to(R)),
        'files':[{'path':str(p),'before_sha256':before[str(p)],'after_sha256':sha(p)} for p,_ in paths],
        'frozen_evidence_modified':False,'note':'STATE/HANDOFF/RUN_LEDGER/RUN_STATE are administrative. Manifest hashes themselves are preserved; changes are not hidden by regenerating the raw freeze.'})
    print(json.dumps({'state_update':'COMPLETE','files':len(paths),'at_kst':kst}))
else:
    if not (O/'FINAL_ANALYSIS_STATE_UPDATE.json').exists():
        raise SystemExit('State update receipt missing.')
    if (O/'COMPLETION_RECEIPT.json').exists():
        raise SystemExit('Completion already sealed; inspect rather than overwrite.')
    update = read(O/'FINAL_ANALYSIS_STATE_UPDATE.json')
    if datetime.fromisoformat(qa['completed_at_utc']) < datetime.fromisoformat(update['at_kst']):
        raise SystemExit('Rerun packet QA after administrative update before sealing.')
    receipt = {'status':'COMPLETE_ANALYSIS_AND_DRAFT_FOR_HUMAN_REVIEW',
        'scientific_outcome':outcome,'task_id':task,'runtime':runtime,
        'completed_at_utc':now.isoformat(),'completed_at_kst':kst,
        'elapsed_task_wall_seconds_through_sealing':elapsed,
        'elapsed_task_wall_definition':'From first actual local turn_context to receipt sealing; includes reading, CPU analysis, drafting and tool waits, not continuous inference or GPU-active time.',
        'independent_arithmetic_script_wall_seconds':read(O/'analysis_20260928/audit.json')['elapsed_seconds'],
        'final_packet_qa_seconds':qa['elapsed_seconds'],'arithmetic_checks_passed':1694,
        'final_packet_checks_passed':qa['checks'],'claims':183,
        'new_training_inference_solver_calibration_test_benchmark_runs':0,
        'actions_completed':['Read root/v3 state and research contracts; reviewed prior v2 audit reports without opening partial outcome arrays.',
            'Hashed complete final evidence inventory and earlier provenance manifests.',
            'Independently reconstructed frozen control endpoints and registered bootstrap intervals in CPU NumPy.',
            'Independently checked all timing quantiles/deadline counts/H1 intervals and 26 saved calibration order statistics.',
            'Reviewed selected checkpoint/source/comparator behavior and all failure-log records.',
            'Identified first-step heat endpoint saturation, protocol mismatches and CUDA percentile-label collision.',
            'Checked originating-task corrected CUDA file against raw arrays.',
            'Reviewed supplied primary literature and authored complete English manuscript and Korean report.',
            'Preserved all baselines/negative results and updated administrative state with before/after receipts.'],
        'prior_v2_audit_is_not_v3_work':True,'earlier_experiment_executor_model':'UNKNOWN',
        'historical_total_gpu_active_hours':None,'authors':'AUTHOR_INPUT_REQUIRED',
        'submission':'NOT_SUBMITTED','external_uploads_or_contacts':0,
        'required_deliverable_sha256':qa['required_deliverable_sha256'],
        'final_qa_sha256':sha(O/'FINAL_PACKET_VALIDATION.json'),
        'administrative_update_sha256':sha(O/'FINAL_ANALYSIS_STATE_UPDATE.json'),
        'raw_freeze_sha256':sha(R/'evidence/raw_evidence_freeze_v3.json')}
    write(O/'COMPLETION_RECEIPT.json',receipt)
    payloads = sorted([R/'manuscript_v3.md', *[p for p in O.rglob('*') if p.is_file() and p.name!='DELIVERABLE_HASHES.json' and '__pycache__' not in p.parts]])
    write(O/'DELIVERABLE_HASHES.json',{'created_at_kst':kst,'scope':'Manuscript and report files, including preserved QA/admin history. Excludes this self-referential hash manifest; frozen source/data hashes remain in original raw freeze.',
        'files':[{'path':str(p.relative_to(R)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in payloads]})
    print(json.dumps({'status':receipt['status'],'outcome':outcome,'task_wall_seconds':elapsed,'final_qa_checks':qa['checks'],'artifact_files_hashed':len(payloads)},indent=2))
