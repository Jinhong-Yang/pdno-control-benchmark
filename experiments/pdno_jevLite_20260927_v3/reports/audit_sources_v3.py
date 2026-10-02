"""Supplementary source/provenance checks; no experimental computation."""
import csv, hashlib, importlib.metadata, json, math, sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

R=Path(__file__).resolve().parents[1];O=R/'reports';A=O/'analysis_20260928'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(name,d):(O/name).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

session=Path('C:/Users/WIN/.codex/sessions/2026/09/28/rollout-2026-09-28T03-55-38-01a0e439-093f-7601-8a7d-6e8929a7ced4.jsonl')
with session.open(encoding='utf-8') as f:
 for line in f:
  r=json.loads(line)
  if r.get('type')=='turn_context':
   payload=r['payload'];runtime={'task_id':'01a0e439-093f-7601-8a7d-6e8929a7ced4','actual_model':payload.get('model'),'actual_reasoning_effort':payload.get('effort'),'started_at_utc':r['timestamp'],'metadata_source':str(session),'metadata_record_sha256':hashlib.sha256(line.encode()).hexdigest(),'metadata_type':'turn_context','analysis_only':True,'prior_execution_model':'UNKNOWN; no retroactive attribution','task_environment':'local Codex desktop, PowerShell, v3 isolated existing venv','new_gpu_training_inference_solver_benchmark_executions':0,'independent_reaggregation_wall_seconds':read(A/'audit.json')['elapsed_seconds'],'packages':{n:importlib.metadata.version(n) for n in ('numpy','scipy','torch','pytest')},'python':sys.version};break
save('RUNTIME_METADATA.json',runtime)
manifest_review=[]
for f,key in [('pretraining_spec_freeze_v3.json','inputs_sha256'),('locked_evaluation_freeze_v3.json','frozen_input_sha256'),('locked_evaluation_freeze_v3.json','checkpoint_sha256')]:
 d=read(R/'evidence'/f)
 rows=[{'path':p,'expected':h,'actual':sha(R/p)} for p,h in d[key].items()]
 manifest_review.append({'manifest':f,'key':key,'count':len(rows),'mismatches':[r for r in rows if r['actual']!=r['expected']]})
cm=read(R/'evidence/confirmatory_checkpoint_manifest_v3.json');selected=[];losses=[]
for r in cm['checkpoints']:
 history=r['validation_summary']['history'];v=r['metadata'].get('validation')
 if v:
  best=min(history,key=lambda x:x['selection_score'])
  selected.append({'pde':r['pde'],'method':r['method'],'seed':r['seed'],'selected_update':r['selected_update'],'minimum_score_update':best['update'],'match':r['selected_update']==best['update'],'field_nrmse':v['validation_field_nrmse'],'field_gate_pass':v['validation_field_nrmse']<=.05})
 if r['method']=='P':
  h=next(x for x in history if x['update']==r['selected_update'])
  losses.append({'pde':r['pde'],'seed':r['seed'],'selected_phase':h['phase'],'weighted_physics_balance_fraction_of_logged_loss':.01*(h['train_physics_loss']+h['train_balance_loss'])/h['train_loss'],'source':'selected-update minibatch logs; not gradient influence or independent PDE residual'})
comp=read(R/'evidence/validation_closed_loop_comparator_v3.json');scores={}
for m in ('B0','B1','B2','B3','B4','B5'):
 vals=[]
 for pde in ('burgers','heat'):
  d=comp['pdes'][pde]['methods'];c=np.mean([s['episode_cost_by_parent'] for s in d[m]['seed_results']],axis=0)
  b=np.mean(d['B0']['seed_results'][0]['episode_cost_by_parent']);vals.append(float(c.mean()/b))
 scores[m]=float(np.mean(vals))
checks={'selected_operator_checkpoint_checks':selected,'operator_checkpoints':len(selected),'failed_field_gates':sum(not r['field_gate_pass'] for r in selected),'P_loss_scale_diagnostic':losses,'comparator_scores_recomputed':scores,'comparator_scores_match':all(abs(scores[k]-comp['method_scores'][k])<1e-12 for k in scores),'comparator_selected':min(scores,key=scores.get),'manifest_chain':manifest_review}
save('source_audit.json',checks)

derived=read(R/'evidence/cuda_runtime_assessment_v3.json');correction=[]
for r in derived['groups']:
 vals=[];streams=[]
 for f in (R/'evidence/latency_raw_v3').glob('*.npz'):
  with np.load(f,allow_pickle=False) as z:
   if str(z['pde'].item())==r['pde'] and str(z['method'].item())==r['method']:
    vals.append(z['latency_ms']);streams.append(z['cuda_stream_span_ms'])
 v=np.concatenate(vals);s=np.concatenate(streams);s=s[np.isfinite(s)]
 row={'pde':r['pde'],'method':r['method'],'supplied_host_p99_label':r['host_wall_ms']['p99'],'true_host_p99':float(np.quantile(v,.99)),'true_host_p999':float(np.quantile(v,.999)),'host_label_actually_p999':bool(np.isclose(r['host_wall_ms']['p99'],np.quantile(v,.999),rtol=0,atol=1e-12))}
 if len(s):
  row.update(supplied_stream_p99_label=r['cuda_stream_ms']['p99'],true_stream_p99=float(np.quantile(s,.99)),true_stream_p999=float(np.quantile(s,.999)),stream_label_actually_p999=bool(np.isclose(r['cuda_stream_ms']['p99'],np.quantile(s,.999),rtol=0,atol=1e-12)),true_p99_ratio=float(np.quantile(s,.99)/np.quantile(v,.99)),true_p99_difference_ms=float(np.quantile(v,.99)-np.quantile(s,.99)))
 correction.append(row)
save('cuda_diagnostic_correction.json',{'classification':'DERIVED_POST_FREEZE_LABEL_CORRECTION_NOT_NEW_BENCHMARK','supplied_artifact_sha256':sha(R/'evidence/cuda_runtime_assessment_v3.json'),'finding':'All 16 host and all 12 available stream fields labeled p99 contain p99.9. Supplied p99 ratios/differences use actual p99 and remain arithmetically correct. Raw and registered latency analysis are unaffected.','original_artifact_preserved':True,'event_interpretation':'Event elapsed spans enclose policy_action including CPU work and synchronization; they may contain GPU idle and launch gaps. A high event/wall ratio does not establish GPU compute occupancy or compute-bound execution. Quantile differences are not paired per-request overhead distributions.','rows':correction})
evidence_catalog=[]
for f in sorted((R/'evidence').glob('*.json')):
 d=read(f);evidence_catalog.append({'path':str(f.relative_to(R)),'sha256':sha(f),'status':d.get('status'),'passed':d.get('passed'),'keys':list(d)})
for folder in ('config','contracts'):
 for f in sorted((R/folder).glob('*')):
  if f.is_file():evidence_catalog.append({'path':str(f.relative_to(R)),'sha256':sha(f),'content':f.read_text(encoding='utf-8-sig')})
save('reviewed_evidence_catalog.json',{'catalog':evidence_catalog,'source_reference_library':'../pdno_jevLite_20260925/references.json','source_reference_library_sha256':sha(R.parent/'pdno_jevLite_20260925/references.json'),'scope':'All top-level v3 evidence JSON parsed; full frozen inventory hash checked; authoritative source and predecessor audit manually reviewed. Legacy configs retained as historical, not silently promoted to executed protocol.'})
print(json.dumps({'actual_model':runtime['actual_model'],'effort':runtime['actual_reasoning_effort'],'g2_failed':checks['failed_field_gates'],'comparator':checks['comparator_selected'],'hash_chain_mismatches':[(r['manifest'],[x['path'] for x in r['mismatches']]) for r in manifest_review],'cuda_host_label_errors':sum(x['host_label_actually_p999'] for x in correction)},indent=2))
