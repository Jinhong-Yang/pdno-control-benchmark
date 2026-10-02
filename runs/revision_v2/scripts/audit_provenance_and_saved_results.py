from pathlib import Path
import json,csv,hashlib,datetime,numpy as np
R=Path(__file__).resolve().parents[1]; ROOT=R.parents[1]; OLD=ROOT/'experiments/pdno_jevLite_20260927_v3';PUB=ROOT/'publications/ieee_access_pdno_revision_v2_20261002'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=[json.loads(l) for l in (OLD/'evidence/failures_and_modifications_v3.jsonl').read_text().splitlines()]
repairs=[dict(line_number=i+1,**d) for i,d in enumerate(rows) if d.get('event') in ['STALE_DATA_PATH_CORRECTED','LOCKED_REPLAY_AXIS_HANDLING_CORRECTED','SYNTAX_ERROR_DURING_REPAIR','MISSING_TRAINING_PROCESS_RECEIPT_REPAIRED']]
control=json.loads((OLD/'state/LOCKED_TEST_ONCE_V3.json').read_text());start=datetime.datetime.strptime(control['opened_at_local'],'%Y-%m-%dT%H:%M:%S%z');end=start+datetime.timedelta(seconds=control['elapsed_seconds'])
source=OLD/'scripts/benchmark_host_e2e_v3.py'; lines=source.read_text().splitlines();snips=[{'line':i+1,'text':l} for i,l in enumerate(lines) if any(s in l for s in ['train_validation_v3','reshape','latency_','flatten'])]
record={'status':'DOCUMENTED_WITH_LIMITATION','control_started':start.isoformat(),'control_completed_inferred':end.isoformat(),'completion_inference':'opened marker plus recorded elapsed_seconds; not an independent timestamp','repairs':repairs,'repaired_code':{'path':str(source),'sha256':sha(source),'relevant_lines':snips},'pre_repair_source_snapshot_available':False,'literal_before_after_diff_available':False,'conclusion':'Repairs concern later serialized latency replay, after the original control campaign had completed. Current runner source and historical log support separation from control-output generation. Full before/after source diffs were not retained; zero effect on the latency measurement itself cannot be independently established. Replay snapshots originate in original test trajectories, so access was not train/validation-only. Do not describe the control test as running after these repairs.'}
(R/'results/P0_2_runtime_repairs.json').write_text(json.dumps(record,indent=2))
reg={'classification':'PRIMARY','external_prospective_registration_found':False,'pretest_git_commit_found':False,'internal_document':{'path':'config/v3_preregistration.yaml','sha256':sha(OLD/'config/v3_preregistration.yaml')},'local_pretraining_seal':'evidence/pretraining_spec_freeze_v3.json','later_public_commit':'b5453468157654a9b1b9c6a68619246c78afd77f','later_public_commit_is_not_preregistration':True,'wording':'primary; original internally specified protocol, no externally verifiable prospective registration','rationale':'Workspace is not a Git checkout. The public release was made after evaluation. Local file/record dates are retained but are not an independently time-stamped pre-test commit.'}
(R/'results/P0_3_registration_audit.json').write_text(json.dumps(reg,indent=2))
original=list((OLD/'evidence/locked_test_raw_v3').rglob('*.npz'))+list((OLD/'evidence/latency_raw_v3').rglob('*.npz'))
assert len(original)==240,len(original)
manifest=[{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size} for p in original]
(R/'evidence/ORIGINAL_240_ARRAYS.json').write_text(json.dumps(manifest,indent=2))
methods=['B0','B1','P','P-no-rank','B4','B5','B2','B3'];roles=['locked_nominal','locked_coefficient_ood','locked_delay_dropout'];out=[];decomp=[];inputs=[]
for role in roles:
 for pde in ['burgers','heat']:
  costs={};ids_ref=None
  for method in methods:
   values=[]
   for seed in ([None] if method in ['B0','B1'] else [11,23,37]):
    p=OLD/f'evidence/locked_test_raw_v3/{role}/{pde}/{method}_s{seed if seed else "na"}.npz';inputs.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p)})
    with np.load(p,allow_pickle=False) as z:
     ids=z['parent_id'];c=z['episode_control_cost'].astype(float);values.append(c)
     if ids_ref is None:ids_ref=ids.copy()
     assert np.array_equal(ids,ids_ref)
     if pde=='heat':
      st=z['state_true'][:,1:].astype(float);a=z['action_applied'].astype(float);goal=z['goal'].astype(float);prev=np.concatenate([np.full((len(a),1,2),.4),a[:,:-1]],axis=1)
      tracking=((st-goal)**2).mean(2).sum(1);effort=.01*(a*a).sum((1,2));slew=.05*((a-prev)**2).sum((1,2));viol=10*(np.maximum(-st,0)**2+np.maximum(st-2.1322593092918396,0)**2).mean(2)
      rec=tracking+effort+slew+viol.sum(1);err=float(np.max(np.abs(rec-c)));assert err<1e-4,err
      decomp.append({'role':role,'method':method,'seed':seed,'parents':len(c),'cost':float(c.mean()),'tracking':float(tracking.mean()),'action':float(effort.mean()),'slew':float(slew.mean()),'violation':float(viol.sum(1).mean()),'first_step_violation_cost':float(viol[:,0].mean()),'violation_fraction_of_cost':float(viol.sum(1).mean()/c.mean()),'max_reconstruction_error':err})
   costs[method]=np.mean(values,axis=0)
  base=costs['B0'];n=len(base);rng=np.random.default_rng(2026100207);ix=rng.integers(n,size=(10000,n));den=base[ix].mean(1)
  for method in methods:
   delta=(costs[method]-base);boot=delta[ix].mean(1);fixed=100*boot/base.mean();joint=100*boot/den
   out.append({'pde':pde,'role':role,'method':method,'parents':n,'estimate_percent':100*float(delta.mean()/base.mean()),'fixed_low':float(np.quantile(fixed,.025)),'fixed_high':float(np.quantile(fixed,.975)),'joint_low':float(np.quantile(joint,.025)),'joint_high':float(np.quantile(joint,.975)),'replicates':10000})
  print(role,pde,'complete',flush=True)
for name,data in [('E6_original_heat_cost_decomposition.csv',decomp),('E7_joint_denominator_bootstrap.csv',out)]:
 with (R/'results'/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
(R/'results/E6_E7_receipt.json').write_text(json.dumps({'status':'COMPLETE_ORIGINAL_CAMPAIGN_ONLY','input_hashes':inputs,'E6_rows':len(decomp),'E7_rows':len(out),'bootstrap_seed':2026100207,'parent_cluster':'training seeds averaged within each parent before bootstrap; numerator and baseline use same sampled parent indices','new_result_scope':'post-hoc reanalysis only; revision experiments must later be added','frozen_inputs_unchanged':all(sha(ROOT/x['path'])==x['sha256'] for x in manifest)},indent=2))
B0={'status':'CODE_DOCUMENTED_TUNING_HISTORY_UNAVAILABLE','modes':7,'ridge':.001,'LQR_Q':'identity','LQR_R':'0.1 identity','dt':.02,'gain':'discrete algebraic Riccati solution per material parameters; zero Burgers constant-mode feedback','heat_feedforward':'least-squares steady input for 7-mode target','action_constraints':'box and slew projection','validation_comparator_selection':'existing comparator selection evaluated B0 versus B1-B5 on 64 parents/PDE; this is method selection, not a documented B0 hyperparameter search','hyperparameter_tuning_budget':'not recorded; no claim of validation-tuned gains/mode count','source_sha256':sha(OLD/'src/pdno/controllers/linear.py')}
(R/'results/E9_B0_tuning_audit.json').write_text(json.dumps(B0,indent=2))
with (PUB/'revision_decision_log.md').open('a',encoding='utf-8') as f:f.write('\n## Phase 0 completed\nP0-1: G-B in both PDEs (100 matched CPU restart updates from seed-11 update-2000 weights). Nonzero physics gradients; tiny weighted/data ratios. Optimizer moments unavailable; exact historical CUDA continuation is not claimed. E4 is mandatory.\nP0-2: repairs occurred in latency replay after control evaluation completed; archived before/after diff absent. Categorical zero impact on timing cannot be proved. Replay inputs include stored test observations.\nP0-3: use primary; local seals exist but no independently time-stamped pre-test commit or external registration was found.\nE6/E7 original-campaign reanalyses completed; E9 code settings documented, tuning budget unavailable. Phase 2 remains unresolved until E1/E2/E5 and required E4 complete.\n')
print('Phase 0 provenance, E6/E7 and E9 complete')
