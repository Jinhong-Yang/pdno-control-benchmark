"""Independently reconstruct completed outcomes, timing quantiles and preservation hashes."""
from pathlib import Path
import json,csv,hashlib,time
import numpy as np
from revision_audit_contract import STATUS, validate_completion, validate_outcome_inventory
R=Path(__file__).resolve().parents[1];ROOT=R.parents[1];OLD=ROOT/'experiments/pdno_jevLite_20260927_v3';started=time.perf_counter();checks=[];rawrows=[];heatrows=[]
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def check(name,condition,detail=None):
    checks.append({'check':name,'pass':bool(condition),'detail':detail})
    if not condition:
        (R/'results/REVISION_RAW_AUDIT_FAILED.json').write_text(json.dumps({'checks':checks},indent=2));raise AssertionError((name,detail))
receipts={}
for name in STATUS:
    check('completion_marker:'+name,(R/'results'/name).exists())
    receipts[name]=json.loads((R/'results'/name).read_text())
validate_completion(receipts)
check('completion_status_and_training_row_inventory',True)
e3_scores={}
for path in sorted((R/'checkpoints/E3').glob('*.summary.json')):
    summary=json.loads(path.read_text()); selected=[v for v in summary['history'] if v['update']==summary['best_step']]
    check(path.name+':unique_selected_validation_row',len(selected)==1)
    e3_scores[path.name.removesuffix('.summary.json')]=selected[0]['validation_field_nrmse']
roles={'locked_nominal':384,'locked_coefficient_ood':128,'locked_delay_dropout':128}
for stage,expected in [('E1',12),('E4',30),('E5',60),('E3',None)]:
    paths=sorted((R/f'results/{stage}_raw').glob('*/*/*.npz'))
    if stage=='E3':expected=3*json.loads((R/'results/E3_EVALUATION.json').read_text())['conditions']
    check(stage+':array_inventory',len(paths)==expected,{'actual':len(paths),'expected':expected})
    validate_outcome_inventory(stage,[(p.parent.parent.name,p.parent.name,p.stem) for p in paths],e3_scores)
    check(stage+':exact_factorial_and_accuracy_gate_inventory',True)
    for path in paths:
        role,pde=path.parent.parent.name,path.parent.name;label=path.stem;rec=json.loads(path.with_suffix('.record.json').read_text());check(str(path.relative_to(R))+':hash',sha(path)==rec['sha256'])
        with np.load(path,allow_pickle=False) as z:
            ids=z['parent_id'];states=z['state_true'];actions=z['action_applied'];costs=z['episode_control_cost'];rmse=z['tracking_rmse'];events=z['truth_constraint_violation'];choices=z['selected_candidate_index'];goal=z['goal'] if 'goal' in z else np.zeros_like(states[:,1:]);fallback=z['fallback_count'];causal=z['causal_timestamp_violations'];components=z['cost_components'] if 'cost_components' in z else None
            vm={k:z[k] for k in ['violation_tolerance_1e7','violation_duration_seconds','violation_max','violation_integrated_mean'] if k in z}
        n=roles[role];name=str(path.relative_to(R));check(name+':shape',states.shape==(n,201,256) and actions.shape==(n,200,2) and len(set(ids))==n)
        check(name+':finite',all(np.isfinite(a).all() for a in [states,actions,costs,rmse]));check(name+':recorded_fallback_and_causality_counters_zero',not np.any(fallback) and not np.any(causal))
        lo,hi,sl,qmax=(-1.,1.,.15,1.2) if pde=='burgers' else (0.,1.,.1,2.1322593092918396)
        prev=np.concatenate([np.full((n,1,2),0. if pde=='burgers' else .4,dtype=np.float32),actions[:,:-1]],axis=1)
        check(name+':box_slew',actions.min()>=lo-1e-7 and actions.max()<=hi+1e-7 and np.max(abs(actions-prev))<=sl+1e-7)
        if not any(label.startswith(m+'_s') for m in ['B0','B2','B3']):check(name+':candidate_indices',choices.min()>=0 and choices.max()<10)
        u=states[:,1:];err=(u-goal).astype(float);v=np.maximum(abs(u)-qmax,0) if pde=='burgers' else np.maximum(-u,0)+np.maximum(u-qmax,0)
        # Float32 spatial penalty and action reductions reproduce stored evaluator terms;
        # float64 time sums expose reduction-roundoff differences explicitly.
        track=np.mean(err*err,axis=2).sum(axis=1);effort=.01*np.sum(actions**2,axis=2,dtype=np.float32).sum(axis=1,dtype=float);slew=.05*np.sum((actions-prev)**2,axis=2,dtype=np.float32).sum(axis=1,dtype=float);penalty_tick=10*np.mean(v**2,axis=2);penalty=penalty_tick.sum(axis=1,dtype=float)
        reconstructed=track+effort+slew+penalty;reconstruction_error=float(np.max(abs(reconstructed-costs)))
        check(name+':cost_reconstruction',np.allclose(reconstructed,costs,atol=1e-5,rtol=1e-6),reconstruction_error)
        check(name+':rmse',np.allclose(np.sqrt(np.mean(err*err,axis=(1,2))),rmse,atol=1e-7,rtol=1e-6))
        peak=np.max(v,axis=2);check(name+':violation_mask',np.array_equal(peak.max(axis=1)>0,events))
        if components is not None:check(name+':stored_cost_components',np.allclose(np.stack([track,effort,slew,penalty],axis=1),components,atol=1e-5,rtol=1e-6))
        if vm:
            check(name+':violation_amounts',np.array_equal(vm['violation_tolerance_1e7'],peak.max(axis=1)>1e-7) and np.allclose(vm['violation_duration_seconds'],(peak>0).sum(axis=1)*.02,atol=1e-8) and np.allclose(vm['violation_max'],peak.max(axis=1),atol=1e-7) and np.allclose(vm['violation_integrated_mean'],np.mean(v,axis=2).sum(axis=1,dtype=float)*.02,atol=1e-7))
        baseline=OLD/f'evidence/locked_test_raw_v3/{role}/{pde}/B0_sna.npz'
        with np.load(baseline,allow_pickle=False) as z:old_ids=z['parent_id']
        check(name+':parent_scope',not set(ids)&set(old_ids) if stage=='E5' else np.array_equal(ids,old_ids))
        if stage=='E5':
            check(name+':initial_feasibility',states[:,0].min()>=0 and states[:,0].max()<=.5000001)
            method,seed=label.rsplit('_s',1);heatrows.append({'role':role,'method':method,'seed':'' if seed=='na' else seed,'parents':n,'cost':float(costs.mean()),'tracking':float(track.mean()),'action':float(effort.mean()),'slew':float(slew.mean()),'violation':float(penalty.mean()),'first_step_violation_cost':float(penalty_tick[:,0].mean()),'violation_fraction_of_cost':float(penalty.mean()/costs.mean()),'max_reconstruction_error':reconstruction_error})
        rawrows.append({'stage':stage,'role':role,'pde':pde,'label':label,'parents':n,'cost_max_reconstruction_error':reconstruction_error,'sha256':rec['sha256']})
    print('Audited',stage,len(paths),'outcome arrays',flush=True)
timing=json.loads((R/'results/E2_TIMING.json').read_text());paths=sorted((R/'results/E2_raw').glob('*.npz'));check('E2:inventory',len(paths)==360 and len(timing['rows'])==360);requests=0
for path in paths:
    rec=json.loads(path.with_suffix('.json').read_text());check(path.stem+':hash',sha(path)==rec['sha256'])
    with np.load(path,allow_pickle=False) as z:t=z['latency_ms'];role=z['source_role'];ix=z['source_index']
    check(path.stem+':count_finite',len(t)==5000 and np.isfinite(t).all() and (t>0).all());check(path.stem+':pool_mix',np.array_equal(np.bincount(role,minlength=3),[3000,1000,1000]) and np.all((ix>=0)&(ix<np.array([7680,2560,2560])[role])))
    check(path.stem+':quantiles',abs(float(np.quantile(t,.5))-rec['p50_ms'])<1e-12 and abs(float(np.quantile(t,.99))-rec['p99_ms'])<1e-12);check(path.stem+':cache_equivalence',rec['equivalence']['max_abs_error']<=1e-6);requests+=len(t)
check('E2:requests',requests==1800000);profiles=sorted((R/'results/E2_profiles').glob('*.json'));check('E2:profile_inventory',len(profiles)==16)
for path in profiles:
    rec=json.loads(path.read_text());check(path.stem+':profile_count',len(rec['stage_ms'])==500);check(path.stem+':profile_finite',all(np.isfinite(list(row.values())).all() and min(row.values())>=-1e-5 for row in rec['stage_ms']))
preserved=[]
for manifest in ['ORIGINAL_240_ARRAYS.json','ORIGINAL_PROTECTION.json']:
    source=json.loads((R/'evidence'/manifest).read_text())
    for item in source:
        p=ROOT/item['path'];check('preserved:'+item['path'],p.exists() and sha(p)==item['sha256']);preserved.append(item['path'])
    print('Verified preservation',manifest,len(source),'files',flush=True)
for name,data in [('E6_new_heat_cost_decomposition.csv',heatrows),('REVISION_RAW_OUTCOME_AUDIT.csv',rawrows)]:
    with (R/'results'/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
result={'status':'PASS','checks':checks,'check_count':len(checks),'outcome_arrays':len(rawrows),'outcome_executions':sum(r['parents'] for r in rawrows),'timing_arrays':360,'timing_requests':requests,'preserved_unique_paths':len(set(preserved)),'elapsed_s':time.perf_counter()-started,'scope':'Arithmetic, exact factorial/gate inventories, arrays, counts, constraints and preservation; recorded fallback/causality counters are not independent full-timestamp evidence. No candidate-forecast argmin reconstruction, control-benefit proof or deployment-safety guarantee.','auditor_sha256':sha(Path(__file__)),'contract_sha256':sha(R/'scripts/revision_audit_contract.py'),'completion_receipt_sha256':{name:sha(R/'results'/name) for name in STATUS}}
(R/'results/REVISION_RAW_AUDIT.json').write_text(json.dumps(result,indent=2));print('Raw revision audit PASS',len(checks),'checks',flush=True)
