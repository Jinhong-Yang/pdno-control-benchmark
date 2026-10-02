from pathlib import Path
import sys,json,time,argparse,hashlib,gc
R=Path(__file__).resolve().parents[1];OLD=R.parents[1]/'experiments/pdno_jevLite_20260927_v3';sys.path.insert(0,str(R/'src'))
import numpy as np,torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
from pdno.evaluation.closed_loop import load_controller
from pdno.evaluation.revision_batch import run_batch
from pdno.data import generate
from pdno.data.revision_heat import positive_heat_initial
parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True,choices=['E4','E5','E3']);args=parser.parse_args()
assert json.loads((R/'results/BATCH_ROLLOUT_TESTS.json').read_text())['status']=='PASS'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();t=time.perf_counter();roles=['locked_nominal','locked_coefficient_ood','locked_delay_dropout'];conditions=[]
if args.stage=='E5':
    freeze=R/'evidence/E5_POLICY_FREEZE.json';f=json.loads(freeze.read_text());assert f['status']=='FROZEN_BEFORE_NEW_TEST_GENERATION'
    for row in f['checkpoints']:assert sha(R/row['path'])==row['sha256']
    for row in f.get('source_and_config',[]):assert sha(R/row['path'])==row['sha256']
    marker=R/'results/E5_TEST_GENERATION.json'
    if not marker.exists():
        assert not any((R/f'data/positive_heat/{role}').exists() for role in roles)
        generate._heat_initial=positive_heat_initial
        rec=generate.generate_roles(R/'config/heat_parent_manifest.json',R/'data/positive_heat',roles=tuple(roles),allow_locked=True,outer_ticks=200)
        marker.write_text(json.dumps({'status':'COMPLETE_AFTER_POLICY_FREEZE','freeze_sha256':sha(freeze),'records':rec},indent=2))
    for method in ['B0','B1','P','B4','B5','P-no-rank','B2','B3']:
        for seed in ([None] if method in ['B0','B1'] else [11,23,37]):conditions.append(('heat',method,seed,None if seed is None else R/f'checkpoints/E5/{method}_s{seed}.pt',f'{method}_s{seed if seed else "na"}'))
elif args.stage=='E4':
    for path in sorted((R/'checkpoints/E4_cuda').glob('*.pt')):conditions.append((path.stem.split('_')[0],'P',11,path,path.stem))
    assert len(conditions)==10
else:
    freeze=json.loads((R/'evidence/E3_POLICY_FREEZE.json').read_text())
    assert freeze['status']=='FROZEN_BEFORE_GATE_TRIGGERED_CONTROL_EVALUATION'
    assert freeze['evaluation_script_sha256']==sha(Path(__file__))
    for row in freeze['checkpoints']:assert sha(R/row['path'])==row['sha256']
    skipped=[]
    for path in sorted((R/'checkpoints/E3').glob('*.pt')):
        saved=torch.load(path,map_location='cpu',weights_only=False);meta=saved.get('metadata',saved)
        # Checkpoint metadata fields are flattened by the original save helper.
        v=meta.get('validation',{});score=v.get('validation_nrmse',v.get('validation_field_nrmse'))
        if score is None:raise RuntimeError('Missing selected-checkpoint validation nRMSE: '+str(meta.keys()))
        method=meta['method'];seed=meta['seed']
        if score<=.05:conditions.append(('burgers',method,seed,path,path.stem))
        else:skipped.append({'checkpoint':str(path.relative_to(R)),'nrmse':score,'reason':'accuracy gate not met; no closed-loop follow-up'})
    (R/'results/E3_GATE_DECISIONS.json').write_text(json.dumps({'selected_for_evaluation':len(conditions),'skipped':skipped},indent=2))
records=[]
for pde,method,seed,path,label in conditions:
    model=load_controller(method,pde,seed,path,torch.device('cuda'));qm=1.2 if pde=='burgers' else 2.1322593092918396
    for role in roles:
        out=R/f'results/{args.stage}_raw/{role}/{pde}/{label}.npz';out.parent.mkdir(parents=True,exist_ok=True)
        if out.exists():
            record=json.loads(out.with_suffix('.record.json').read_text());assert record['sha256']==sha(out);records.append(record);continue
        source=(R/f'data/positive_heat/{role}/heat/trajectories.npz') if args.stage=='E5' else OLD/f'data/locked_v3/{role}/{pde}/trajectories.npz'
        with np.load(source,allow_pickle=False) as z:data={k:z[k] for k in z.files if k in ['parent_id','role','nu','kappa','decay','disturbance_seed','goal','goal_step_field','goal_step_tick']};data['state_true']=z['state_true'][:,:1].copy()
        blocks=[]
        for start in range(0,len(data['parent_id']),64):
            if (R/'state/STOP_REVISION').exists():raise SystemExit('Stopped at evaluation chunk boundary')
            blocks.append(run_batch(data,range(start,min(start+64,len(data['parent_id']))),pde,method,model,qm));gc.collect();torch.cuda.empty_cache()
        arrays={k:np.concatenate([b[k] for b in blocks]) for k in blocks[0]};np.savez_compressed(out,**arrays)
        record={'stage':args.stage,'pde':pde,'method':method,'seed':seed,'label':label,'role':role,'parents':len(data['parent_id']),'sha256':sha(out),'input_sha256':sha(source),'checkpoint_sha256':None if path is None else sha(path),'elapsed_s':time.perf_counter()-t,'post_hoc_original_parents':args.stage!='E5','latency_measurement':False,'mean_cost':float(arrays['episode_control_cost'].mean())}
        out.with_suffix('.record.json').write_text(json.dumps(record,indent=2));records.append(record);print(json.dumps(record),flush=True)
    del model;gc.collect();torch.cuda.empty_cache()
(R/f'results/{args.stage}_EVALUATION.json').write_text(json.dumps({'status':'COMPLETE','conditions':len(conditions),'rows':records,'elapsed_s':time.perf_counter()-t},indent=2))
if args.stage=='E4':
    import subprocess
    for script in ['e4_selected_gradient_diagnostics.py','audit_choice_frequencies.py','summarize_revision.py','summarize_e2.py','export_training_evidence.py','audit_revision_raw.py']:
        subprocess.run([sys.executable,'-B','-X','utf8',str(R/'scripts'/script)],check=True)
