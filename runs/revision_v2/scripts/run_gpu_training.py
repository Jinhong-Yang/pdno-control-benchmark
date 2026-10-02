"""Sequential GPU fits; validation-only selection; immutable condition outputs."""
from pathlib import Path
import sys,json,time,argparse,hashlib,datetime
R=Path(__file__).resolve().parents[1];OLD=R.parents[1]/'experiments/pdno_jevLite_20260927_v3';sys.path.insert(0,str(R/'src'))
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
from pdno.training.confirmatory import train_operator_staged,train_observer,train_direct_staged,train_b3_staged
parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['E4','E5','E3'],required=True);args=parser.parse_args();t=time.perf_counter();rows=[]
assert torch.cuda.is_available()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def run(output,fn,label):
 if (R/'state/STOP_REVISION').exists():raise SystemExit('Stop requested at condition boundary')
 summary=output.with_suffix('.summary.json')
 if summary.exists():row=json.loads(summary.read_text());assert output.exists()
 else:
  if output.exists():raise RuntimeError('Incomplete checkpoint found; inspect bounded recovery before retry: '+str(output))
  row=fn()
 rows.append({'condition':label,**row});(R/'state'/f'{args.stage}_GPU_PROGRESS.json').write_text(json.dumps({'status':'RUNNING','completed':len(rows),'elapsed_s':time.perf_counter()-t,'rows':rows},indent=2));print(json.dumps({'stage':args.stage,'condition':label,'completed':len(rows),'wall_seconds':row.get('elapsed_seconds'),'update':row.get('best_step')}),flush=True)
 return output
if args.stage=='E4':
 for pde in ['burgers','heat']:
  for phys in [.01,.1,1.]:
   for rank in [.1,1.]:
    if phys==.01 and rank==.1:continue
    label=f'{pde}_phys{phys:g}_rank{rank:g}_s11';out=R/f'checkpoints/E4_cuda/{label}.pt'
    run(out,lambda:train_operator_staged(OLD/f'data/queries_v3/train/{pde}/teacher_queries.npz',OLD/f'data/queries_v3/validation/{pde}/teacher_queries.npz',OLD/f'runs/confirmatory_v3_training/observer/{pde}_seed7.pt',pde,'P',11,out,lambda_phys=phys,lambda_balance=phys,lambda_rank=rank,field_updates=2000,physics_updates=3000,eval_interval=500,batch_size=64,device_name='cuda'),label)
elif args.stage=='E5':
 train=R/'data/positive_heat_queries/train/heat/teacher_queries.npz';val=R/'data/positive_heat_queries/validation/heat/teacher_queries.npz';base=R/'checkpoints/E5';observer=base/'observer/heat_seed7.pt'
 run(observer,lambda:train_observer(train,val,'heat',observer,seed=7,max_updates=1000,eval_interval=100,batch_size=64,device_name='cuda'),'heat_observer')
 for seed in [11,23,37]:
  for method in ['P','B4','B5','P-no-rank']:
   out=base/f'{method}_s{seed}.pt';run(out,lambda:train_operator_staged(train,val,observer,'heat',method,seed,out,lambda_phys=.01,lambda_balance=.01,field_updates=2000,physics_updates=3000,eval_interval=500,batch_size=64,device_name='cuda'),f'{method}_s{seed}')
  b2=base/f'B2_s{seed}.pt';run(b2,lambda:train_direct_staged(train,val,observer,'heat',seed,b2,max_updates=3000,eval_interval=500,batch_size=64,device_name='cuda'),f'B2_s{seed}')
  b3=base/f'B3_s{seed}.pt';run(b3,lambda:train_b3_staged(train,val,b2,base/f'B4_s{seed}.pt','heat',seed,b3,max_updates=1000,eval_interval=500,device_name='cuda'),f'B3_s{seed}')
 # This local seal precedes any new heat test-target generation.
 paths=sorted(base.rglob('*.pt'));assert len(paths)==19
 freeze=R/'evidence/E5_POLICY_FREEZE.json'
 weights=[{'path':str(p.relative_to(R)),'sha256':sha(p)} for p in paths]
 if freeze.exists():
  prior=json.loads(freeze.read_text());assert prior['checkpoints']==weights
 else:
  assert not any((R/f'data/positive_heat/{role}').exists() for role in ['locked_nominal','locked_coefficient_ood','locked_delay_dropout'])
  sources=[R/'config/PHASE1_SPEC.json',R/'config/heat_parent_manifest.json',R/'heat_ic_spec.md',R/'scripts/evaluate_revision.py',*sorted((R/'src').rglob('*.py'))]
  freeze.write_text(json.dumps({'status':'FROZEN_BEFORE_NEW_TEST_GENERATION','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'new_heat_test_generated':False,'checkpoints':weights,'source_and_config':[{'path':str(p.relative_to(R)),'sha256':sha(p)} for p in sources],'train_targets_sha256':sha(train),'validation_targets_sha256':sha(val),'selection':'original per-method validation criteria; all six learned families, 3 seeds, shared retrained observer','q_max':2.1322593092918396},indent=2))
else:
 for factor in [1,4,16]:
  train=OLD/'data/queries_v3/train/burgers/teacher_queries.npz' if factor==1 else R/f'data/E3_queries/x{factor}/teacher_queries.npz';val=OLD/'data/queries_v3/validation/burgers/teacher_queries.npz';observer=R/f'checkpoints/E3/observer/burgers_x{factor}_seed7.pt'
  # The upstream observer must also see the enlarged training-target set.
  # It is refit once per factor, then shared/frozen across P/B4 and all three seeds.
  run(observer,lambda:train_observer(train,val,'burgers',observer,seed=7,max_updates=4000,eval_interval=100,batch_size=64,device_name='cuda'),f'burgers_x{factor}_observer_s7')
  for seed in [11,23,37]:
   for method in ['P','B4']:
    label=f'burgers_x{factor}_{method}_s{seed}';out=R/f'checkpoints/E3/{label}.pt';run(out,lambda:train_operator_staged(train,val,observer,'burgers',method,seed,out,lambda_phys=.01,lambda_balance=.01,field_updates=2000,physics_updates=18000,eval_interval=500,batch_size=64,device_name='cuda'),label)
 # Same early-stopping rule is retained; actual update counts, not caps, must be reported.
 paths=sorted((R/'checkpoints/E3').rglob('*.pt'));assert len(paths)==21
 freeze=R/'evidence/E3_POLICY_FREEZE.json';weights=[{'path':str(p.relative_to(R)),'sha256':sha(p)} for p in paths]
 if freeze.exists():assert json.loads(freeze.read_text())['checkpoints']==weights
 else:freeze.write_text(json.dumps({'status':'FROZEN_BEFORE_GATE_TRIGGERED_CONTROL_EVALUATION','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checkpoints':weights,'amendment_sha256':sha(R/'config/E3_OBSERVER_SCOPE_AMENDMENT.json'),'accuracy_gate':.05,'selection':'validation-only selection; existing test parents reused only for gate-passing checkpoints','evaluation_script_sha256':sha(R/'scripts/evaluate_revision.py')},indent=2))
result={'status':'TRAINING_COMPLETE_EVALUATION_PENDING','stage':args.stage,'device':'cuda','process_wall_seconds':time.perf_counter()-t,'gpu_active_seconds':'not directly measured; do not relabel process wall time','rows':rows}
(R/'results'/f'{args.stage}_GPU_TRAINING.json').write_text(json.dumps(result,indent=2))
