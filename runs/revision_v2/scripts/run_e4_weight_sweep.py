from pathlib import Path
import os,sys,json,time
os.environ['CUDA_VISIBLE_DEVICES']='-1'
R=Path(__file__).resolve().parents[1];OLD=R.parents[1]/'experiments/pdno_jevLite_20260927_v3';sys.path.insert(0,str(R/'src'))
import torch
torch.set_num_threads(2);torch.set_num_interop_threads(1)
from pdno.training.confirmatory import train_operator_staged
spec=R/'config/PHASE1_SPEC.json';assert spec.exists();started=time.perf_counter();rows=[]
for pde in ['burgers','heat']:
 for phys in [.01,.1,1.]:
  for rank in [.1,1.]:
   if phys==.01 and rank==.1:continue
   name=f'{pde}_phys{phys:g}_rank{rank:g}_s11';out=R/'checkpoints/E4'/f'{name}.pt'
   if out.with_suffix('.summary.json').exists():raise RuntimeError('Refusing to overwrite completed sensitivity run')
   if (R/'state/STOP_REVISION').exists():raise SystemExit('Stopped at requested safe boundary')
   row=train_operator_staged(OLD/f'data/queries_v3/train/{pde}/teacher_queries.npz',OLD/f'data/queries_v3/validation/{pde}/teacher_queries.npz',OLD/f'runs/confirmatory_v3_training/observer/{pde}_seed7.pt',pde,'P',11,out,lambda_phys=phys,lambda_balance=phys,lambda_rank=rank,field_updates=2000,physics_updates=3000,eval_interval=500,batch_size=64,device_name='cpu')
   rows.append({'condition':name,'physics_balance_weight':phys,'ranking_weight':rank,**row});(R/'state/E4_PROGRESS.json').write_text(json.dumps({'status':'TRAINING','completed':len(rows),'total':10,'elapsed_s':time.perf_counter()-started,'rows':rows},indent=2));print(json.dumps({'completed':len(rows),'condition':name,'seconds':row['elapsed_seconds'],'best_step':row['best_step']}),flush=True)
(R/'results/E4_TRAINING.json').write_text(json.dumps({'status':'TRAINING_COMPLETE_EVALUATION_PENDING','device':'cpu','new_GPU_use':0,'elapsed_seconds':time.perf_counter()-started,'rows':rows},indent=2))
