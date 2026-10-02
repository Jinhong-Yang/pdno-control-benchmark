from pathlib import Path
import os,sys,json,time,hashlib,argparse
os.environ['CUDA_VISIBLE_DEVICES']='-1'
R=Path(__file__).resolve().parents[1];OLD=R.parents[1]/'experiments/pdno_jevLite_20260927_v3';sys.path.insert(0,str(R/'src'))
import numpy as np,torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
from pdno.evaluation.closed_loop import run_episode
parser=argparse.ArgumentParser();parser.add_argument('--pde',choices=['heat','burgers'],required=True);parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
roles=['locked_nominal','locked_coefficient_ood','locked_delay_dropout'];start=time.perf_counter();total=0
for role in roles:
 p=OLD/f'data/locked_v3/{role}/{args.pde}/trajectories.npz'
 with np.load(p,allow_pickle=False) as z:data={k:z[k] for k in ['parent_id','role','state_true','disturbance_seed','nu'] if k in z};data.update({k:z[k] for k in ['kappa','decay','goal','goal_step_field','goal_step_tick'] if k in z})
 for method in ['O-cand-state','O-cand-obs']:
  rows=[];count=1 if args.smoke else len(data['parent_id']);outdir=R/('smoke/E1' if args.smoke else 'results/E1_raw')/role/args.pde;outdir.mkdir(parents=True,exist_ok=True);out=outdir/f'{method}.npz'
  if out.exists():raise RuntimeError('Refusing overwrite '+str(out))
  for parent in range(count):
   if (R/'state/STOP_REVISION').exists():raise SystemExit('Stopped by user at parent boundary')
   row=run_episode(data,parent,args.pde,method,None,torch.device('cpu'),1.2 if args.pde=='burgers' else 2.1322593092918396,query_tick=-1,ticks=200)
   assert row['fallback_count']==0,row['fallback_errors'];assert row['causal_timestamp_violations']==0;assert np.isfinite(row['state_true']).all()
   a=row['action_applied'].astype(float);prev=np.concatenate([np.full((1,2),.4) if args.pde=='heat' else np.zeros((1,2)),a[:-1]],axis=0);lo,hi,sl=(0,1,.1) if args.pde=='heat' else (-1,1,.15)
   assert np.all((a>=lo-1e-7)&(a<=hi+1e-7));assert np.max(np.abs(a-prev))<=sl+1e-7
   rows.append({k:row[k] for k in ['parent_id','episode_control_cost','tracking_rmse','truth_constraint_violation','fallback_count','causal_timestamp_violations','action_applied','state_true','goal','selected_candidate_index']});total+=1
   if parent%16==0:print(json.dumps({'pde':args.pde,'role':role,'method':method,'parent':parent+1,'count':count,'elapsed_s':time.perf_counter()-start}),flush=True)
  arrays={k:np.asarray([r[k] for r in rows]) for k in rows[0]};np.savez_compressed(out,**arrays)
  (out.with_suffix('.record.json')).write_text(json.dumps({'status':'COMPLETE','parents':count,'pde':args.pde,'method':method,'role':role,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'seconds_since_start':time.perf_counter()-start,'test_reuse':'post-hoc oracle diagnostic; not fresh confirmatory evidence'},indent=2))
  if args.smoke:print('smoke',args.pde,method,'passed',flush=True)
 if args.smoke:break
(R/('smoke' if args.smoke else 'results')/f'E1_{args.pde}_COMPLETE.json').write_text(json.dumps({'status':'SMOKE_PASS' if args.smoke else 'COMPLETE','executions':total,'seconds':time.perf_counter()-start,'device':'cpu'},indent=2))
