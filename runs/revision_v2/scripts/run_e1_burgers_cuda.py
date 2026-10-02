from pathlib import Path
import sys,json,time,argparse,hashlib,gc
R=Path(__file__).resolve().parents[1];OLD=R.parents[1]/'experiments/pdno_jevLite_20260927_v3';sys.path.insert(0,str(R/'src'))
import numpy as np,torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
from pdno.controllers.revision_oracle_cuda import BurgersCandidateCUDA
from pdno.controllers.revision_oracle import observer_initial
from pdno.evaluation.closed_loop import _schedule,_sensor_at,_image_at,_obs_for_tick
from pdno.controllers.linear import nominal_lqr_action
from pdno.controllers.actions import project_box_slew
from pdno.data.teacher_queries import feasible_candidates
parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args();started=time.perf_counter();allcount=0
for role in ['locked_nominal','locked_coefficient_ood','locked_delay_dropout']:
 src=OLD/f'data/locked_v3/{role}/burgers/trajectories.npz'
 with np.load(src,allow_pickle=False) as z:data={k:z[k] for k in ['parent_id','nu','disturbance_seed']};data['initial']=z['state_true'][:,0].copy()
 for method in ['O-cand-state','O-cand-obs']:
  count=1 if args.smoke else len(data['parent_id']);blocks=[]
  for start in range(0,count,64):
   if (R/'state/STOP_REVISION').exists():raise SystemExit('Stopped at chunk boundary')
   end=min(count,start+64);B=end-start;nu=data['nu'][start:end].astype(float);parentids=data['parent_id'][start:end];states=data['initial'][start:end].copy();goal=np.zeros(256,np.float32);materials=[np.array([n,-1,1,.15,.02,.16,1.2,128],np.float32) for n in nu]
   # The reference uses float(material[0]); preserve its FP32 parameter representation.
   solver=BurgersCandidateCUDA(np.array([m[0] for m in materials],float));truth=BurgersCandidateCUDA(np.array([m[0] for m in materials],float),K=1)
   schedules=[_schedule(str(pid),200,role=='locked_delay_dropout') for pid in parentids];sensors=[np.zeros((200,16),np.float32) for _ in range(B)];images=[[None]*200 for _ in range(B)];actions=[[] for _ in range(B)]
   history=np.empty((B,201,256),np.float32);history[:,0]=states;ah=np.empty((B,200,2),np.float32);choiceh=np.empty((B,200),np.int16);cost=np.zeros(B);sq=np.zeros(B);viol=np.zeros(B,bool);pulses=np.zeros((B,256));haspulse=False
   x=np.arange(256)/256
   for b in range(B):
    ds=int(data['disturbance_seed'][start+b])
    if ds%4==0:
     rng=np.random.default_rng(ds);c=rng.random();d=np.minimum(abs(x-c),1-abs(x-c));pulse=.05*np.exp(-.5*(d/.025)**2);pulse-=pulse.mean();pulses[b]=pulse;haspulse=True
   for tick in range(200):
    cand=[];prev=[];initial=[]
    for b in range(B):
     schedule=schedules[b];sensors[b][tick]=_sensor_at(states[b],'burgers',schedule['sensor_noise'][tick])
     if not schedule['image_drop'][tick]:images[b][tick]=_image_at(states[b],'burgers',schedule,tick)
     obs,pr=_obs_for_tick(actions[b],tick,'burgers',materials[b],goal,schedule,sensors[b],images[b]);nom=nominal_lqr_action(obs,'burgers',pr);cand.append(feasible_candidates('burgers',pr,nom));prev.append(pr)
     initial.append(states[b] if method=='O-cand-state' else observer_initial(obs,'burgers'))
     assert not np.any(obs['sensor_mask'] & (obs['sensor_receive_time']>tick))
    cand=np.stack(cand);prev=np.stack(prev);fields=solver(np.stack(initial),cand).cpu().numpy()
    tracking=(fields*fields).mean(axis=(-1,-2)).astype(float);effort=.01*(cand*cand).sum(-1).astype(float);slew=.05*((cand-prev[:,None,:])**2).sum(-1).astype(float);vc=10*(np.maximum(np.abs(fields)-1.2,0)**2).mean(axis=(-1,-2)).astype(float)
    choices=(tracking+effort+slew+vc).argmin(1);selected=cand[np.arange(B),choices];applied=np.stack([project_box_slew(selected[b],prev[b],-1,1,.15) for b in range(B)])
    nxt=truth.advance_one_tick(states,applied,pulses if haspulse and tick==100 else None)
    assert np.isfinite(nxt).all();assert np.max(abs(applied-prev))<=.1500001;assert np.max(abs(applied))<=1.0000001
    err=nxt.astype(float);cost+=np.mean(err*err,axis=-1)+.01*(applied**2).sum(-1)+.05*((applied-prev)**2).sum(-1)+10*np.mean(np.maximum(np.abs(nxt)-1.2,0)**2,axis=-1)
    sq+=(err*err).sum(-1);viol|=np.max(abs(nxt),axis=-1)>1.2
    for b in range(B):actions[b].append(applied[b].copy())
    states=nxt;history[:,tick+1]=states;ah[:,tick]=applied;choiceh[:,tick]=choices
   blocks.append({'parent_id':parentids,'episode_control_cost':cost,'tracking_rmse':np.sqrt(sq/(200*256)),'truth_constraint_violation':viol,'state_true':history,'action_applied':ah,'selected_candidate_index':choiceh,'fallback_count':np.zeros(B,int),'causal_timestamp_violations':np.zeros(B,int)});allcount+=B
   del solver,truth;gc.collect();torch.cuda.empty_cache();print(json.dumps({'role':role,'method':method,'parents':end,'total':count,'elapsed_s':time.perf_counter()-started}),flush=True)
  outdir=R/('smoke/E1_cuda' if args.smoke else 'results/E1_raw')/role/'burgers';outdir.mkdir(parents=True,exist_ok=True);out=outdir/f'{method}.npz';assert not out.exists()
  arrays={k:np.concatenate([b[k] for b in blocks]) for k in blocks[0]};np.savez_compressed(out,**arrays);(out.with_suffix('.record.json')).write_text(json.dumps({'status':'COMPLETE','parents':count,'precision':'FP64 CUDA reference with graph replay; original FP32 tick states','role':role,'method':method,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'source_data_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'post_hoc':True},indent=2))
 if args.smoke:break
(R/('smoke' if args.smoke else 'results')/'E1_burgers_cuda_COMPLETE.json').write_text(json.dumps({'status':'SMOKE_PASS' if args.smoke else 'COMPLETE','executions':allcount,'elapsed_s':time.perf_counter()-started},indent=2))
