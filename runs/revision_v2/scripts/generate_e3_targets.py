"""Nested train-only target expansion using the validated FP64 CUDA reference."""
from pathlib import Path
import sys,json,time,hashlib,gc
R=Path(__file__).resolve().parents[1];OLD=R.parents[1]/'experiments/pdno_jevLite_20260927_v3';sys.path.insert(0,str(R/'src'))
import numpy as np,torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
from pdno.data.teacher_queries import _snapshot_observation,feasible_candidates,restrict_field,_cost
from pdno.controllers.linear import nominal_lqr_action
from pdno.controllers.revision_oracle_cuda import BurgersCandidateCUDA
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
t=time.perf_counter();source=OLD/'data/train_validation_v3/train/burgers/trajectories.npz'
with np.load(source,allow_pickle=False) as z:data={k:z[k] for k in z.files}
with np.load(OLD/'data/queries_v3/train/burgers/teacher_queries.npz',allow_pickle=False) as z:original={k:z[k] for k in z.files}
with np.load(OLD/'data/queries_v3/validation/burgers/teacher_queries.npz',allow_pickle=False) as z:assert not set(data['parent_id'])&set(z['parent_id'])
indices=list(np.linspace(8,120,4,dtype=int))
while len(indices)<64:
    choices=[i for i in range(8,121) if i not in indices]
    indices.append(max(choices,key=lambda i:(min(abs(i-j) for j in indices),-i)))
spec={'algorithm':'start original four ticks; repeatedly choose largest minimum distance, lower tick breaks ties','ordered_ticks':list(map(int,indices)),'observer':'refit on each factor at 4000-update cap; then shared/frozen for matched P/B4 fits; see pre-execution E3_OBSERVER_SCOPE_AMENDMENT.json','train_parents':len(data['parent_id']),'source_sha256':sha(source),'test_used':False}
specpath=R/'config/E3_TARGET_EXPANSION.json'
assert not specpath.exists();specpath.write_text(json.dumps(spec,indent=2))
N=len(data['parent_id'])*64;arrays={k:np.empty((N,*v.shape[1:]),dtype=v.dtype) for k,v in original.items()}
original_lookup={(str(p),int(tick)):i for i,(p,tick) in enumerate(zip(original['parent_id'],original['decision_tick']))}
maxerr=0.
for start in range(0,N,64):
    if (R/'state/STOP_REVISION').exists():raise SystemExit('Stop at chunk boundary')
    rows=list(range(start,min(N,start+64)));parents=[i//64 for i in rows];ticks=[indices[i%64] for i in rows];cand=[];initial=[];previous=[]
    for i,p,tick in zip(rows,parents,ticks):
        nu=float(data['nu'][p]);material=np.array([nu,-1,1,.15,.02,.16,1.2,128],np.float32)
        obs,prev=_snapshot_observation(data,p,int(tick),'burgers',material,np.zeros(256,np.float32))
        for k,v in obs.items():arrays[k][i]=v
        cs=feasible_candidates('burgers',prev,nominal_lqr_action(obs,'burgers',prev));cand.append(cs);initial.append(data['state_true'][p,tick]);previous.append(prev)
        arrays['parent_id'][i]=data['parent_id'][p];arrays['decision_tick'][i]=tick;arrays['initial_field'][i]=restrict_field(initial[-1],'burgers');arrays['candidate_action'][i]=cs;arrays['goal'][i]=0;arrays['q_min'][i]=-1.2;arrays['q_max'][i]=1.2
    solver=BurgersCandidateCUDA(data['nu'][parents].astype(float));fields=solver(np.stack(initial),np.stack(cand)).cpu().numpy()
    for b,i in enumerate(rows):
        arrays['future_field'][i]=fields[b]
        cp=[_cost(fields[b,j],arrays['goal'][i],cand[b][j],previous[b],1.2) for j in range(10)]
        arrays['teacher_cost'][i]=[v[0] for v in cp];arrays['teacher_peak'][i]=[v[1] for v in cp];arrays['teacher_best_index'][i]=int(np.argmin(arrays['teacher_cost'][i]))
        key=(str(arrays['parent_id'][i]),int(arrays['decision_tick'][i]))
        if key in original_lookup:
            oi=original_lookup[key];err=float(np.max(abs(arrays['future_field'][i]-original['future_field'][oi])));maxerr=max(maxerr,err)
            assert err<=1e-6;assert np.array_equal(arrays['candidate_action'][i],original['candidate_action'][oi])
    del solver;gc.collect();torch.cuda.empty_cache()
    if start%1024==0:print(json.dumps({'queries':start+len(rows),'total':N,'elapsed_s':time.perf_counter()-t}),flush=True)
records=[]
for factor,snap in [(4,16),(16,64)]:
    out=R/f'data/E3_queries/x{factor}/teacher_queries.npz';out.parent.mkdir(parents=True,exist_ok=True);assert not out.exists()
    select=np.flatnonzero(np.arange(N)%64<snap);np.savez_compressed(out,**{k:v[select] for k,v in arrays.items()})
    records.append({'factor':factor,'queries':len(select),'parents':len(data['parent_id']),'sha256':sha(out),'path':str(out.relative_to(R))})
(R/'results/E3_TARGET_GENERATION.json').write_text(json.dumps({'status':'COMPLETE','source_sha256':sha(source),'records':records,'original_overlap_max_abs_error':maxerr,'elapsed_s':time.perf_counter()-t,'device':'cuda FP64','test_used':False},indent=2))
