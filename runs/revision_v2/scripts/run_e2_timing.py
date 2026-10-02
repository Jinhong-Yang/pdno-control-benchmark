"""Exclusive sequential host-ready cache/K campaign; same original transfer boundary."""
from pathlib import Path
import sys,json,time,hashlib,random,gc
R=Path(__file__).resolve().parents[1];OLD=R.parents[1]/'experiments/pdno_jevLite_20260927_v3';sys.path.insert(0,str(R/'src'))
import numpy as np,torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
from pdno.evaluation.closed_loop import load_controller
from pdno.controllers.actions import project_box_slew
from pdno.controllers.linear import nominal_lqr_action
from pdno.controllers.objectives import candidate_cost
from pdno.data.teacher_queries import restrict_field
from pdno.models.revision_cache import CachedOperator,expanded_candidates
from pdno.training.fit import _grid
KEYS=('sensor_value','sensor_mask','sensor_age','instrument_image','image_mask','goal_coefficients','material_context','previous_applied_action','applied_action_history','image_age','image_valid','goal_field')
ROLES=['locked_nominal','locked_coefficient_ood','locked_delay_dropout'];dev=torch.device('cuda');N=5000;WARMUP=50
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();started=time.perf_counter();rows=[];equivalence=[]

def sample(pde,method,seed,session):
    rng=np.random.default_rng(2026100202+session*100000+seed+(50000 if pde=='heat' else 0));blocks={k:[] for k in KEYS};roles=[];indices=[];hashes={}
    for ri,(role,n) in enumerate(zip(ROLES,[3000,1000,1000])):
        path=OLD/f'evidence/locked_test_raw_v3/{role}/{pde}/{method}_s{seed}.npz';hashes[str(path.relative_to(OLD))]=sha(path)
        with np.load(path,allow_pickle=False) as z:
            pool=z['latency_sensor_value'].shape[0]*20;ix=rng.permutation(pool)[:n]
            for k in KEYS:
                a=z['latency_'+k];blocks[k].append(a.reshape((-1,)+a.shape[2:])[ix])
        roles.extend([ri]*n);indices.extend(ix)
    order=rng.permutation(N)
    return {k:np.concatenate(v)[order] for k,v in blocks.items()},np.array(roles,np.int8)[order],np.array(indices,np.int32)[order],hashes

@torch.no_grad()
def request(replay,i,pde,K,m,profile=False):
    stages={};hooks=[]
    def mark(name,start):
        if profile:torch.cuda.synchronize();stages[name]=(time.perf_counter_ns()-start)/1e6
    t=time.perf_counter_ns();obs={k:replay[k][i] for k in KEYS};prev=obs['previous_applied_action'];nom=nominal_lqr_action(obs,pde,prev);candidates=expanded_candidates(pde,prev,nom,K);mark('request_preparation',t)
    t=time.perf_counter_ns();td={k:torch.as_tensor(v,device=dev) for k,v in obs.items() if k!='goal_field'};td={k:(v.unsqueeze(0) if v.ndim else v.reshape(1)) for k,v in td.items()};acts=torch.as_tensor(candidates,dtype=torch.float32,device=dev).unsqueeze(0);x,tau=_grid(pde,dev);mark('H2D_and_grid',t)
    module=m.model if isinstance(m,CachedOperator) else m
    if profile:
        for name,layer in [('encoder',module.encoder),('branch',module.response_branch if hasattr(module,'response_branch') else module.action_branch),('trunk',module.trunk)]:
            clock={}
            def pre(_module,_input,key=name,timer=clock):torch.cuda.synchronize();timer['t']=time.perf_counter_ns()
            def post(_module,_input,_output,key=name,timer=clock):torch.cuda.synchronize();stages[key]=(time.perf_counter_ns()-timer['t'])/1e6
            hooks.extend([layer.register_forward_pre_hook(pre),layer.register_forward_hook(post)])
    t=time.perf_counter_ns();pred=m(td,acts,x,tau);mark('model_total',t)
    for hook in hooks:hook.remove()
    if profile:
        stages.setdefault('trunk',0.);stages['field_assembly_and_other']=stages.pop('model_total')-sum(stages.get(k,0) for k in ['encoder','branch','trunk'])
    t=time.perf_counter_ns();goal=torch.as_tensor(restrict_field(obs['goal_field'],pde),dtype=torch.float32,device=dev).unsqueeze(0);cost=candidate_cost(pred,goal,acts,td['previous_applied_action'].float(),1.2 if pde=='burgers' else 2.1322593092918396,q_min=0. if pde=='heat' else None);selected=cost.argmin(1);mark('scoring_including_goal_transfer',t)
    t=time.perf_counter_ns();j=int(selected.item());unused_selected_forecast=pred[0,j].cpu().numpy();proposed=candidates[j];mark('D2H_selected_index_and_forecast',t)
    t=time.perf_counter_ns();lo,hi,sl=(-1,1,.15) if pde=='burgers' else (0,1,.1);applied=project_box_slew(proposed,prev,lo,hi,sl);mark('projection',t)
    t=time.perf_counter_ns();assert applied.shape==(2,) and np.isfinite(applied).all();mark('verification',t)
    return stages

def test_equivalence(replay,pde,method,seed,K,model,cached):
    with torch.no_grad():
        x,tau=_grid(pde,dev);error=0.
        for i in range(4):
            obs={k:replay[k][i] for k in KEYS};prev=obs['previous_applied_action'];ca=expanded_candidates(pde,prev,nominal_lqr_action(obs,pde,prev),K);td={k:torch.as_tensor(v,device=dev) for k,v in obs.items() if k!='goal_field'};td={k:(v.unsqueeze(0) if v.ndim else v.reshape(1)) for k,v in td.items()};acts=torch.as_tensor(ca,device=dev).unsqueeze(0)
            error=max(error,float((model(td,acts,x,tau)-cached(td,acts,x,tau)).abs().max()))
        assert error<=1e-6,(pde,method,seed,K,error)
        return {'pde':pde,'method':method,'seed':seed,'K':K,'max_abs_error':error}

for session in range(1,4):
    variants=[(pde,method,seed,K,cache) for pde in ['burgers','heat'] for method in ['P','B4'] for seed in [11,23,37] for K in [10,25,50,100,200] for cache in [False,True]]
    random.Random(2026100202+session).shuffle(variants)
    for pde,method,seed,K,cache in variants:
        if (R/'state/STOP_REVISION').exists():raise SystemExit('Stop at timing condition boundary')
        label=f'session{session}_{pde}_{method}_s{seed}_K{K}_cache{int(cache)}';out=R/f'results/E2_raw/{label}.npz';out.parent.mkdir(parents=True,exist_ok=True)
        if out.exists():
            row=json.loads(out.with_suffix('.json').read_text());assert row['sha256']==sha(out);rows.append(row);continue
        replay,role_index,source_index,inputhashes=sample(pde,method,seed,session);checkpoint=OLD/f'runs/confirmatory_v3_training/{pde}/{method}_s{seed}.pt';model=load_controller(method,pde,seed,checkpoint,dev)
        x,tau=_grid(pde,dev);torch.cuda.synchronize();ct=time.perf_counter();cached=CachedOperator(model,x,tau);torch.cuda.synchronize();cache_init=(time.perf_counter()-ct)*1000
        check=test_equivalence(replay,pde,method,seed,K,model,cached);equivalence.append(check);selected=cached if cache else model
        for i in range(WARMUP):request(replay,i,pde,K,selected)
        times=np.empty(N);stream=np.empty(N)
        for i in range(N):
            torch.cuda.synchronize();ts=time.perf_counter_ns();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();request(replay,i,pde,K,selected);b.record();torch.cuda.synchronize();stream[i]=a.elapsed_time(b);times[i]=(time.perf_counter_ns()-ts)/1e6
        np.savez_compressed(out,latency_ms=times,cuda_stream_span_ms=stream,source_role=role_index,source_index=source_index)
        row={'session':session,'pde':pde,'method':method,'seed':seed,'K':K,'cache':cache,'requests':N,'warmup':WARMUP,'p50_ms':float(np.quantile(times,.5)),'p99_ms':float(np.quantile(times,.99)),'cache_init_ms':cache_init,'sha256':sha(out),'checkpoint_sha256':sha(checkpoint),'input_sha256':inputhashes,'equivalence':check}
        out.with_suffix('.json').write_text(json.dumps(row,indent=2));rows.append(row)
        if session==1 and seed==11 and K in [10,200]:
            prof=[request(replay,i,pde,K,selected,profile=True) for i in range(500)];pr=R/f'results/E2_profiles/{label}.json';pr.parent.mkdir(parents=True,exist_ok=True);pr.write_text(json.dumps({'stage_ms':prof,'profiling':'separately synchronized instrumentation; overhead changes execution and stage quantiles must not be added','requests':500,'pde':pde,'method':method,'K':K,'cache':cache},indent=2))
        (R/'state/E2_PROGRESS.json').write_text(json.dumps({'completed':len(rows),'expected':len(variants)*3,'elapsed_s':time.perf_counter()-started,'last':{k:row[k] for k in ['pde','method','K','cache','p99_ms']}},indent=2));print(json.dumps({'completed':len(rows),'label':label,'p99_ms':row['p99_ms'],'elapsed_s':time.perf_counter()-started}),flush=True)
        del selected,cached,model,replay;gc.collect();torch.cuda.empty_cache()
(R/'results/E2_TIMING.json').write_text(json.dumps({'status':'COMPLETE','rows':rows,'expected_conditions':360,'elapsed_s':time.perf_counter()-started,'requests_per_condition':N,'boundary':'stored observation to validated host action including transfer of selected forecast as in original implementation; one-time cache setup excluded','platform':'Windows WDDM; desktop applications remain open','equivalence':equivalence},indent=2))
