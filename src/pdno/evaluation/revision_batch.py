"""Batched learned-controller evaluation; scientific outputs, never latency samples."""
import numpy as np
import torch
from functools import lru_cache
from pdno.evaluation import closed_loop as cl
from pdno.controllers.actions import project_box_slew
from pdno.controllers.linear import nominal_lqr_action
from pdno.controllers.objectives import candidate_cost
from pdno.data.teacher_queries import feasible_candidates,restrict_field
from pdno.training.fit import _grid
from pdno.controllers.revision_oracle_cuda import BurgersCandidateCUDA

KEYS={'sensor_value','sensor_mask','sensor_age','instrument_image','image_mask','goal_coefficients','material_context','previous_applied_action','applied_action_history','image_age','image_valid'}
_original_goal_coeff=cl._goal_coeff
@lru_cache(maxsize=2048)
def _goal_cached(raw,pde):return _original_goal_coeff(np.frombuffer(raw,dtype=np.float32),pde)
cl._goal_coeff=lambda goal,pde:_goal_cached(np.asarray(goal,dtype=np.float32).tobytes(),pde)

@torch.no_grad()
def batch_actions(observations,pde,method,model,qmax):
    prev=np.stack([o['previous_applied_action'] for o in observations])
    nominal=np.stack([nominal_lqr_action(o,pde,p) for o,p in zip(observations,prev)])
    if method=='B0':return nominal,np.full(len(prev),-1,np.int16)
    if method=='B1':
        out=[cl.policy_action(method,pde,o,None,torch.device('cpu'),qmax) for o in observations]
        return np.stack([v[0] for v in out]),np.array([v[2] for v in out],np.int16)
    obs={k:torch.as_tensor(np.stack([o[k] for o in observations]),device='cuda') for k in KEYS}
    if method in {'B2','B3'}:return model(obs).cpu().numpy(),np.full(len(prev),-1,np.int16)
    candidates=np.stack([feasible_candidates(pde,p,n) for p,n in zip(prev,nominal)])
    actions=torch.as_tensor(candidates,device='cuda');x,tau=_grid(pde,torch.device('cuda'));pred=model(obs,actions,x,tau)
    goal=torch.as_tensor(np.stack([restrict_field(o['goal_field'],pde) for o in observations]),device='cuda')
    cost=candidate_cost(pred,goal,actions,obs['previous_applied_action'].float(),qmax,q_min=0. if pde=='heat' else None)
    choice=cost.argmin(1).cpu().numpy();return candidates[np.arange(len(prev)),choice],choice.astype(np.int16)

def run_batch(data,indices,pde,method,model,qmax,ticks=200):
    ids=np.asarray(indices);B=len(ids);pid=data['parent_id'][ids];states=data['state_true'][ids,0].astype(np.float32).copy()
    materials=[np.array([float(data['nu'][p]),-1,1,.15,.02,.16,1.2,128],np.float32) if pde=='burgers' else np.array([float(data['kappa'][p]),float(data['decay'][p]),0,1,.1,.02,.16,128],np.float32) for p in ids]
    goals=data['goal'][ids].copy() if pde=='heat' else np.zeros((B,256),np.float32)
    goals_after=data.get('goal_step_field',data.get('goal'))[ids].copy() if pde=='heat' else goals
    step_ticks=data.get('goal_step_tick',np.full(len(data['parent_id']),-1))[ids] if pde=='heat' else np.full(B,-1)
    schedules=[cl._schedule(str(p),ticks,str(data['role'][j])=='locked_delay_dropout') for p,j in zip(pid,ids)]
    sensors=[np.zeros((ticks,16),np.float32) for _ in ids];images=[[None]*ticks for _ in ids];actions=[[] for _ in ids]
    hist=np.empty((B,ticks+1,256),np.float32);hist[:,0]=states;ah=np.empty((B,ticks,2),np.float32);gh=np.empty((B,ticks,256),np.float32);ch=np.full((B,ticks),-1,np.int16)
    terms=np.zeros((B,4),float);sq=np.zeros(B);vany=np.zeros(B,bool);vtol=np.zeros(B,bool);duration=np.zeros(B);vmax=np.zeros(B);integral=np.zeros(B);causal=np.zeros(B,int)
    pulses=np.zeros((B,256));x=np.arange(256)/256
    for b,p in enumerate(ids):
        ds=int(data['disturbance_seed'][p])
        if ds%4==0:
            center=np.random.default_rng(ds).random();d=np.minimum(abs(x-center),1-abs(x-center));pulse=.05*np.exp(-.5*(d/.025)**2);pulse-=pulse.mean();pulses[b]=pulse
    truth=BurgersCandidateCUDA(np.array([m[0] for m in materials],float),K=1) if pde=='burgers' else None
    for tick in range(ticks):
        observations=[];targets=np.where(((step_ticks>=0)&(tick>=step_ticks))[:,None],goals_after,goals)
        for b in range(B):
            sc=schedules[b];sensors[b][tick]=cl._sensor_at(states[b],pde,sc['sensor_noise'][tick])
            if not sc['image_drop'][tick]:images[b][tick]=cl._image_at(states[b],pde,sc,tick)
            obs,prev=cl._obs_for_tick(actions[b],tick,pde,materials[b],targets[b],sc,sensors[b],images[b]);observations.append(obs)
            causal[b]+=int(np.sum(obs['sensor_mask'] & (obs['sensor_receive_time']>tick)))+int(obs['image_valid'] and obs['image_receive_time']>tick)
        previous=np.stack([o['previous_applied_action'] for o in observations]);proposed,choices=batch_actions(observations,pde,method,model,qmax)
        lo,hi,sl=(-1,1,.15) if pde=='burgers' else (0,1,.1)
        applied=np.stack([project_box_slew(a,p,lo,hi,sl) for a,p in zip(proposed,previous)])
        assert np.isfinite(applied).all() and np.max(abs(applied-previous))<=sl+1e-6
        assert np.min(applied)>=lo-1e-6 and np.max(applied)<=hi+1e-6
        nxt=truth.advance_one_tick(states,applied,pulses if tick==100 else None) if pde=='burgers' else np.stack([cl._advance(states[b],pde,applied[b],materials[b],tick,str(pid[b]),None) for b in range(B)])
        assert np.isfinite(nxt).all()
        error=(nxt-targets).astype(float);v=np.maximum(abs(nxt)-qmax,0) if pde=='burgers' else np.maximum(-nxt,0)+np.maximum(nxt-qmax,0)
        terms[:,0]+=np.mean(error**2,axis=1);terms[:,1]+=.01*np.sum(applied**2,axis=1);terms[:,2]+=.05*np.sum((applied-previous)**2,axis=1);terms[:,3]+=10*np.mean(v**2,axis=1)
        sq+=np.sum(error**2,axis=1);peak=np.max(v,axis=1);vany|=peak>0;vtol|=peak>1e-7;duration+=(peak>0)*.02;vmax=np.maximum(vmax,peak);integral+=np.mean(v,axis=1)*.02
        for b in range(B):actions[b].append(applied[b].copy())
        states=nxt;hist[:,tick+1]=states;ah[:,tick]=applied;gh[:,tick]=targets;ch[:,tick]=choices
    assert not np.any(causal)
    return {'parent_id':pid,'state_true':hist,'action_applied':ah,'goal':gh,'selected_candidate_index':ch,'episode_control_cost':terms.sum(1),'cost_components':terms,'tracking_rmse':np.sqrt(sq/(ticks*256)),'truth_constraint_violation':vany,'violation_tolerance_1e7':vtol,'violation_duration_seconds':duration,'violation_max':vmax,'violation_integrated_mean':integral,'fallback_count':np.zeros(B,int),'causal_timestamp_violations':causal}
