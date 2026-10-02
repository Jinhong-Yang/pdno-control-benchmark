"""Deterministic causal observations and shared-parent closed-loop rollouts."""
from __future__ import annotations

import hashlib
from functools import lru_cache
import time
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

from pdno.controllers.actions import project_box_slew
from pdno.controllers.linear import nominal_lqr_action
from pdno.controllers.objectives import candidate_cost
from pdno.controllers.rom import rom_candidate_costs
from pdno.data.generate import _dirichlet_actuators, _periodic_actuators
from pdno.data.teacher_queries import feasible_candidates, restrict_field, _future_burgers, _future_heat, _cost
from pdno.models.operators import ActionFactorizedOperator, CandidateConditionedOperator
from pdno.models.policies import DirectPolicy
from pdno.models.encoders import load_state_with_normalizer_defaults
from pdno.physics.burgers import burgers_imex_split_step, periodic_grid
from pdno.physics.heat import heat_cn_step
from pdno.training.fit import _grid


@lru_cache(maxsize=2048)
def _schedule(parent_id: str, ticks: int, stress: bool) -> dict[str, np.ndarray]:
    seed = int.from_bytes(hashlib.sha256(("observation|" + parent_id).encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    result={
        "sensor_keep": rng.random((ticks, 16)) >= (0.20 if stress else 0.03),
        "sensor_delay": rng.choice([0, 0, 0, 1], (ticks, 16)),
        "sensor_noise": rng.normal(0, 0.01, (ticks, 16)),
        "image_delay": rng.choice([0, 2, 4, 8] if stress else [0, 0, 1, 2], ticks),
        "image_drop": rng.random(ticks) < 0.03,
        "image_gain": rng.normal(1.0, 0.03, (ticks, 16, 1)),
        "image_noise": rng.normal(0, 0.01, (ticks, 16, 64)),
        "image_occlude": rng.random(ticks) < 0.1,
        "image_col": rng.integers(0, 56, ticks),
    }
    sensor_capture=np.full((ticks,16),-1,dtype=np.int16); sensor_receive=np.full((ticks,16),-1,dtype=np.int16)
    latest=np.full(16,-1,dtype=np.int16)
    for decision in range(ticks):
        if decision>0:
            arriving=(result["sensor_keep"][decision-1] & (result["sensor_delay"][decision-1]==1) & (decision-1>latest))
            latest[arriving]=decision-1
        immediate=result["sensor_keep"][decision] & (result["sensor_delay"][decision]==0)
        latest[immediate]=decision
        sensor_capture[decision]=latest
        valid=latest>=0
        sensor_receive[decision,valid]=latest[valid]+result["sensor_delay"][latest[valid],np.flatnonzero(valid)]
    image_capture=np.full(ticks,-1,dtype=np.int16); image_receive=np.full(ticks,-1,dtype=np.int16)
    image_events=[]
    for decision in range(ticks):
        if decision>=0 and not result["image_drop"][decision]:
            image_events.append((decision,decision+int(result["image_delay"][decision])))
        available=[event for event in image_events if event[0]<=decision and event[1]<=decision]
        if available:
            cap,rec=max(available,key=lambda e:(e[0],e[1])); image_capture[decision]=cap; image_receive[decision]=rec
    result.update({"sensor_capture":sensor_capture,"sensor_receive":sensor_receive,
                   "image_capture":image_capture,"image_receive":image_receive})
    return result


def _sensor_at(field: np.ndarray, pde: str, noise: np.ndarray) -> np.ndarray:
    n = field.size
    positions = np.linspace(0, n, 16, endpoint=False) if pde == "burgers" else np.linspace(0, n - 1, 16)
    if pde == "burgers":
        lo = np.floor(positions).astype(int); frac = positions - lo
        values = field[lo] * (1-frac) + field[(lo+1) % n] * frac
    else:
        values = np.interp(positions, np.arange(n), field)
    return values + noise


def _image_at(field: np.ndarray, pde: str, spec: dict, tick: int) -> tuple[np.ndarray, np.ndarray]:
    n = field.size
    if pde == "burgers":
        x_old, x_new, scale, mode = np.arange(n)/n, np.arange(64)/64, 1.5, "wrap"
    else:
        x_old, x_new, scale, mode = np.arange(1,n+1)/(n+1), np.arange(1,65)/65, 1.0, "nearest"
    line = gaussian_filter1d(np.interp(x_new,x_old,field),sigma=0.8,mode=mode)
    image = np.clip(0.5+0.5*line/scale,0,1)[None,:] * spec["image_gain"][tick]
    image = image + spec["image_noise"][tick]
    mask = np.ones((16,64),dtype=bool)
    if spec["image_occlude"][tick]:
        col=int(spec["image_col"][tick]); image[:,col:col+8]=0; mask[:,col:col+8]=False
    return image.astype(np.float32),mask


def _goal_coeff(goal: np.ndarray, pde: str) -> np.ndarray:
    if pde == "burgers": return np.zeros(33,dtype=np.float32)
    x=np.arange(1,len(goal)+1)/(len(goal)+1)
    basis=np.stack([np.sin(k*np.pi*x) for k in range(1,33)],axis=1)
    return np.linalg.lstsq(basis,goal,rcond=None)[0].astype(np.float32)


def _obs_for_tick(actions: list[np.ndarray], tick: int, pde: str, material: np.ndarray,
                  goal: np.ndarray, schedule: dict, sensor_values_by_capture: np.ndarray,
                  image_cache: list) -> tuple[dict, np.ndarray]:
    sensor_history=[]
    for decision in range(max(0,tick-7),tick+1):
        v=np.zeros(16,dtype=np.float32); m=np.zeros(16,dtype=bool); age=np.zeros(16,dtype=np.int16)
        cap=np.full(16,-1,dtype=np.int32); rec=np.full(16,-1,dtype=np.int32)
        for s in range(16):
            c=int(schedule["sensor_capture"][decision,s])
            if c>=0:
                v[s]=sensor_values_by_capture[c,s]
                m[s]=True; age[s]=decision-c; cap[s]=c; rec[s]=schedule["sensor_receive"][decision,s]
        sensor_history.append((v,m,age,cap,rec))
    pad=8-len(sensor_history)
    sv=np.zeros((8,16),np.float32); sm=np.zeros((8,16),bool); sa=np.zeros((8,16),np.int16)
    sc=np.full((8,16),-1,np.int32); sr=np.full((8,16),-1,np.int32)
    for j,row in enumerate(sensor_history):
        sv[pad+j],sm[pad+j],sa[pad+j],sc[pad+j],sr[pad+j]=row
    ic=int(schedule["image_capture"][tick]); ir=int(schedule["image_receive"][tick])
    if ic>=0:
        image,imask=image_cache[ic]; image=image.copy(); imask=imask.copy(); valid=True
        iage=tick-ic
    else:
        image=np.zeros((16,64),np.float32); imask=np.zeros((16,64),bool); ic=ir=-1; iage=0; valid=False
    previous=(np.zeros(2,np.float32) if pde=="burgers" else np.full(2,0.4,np.float32)) if tick==0 else actions[-1].astype(np.float32)
    hist=np.zeros((8,2),np.float32); hist[:, :]=previous if tick<8 and pde=="heat" else 0
    if actions:
        n=min(8,len(actions)); hist[-n:]=np.asarray(actions[-n:],np.float32)
    observation={"sensor_value":sv,"sensor_mask":sm,"sensor_age":sa,"sensor_capture_time":sc,"sensor_receive_time":sr,
        "instrument_image":image[None],"image_mask":imask[None],"image_age":np.int16(iage),"image_valid":np.bool_(valid),
        "image_capture_time":np.int32(ic),"image_receive_time":np.int32(ir),"previous_applied_action":previous,
        "applied_action_history":hist,"material_context":material.astype(np.float32),"goal_field":goal.astype(np.float32),
        "goal_coefficients":_goal_coeff(goal,pde)}
    return observation,previous


def load_controller(method: str,pde: str,seed: int|None,checkpoint: Path|None,device: torch.device):
    if method in {"P","P-no-rank","B5"}: model=ActionFactorizedOperator(pde,33 if pde=="burgers" else 32,rank=32)
    elif method=="B4": model=CandidateConditionedOperator(pde,33 if pde=="burgers" else 32,rank=32)
    elif method in {"B2","B3"}: model=DirectPolicy(33 if pde=="burgers" else 32,pde)
    else:return None
    saved=torch.load(checkpoint,map_location=device,weights_only=False)
    load_state_with_normalizer_defaults(model, saved["state_dict"]); model.to(device).eval()
    for par in model.parameters(): par.requires_grad_(False)
    return model


@torch.no_grad()
def policy_action(method: str,pde: str,obs_np: dict,model,device: torch.device,q_max: float, diagnostic_state=None):
    prev=obs_np["previous_applied_action"]
    nominal=nominal_lqr_action(obs_np,pde,prev)
    candidates=feasible_candidates(pde,prev,nominal)
    if method in {"O-cand-state", "O-cand-obs"}:
        from pdno.controllers.revision_oracle import future_candidates, observer_initial
        initial = diagnostic_state if method == "O-cand-state" else observer_initial(obs_np, pde)
        if initial is None: raise ValueError("Oracle state missing")
        fields = future_candidates(initial, pde, obs_np["material_context"], candidates)
        target = restrict_field(obs_np["goal_field"], pde)
        costs = [_cost(fields[j], target, candidates[j], prev, q_max, q_min=0. if pde=="heat" else None)[0] for j in range(len(candidates))]
        choice = int(np.argmin(costs))
        return candidates[choice], fields[choice], choice, candidates
    if method=="B0": return nominal,None,None,candidates
    if method=="B1":
        model_goal=obs_np["goal_field"][::2] if pde=="burgers" else restrict_field(obs_np["goal_field"],"heat")
        costs,fields=rom_candidate_costs(obs_np,pde,candidates,model_goal,q_max)
        k=int(np.argmin(costs)); return candidates[k],fields[k],k,candidates
    td={k:torch.as_tensor(v,device=device) for k,v in obs_np.items() if k in {"sensor_value","sensor_mask","sensor_age","instrument_image","image_mask","goal_coefficients","material_context","previous_applied_action","applied_action_history","image_age","image_valid"}}
    td={k:(v.unsqueeze(0) if v.ndim>0 else v.reshape(1)) for k,v in td.items()}
    if method in {"B2","B3"}:
        raw=model(td)[0].cpu().numpy(); return raw,None,None,candidates
    acts=torch.as_tensor(candidates,dtype=torch.float32,device=device).unsqueeze(0)
    x,tau=_grid(pde,device); predicted=model(td,acts,x,tau)
    goal=torch.as_tensor(obs_np["goal_field"][::2] if pde=="burgers" else restrict_field(obs_np["goal_field"],"heat"),dtype=torch.float32,device=device).unsqueeze(0)
    costs=candidate_cost(predicted,goal,acts,td["previous_applied_action"].float(),q_max,
                         q_min=0.0 if pde=="heat" else None)
    k=int(costs.argmin(dim=1).item()); return candidates[k],predicted[0,k].cpu().numpy(),k,candidates


def _advance(state,pde,action,material,tick,parent_meta,disturbance):
    n=state.size
    basis=_periodic_actuators(n) if pde=="burgers" else _dirichlet_actuators(n)
    if pde=="burgers":
        force=action@basis
        if disturbance is not None and tick==100: force=force+disturbance
        u=state.astype(np.float64).copy()
        for _ in range(80): u=burgers_imex_split_step(u,float(material[0]),2.5e-4,force)
        return u.astype(np.float32)
    return heat_cn_step(state.astype(np.float64),1.0/(n+1),0.02,float(material[0]),float(material[1]),action@basis).astype(np.float32)


def run_episode(data: dict[str,np.ndarray],parent: int,pde: str,method: str,model,device: torch.device,
                q_max: float,query_tick: int=100,ticks: int=200) -> dict:
    parent_id=str(data["parent_id"][parent]); stress=data["role"][parent]=="locked_delay_dropout"
    state=data["state_true"][parent,0].astype(np.float32).copy()
    material=np.asarray([float(data["nu"][parent]),-1,1,.15,.02,.16,1.2,128],np.float32) if pde=="burgers" else np.asarray([float(data["kappa"][parent]),float(data["decay"][parent]),0,1,.10,.02,.16,128],np.float32)
    goal=data["goal"][parent].astype(np.float32).copy() if pde=="heat" else np.zeros(256,np.float32)
    goal_after=data.get("goal_step_field",data.get("goal",np.zeros((len(data["parent_id"]),256),np.float32)))[parent].astype(np.float32).copy() if pde=="heat" else goal
    step_tick=int(data.get("goal_step_tick",np.full(len(data["parent_id"]),-1,dtype=np.int16))[parent]) if pde=="heat" else -1
    applied=np.full(2,.4,np.float64) if pde=="heat" else np.zeros(2,np.float64)
    states=[state.copy()]; actions=[]; goals=[]; proposals=[]; projected=[]
    obs_sensor_mask=[]; obs_sensor_age=[]; obs_sensor_capture=[]; obs_sensor_receive=[]
    obs_image_valid=[]; obs_image_age=[]; obs_image_capture=[]; obs_image_receive=[]; causal_bad=0
    replay=[]
    total_cost=0.0; sqerr=0.0; nerr=0; violation=False; fallback=0; fallback_errors=[]; predn=[]; regret=None; reg_choice=None
    controller_latency_ms=[]
    selected_candidate_history=[]
    d_rng=np.random.default_rng(int(data["disturbance_seed"][parent]))
    pulse=None
    if data["disturbance_seed"][parent]%4==0:
        x=periodic_grid(state.size); center=d_rng.random(); d=np.minimum(np.abs(x-center),1-np.abs(x-center))
        pulse=.05*np.exp(-.5*(d/.025)**2); pulse-=pulse.mean()
    schedule=_schedule(parent_id,ticks,stress)
    sensor_values_by_capture=np.zeros((ticks,16),np.float32); image_cache=[None]*ticks
    for tick in range(ticks):
        sensor_values_by_capture[tick]=_sensor_at(states[-1],pde,schedule["sensor_noise"][tick])
        if not schedule["image_drop"][tick]:
            image_value,image_mask=_image_at(states[-1],pde,schedule,tick)
            image_cache[tick]=(image_value,image_mask)
        target=goal_after if step_tick>=0 and tick>=step_tick else goal
        obs,previous=_obs_for_tick(actions,tick,pde,material,target,schedule,sensor_values_by_capture,image_cache)
        obs_sensor_mask.append(obs["sensor_mask"][-1].copy()); obs_sensor_age.append(obs["sensor_age"][-1].copy())
        obs_sensor_capture.append(obs["sensor_capture_time"][-1].copy()); obs_sensor_receive.append(obs["sensor_receive_time"][-1].copy())
        obs_image_valid.append(bool(obs["image_valid"])); obs_image_age.append(int(obs["image_age"]))
        obs_image_capture.append(int(obs["image_capture_time"])); obs_image_receive.append(int(obs["image_receive_time"]))
        causal_bad+=int(np.sum(obs["sensor_mask"][-1] & ((obs["sensor_capture_time"][-1]>tick)|(obs["sensor_receive_time"][-1]>tick))))
        causal_bad+=int(bool(obs["image_valid"]) and (int(obs["image_capture_time"])>tick or int(obs["image_receive_time"])>tick))
        predicted=None; choice=None; candidates=None
        try:
            controller_start=time.perf_counter()
            proposed,predicted,choice,candidates=policy_action(method,pde,obs,model,device,q_max, diagnostic_state=states[-1] if method=="O-cand-state" else None)
            controller_latency_ms.append((time.perf_counter()-controller_start)*1000.0)
            if not np.isfinite(proposed).all(): raise ValueError("nonfinite proposed action")
        except Exception as exc:
            fallback+=1
            fallback_errors.append(f"tick={tick}:{type(exc).__name__}:{exc}")
            proposed=nominal_lqr_action(obs,pde,previous) if method!="B0" else previous.copy()
            candidates=feasible_candidates(pde,previous,proposed)
        selected_candidate_history.append(-1 if choice is None else int(choice))
        if tick%10==0:
            replay.append({k:np.asarray(obs[k]).copy() for k in ("sensor_value","sensor_mask","sensor_age","instrument_image","image_mask","goal_coefficients","material_context","previous_applied_action","applied_action_history","image_age","image_valid","goal_field")})
            replay[-1]["candidate_action"]=np.asarray(candidates,np.float32).copy()
        low,high,slew=(-1.,1.,.15) if pde=="burgers" else (0.,1.,.10)
        action=project_box_slew(proposed,previous,low,high,slew)
        proposals.append(proposed); projected.append(action); actions.append(action.copy()); goals.append(target.copy())
        if tick==query_tick:
            initial=states[-1]
            best_cost=np.inf
            for j,cand in enumerate(candidates):
                future=_future_burgers(initial,float(material[0]),cand) if pde=="burgers" else _future_heat(initial,float(material[0]),float(material[1]),cand)
                c,_=_cost(future,target[::2] if pde=="burgers" else restrict_field(target,"heat"),cand,previous,q_max,
                          q_min=0.0 if pde=="heat" else None)
                if c<best_cost: best_cost=c; reg_choice=j
            chosen_future=_future_burgers(initial,float(material[0]),action) if pde=="burgers" else _future_heat(initial,float(material[0]),float(material[1]),action)
            selected_cost,_=_cost(chosen_future,target[::2] if pde=="burgers" else restrict_field(target,"heat"),action,previous,q_max,
                                  q_min=0.0 if pde=="heat" else None)
            regret=float(selected_cost-best_cost)
            if predicted is not None:
                truth=(_future_burgers(initial,float(material[0]),candidates[choice]) if pde=="burgers" else _future_heat(initial,float(material[0]),float(material[1]),candidates[choice]))
                predn.append(float(np.sqrt(np.mean((predicted-truth)**2)/(np.mean(truth**2)+1e-12))))
        nxt=_advance(states[-1],pde,action,material,tick,parent_id,pulse)
        states.append(nxt)
        err=nxt-target; sqerr+=float(np.sum(err.astype(np.float64)**2)); nerr+=err.size
        effort=.01*float(np.sum(action**2)); slew_cost=.05*float(np.sum((action-previous)**2))
        if pde == "heat":
            violation_cost=10*float(np.mean(np.maximum(-nxt,0)**2 + np.maximum(nxt-q_max,0)**2))
            violation |= bool(np.min(nxt) < 0.0 or np.max(nxt) > q_max)
        else:
            violation_cost=10*float(np.mean(np.maximum(np.abs(nxt)-q_max,0)**2))
            violation |= bool(np.max(np.abs(nxt)) > q_max)
        total_cost+=float(np.mean(err.astype(np.float64)**2))+effort+slew_cost+violation_cost
        applied=action
    result={"parent_id":parent_id,"episode_control_cost":total_cost,"tracking_rmse":float(np.sqrt(sqerr/nerr)),
        "truth_constraint_violation":bool(violation),"fallback_rate":fallback/ticks,"fallback_count":fallback,"fallback_errors":fallback_errors,
        "controller_latency_ms":np.asarray(controller_latency_ms,np.float64),
        "causal_timestamp_violations":causal_bad,"sensor_mask":np.asarray(obs_sensor_mask,bool),"sensor_age":np.asarray(obs_sensor_age,np.int16),
        "sensor_capture_time":np.asarray(obs_sensor_capture,np.int16),"sensor_receive_time":np.asarray(obs_sensor_receive,np.int16),
        "image_valid":np.asarray(obs_image_valid,bool),"image_age":np.asarray(obs_image_age,np.int16),
        "image_capture_time":np.asarray(obs_image_capture,np.int16),"image_receive_time":np.asarray(obs_image_receive,np.int16),
        "field_nrmse_at_snapshot":float(np.mean(predn)) if predn else np.nan,"candidate_regret_at_snapshot":regret if regret is not None else np.nan,
        "snapshot_candidate_best_index":reg_choice,"action_proposed":np.asarray(proposals,np.float32),"action_projected":np.asarray(projected,np.float32),
        "action_applied":np.asarray(actions,np.float32),"state_true":np.asarray(states,np.float32),"goal":np.asarray(goals,np.float32)}
    result["selected_candidate_index"]=np.asarray(selected_candidate_history,np.int16)
    result["latency_replay"]={k:np.stack([row[k] for row in replay]) for k in replay[0]} if replay else {}
    return result
