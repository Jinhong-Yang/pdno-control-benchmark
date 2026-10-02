from pathlib import Path
import os,sys,json,csv,time,hashlib,math
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import numpy as np
import torch
from torch.nn import functional as F
R=Path(__file__).resolve().parents[1]; ROOT=R.parents[1]; OLD=ROOT/'experiments/pdno_jevLite_20260927_v3';sys.path.insert(0,str(R/'src'))
from pdno.models.operators import ActionFactorizedOperator
from pdno.training.fit import _load,_obs,_grid
from pdno.training.losses import observation_consistency_loss,pde_residual_loss,boundary_condition_diagnostic,ranking_kl
from pdno.controllers.objectives import candidate_cost
torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=[];reports=[];started=time.perf_counter()
for pde in ['burgers','heat']:
    seed=11;torch.manual_seed(seed);device=torch.device('cpu')
    trainpath=OLD/f'data/queries_v3/train/{pde}/teacher_queries.npz';data=_load(trainpath,device);x,tau=_grid(pde,device)
    models={};optim={}; paths={}
    lr=1e-3*(1+math.cos(math.pi*2000/5000))/2
    for m in ['P','B5']:
        p=OLD/f'runs/confirmatory_v3_candidates/{pde}/{m}_s11/update_02000.pt'; saved=torch.load(p,map_location='cpu',weights_only=False);paths[m]={'path':str(p),'sha256':sha(p),'keys':list(saved)}
        model=ActionFactorizedOperator(pde,33 if pde=='burgers' else 32);model.load_state_dict(saved['state_dict'])
        for par in list(model.encoder.parameters())+list(model.initial_head.parameters()):par.requires_grad_(False)
        model.train();models[m]=model;optim[m]=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=lr,weight_decay=1e-4)
    pars={m:[p for p in v.parameters() if p.requires_grad] for m,v in models.items()}
    initial_diff=float(torch.sqrt(sum((a-b).double().square().sum() for a,b in zip(pars['P'],pars['B5']))))
    qscale=max(float(data['initial_field'].float().square().mean().sqrt()),1e-3);mat=data['material_context'].float()
    rs=max(qscale/.16,qscale**2,float(mat[:,0].max())*(32*np.pi)**2*qscale,1e-3) if pde=='burgers' else max(qscale/.16,float(mat[:,0].max())*(32*np.pi)**2*qscale,float(mat[:,1].max())*qscale,1e-3)
    iqr=float(torch.quantile(data['teacher_cost'].float().flatten(),.75)-torch.quantile(data['teacher_cost'].float().flatten(),.25));temp=max(.1*iqr,1e-4)
    def gradnorm(term, parameters):
        if not term.requires_grad:return 0.0
        g=torch.autograd.grad(term,parameters,retain_graph=True,allow_unused=True)
        return float(torch.sqrt(sum(v.detach().double().square().sum() for v in g if v is not None)))
    for step in range(1,101):
        torch.manual_seed(20261002+step);ids=torch.randint(len(data['candidate_action']),(64,));obs=_obs(data,ids);acts=data['candidate_action'][ids].float();target=data['future_field'][ids].float()
        ri=torch.randint(acts.shape[1],(len(ids),));sampled=acts[torch.arange(len(ids)),ri][:,None,:];cx=torch.rand(len(ids),32);ct=torch.rand(len(ids),32).clamp_min(.02)
        for m,model in models.items():
            torch.manual_seed(917000+step);optim[m].zero_grad(set_to_none=True)
            pred=model(obs,acts,x,tau);field=F.mse_loss(pred,target)/qscale**2
            ob1,ob2=observation_consistency_loss(model,obs,data['initial_field'][ids].float(),qscale)
            costs=candidate_cost(pred,data['goal'][ids].float(),acts,obs['previous_applied_action'].float(),float(data['q_max'].max()),q_min=0. if pde=='heat' else None)
            rank=.1*ranking_kl(costs,data['teacher_cost'][ids].float(),temp)
            ph,ba,bc=pde_residual_loss(model,obs,sampled,cx,ct);ramp=step/600.;w=.01 if m=='P' else 0.
            physics=w*ramp*ph/rs**2;balance=w*ramp*ba/rs**2
            row={'pde':pde,'seed':seed,'method':m,'restart_step':step,'original_update':2000+step,'ramp':ramp,'residual_scale':rs,'data_grad':gradnorm(field,pars[m]),'physics_grad':gradnorm(physics,pars[m]),'balance_grad':gradnorm(balance,pars[m]),'ranking_grad':gradnorm(rank,pars[m]),'unweighted_physics_grad':gradnorm(ph,pars[m]),'unweighted_balance_grad':gradnorm(ba,pars[m]),'field_loss':float(field.detach()),'physics_weighted_loss':float(physics.detach()),'balance_weighted_loss':float(balance.detach()),'ranking_weighted_loss':float(rank.detach())}
            row['physics_data_gradient_ratio']=row['physics_grad']/max(row['data_grad'],1e-30)
            row['balance_data_gradient_ratio']=row['balance_grad']/max(row['data_grad'],1e-30)
            loss=field+ob1+ob2+bc+rank+physics+balance;loss.backward();torch.nn.utils.clip_grad_norm_(pars[m],1.0);optim[m].step()
            for group in optim[m].param_groups:group['lr']=1e-3*(1+math.cos(math.pi*(2000+step)/5000))/2
            rows.append(row)
        diff=float(torch.sqrt(sum((a-b).double().square().sum() for a,b in zip(pars['P'],pars['B5']))))
        for row in rows[-2:]:row['parameter_difference_l2']=diff
        if step%10==0: print(json.dumps({'pde':pde,'step':step,'ratio':rows[-2]['physics_data_gradient_ratio'],'parameter_delta':diff,'elapsed_s':time.perf_counter()-started}),flush=True)
    p_rows=[r for r in rows if r['pde']==pde and r['method']=='P'];ratios=[r['physics_data_gradient_ratio'] for r in p_rows]
    report={'pde':pde,'seed':seed,'updates':100,'initial_parameter_difference':initial_diff,'final_parameter_difference':diff,'physics_gradient_nonzero_updates':sum(r['physics_grad']>0 for r in p_rows),'median_weighted_physics_data_gradient_ratio':float(np.median(ratios)),'max_weighted_physics_data_gradient_ratio':max(ratios),'median_balance_data_gradient_ratio':float(np.median([r['balance_data_gradient_ratio'] for r in p_rows])),'classification':'G-B' if np.median(ratios)<1e-3 and all(r['unweighted_physics_grad']>0 for r in p_rows) else 'G-C' if all(r['unweighted_physics_grad']>0 for r in p_rows) else 'G-A_NEEDS_CAUSAL_CONFIRMATION','checkpoints':paths,'train_sha256':sha(trainpath)};reports.append(report)
with (R/'results/P0_1_gradient_trace.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
result={'status':'COMPLETE','device':'cpu','threads':2,'elapsed_seconds':time.perf_counter()-started,'restart_limitation':'Saved update-2000 checkpoints contain model weights and metadata only, no Adam moments or RNG state. Both optimizers were identically reinitialized at the scheduled update-2000 learning rate. This is a matched weight-restart diagnostic, not bitwise continuation of the original CUDA training.','training_data_only':True,'test_access':False,'reports':reports,'paths_findings':{'coordinate_detach':'only x/tau are detached then require gradients; prediction graph retains model parameters','create_graph':True,'observer_and_initial_head':'intentionally frozen for both methods; response branch and trunk remain trainable','autocast':'explicitly rejected; float casts retain autograd','physics_scaling':'both physics and balance divided by residual_scale squared then multiplied by 0.01 and phase ramp'}}
(R/'results/P0_1_gradient_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
