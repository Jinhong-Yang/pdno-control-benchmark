"""Train-only component gradients at selected checkpoints, at full ramp, without updates."""
from pathlib import Path
import sys,json,csv,os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
R=Path(__file__).resolve().parents[1];OLD=R.parents[1]/'experiments/pdno_jevLite_20260927_v3';sys.path.insert(0,str(R/'src'))
import numpy as np,torch
from torch.nn import functional as F
torch.set_num_threads(2)
from pdno.models.operators import ActionFactorizedOperator
from pdno.training.fit import _load,_obs,_grid
from pdno.training.losses import pde_residual_loss,ranking_kl
from pdno.controllers.objectives import candidate_cost
rows=[]
for pde in ['burgers','heat']:
    paths=[OLD/f'runs/confirmatory_v3_training/{pde}/P_s11.pt']+sorted((R/'checkpoints/E4_cuda').glob(pde+'*.pt'))
    data=_load(OLD/f'data/queries_v3/train/{pde}/teacher_queries.npz',torch.device('cpu'));x,tau=_grid(pde,torch.device('cpu'));qscale=max(float(data['initial_field'].float().square().mean().sqrt()),1e-3)
    for path in paths:
        saved=torch.load(path,map_location='cpu',weights_only=False);meta=saved['metadata'];model=ActionFactorizedOperator(pde,33 if pde=='burgers' else 32);model.load_state_dict(saved['state_dict']);model.eval()
        for p in list(model.encoder.parameters())+list(model.initial_head.parameters()):p.requires_grad_(False)
        pars=[p for p in model.parameters() if p.requires_grad];rs=meta['residual_scale'];temp=meta['temperature']
        for index in range(20):
            torch.manual_seed(2026100240+index);ix=torch.randint(len(data['candidate_action']),(64,));obs=_obs(data,ix);acts=data['candidate_action'][ix].float();pred=model(obs,acts,x,tau);field=F.mse_loss(pred,data['future_field'][ix].float())/qscale**2
            chosen=acts[torch.arange(64),torch.randint(10,(64,))][:,None,:];px=torch.rand(64,32);pt=torch.rand(64,32).clamp_min(.02);ph,ba,_=pde_residual_loss(model,obs,chosen,px,pt)
            costs=candidate_cost(pred,data['goal'][ix].float(),acts,obs['previous_applied_action'].float(),float(data['q_max'].max()),q_min=0. if pde=='heat' else None);rank=ranking_kl(costs,data['teacher_cost'][ix].float(),temp)
            terms={'data':field,'physics':meta['lambda_phys']*ph/rs**2,'balance':meta['lambda_balance']*ba/rs**2,'ranking':meta['lambda_rank']*rank};norms={}
            for name,term in terms.items():
                grads=torch.autograd.grad(term,pars,retain_graph=True,allow_unused=True);norms[name]=float(torch.sqrt(sum(g.detach().double().square().sum() for g in grads if g is not None)))
            rows.append({'pde':pde,'checkpoint':path.stem,'batch':index,'physics_weight':meta['lambda_phys'],'ranking_weight':meta['lambda_rank'],'selected_update':meta['steps'],'ramp':1.,**{k+'_grad':v for k,v in norms.items()},'physics_data_ratio':norms['physics']/max(norms['data'],1e-30),'balance_data_ratio':norms['balance']/max(norms['data'],1e-30)})
with (R/'results/E4_selected_gradient_diagnostics.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(R/'results/E4_gradient_record.json').write_text(json.dumps({'status':'COMPLETE','rows':len(rows),'batches_per_checkpoint':20,'data':'train only, common seed batches across conditions','ramp':'full ramp 1 at validation-selected weights; not historical per-update training gradients','device':'cpu FP32','updates_performed':0},indent=2))
