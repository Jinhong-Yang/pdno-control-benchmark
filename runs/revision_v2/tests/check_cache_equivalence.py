from pathlib import Path
import os,sys,json,time
os.environ['CUDA_VISIBLE_DEVICES']='-1'
R=Path(__file__).resolve().parents[1];OLD=R.parents[1]/'experiments/pdno_jevLite_20260927_v3';sys.path.insert(0,str(R/'src'))
import torch,numpy as np
torch.set_num_threads(2)
from pdno.models.revision_cache import CachedOperator,expanded_candidates
from pdno.evaluation.closed_loop import load_controller
from pdno.training.fit import _grid,_obs
from pdno.controllers.linear import nominal_lqr_action
rows=[]
with torch.no_grad():
 for pde in ['burgers','heat']:
  with np.load(OLD/f'data/queries_v3/train/{pde}/teacher_queries.npz',allow_pickle=False) as z:data={k:torch.as_tensor(z[k][:4]) for k in z.files if k!='parent_id'}
  obs=_obs(data,torch.arange(4));x,tau=_grid(pde,torch.device('cpu'))
  for method in ['P','B4']:
   m=load_controller(method,pde,11,OLD/f'runs/confirmatory_v3_training/{pde}/{method}_s11.pt',torch.device('cpu'));c=CachedOperator(m,x,tau)
   for K in [10,25,50,100,200]:
    actions=torch.stack([torch.as_tensor(expanded_candidates(pde,data['previous_applied_action'][i].numpy(),data['candidate_action'][i,-1].numpy(),K)) for i in range(4)])
    a=m(obs,actions,x,tau);b=c(obs,actions,x,tau);err=float((a-b).abs().max());assert err<=1e-6,err
    rows.append({'pde':pde,'method':method,'K':K,'observations':4,'max_absolute_error':err})
(R/'results/E2_cache_CPU_tests.json').write_text(json.dumps({'status':'PASS','device':'cpu','cases':rows,'cuda_test_required_before_timing':True,'not_a_latency_benchmark':True},indent=2));print('20 fixed-grid cache cases PASS, maximum error',max(x['max_absolute_error'] for x in rows))
