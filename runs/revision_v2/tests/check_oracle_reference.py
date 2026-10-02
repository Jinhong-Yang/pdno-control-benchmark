from pathlib import Path
import os,sys,json,time
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from pdno.controllers.revision_oracle import future_candidates
from pdno.data.teacher_queries import _future_burgers,_future_heat,feasible_candidates
from pdno.data.generate import _burgers_initial,_heat_initial
from pdno.controllers.actions import project_box_slew
rng=np.random.default_rng(20261002);checks=[]
for pde in ['burgers','heat']:
 q=(_burgers_initial if pde=='burgers' else _heat_initial)(rng,256);m=np.array([.02,0] if pde=='burgers' else [.012,.3]);prev=np.zeros(2) if pde=='burgers' else np.full(2,.4);a=feasible_candidates(pde,prev,prev)
 start=time.perf_counter();batch=future_candidates(q,pde,m,a);elapsed=time.perf_counter()-start
 ref=np.stack([_future_burgers(q,m[0],v) if pde=='burgers' else _future_heat(q,m[0],m[1],v) for v in a]);error=float(np.max(np.abs(batch-ref)));assert error<=1e-6
 checks.append({'pde':pde,'K':10,'max_abs_error':error,'reference_equivalence':True,'candidate_rollout_cpu_seconds':elapsed,'warning':'single-call smoke under other-workload contention, not benchmark timing'})
(R/'results/E1_reference_equivalence.json').write_text(json.dumps({'status':'PASS','checks':checks},indent=2));print(json.dumps(checks))
