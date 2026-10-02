from pathlib import Path
import sys,json,time,numpy as np,torch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'));torch.set_num_threads(1)
from pdno.controllers.revision_oracle_cuda import BurgersCandidateCUDA
from pdno.controllers.revision_oracle import future_candidates
from pdno.data.generate import _burgers_initial
from pdno.data.teacher_queries import feasible_candidates
rng=np.random.default_rng(1002);q=np.stack([_burgers_initial(rng,256) for _ in range(2)]);nu=np.array([.01,.03]);a=np.stack([feasible_candidates('burgers',np.zeros(2),np.zeros(2)) for _ in nu]);t=time.perf_counter()
solver=BurgersCandidateCUDA(nu);output=solver(q,a).cpu().numpy();ref=np.stack([future_candidates(q[i],'burgers',[nu[i]],a[i]) for i in range(2)]);err=float(np.max(np.abs(output-ref)));assert err<=1e-6,err
(R/'results/E1_cuda_reference_equivalence.json').write_text(json.dumps({'status':'PASS','max_abs_error':err,'pde':'burgers','parents':2,'candidate_count':10,'precision':'FP64 solver, FP32 restricted outputs matching original','support_elapsed_seconds':time.perf_counter()-t,'not_benchmark_latency':True},indent=2));print('CUDA reference-equivalence PASS',err,'seconds',time.perf_counter()-t)
