from pathlib import Path
import sys,json,time,gc
R=Path(__file__).resolve().parents[1];OLD=R.parents[1]/'experiments/pdno_jevLite_20260927_v3';sys.path.insert(0,str(R/'src'))
import numpy as np,torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
from pdno.evaluation.closed_loop import load_controller,run_episode
from pdno.evaluation.revision_batch import run_batch
rows=[];t=time.perf_counter()
for pde in ['burgers','heat']:
    with np.load(OLD/f'data/locked_v3/locked_nominal/{pde}/trajectories.npz',allow_pickle=False) as z:data={k:z[k][:2] for k in z.files}
    for method in ['B0','B1','P','B4','B2','B3']:
        model=load_controller(method,pde,11,OLD/f'runs/confirmatory_v3_training/{pde}/{method}_s11.pt',torch.device('cuda'))
        qm=1.2 if pde=='burgers' else 2.1322593092918396
        got=run_batch(data,[0,1],pde,method,model,qm)
        refs=[run_episode(data,p,pde,method,model,torch.device('cuda'),qm,query_tick=-1) for p in [0,1]]
        se=float(np.max(abs(got['state_true']-np.stack([r['state_true'] for r in refs]))));ae=float(np.max(abs(got['action_applied']-np.stack([r['action_applied'] for r in refs]))));ce=float(np.max(abs(got['episode_control_cost']-np.array([r['episode_control_cost'] for r in refs]))))
        # Direct neural policies may differ slightly with batch GEMM kernels.
        assert se<1e-4 and ae<1e-4 and ce<1e-3,(pde,method,se,ae,ce)
        rows.append({'pde':pde,'method':method,'parents':2,'ticks':200,'max_state_error':se,'max_action_error':ae,'max_cost_error':ce});print(rows[-1],flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
(R/'results/BATCH_ROLLOUT_TESTS.json').write_text(json.dumps({'status':'PASS','rows':rows,'elapsed_s':time.perf_counter()-t,'not_latency_benchmark':True},indent=2))
