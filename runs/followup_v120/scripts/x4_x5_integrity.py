"""Read-only training-objective inspection and three specified source-integrity checks."""
from pathlib import Path
from datetime import datetime,timezone
import json,csv,hashlib,ast
import numpy as np
import torch
R=Path(__file__).resolve().parents[1];ROOT=R.parents[1];OLD=ROOT/'experiments/pdno_jevLite_20260927_v3';V=ROOT/'runs/revision_v2'
OUT=R/'results/X4_X5';OUT.mkdir(parents=True,exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
sources={}
def source(p):sources[p.relative_to(ROOT).as_posix()]=sha(p);return p
def csvrows(p):return list(csv.DictReader(source(p).open(encoding='utf-8-sig',newline='')))
code=source(V/'src/pdno/training/confirmatory.py');text=code.read_text();tree=ast.parse(text)
funcs={f.name:ast.get_source_segment(text,f) for f in tree.body if isinstance(f,ast.FunctionDef)}
assert 'F.mse_loss(model(obs), target)' in funcs['train_direct_staged']
assert 'teacher_best_index' in funcs['train_direct_staged']
assert 'loss = objective + 0.1 * bc_loss' in funcs['train_b3_staged']
assert 'surrogate.parameters(): parameter.requires_grad_(False)' in funcs['train_b3_staged']
query=source(V/'src/pdno/data/teacher_queries.py');qtext=query.read_text();assert '"teacher_best_index": int(np.argmin(costs))' in qtext
objective=source(V/'src/pdno/controllers/objectives.py')
x4=dict(status='PASS_IMPLEMENTATION_SOURCE_OBJECTIVES_DOCUMENTED',B2='Supervised action MSE against the minimum-reference-cost candidate action; the observation encoder is frozen. This is not direct imitation of B0.',B3='B2 initialization; optimize constant-action cost predicted through frozen B4 plus 0.1 action-MSE regularization to the same reference-solver candidate target; policy encoder frozen. Validation selects that combined surrogate-cost-plus-MSE score.',source_functions=[dict(function=f,line=next(n.lineno for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==f),source=code.relative_to(ROOT).as_posix()) for f in ['train_direct_staged','train_b3_staged']],limits='Actual executed staged-trainer definitions; no claim these training objectives equal realized closed-loop cost.')
ratios=csvrows(V/'results/E2_ratio_bootstrap.csv');dup=[]
for K in [100,200]:
 per=[]
 for seed in [11,23,37]:
  arrays={}
  for method in ['P','B4']:
   vals=[]
   for session in [1,2,3]:
    p=source(V/f'results/E2_raw/session{session}_burgers_{method}_s{seed}_K{K}_cache1.npz')
    with np.load(p,allow_pickle=False) as z:vals.append(z['latency_ms'])
   arrays[method]=np.concatenate(vals)
  per.append(float(np.quantile(arrays['P'],.99)/np.quantile(arrays['B4'],.99)))
 point=float(np.mean(per));row=next(r for r in ratios if r['pde']=='burgers' and int(r['K'])==K and r['cache']=='True')
 assert abs(point-float(row['mean_per_seed_p99_ratio']))<1e-14
 dup.append(dict(K=K,mean_seed_ratio=point,seed_ratios=per,formatted4=f'{point:.4f}'))
assert dup[0]['mean_seed_ratio']!=dup[1]['mean_seed_ratio'] and dup[0]['formatted4']==dup[1]['formatted4']=='0.7804'
training=csvrows(V/'results/REVISION_TRAINING_SELECTION.csv');weights=[]
for seed in [11,23,37]:
 selected={m:next(r for r in training if r['stage']=='E5' and r['method']==m and int(r['seed'])==seed and r['pde']=='heat') for m in ['P','B5']}
 paths={m:source(V/f'checkpoints/E5/{m}_s{seed}.pt') for m in ['P','B5']}
 hashes={m:sha(p) for m,p in paths.items()};assert hashes['P']!=hashes['B5']
 states={m:torch.load(p,map_location='cpu',weights_only=False)['state_dict'] for m,p in paths.items()};assert states['P'].keys()==states['B5'].keys()
 differences=[(states['P'][k].double()-states['B5'][k].double()) for k in states['P']]
 different=sum(not torch.equal(states['P'][k],states['B5'][k]) for k in states['P'])
 values={m:{k:float(selected[m][k]) for k in ['validation_field_nrmse','selection_score']} for m in selected}
 weights.append(dict(seed=seed,checkpoint_sha256=hashes,different_tensors=different,total_tensors=len(differences),max_abs_state_difference=max(float(d.abs().max()) for d in differences),state_difference_l2=float(torch.sqrt(sum(d.square().sum() for d in differences))),metrics=values,formatted6={m:{k:f'{v:.6f}' for k,v in v.items()} for m,v in values.items()}))
# The archived baseline cell explicitly reuses primary P seed11. Reconstruct from arrays.
means=[]
for seed in [11,23,37]:
 p=source(OLD/f'evidence/locked_test_raw_v3/locked_nominal/heat/P_s{seed}.npz')
 with np.load(p,allow_pickle=False) as z:means.append(float(z['episode_control_cost'].mean()))
p=source(OLD/'evidence/locked_test_raw_v3/locked_nominal/heat/B0_sna.npz')
with np.load(p,allow_pickle=False) as z:b0=float(z['episode_control_cost'].mean())
summary=csvrows(V/'results/REVISION_COST_SUMMARY.csv');baseline=next(r for r in summary if r['stage']=='E4' and r['pde']=='heat' and r['role']=='locked_nominal' and r['method']=='heat_phys0.01_rank0.1_s11')
single=means[0]/b0-1;pooled=float(np.mean(means))/b0-1
assert abs(single-float(baseline['relative_excess']))<1e-12
assert baseline['source_kind']=='original_P_seed11_baseline'
x5=dict(status='PASS_SPECIFIED_SOURCE_INTEGRITY_CHECKS',rounded_timing_ratios=dup,ratio_interpretation='Distinct raw-derived point estimates round to the same four-decimal display; no copied value detected.',NH_checkpoint_comparison=weights,NH_metric_interpretation='Checkpoint hashes and tensors differ. Inspect per-seed formatted values rather than assuming all composite/nRMSE pairs agree to six decimals.',SH_weight_baseline=dict(seed11_mean_cost=means[0],three_seed_means=means,three_seed_mean_cost=float(np.mean(means)),B0_mean_cost=b0,seed11_excess=single,three_seed_excess=pooled,baseline_source_kind=baseline['source_kind']),weight_interpretation='The0.01physics/0.1ranking reference cell is primary P seed11, not a new fit. Its14.07percent and three-seed13.70percent use different seed aggregation. Other weight cells remain genuine single-seed sensitivity results.')
receipt=dict(utc=datetime.now(timezone.utc).isoformat(),X4=x4,X5=x5,input_sha256=sources,script_sha256=sha(Path(__file__)),limits='Read-only arithmetic/source review; no training, GPU timing or primary-endpoint changes.')
(OUT/'X4_X5_REPORT.json').write_text(json.dumps(receipt,indent=2)+'\n')
for name,rows in [('X5_timing_rounding.csv',dup),('X5_NH_checkpoint_metrics.csv',[dict(seed=r['seed'],method=m,checkpoint_sha256=r['checkpoint_sha256'][m],different_tensors=r['different_tensors'],**r['metrics'][m]) for r in weights for m in ['P','B5']])]:
 with (OUT/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps(dict(X4=x4['status'],X5=x5['status'],ratios=dup,weights=weights,seed11_excess=single,three_seed_excess=pooled)))
