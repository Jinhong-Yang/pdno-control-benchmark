"""Paired-parent, seed-conditional summaries of completed revision outputs."""
from pathlib import Path
import json,csv,hashlib
import numpy as np
R=Path(__file__).resolve().parents[1];OLD=R.parents[1]/'experiments/pdno_jevLite_20260927_v3';rows=[];agreements=[];inputhash={}
def load(path):
    inputhash[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files if k in ['parent_id','episode_control_cost','tracking_rmse','truth_constraint_violation','violation_tolerance_1e7','cost_components','violation_duration_seconds','violation_max','violation_integrated_mean','action_applied']}
def interval(cost,base,seed):
    rng=np.random.default_rng(seed);ix=rng.integers(0,len(base),(10000,len(base)));d=cost-base;den=max(float(base.mean()),1e-12);num=d[ix].mean(1)
    return (float(d.mean()/den),*np.quantile(num/den,[.025,.975]).tolist(),*np.quantile(num/np.maximum(base[ix].mean(1),1e-12),[.025,.975]).tolist())
for stage in ['E1','E3','E4','E5']:
    root=R/f'results/{stage}_raw'
    if not root.exists():continue
    for role in ['locked_nominal','locked_coefficient_ood','locked_delay_dropout']:
        for pde in ['burgers','heat']:
            paths=sorted((root/role/pde).glob('*.npz'))
            if not paths:continue
            basepath=root/role/pde/'B0_sna.npz' if stage=='E5' else OLD/f'evidence/locked_test_raw_v3/{role}/{pde}/B0_sna.npz'
            if not basepath.exists():continue
            baseline=load(basepath);groups={}
            for path in paths:
                if stage=='E5':label=path.stem.rsplit('_s',1)[0]
                elif stage=='E3':label=path.stem.rsplit('_s',1)[0]
                else:label=path.stem
                groups.setdefault(label,[]).append(path)
            if stage=='E4':
                # The sixth grid cell is the preserved original P seed11 run.
                # It is summarized here, not rerun or counted as a new evaluation.
                groups[f'{pde}_phys0.01_rank0.1_s11']=[OLD/f'evidence/locked_test_raw_v3/{role}/{pde}/P_s11.npz']
            for label,gpaths in groups.items():
                arrays=[load(p) for p in gpaths]
                for a in arrays:assert np.array_equal(a['parent_id'],baseline['parent_id'])
                costs=np.mean([a['episode_control_cost'] for a in arrays],axis=0);values=interval(costs,baseline['episode_control_cost'],2026100270)
                row={'stage':stage,'role':role,'pde':pde,'method':label,'parents':len(costs),'training_seed_count':len(arrays) if stage in ['E3','E5'] and label not in ['B0','B1'] else (1 if stage=='E4' else 0),'mean_cost':float(costs.mean()),'B0_mean_cost':float(baseline['episode_control_cost'].mean()),'relative_excess':values[0],'fixed_ci_low':values[1],'fixed_ci_high':values[2],'joint_ci_low':values[3],'joint_ci_high':values[4],'tracking_rmse_mean':float(np.mean([a['tracking_rmse'].mean() for a in arrays])),'any_violation_rate':float(np.mean([a['truth_constraint_violation'].mean() for a in arrays]))}
                for k in ['violation_duration_seconds','violation_max','violation_integrated_mean']:
                    row[k]=float(np.mean([a[k].mean() for a in arrays])) if k in arrays[0] else None
                for j,k in enumerate(['tracking_cost','action_cost','slew_cost','violation_cost']):row[k]=float(np.mean([a['cost_components'][:,j].mean() for a in arrays])) if 'cost_components' in arrays[0] else None
                rows.append(row)
                row['source_kind']='original_P_seed11_baseline' if stage=='E4' and all(p.is_relative_to(OLD) for p in gpaths) else 'revision_execution'
                row['violation_rate_tolerance_1e7']=float(np.mean([a['violation_tolerance_1e7'].mean() for a in arrays])) if 'violation_tolerance_1e7' in arrays[0] else None
                if stage=='E4':
                    orig=load(OLD/f'evidence/locked_test_raw_v3/{role}/{pde}/P_s11.npz');a=arrays[0]['action_applied'];b=orig['action_applied'];assert a.shape==b.shape
                    agreements.append({'role':role,'pde':pde,'condition':label,'parents':len(a),'exact_history_agreement':float(np.mean(np.all(a==b,axis=(1,2)))),'history_agreement_at_1e7':float(np.mean(np.all(abs(a-b)<=1e-7,axis=(1,2)))),'exact_action_agreement':float(np.mean(np.all(a==b,axis=2)))})
for name,data in [('REVISION_COST_SUMMARY',rows),('E4_ACTION_AGREEMENT',agreements)]:
    if data:
        with (R/f'results/{name}.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
(R/'results/REVISION_SUMMARY_RECEIPT.json').write_text(json.dumps({'rows':len(rows),'stages_available':sorted(set(r['stage'] for r in rows)),'status':'AVAILABLE_OUTPUTS_ONLY_CHECK_COMPLETION_MARKERS','bootstrap_draws':10000,'paired_unit':'parent, training seeds averaged within each parent','denominator_comparison':'same paired parent sample in numerator and denominator for joint sensitivity; primary fixed denominator','input_sha256':inputhash},indent=2))
print('Summary rows',len(rows))
