"""Read-only, CPU-only arithmetic audit of completed v3 evidence.

Never imports experiment modules, loads checkpoints, runs inference/solvers, or
writes under evidence/config/src/runs/data. Output must be a NEW reports subdir.
"""
from __future__ import annotations
import argparse, csv, hashlib, itertools, json, math, platform, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('B0','B1','P','P-no-rank','B4','B5','B2','B3')
ROLES = ('locked_nominal','locked_coefficient_ood','locked_delay_dropout')
SEEDS = (11,23,37)
access = {}
checks = []

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def js(rel):
    p=ROOT/rel; access[rel]=sha(p)
    return json.loads(p.read_text(encoding='utf-8-sig'))

def check(name, actual, expected, tol=1e-11):
    ok=bool(np.allclose(actual,expected,rtol=tol,atol=tol,equal_nan=True))
    checks.append({'check':name,'pass':ok})
    if not ok: checks[-1].update(actual=np.asarray(actual).tolist(),expected=np.asarray(expected).tolist())

def csvout(out,name,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (out/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);args=ap.parse_args()
    out=(ROOT/args.output).resolve()
    if not out.is_relative_to(ROOT/'reports') or out.exists():
        raise SystemExit('Output must be a new directory under reports; refusing overwrite')
    out.mkdir(parents=True)
    start=time.perf_counter()
    freeze=js('evidence/raw_evidence_freeze_v3.json')
    inventory={r['path']:r['sha256'] for group in ('experiment_tree_inventory','context_and_procedure_files','locked_raw_artifacts','host_latency_raw_artifacts') for r in freeze[group]}
    hashes=[]
    for rel,expected in inventory.items():
        p=ROOT/rel; actual=sha(p) if p.is_file() else None
        hashes.append({'path':rel,'expected':expected,'actual':actual,'match':actual==expected,'bytes':p.stat().st_size if p.is_file() else None})
    csvout(out,'freeze_hash_verification.csv',hashes)
    print('Hash inventory checked',len(hashes),'mismatches',[r['path'] for r in hashes if not r['match']],flush=True)
    a=js('evidence/locked_test_analysis_v3.json');lat=js('evidence/latency_analysis_v3.json'); lm=js('evidence/latency_summary_v3.json')
    cm=js('evidence/confirmatory_checkpoint_manifest_v3.json');cal=js('evidence/calibration_lock_v3.json')
    marker=js('state/LOCKED_TEST_ONCE_V3.json')
    data={}; per_seed=[]; mean_rows=[]; heat_bounds=[]; equivalence=[]
    qmax_by_pde={'burgers':1.2}
    heatmax=1.2
    for role in ('train','validation'):
        with np.load(ROOT/f'data/train_validation_v3/{role}/heat/trajectories.npz',allow_pickle=False) as z:
            heatmax=max(heatmax,1.25*float(z['goal'].max()))
    qmax_by_pde['heat']=heatmax
    rng=np.random.default_rng(20260926)
    def boot(x):
        ix=rng.integers(0,len(x),size=(10000,len(x)))
        v=x[ix].mean(axis=1)
        return float(x.mean()),np.quantile(v,[.025,.975]).tolist()
    for g in a['groups']:
        role,pde=g['role'],g['pde']; n=g['parent_count'];prefix=f'{role}/{pde}'
        for method in METHODS:
            stored=next(r for r in g['controller_metrics'] if r['method']==method)
            for seed in ((None,) if method in ('B0','B1') else SEEDS):
                rel=f"evidence/locked_test_raw_v3/{role}/{pde}/{method}_s{'na' if seed is None else seed}.npz"
                with np.load(ROOT/rel,allow_pickle=False) as z:
                    d={k:z[k] for k in ('parent_id','episode_control_cost','tracking_rmse','truth_constraint_violation','fallback_rate','fallback_count','causal_timestamp_violations','field_nrmse_at_snapshot','candidate_regret_at_snapshot','action_applied')}
                    state=z['state_true'];goal=z['goal'];proposed=z['action_proposed'];projected=z['action_projected']
                    check(f'{rel}/projected_applied',projected,d['action_applied'],0)
                    qmax=qmax_by_pde[pde]; nxt=state[:,1:]
                    if pde=='heat':
                        lower=(nxt<0).any(axis=(1,2));upper=(nxt>qmax).any(axis=(1,2))
                        init=(state[:,0]<0).any(axis=1);first=(nxt[:,0]<0).any(axis=1)
                        violations=lower|upper
                        heat_bounds.append({'role':role,'method':method,'seed':seed,'n':n,'qmax':qmax,'initial_lower_count':int(init.sum()),'first_step_lower_count':int(first.sum()),'ever_lower_count':int(lower.sum()),'ever_upper_count':int(upper.sum()),'late_new_lower_count':int((lower&~first).sum()),'minimum_state':float(state.min()),'minimum_first_step':float(nxt[:,0].min()),'maximum_state':float(state.max())})
                        penalty=np.maximum(-nxt,0)**2+np.maximum(nxt-qmax,0)**2
                    else:
                        violations=(abs(nxt)>qmax).any(axis=(1,2));penalty=np.maximum(abs(nxt)-qmax,0)**2
                    check(f'{rel}/truth_violation',violations,d['truth_constraint_violation'],0)
                    acts=d['action_applied'].astype(np.float64)
                    prev=np.concatenate([np.full((n,1,2),0 if pde=='burgers' else .4),acts[:,:-1]],axis=1)
                    rebuilt=((nxt-goal).astype(np.float64)**2).mean(axis=2).sum(axis=1)+.01*(acts**2).sum(axis=(1,2))+.05*((acts-prev)**2).sum(axis=(1,2))+10*penalty.mean(axis=2).sum(axis=1)
                    check(f'{rel}/raw_cost_reconstruction',rebuilt,d['episode_control_cost'],2e-6)
                    check(f'{rel}/raw_tracking_reconstruction',np.sqrt(((nxt-goal).astype(np.float64)**2).mean(axis=(1,2))),d['tracking_rmse'],2e-6)
                data[(role,pde,method,seed)]=d
                s=next(r for r in stored['seed_results'] if r['seed']==seed)
                for k,array in [('mean_control_cost','episode_control_cost'),('mean_tracking_rmse','tracking_rmse'),('violation_rate','truth_constraint_violation'),('mean_fallback_rate','fallback_rate')]:check(f'{rel}/{k}',np.mean(d[array]),s[k])
                row={'role':role,'pde':pde,'method':method,'seed':seed,'parent_count':n,**{k:v for k,v in s.items() if k!='seed'}}
                per_seed.append(row)
            ds=[data[(role,pde,method,s)] for s in ((None,) if method in ('B0','B1') else SEEDS)]
            ids=ds[0]['parent_id']; b0=data[(role,pde,'B0',None)]
            checks.append({'check':prefix+'/'+method+'/parent_order','pass':all(np.array_equal(d['parent_id'],b0['parent_id']) for d in ds)})
            cost=np.mean([d['episode_control_cost'] for d in ds],axis=0)
            viol=np.mean([d['truth_constraint_violation'] for d in ds],axis=0)
            diff=(cost-b0['episode_control_cost'])/max(float(b0['episode_control_cost'].mean()),1e-12)
            mean,ci=boot(diff);vm,vci=boot(viol-b0['truth_constraint_violation'])
            pair=g['paired_vs_h3_comparator'][method]
            check(prefix+'/'+method+'/cost_mean',cost.mean(),stored['mean_control_cost_over_parent_and_seed'])
            check(prefix+'/'+method+'/cost_normalization',mean,pair['normalized_cost_difference_mean'])
            check(prefix+'/'+method+'/cost_ci',ci,pair['normalized_cost_difference_ci95'])
            check(prefix+'/'+method+'/violation_ci',vci,pair['violation_rate_difference_ci95'])
            mean_rows.append({'role':role,'pde':pde,'method':method,'parent_count':n,'mean_cost':float(cost.mean()),'tracking_rmse':stored['mean_tracking_rmse_over_parent_and_seed'],'relative_cost_difference_vs_B0':mean,'ci_low':ci[0],'ci_high':ci[1],'violation_rate':float(viol.mean()),'violation_difference_ci_low':vci[0],'violation_difference_ci_high':vci[1],'tick100_field_nrmse':stored['mean_field_nrmse_at_tick100']})
        for label,left,right in [('B1_vs_B0_cost_gap','B1','B0'),('P_vs_B5_cost_gap','P','B5')]:
            def costs(m):return np.mean([data[(role,pde,m,s)]['episode_control_cost'] for s in ((None,) if m in ('B0','B1') else SEEDS)],axis=0)
            mean,ci=boot(costs(left)-costs(right));reg=g[label]['paired_parent_bootstrap']
            check(prefix+'/'+label+'/mean',mean,reg['mean']);check(prefix+'/'+label+'/ci',ci,reg['ci95_percentile'])
        for seed in SEEDS:
            p=data[(role,pde,'P',seed)];b=data[(role,pde,'B5',seed)]
            equivalence.append({'role':role,'pde':pde,'seed':seed,'max_abs_cost_difference':float(abs(p['episode_control_cost']-b['episode_control_cost']).max()),'max_abs_applied_action_difference':float(abs(p['action_applied']-b['action_applied']).max()),'identical_action_parent_count':int((p['action_applied']==b['action_applied']).all(axis=(1,2)).sum()),'parent_count':n})
        print('Raw endpoint audit',prefix,flush=True)
    csvout(out,'control_all_methods.csv',mean_rows);csvout(out,'control_per_seed.csv',per_seed);csvout(out,'heat_constraint_diagnostic.csv',heat_bounds);csvout(out,'P_B5_matched_actions.csv',equivalence)
    selected=[]
    for r in cm['checkpoints']:
        v=r['metadata'].get('validation',{})
        row={k:r.get(k) for k in ('pde','method','seed','selected_update','selection_status','sha256')};row.update(v)
        selected.append(row)
    csvout(out,'selected_checkpoints.csv',selected)
    calrows=[]
    for r in cal['groups']:
        values=np.array(list(r['parent_scores'].values()));ix=math.ceil((len(values)+1)*.95)-1
        check(f"cal/{r['pde']}/{r['method']}/{r['seed']}",np.sort(values)[ix],r['margin'])
        calrows.append({k:r[k] for k in ('pde','method','seed','parent_count','query_count','finite_sample_order_index_zero_based','margin','action_selection_uses_margin')})
    csvout(out,'calibration_margins.csv',calrows)
    arrays={};streams={}
    for rel,expected in lm['raw_files_sha256'].items():
        with np.load(ROOT/rel,allow_pickle=False) as z:
            key=(str(z['pde'].item()),str(z['method'].item()),int(z['seed'].item()),int(z['session'].item()))
            arrays[key]=z['latency_ms'];streams[key]=z['cuda_stream_span_ms']
            check(rel+'/n',len(arrays[key]),20000,0)
    for r in lat['controller_groups']:
        seed=-1 if r['seed'] is None else r['seed'];values=np.concatenate([arrays[(r['pde'],r['method'],seed,s)] for s in (1,2,3)])
        for k,q in [('p50_ms',.5),('p95_ms',.95),('p99_ms',.99),('p999_ms',.999)]:check(f"lat/{r['pde']}/{r['method']}/{seed}/{k}",np.quantile(values,q),r[k])
        for ms in (1,2,5,10):check(f"lat/{r['pde']}/{r['method']}/{seed}/{ms}",np.sum(values>ms),r['deadline_misses'][f'{ms}ms'],0)
    csvout(out,'latency_per_seed.csv',[{**{k:v for k,v in r.items() if k!='deadline_misses'},**{f'misses_{k}':v for k,v in r['deadline_misses'].items()}} for r in lat['controller_groups']])
    pooled=[]
    for pde in ('burgers','heat'):
        for m in METHODS:
            vals=np.concatenate([v for k,v in arrays.items() if k[:2]==(pde,m)])
            sv=np.concatenate([v for k,v in streams.items() if k[:2]==(pde,m)]);sv=sv[np.isfinite(sv)]
            pooled.append({'pde':pde,'method':m,'n_requests':len(vals),'p50_ms':float(np.quantile(vals,.5)),'p95_ms':float(np.quantile(vals,.95)),'p99_ms':float(np.quantile(vals,.99)),'p999_ms':float(np.quantile(vals,.999)),'miss_count_2ms':int((vals>2).sum()),'miss_count_5ms':int((vals>5).sum()),'miss_rate_5ms':float((vals>5).mean()),'cuda_stream_p99_ms':float(np.quantile(sv,.99)) if len(sv) else None})
    csvout(out,'latency_pooled_descriptive.csv',pooled)
    rnglat=np.random.default_rng(20260926);h1=[]
    for pde in ('burgers','heat'):
        cache={}
        def ratio(seed,sessions):
            key=(int(seed),tuple(sorted(int(s) for s in sessions)))
            if key not in cache:
                v=[np.concatenate([arrays[(pde,m,key[0],s)] for s in key[1]]) for m in ('P','B4')]
                cache[key]=float(np.quantile(v[0],.99)/np.quantile(v[1],.99))
            return cache[key]
        pts={str(s):ratio(s,(1,2,3)) for s in SEEDS};boots=[]
        for i in range(10000):
            chosen=rnglat.choice(np.array(SEEDS),size=3,replace=True)
            boots.append(np.mean([ratio(s,rnglat.integers(1,4,size=3)) for s in chosen]))
        ci=np.quantile(boots,[.025,.975]).tolist();reg=next(r for r in lat['H1_p99_ratio_analysis'] if r['pde']==pde)
        check('H1/'+pde+'/point',np.mean(list(pts.values())),reg['mean_per_seed_ratio']);check('H1/'+pde+'/ci',ci,reg['hierarchical_seed_session_bootstrap_ci95'])
        h1.append({'pde':pde,'mean_per_seed_ratio':float(np.mean(list(pts.values()))),'ci95':ci,'per_seed_ratio':pts})
    failures=[json.loads(s) for s in (ROOT/'evidence/failures_and_modifications_v3.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    csvout(out,'failure_event_index.csv',[{'line':i+1,'timestamp':r.get('at_kst'),'stage':r.get('stage'),'event':r.get('event'),'detail':r.get('detail'),'full_record_json':json.dumps(r,ensure_ascii=False)} for i,r in enumerate(failures)])
    result={'status':'PASS' if all(c['pass'] for c in checks) else 'CHECK_FAILURE','analysis_only':True,'gpu_operations':0,'new_model_or_solver_executions':0,'raw_test_shards_read':len(data),'independent_test_parents':sum(g['parent_count'] for g in a['groups']),'control_executions':sum(len(d['parent_id']) for d in data.values()),'latency_rows':len(arrays),'latency_requests':sum(len(v) for v in arrays.values()),'freeze_inventory_checked':len(hashes),'freeze_mismatches':[r for r in hashes if not r['match']],'arithmetic_checks':len(checks),'failed_checks':[c for c in checks if not c['pass']],'all_checks':checks,'qmax':qmax_by_pde,'H1_recomputed':h1,'source_json_sha256':access,'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'elapsed_seconds':time.perf_counter()-start,'completed_at_utc':datetime.now(timezone.utc).isoformat(),'interpretation':'Registered parent bootstraps condition on three fitted seeds and fixed comparator mean denominator. Heat constraint decomposition is post-freeze descriptive, never a replacement endpoint.'}
    (out/'audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('all_checks','source_json_sha256','freeze_mismatches','H1_recomputed')},indent=2))
    return 0 if result['status']=='PASS' else 1

if __name__=='__main__':raise SystemExit(main())
