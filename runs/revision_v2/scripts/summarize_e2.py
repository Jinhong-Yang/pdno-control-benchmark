"""Complete-sweep inference summaries; preserve original seed/session statistic."""
from pathlib import Path
import json,csv,hashlib,itertools
import numpy as np
R=Path(__file__).resolve().parents[1];OLD=R.parents[1]/'experiments/pdno_jevLite_20260927_v3'
marker=R/'results/E2_TIMING.json';manifest=json.loads(marker.read_text());assert manifest['status']=='COMPLETE' and len(manifest['rows'])==360
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();arrays={};hashes={};perseed=[];pooled=[];ratios=[];profiles=[]
for row in manifest['rows']:
    key=(row['pde'],row['method'],row['seed'],row['K'],row['cache'],row['session'])
    path=R/f"results/E2_raw/session{row['session']}_{row['pde']}_{row['method']}_s{row['seed']}_K{row['K']}_cache{int(row['cache'])}.npz"
    assert sha(path)==row['sha256'];hashes[str(path.relative_to(R))]=row['sha256']
    with np.load(path,allow_pickle=False) as z:a=z['latency_ms']
    assert len(a)==5000 and np.isfinite(a).all();assert key not in arrays;arrays[key]=a
expected=set(itertools.product(['burgers','heat'],['P','B4'],[11,23,37],[10,25,50,100,200],[False,True],[1,2,3]));assert set(arrays)==expected
def stats(values):return {'requests':len(values),'p50_ms':float(np.quantile(values,.5)),'p95_ms':float(np.quantile(values,.95)),'p99_ms':float(np.quantile(values,.99)),'p999_ms':float(np.quantile(values,.999)),'miss_count_5ms':int(np.sum(values>5)),'miss_rate_5ms':float(np.mean(values>5))}
# Preserve session-level evidence rather than hiding temporal variability in pools.
persession=[dict(zip(['pde','method','seed','K','cache','session'],key),**stats(values)) for key,values in arrays.items()]
for pde,method,K,cache in itertools.product(['burgers','heat'],['P','B4'],[10,25,50,100,200],[False,True]):
    for seed in [11,23,37]:
        values=np.concatenate([arrays[pde,method,seed,K,cache,s] for s in [1,2,3]]);perseed.append({'pde':pde,'method':method,'seed':seed,'K':K,'cache':cache,**stats(values)})
    values=np.concatenate([arrays[pde,method,seed,K,cache,s] for seed in [11,23,37] for s in [1,2,3]]);pooled.append({'pde':pde,'method':method,'K':K,'cache':cache,**stats(values)})
rng=np.random.default_rng(2026100220)
for pde,K,cache in itertools.product(['burgers','heat'],[10,25,50,100,200],[False,True]):
    memo={}
    def ratio(seed,sessions):
        key=(int(seed),tuple(sorted(int(s) for s in sessions)))
        if key not in memo:
            p=np.concatenate([arrays[pde,'P',key[0],K,cache,s] for s in key[1]]);b=np.concatenate([arrays[pde,'B4',key[0],K,cache,s] for s in key[1]]);memo[key]=float(np.quantile(p,.99)/max(np.quantile(b,.99),1e-12))
        return memo[key]
    points=[ratio(s,[1,2,3]) for s in [11,23,37]];boot=[]
    for _ in range(10000):
        seeds=rng.choice(np.array([11,23,37]),3,replace=True);boot.append(float(np.mean([ratio(s,rng.integers(1,4,3)) for s in seeds])))
    lo,hi=np.quantile(boot,[.025,.975]);ratios.append({'pde':pde,'K':K,'cache':cache,'mean_per_seed_p99_ratio':float(np.mean(points)),'seed11_ratio':points[0],'seed23_ratio':points[1],'seed37_ratio':points[2],'ci_low':float(lo),'ci_high':float(hi),'bootstrap_replicates':10000,'point_ratio_at_most_075':bool(np.mean(points)<=.75),'pointwise_upper_below_one':bool(hi<1)})
for path in sorted((R/'results/E2_profiles').glob('*.json')):
    r=json.loads(path.read_text());hashes[str(path.relative_to(R))]=sha(path);assert len(r['stage_ms'])==500
    for stage in r['stage_ms'][0]:
        vals=np.array([row[stage] for row in r['stage_ms']]);assert np.isfinite(vals).all();profiles.append({'pde':r['pde'],'method':r['method'],'K':r['K'],'cache':r['cache'],'stage':stage,'requests':len(vals),'p50_ms':float(np.quantile(vals,.5)),'p99_ms':float(np.quantile(vals,.99)),'mean_ms':float(vals.mean())})
def csvout(name,rows):
    with (R/'results'/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for name,rows in [('E2_per_session.csv',persession),('E2_per_seed.csv',perseed),('E2_pooled.csv',pooled),('E2_ratio_bootstrap.csv',ratios),('E2_stage_profile.csv',profiles)]:csvout(name,rows)
# Log-spaced histogram bins include all observed values; campaigns stay separate.
hist_sources={}
for pde in ['burgers','heat']:
    for method in ['P','B4','B2']:
        vals=[]
        for seed,s in itertools.product([11,23,37],[1,2,3]):
            path=OLD/f'evidence/latency_raw_v3/session{s}_{pde}_{method}_s{seed}.npz'
            with np.load(path,allow_pickle=False) as z:vals.append(z['latency_ms'])
            hashes[str(path)]=sha(path)
        hist_sources['original',pde,method,False]=np.concatenate(vals)
    for method,cache in itertools.product(['P','B4'],[False,True]):hist_sources['revision_K10',pde,method,cache]=np.concatenate([arrays[pde,method,seed,10,cache,s] for seed,s in itertools.product([11,23,37],[1,2,3])])
bins=np.geomspace(min(v.min() for v in hist_sources.values())*.99,max(v.max() for v in hist_sources.values())*1.01,101);hist=[]
for (campaign,pde,method,cache),vals in hist_sources.items():
    counts,_=np.histogram(vals,bins);assert counts.sum()==len(vals)
    for j,count in enumerate(counts):hist.append({'campaign':campaign,'pde':pde,'method':method,'cache':cache,'bin_left_ms':float(bins[j]),'bin_right_ms':float(bins[j+1]),'count':int(count),'fraction':float(count/len(vals)),'density_per_log10_ms':float(count/len(vals)/(np.log10(bins[j+1])-np.log10(bins[j]))),'total_requests':len(vals)})
csvout('E2_latency_histograms.csv',hist)
(R/'results/E2_ANALYSIS.json').write_text(json.dumps({'status':'COMPLETE','input_sha256':hashes,'seed_rows':len(perseed),'pooled_rows':len(pooled),'ratio_cells':len(ratios),'profile_rows':len(profiles),'interpretation':'Exploratory pointwise intervals conditional on3trainingseeds and3sequentialsessions; no simultaneous coverage, independent environment, or service guarantee. All timing outliers retained. Stage profiles are separately synchronized and stage quantiles must not be added.','statistic':'mean over seeds of pooled-three-session P/B4 p99 ratios; matched session resampling within each sampled seed, same functional as original H1','bootstrap_seed':2026100220},indent=2));print('Complete E2 sweep summarized;20 pointwise ratio cells')
