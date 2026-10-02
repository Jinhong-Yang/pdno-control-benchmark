"""Independent keyed checks of newly displayed tables and current figure hashes."""
from pathlib import Path
import csv,json,re,hashlib
import math
import argparse
import numpy as np
O=Path(__file__).resolve().parents[1];D=O/'data';T=O/'tables'
parser=argparse.ArgumentParser(description='CPU table audit; X2b/X1 checks require the raw experiment checkout.')
parser.add_argument('--repo-root',type=Path,help='experiment checkout containing runs/ and experiments/; autodetected from ancestors when omitted')
args=parser.parse_args()
if args.repo_root is not None:
    REPO=args.repo_root.expanduser().resolve()
    if not (REPO/'runs').is_dir() or not (REPO/'experiments').is_dir():
        parser.error(f'--repo-root must contain both runs/ and experiments/: {REPO}')
else:
    REPO=next((p for p in (O,*O.parents) if (p/'runs').is_dir() and (p/'experiments').is_dir()),None)
    if REPO is None:
        parser.error('CPU raw-data audit requires the experiment checkout (runs/ and experiments/). From an extracted Overleaf folder, pass --repo-root PATH_TO_CHECKOUT.')
def raw_path(relative):return REPO/Path(str(relative).replace('\\','/'))
def require_raw(path):
    if not path.is_file():
        raise FileNotFoundError(f'CPU raw-data audit requires this source file: {path}; provide the complete experiment checkout with --repo-root')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return list(csv.DictReader(p.open(encoding='utf-8-sig',newline='')))
def tabrows(p):
    return [[c.strip() for c in line[:-2].split('&')] for line in p.read_text().splitlines() if line.endswith(r'\\') and '&' in line]
checks=[]
def eq(found,value,digits,scale=1,signed=False,key=''):
    wanted=format(float(value)*scale,('+' if signed else '')+'.'+str(digits)+'f')
    assert found==wanted,(key,found,wanted)
    checks.append({'key':key,'display':found,'raw':value,'scale':scale,'decimals':digits})
mapping={'P-nr':'P-no-rank','RS-true':'O-cand-state','RS-obs':'O-cand-obs'}
primary=read(D/'control_all_methods.csv');follow=read(D/'revision/REVISION_COST_SUMMARY.csv')
roles=['locked_nominal','locked_coefficient_ood','locked_delay_dropout']
display=[r for r in tabrows(T/'nominal.tex') if r[0] in ['B0','B1','B2','B3','B4','B5','P-nr','P','RS-true','RS-obs']]
assert len(display)==18
for index,cells in enumerate(display):
    pop='burgers' if index<10 else 'heat';method=mapping.get(cells[0],cells[0])
    for j,role in enumerate(roles):
        source=primary if pop=='burgers' and not method.startswith('O-') else follow
        selected=[r for r in source if (r['pde'],r['method'],r['role'])==(pop,method,role) and (source is primary or r['stage']==('E1' if pop=='burgers' else 'E5'))]
        assert len(selected)==1; row=selected[0]
        key=f'nominal/{pop}/{method}/{role}'
        eq(cells[1+2*j],row['mean_cost'],6,key=key+'/cost')
        eq(cells[2+2*j],row['relative_cost_difference_vs_B0' if source is primary else 'relative_excess'],2,100,True,key+'/excess')

profiles=read(D/'followup/X2/X2a_profile_totals.csv')
display=[r for r in tabrows(T/'followup_stage_shares.tex') if r[0] in ['Burgers','SH heat']]
assert len(display)==16
for c in display:
    selected=[r for r in profiles if (r['pde'],r['method'],r['K'],r['cache'])==('burgers' if c[0]=='Burgers' else 'heat',c[1],c[2],'True' if c[3]=='on' else 'False')]
    assert len(selected)==1;r=selected[0];key='/'.join(c[:4])
    eq(c[4],r['mean_summed_instrumented_stages_ms'],4,key=key+'/stage_sum')
    eq(c[5],r['branch_plus_trunk_fraction_of_summed_instrumented_stages'],2,100,key=key+'/share')

errors=json.loads((D/'followup/X3/X3_OPERATOR_SUMMARIES.json').read_text())['rows']
display=[r for r in tabrows(T/'followup_error_decomposition.tex') if r[0] in ['Burgers','SH heat','NH heat']]
assert len(display)==len(errors)==54
for c in display:
    selected=[r for r in errors if r['pde']==('burgers' if c[0]=='Burgers' else 'heat') and r['method']==mapping.get(c[1],c[1]) and str(r['seed'])==c[2] and str(r['target_factor'] or '--')==c[3] and (c[0]!='NH heat' or r['stage']=='nonnegative_heat') and (c[0]!='SH heat' or r['stage']=='primary')]
    assert len(selected)==1,(c,selected);r=selected[0]
    for col,k in enumerate(['E_obs','E_prop','E_total','E_op','persistence','E_prop_over_E_total'],4):eq(c[col],r[k],4,key='/'.join(c[:4])+'/'+k)

# X2b: independently recompute seed-pooled p99s from all three saved request
# arrays per method/seed/archive, then match the displayed table.
x2_source=read(D/'followup/X2/X2b_row_statistics.csv')
x2_display=tabrows(T/'followup_timing_reconciliation.tex')
assert len(x2_display)==8, len(x2_display)
x2_cases={};x2_raw_hashes=[]
for archive in ('primary','followup'):
    for seed in ('11','23','37'):
        method_values={}
        for method in ('P','B4'):
            selected=[r for r in x2_source if r['archive']==archive and r['pde']=='burgers' and r['method']==method and r['seed']==seed and r['K']=='10' and r['cache'].lower() in ('false','0')]
            assert len(selected)==3,(archive,seed,method,len(selected))
            values=[]
            for r in selected:
                raw=raw_path(r['raw_file']);require_raw(raw)
                assert sha(raw)==r['raw_sha256'],raw
                x2_raw_hashes.append({'path':str(raw.relative_to(REPO)),'sha256':r['raw_sha256']})
                with np.load(raw,allow_pickle=False) as z:
                    assert 'latency_ms' in z.files,raw
                    values.append(np.asarray(z['latency_ms'],dtype=np.float64))
            method_values[method]=float(np.quantile(np.concatenate(values),.99))
        x2_cases[(archive,seed)]=(method_values['P'],method_values['B4'],method_values['P']/method_values['B4'])
assert len(x2_raw_hashes)==36, f'expected 36 X2b session arrays, got {len(x2_raw_hashes)}'
x2_unique_hashes={r['path']:r['sha256'] for r in x2_raw_hashes}
assert len(x2_unique_hashes)==len(x2_raw_hashes), 'duplicate X2b source-array path in the 36 session records'
for archive,shown in (('primary','Primary'),('followup','Follow-up')):
    archive_rows=[];started=False
    for c in x2_display:
        if c[0]==shown:
            started=True;archive_rows.append(c)
        elif c[0]=='Mean' and started and len(archive_rows)==3:
            archive_rows.append(c);break
    assert len(archive_rows)==4,(shown,archive_rows)
    ratios=[]
    for c in archive_rows[:3]:
        seed=c[1];p99p,p99b,ratio=x2_cases[(archive,seed)];ratios.append(ratio)
        key=f'X2b/{archive}/seed{seed}'
        eq(c[2],p99p,4,key=key+'/P_p99_ms')
        eq(c[3],p99b,4,key=key+'/B4_p99_ms')
        eq(c[4],ratio,2,key=key+'/P_over_B4_p99_ratio')
    mean_row=archive_rows[3]
    assert mean_row[1:4]==['--','--','--'],(shown,mean_row)
    eq(mean_row[4],float(np.mean(ratios)),2,key=f'X2b/{archive}/mean_of_seed_ratios')

# X1 remains conditional until the source analysis CSV and both generated
# tables are staged. If present, reconstruct table metrics from raw arrays;
# heat no-change uses the correct initial previous action [0.4, 0.4].
x1_csv=D/'followup/X1/X1_ANALYSIS.csv'
x1_design_table=T/'followup_candidate_design.tex'
x1_choice_table=T/'followup_candidate_choices.tex'
x1_status='PENDING_X1_ANALYSIS_OR_TABLES'
x1_checked_rows=0
x1_raw_checks=[]
if x1_csv.is_file() and x1_design_table.is_file() and x1_choice_table.is_file():
    x1_rows=read(x1_csv)
    assert len(x1_rows)==19, f'expected 19 X1 rows including full384 anchor, got {len(x1_rows)}'
    matched=[r for r in x1_rows if r['n']=='128' and r['population'] in ('burgers_nominal','heat_nonnegative_nominal')]
    anchors=[r for r in x1_rows if r['population']=='burgers_anchor_full384']
    assert len(matched)==18 and len(anchors)==1 and anchors[0]['n']=='384'
    for row in x1_rows:
        n=int(row['n']);pop=row['population'];pde=row['pde'];cell=row['cell']
        if pop=='burgers_anchor_full384':
            raw=REPO/'runs/followup_v120/results/X1/burgers_nominal/K10_H8/n384.npz'
        else:
            runpop='burgers_nominal' if pde=='burgers' else 'heat_nonnegative_nominal'
            raw=REPO/f'runs/followup_v120/results/X1/{runpop}/{cell}/n128.npz'
        require_raw(raw)
        assert sha(raw)==row['source_result_sha256'],raw
        with np.load(raw,allow_pickle=False) as z:
            a={k:z[k] for k in ('parent_id','episode_control_cost','tracking_rmse','selected_candidate_index','action_applied','mean_abs_action_change')}
        assert len(a['parent_id'])==n,(raw,len(a['parent_id']),n)
        choices=a['selected_candidate_index'];actions=a['action_applied'].astype(np.float32)
        init=np.zeros(2,dtype=np.float32) if pde=='burgers' else np.full(2,.4,dtype=np.float32)
        prev=np.concatenate([np.broadcast_to(init,(n,1,2)),actions[:,:-1]],axis=1)
        computed={
            'mean_episode_cost':float(np.mean(a['episode_control_cost'])),
            'mean_tracking_rmse':float(np.mean(a['tracking_rmse'])),
            'hold_selection_rate':float(np.mean(choices==(4 if int(row['K'])==10 else 48))),
            'b0_selection_rate':float(np.mean(choices==(int(row['K'])-1))),
            'applied_no_change_rate':float(np.mean(np.all(actions==prev,axis=2))),
            'mean_abs_action_change':float(np.mean(a['mean_abs_action_change'])),
        }
        if row['feedback_rollout_selection_rate']:
            computed['feedback_rollout_selection_rate']=float(np.mean(choices==int(row['K'])))
        for field,value in computed.items():
            assert math.isclose(float(row[field]),value,rel_tol=0,abs_tol=1e-12),(pop,cell,field,row[field],value)
        x1_raw_checks.append({'population':pop,'cell':cell,'n':n,'raw_sha256':sha(raw),'metric_fields_checked':sorted(computed)})
    # The two candidate tables contain 18 matched n=128 cells; the n=384
    # Burgers anchor is validated above as a distinct source row, not pooled.
    for table_path in (x1_design_table,x1_choice_table):
        shown=tabrows(table_path);assert len(shown)==18,(table_path,len(shown))
        seen=set()
        for idx,c in enumerate(shown):
            cell_id=c[0];feedback=cell_id.endswith('+FB');base=cell_id[:-3] if feedback else cell_id
            K,H=base.split('/')
            population='burgers_nominal' if idx<9 else 'heat_nonnegative_nominal'
            selected=[r for r in matched if r['population']==population and r['K']==K and r['H']==H and (r['feedback_rollout'].lower() in ('true','1'))==feedback]
            assert len(selected)==1,(table_path,cell_id,population,selected)
            r=selected[0];seen.add((population,K,H,feedback))
            if table_path==x1_design_table:
                eq(c[1],r['n'],0,key=f'X1/{population}/{cell_id}/n')
                eq(c[2],r['mean_episode_cost'],6,key=f'X1/{population}/{cell_id}/cost')
                eq(c[3],r['relative_excess_fixed'],2,100,True,key=f'X1/{population}/{cell_id}/excess')
                for j,(lo,hi) in enumerate((('fixed_ci_low','fixed_ci_high'),('paired_ci_low','paired_ci_high')),4):
                    want=f"[{100*float(r[lo]):+.2f}, {100*float(r[hi]):+.2f}]"
                    assert c[j]==want,(table_path,cell_id,c[j],want)
                    checks.append({'key':f'X1/{population}/{cell_id}/{lo}_{hi}','display':c[j],'raw':[r[lo],r[hi]],'scale':100,'decimals':2})
                eq(c[6],r['mean_tracking_rmse'],6,key=f'X1/{population}/{cell_id}/rmse')
            else:
                fields=('hold_selection_rate','b0_selection_rate','feedback_rollout_selection_rate','applied_no_change_rate','mean_abs_action_change')
                for j,field in enumerate(fields,1):
                    if field=='feedback_rollout_selection_rate' and not r[field]:
                        assert c[j]=='--',(table_path,cell_id,c[j])
                    else:
                        eq(c[j],r[field],6 if field=='mean_abs_action_change' else 2,
                           1 if field=='mean_abs_action_change' else 100,key=f'X1/{population}/{cell_id}/{field}')
        assert len(seen)==18
    x1_checked_rows=len(x1_rows);x1_status='PASS_X1_TABLES_AND_RAW_ARRAY_MATCH'

figure_checks=[]
for r in json.loads((D/'figure_provenance.json').read_text()):
    for source in r['input_sha256']:assert sha(O/source['path'])==source['sha256'],source
    assert sha(O/'figures'/(r['id']+'.pdf'))==r['pdf_sha256'],r['id']
    figure_checks.append(r['id'])
sources=[D/'control_all_methods.csv',D/'revision/REVISION_COST_SUMMARY.csv',D/'followup/X2/X2a_profile_totals.csv',D/'followup/X2/X2b_row_statistics.csv',D/'followup/X2/X2b_seed_pooled_ratios.csv',D/'followup/X3/X3_OPERATOR_SUMMARIES.json']
status='PASS_NEW_TABLE_TRANSCRIPTION_FIGURE_HASHES_X1_COMPLETE' if x1_status.startswith('PASS') else 'PASS_EXISTING_AND_X2B_X1_PENDING'
out={'status':status,'repo_root':str(REPO),'cpu_raw_data_required':True,'cpu_raw_data_requirement':'X2b raw request latency arrays and (when available) X1 raw result NPZ files must be present under this checkout; use --repo-root for an extracted Overleaf tree.','numeric_cells_checked':len(checks),'checks':checks,'x2b_rows_checked':20,'x2b_source_arrays_checked':len(x2_raw_hashes),'x2b_unique_source_arrays_checked':len(x2_unique_hashes),'x2b_raw_source_hashes':x2_raw_hashes,'x1_status':x1_status,'x1_analysis_rows_checked':x1_checked_rows,'x1_raw_array_checks':x1_raw_checks,'figure_hashes_verified':figure_checks,'source_hashes':{str(p.relative_to(O)):sha(p) for p in sources},'table_hashes':{p.name:sha(p) for p in T.glob('*.tex')},'limits':'Keyed checks of replacement main table, X2a/X2b timing tables and X3 norm tables; conditional X1 numeric tables are explicitly pending until both the source CSV and tables arrive. Legacy tables have separate pre-presentation audits and numeric-cell conservation receipt. Does not certify all prose or PDF layout.'}
(D/'NEW_TABLE_NUMERIC_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n')
print(out['status'],len(checks),'numeric cells;',len(figure_checks),'figure hashes')
