"""Submission presentation layer; frozen numerical tables are built/audited first."""
from pathlib import Path
import csv,json,re,hashlib
O=Path(__file__).resolve().parents[1]; D=O/'data'; T=O/'tables'
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def numeric_cells(text):
    return [c.strip() for line in text.splitlines() if '&' in line and not any(x in line for x in ['caption','multicolumn','Method','PDE','Seed','Cost','Stage','Mean','Update']) for c in line.replace(r'\\','').split('&') if re.fullmatch(r'[+\-\d.,\[\] %]+',c.strip())]
def normalize(s):
    # Cross-reference identifiers are internal, not displayed terminology.
    keys=[]
    def protect(m):
        keys.append(m.group(0));return 'INTERNALKEY'+chr(65+len(keys)-1)+'TOKEN'
    s=re.sub(r'\\(?:label|ref)\{[^}]+\}',protect,s)
    replacements=[('Oracle: Observed','RS-obs'),('Oracle: State','RS-true'),('O-cand-state','RS-true'),('O-cand-obs','RS-obs'),('Oracle','Reference-solver'),('oracle','reference-solver'),('host-ready','request'),('Host-ready','Request'),('paired-parent','paired-scenario'),('parent','scenario'),('Parent','Scenario'),('Original heat','SH heat'),('original heat','SH heat'),('New heat','NH heat'),('new heat','NH heat'),('Original nominal','Primary nominal'),('original nominal','primary nominal'),('original parents','primary scenarios'),('Original','Primary'),('original','primary'),('Revision','Follow-up'),('revision','follow-up'),('H4','unexecuted no-age ablation')]
    for a,b in replacements:s=s.replace(a,b)
    s=re.sub(r'\bE([1-5])\b',r'D\1',s)
    for i,key in enumerate(keys):s=s.replace('INTERNALKEY'+chr(65+i)+'TOKEN',key)
    return s
receipt={'scope':'Display labels normalized after frozen table audits; numeric cells conserved except separately sourced replacement main table.','tables':[]}
for p in sorted(T.glob('*.tex')):
    if p.name.startswith('followup_') or p.name in ['nominal.tex','evidence_numbers.tex']:continue
    old=p.read_text(encoding='utf-8'); new=normalize(old)
    assert numeric_cells(old)==numeric_cells(new),p.name
    dest=p.with_name(p.name.replace('revision_','followup_'))
    dest.write_text(new,encoding='utf-8')
    receipt['tables'].append({'input':p.name,'output':dest.name,'sha256':sha(dest),'numeric_cells_conserved':len(numeric_cells(old))})

methods=['B0','B1','B2','B3','B4','B5','P-no-rank','P']
roles=['locked_nominal','locked_coefficient_ood','locked_delay_dropout']
primary=rows(D/'control_all_methods.csv'); follow=rows(D/'revision/REVISION_COST_SUMMARY.csv')
checks=[]
out=[r'\begin{table*}[t]',r'\caption{Mean episode cost and excess cost relative to B0 across nominal and stress conditions. Burgers is the primary population; NH heat is the separate nonnegative follow-up population. Learned seeds are averaged within scenario. RS rows are follow-up reference-solver controls on the same Burgers scenarios. Paired intervals and tracking errors are in the supplementary tables.}\label{tab:nominal}',r'\centering\small',r'\begin{tabular}{@{}lrrrrrr@{}}\toprule',r'& \multicolumn{2}{c}{Nominal} & \multicolumn{2}{c}{Coefficient shift} & \multicolumn{2}{c}{Delay/dropout}\\',r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}',r'Method & Cost & Excess (\%) & Cost & Excess (\%) & Cost & Excess (\%)\\\midrule']
for population in ['burgers','NH']:
    out.append(r'\multicolumn{7}{l}{\textit{'+('Primary Burgers' if population=='burgers' else 'Follow-up NH heat')+r'}}\\')
    for m in methods+(['O-cand-state','O-cand-obs'] if population=='burgers' else []):
        vals=[{'P-no-rank':'P-nr','O-cand-state':'RS-true','O-cand-obs':'RS-obs'}.get(m,m)]
        for role in roles:
            if population=='burgers' and not m.startswith('O-'):
                source='control_all_methods.csv'; data=primary; filt={'pde':'burgers','role':role,'method':m}; col='relative_cost_difference_vs_B0'
            else:
                source='revision/REVISION_COST_SUMMARY.csv';data=follow;filt={'stage':'E1' if population=='burgers' else 'E5','pde':'burgers' if population=='burgers' else 'heat','role':role,'method':m};col='relative_excess'
            matched=[(i+2,r) for i,r in enumerate(data) if all(r[k]==v for k,v in filt.items())]
            assert len(matched)==1,(population,m,role,matched)
            line,r=matched[0]
            cost=f"{float(r['mean_cost']):.6f}"; excess=f"{100*float(r[col]):+.2f}"
            vals += [cost,excess]
            checks.append({'table':'nominal.tex','population':population,'method':m,'role':role,'source':'data/'+source,'csv_line':line,'cost_column':'mean_cost','excess_column':col,'display_cost':cost,'display_excess':excess,'source_sha256':sha(D/source)})
        out.append(' & '.join(vals)+r'\\')
    out.append(r'\midrule' if population=='burgers' else r'\bottomrule')
out += [r'\end{tabular}',r'\end{table*}']
(T/'nominal.tex').write_text('\n'.join(out)+'\n',encoding='utf-8')
receipt['main_replacement_cells']=checks

# Reformat primary cost percentages directly from full-precision CSV (never
# from an already rounded display value).
out=[]
for role in roles:
    for pde in ['burgers','heat']:
        out += [r'\begin{table}[ht]',r'\centering\small',r'\caption{'+('Primary Burgers' if pde=='burgers' else 'Primary SH heat')+': '+{'locked_nominal':'nominal','locked_coefficient_ood':'coefficient shift','locked_delay_dropout':'delay/dropout'}[role]+r'. Cost differences and paired-scenario intervals relative to B0.}',r'\begin{tabular}{@{}lrrrrr@{}}\toprule',r'Method & Cost & RMSE & Excess (\%) & 95\% interval (\%) & Violation (\%)\\\midrule']
        for m in methods:
            r=next(x for x in primary if (x['role'],x['pde'],x['method'])==(role,pde,m))
            out.append(' & '.join([m.replace('P-no-rank','P-nr'),f"{float(r['mean_cost']):.6f}",f"{float(r['tracking_rmse']):.6f}",f"{100*float(r['relative_cost_difference_vs_B0']):+.2f}",f"[{100*float(r['ci_low']):+.2f}, {100*float(r['ci_high']):+.2f}]",f"{100*float(r['violation_rate']):.2f}"])+r'\\')
        out += [r'\bottomrule\end{tabular}',r'\end{table}']
    out.append(r'\clearpage')
(T/'supp_control.tex').write_text('\n'.join(out)+'\n')
receipt['primary_supplement_percent_format']='2 decimal places regenerated from control_all_methods.csv; raw cost and RMSE remain 6 places.'

p=D/'followup/X2/X2a_profile_totals.csv'
if p.exists():
    out=[r'\begin{table}[ht]',r'\centering\small',r'\caption{Follow-up mean stage budgets. The denominator is the sum of ten instrumented mean stage durations, not an independently timed whole request. The final column is the branch-plus-trunk share, an ideal deletion bound only within this budget.}',r'\begin{tabular}{@{}llrrrr@{}}\toprule',r'PDE & Method & $K$ & Cache & Sum (ms) & Branch+trunk (\%)\\\midrule']
    for r in rows(p):out.append(' & '.join(['Burgers' if r['pde']=='burgers' else 'SH heat',r['method'],r['K'],'on' if r['cache']=='True' else 'off',f"{float(r['mean_summed_instrumented_stages_ms']):.4f}",f"{100*float(r['branch_plus_trunk_fraction_of_summed_instrumented_stages']):.2f}"])+r'\\')
    out += [r'\bottomrule\end{tabular}',r'\end{table}']
    (T/'followup_stage_shares.tex').write_text('\n'.join(out)+'\n')

p=D/'followup/X2/X2b_seed_pooled_ratios.csv'
if p.exists():
    out=[r'\begin{table}[ht]',r'\centering\small',r'\caption{Matched Burgers request-latency comparison at $K=10$, cache off. Sessions are pooled within each seed before computing p99. The mean row averages the three seed-specific ratios; it is not a ratio of average percentiles. All recorded requests are retained.}',r'\begin{tabular}{@{}llrrr@{}}\toprule',r'Archive & Seed & P p99 (ms) & B4 p99 (ms) & P/B4 ratio\\\midrule']
    for archive in ['primary','followup']:
        selected=[r for r in rows(p) if r['archive']==archive]
        assert len(selected)==3
        for r in selected:out.append(' & '.join(['Primary' if archive=='primary' else 'Follow-up',r['seed'],f"{float(r['P_p99_ms']):.4f}",f"{float(r['B4_p99_ms']):.4f}",f"{float(r['P_over_B4_p99_ratio']):.2f}"])+r'\\')
        out.append(' & '.join(['Mean','--','--','--',f"{sum(float(r['P_over_B4_p99_ratio']) for r in selected)/3:.2f}"])+r'\\')
    out += [r'\bottomrule\end{tabular}',r'\end{table}']
    (T/'followup_timing_reconciliation.tex').write_text('\n'.join(out)+'\n')

p=D/'followup/X3/X3_OPERATOR_SUMMARIES.json'
if p.exists():
    data=json.loads(p.read_text());assert data['operator_count']==54
    out=[]
    for stage in ['primary','expanded_burgers','nonnegative_heat']:
        selected=[r for r in data['rows'] if r['stage']==stage]
        if not selected:continue
        out += [r'\begin{table}[ht]',r'\centering\small',r'\caption{Follow-up error decomposition: '+{'primary':'primary checkpoints','expanded_burgers':'expanded-target Burgers checkpoints','nonnegative_heat':'NH heat checkpoints'}[stage]+r'. CPU diagnostic norms are not additive; the propagated-error ratio is not explained variance. Persist denotes the persistence forecast error.}',r'\begin{tabular}{@{}llrrrrrrrr@{}}\toprule',r'PDE & Method & Seed & Factor & $E_{\rm obs}$ & $E_{\rm prop}$ & $E_{\rm total}$ & $E_{\rm op}$ & Persist & Ratio\\\midrule']
        for r in selected:
            out.append(' & '.join(['Burgers' if r['pde']=='burgers' else ('NH heat' if stage=='nonnegative_heat' else 'SH heat'),r['method'].replace('P-no-rank','P-nr'),str(r['seed']),str(r['target_factor'] or '--')]+[f"{r[k]:.4f}" for k in ['E_obs','E_prop','E_total','E_op','persistence','E_prop_over_E_total']])+r'\\')
        out += [r'\bottomrule\end{tabular}',r'\end{table}']
    (T/'followup_error_decomposition.tex').write_text('\n'.join(out)+'\n')

p=D/'followup/X1/X1_ANALYSIS.csv'
if p.exists():
    data=rows(p);assert len(data)==19
    out=[];choices=[]
    for population,label in [('burgers_nominal','Burgers'),('heat_nonnegative_nominal','NH heat')]:
        selected=[r for r in data if r['population']==population];assert len(selected)==9
        out += [r'\begin{table}[ht]',r'\centering\small',r'\caption{D6 '+label+r': the same first 128 lexically sorted scenarios for every design. $K/H$ denotes candidate count and horizon; +FB adds one feedback-rollout candidate. Fixed and joint denote B0-denominator bootstrap treatments for pointwise 95\% intervals.}',r'\begin{tabular}{@{}lrrrrrr@{}}\toprule',r'$K/H$ & $n$ & Cost & Excess (\%) & Fixed interval (\%) & Joint interval (\%) & RMSE\\\midrule']
        choices += [r'\begin{table}[ht]',r'\centering\small',r'\caption{D6 '+label+r': candidate-index selection frequencies and realized action changes on the matched subset. Hold slot counts the designated candidate index; duplicate zero-offset candidates make this distinct from an unchanged applied action. Feedback rollout has its own selection index.}',r'\begin{tabular}{@{}lrrrrr@{}}\toprule',r'$K/H$ & Hold slot (\%) & B0 slot (\%) & FB slot (\%) & No action change (\%) & Mean $|\Delta a|$\\\midrule']
        for r in selected:
            cell=r['K']+'/'+r['H']+('+FB' if r['feedback_rollout'].lower()=='true' else '')
            out.append(' & '.join([cell,r['n'],f"{float(r['mean_episode_cost']):.6f}",f"{100*float(r['relative_excess_fixed']):+.2f}",f"[{100*float(r['fixed_ci_low']):+.2f}, {100*float(r['fixed_ci_high']):+.2f}]",f"[{100*float(r['paired_ci_low']):+.2f}, {100*float(r['paired_ci_high']):+.2f}]",f"{float(r['mean_tracking_rmse']):.6f}"])+r'\\')
            choices.append(' & '.join([cell,f"{100*float(r['hold_selection_rate']):.2f}",f"{100*float(r['b0_selection_rate']):.2f}",f"{100*float(r['feedback_rollout_selection_rate']):.2f}" if r['feedback_rollout_selection_rate'] else '--',f"{100*float(r['applied_no_change_rate']):.2f}",f"{float(r['mean_abs_action_change']):.6f}"])+r'\\')
        out += [r'\bottomrule\end{tabular}',r'\end{table}']
        choices += [r'\bottomrule\end{tabular}',r'\end{table}']
    (T/'followup_candidate_design.tex').write_text('\n'.join(out)+'\n')
    (T/'followup_candidate_choices.tex').write_text('\n'.join(choices)+'\n')
(D/'SUBMISSION_TABLE_AUDIT.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('Submission table normalization and main replacement:',len(checks),'source rows')
