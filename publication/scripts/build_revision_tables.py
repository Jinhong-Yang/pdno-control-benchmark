"""Portable TeX tables from revision CSV/JSON evidence; no experimental execution."""
from pathlib import Path
import csv,json,hashlib
O=Path(__file__).resolve().parents[1];D=O/'data/revision';T=O/'tables';records=[]
def read(name):return list(csv.DictReader((D/name).open(newline='',encoding='utf-8')))
def write(name,text,inputs):
    path=T/name;path.write_text(text+'\n',encoding='utf-8');records.append({'table':str(path.relative_to(O)),'inputs':[{ 'path':str((D/p).relative_to(O)),'sha256':hashlib.sha256((D/p).read_bytes()).hexdigest()} for p in inputs],'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
def table(caption,label,cols,header,body):
    return '\\begin{table}[ht]\n\\caption{'+caption+'}\\label{'+label+'}\n\\centering\\small\n\\begin{tabular}{@{}'+cols+'@{}}\\toprule\n'+header+'\\\\\\midrule\n'+'\n'.join(body)+'\n\\bottomrule\\end{tabular}\n\\end{table}'
def sci(v):
    m,e=f'{v:.3e}'.split('e');return '$'+m+'\\times10^{'+str(int(e))+'}$'
p0=json.loads((D/'P0_1_gradient_audit.json').read_text());body=[]
for r in p0['reports']:
    body.append(r['pde'].capitalize()+' & '+sci(r['median_weighted_physics_data_gradient_ratio'])+' & '+sci(r['median_balance_data_gradient_ratio'])+' & '+sci(r['final_parameter_difference'])+' & '+str(r['physics_gradient_nonzero_updates'])+'/100 \\\\')
write('revision_gradients.tex',table('Matched restart diagnostic at updates 2,001--2,100. Ratios are medians of weighted component-gradient norms divided by the data-gradient norm; parameter differences are final Euclidean norms.','tab:revision_gradients','lrrrr','PDE & Physics/data & Balance/data & Parameter difference & Nonzero',body),['P0_1_gradient_audit.json'])
e1=[r for r in read('REVISION_COST_SUMMARY.csv') if r['stage']=='E1'];body=[];roles={'locked_nominal':'Nominal','locked_coefficient_ood':'Coefficient shift','locked_delay_dropout':'Delay/dropout'}
for r in e1:
    method='State' if r['method']=='O-cand-state' else 'Observed'
    body.append(f"{r['pde'].capitalize()} & {roles[r['role']]} & {method} & {float(r['mean_cost']):.6f} & {100*float(r['relative_excess']):.2f} & [{100*float(r['fixed_ci_low']):.2f}, {100*float(r['fixed_ci_high']):.2f}] \\\\")
write('revision_oracle.tex',table('Reference-solver candidate controllers on the original parents. Excess cost and fixed-denominator paired-parent 95\\% intervals are in percent relative to B0. State uses privileged current truth; Observed uses B0 modal reconstruction.','tab:revision_oracles','lllrrl','PDE & Condition & Oracle & Cost & Excess (\\%) & 95\\% interval',body),['REVISION_COST_SUMMARY.csv'])
orig=list(csv.DictReader((O/'data/control_all_methods.csv').open(newline='',encoding='utf-8-sig')));e7=read('E7_joint_denominator_bootstrap.csv');sections=[]
for pde in ['burgers','heat']:
    for role,rolelabel in roles.items():
        body=[]
        for r in [x for x in e7 if x['pde']==pde and x['role']==role]:
            old=next(x for x in orig if (x['pde'],x['role'],x['method'])==(pde,role,r['method']));method=r['method'].replace('P-no-rank','P-nr')
            body.append(f"{method} & {float(r['estimate_percent']):.2f} & [{100*float(old['ci_low']):.2f}, {100*float(old['ci_high']):.2f}] & [{float(r['joint_low']):.2f}, {float(r['joint_high']):.2f}] \\\\")
        sections.append(table(f'Original {pde}, {rolelabel.lower()}: excess cost (percent) and 95\\% intervals. The archived primary interval is retained; the sensitivity interval jointly resamples numerator and denominator.',f'tab:joint_{pde}_{role}','lrll','Method & Excess (\\%) & Archived fixed & Joint denominator',body))
write('revision_bootstrap.tex','\n'.join(sections),['E7_joint_denominator_bootstrap.csv'])
records[-1]['inputs'].append({'path':'data/control_all_methods.csv','sha256':hashlib.sha256((O/'data/control_all_methods.csv').read_bytes()).hexdigest()})
e6=read('E6_original_heat_cost_decomposition.csv');sections=[]
for role,rolelabel in roles.items():
    body=[]
    for method in ['B0','B1','B2','B3','B4','B5','P-no-rank','P']:
        group=[r for r in e6 if r['role']==role and r['method']==method];means=[sum(float(r[k]) for r in group)/len(group) for k in ['cost','tracking','action','slew','violation','first_step_violation_cost']]
        body.append(method.replace('P-no-rank','P-nr')+' & '+' & '.join(f'{v:.6f}' for v in means)+' \\\\')
    sections.append(table(f'Original heat benchmark, {rolelabel.lower()}: additive mean episode-cost components. First-step column is the first-step portion of the violation component, not an additional cost.',f'tab:costparts_{role}','lrrrrrr','Method & Total & Tracking & Action & Slew & Violation & First step',body))
write('revision_heat_cost.tex','\n'.join(sections),['E6_original_heat_cost_decomposition.csv'])
freq=read('E1_candidate_frequency.csv');sections=[]
for pde in ['burgers','heat']:
    body=[]
    for role,rolelabel in roles.items():
        for method in ['P','B4','B5','P-no-rank','O-cand-state','O-cand-obs']:
            group=[r for r in freq if r['source']!='E5_raw_recorded_indices' and r['pde']==pde and r['role']==role and (r['method_seed']==method or r['method_seed'].rsplit('_s',1)[0]==method)]
            fields=[]
            for index in [9,4]:
                rr=[r for r in group if int(r['candidate'])==index];n=sum(int(r['requests']) for r in rr)
                lo=sum(float(r['frequency_lower'])*int(r['requests']) for r in rr)/n;hi=sum(float(r['frequency_upper'])*int(r['requests']) for r in rr)/n
                fields.append(f'{100*lo:.2f}' if lo==hi else f'[{100*lo:.2f}, {100*hi:.2f}]')
            body.append(rolelabel+' & '+method.replace('P-no-rank','P-nr')+' & '+' & '.join(fields)+' \\\\')
    sections.append(table(f'{pde.capitalize()}: B0-command and hold-candidate frequencies (percent). Intervals are action-compatibility bounds at absolute tolerance $10^{{-7}}$ per actuator, not confidence intervals. Oracles use recorded indices at all 200 ticks; original learned methods use 20 stored snapshots per parent.',f'tab:choices_{pde}','llll','Condition & Method & B0 candidate & Hold candidate',body))
write('revision_choices.tex','\n'.join(sections),['E1_candidate_frequency.csv'])
if any(r['source']=='E5_raw_recorded_indices' for r in freq):
    body=[]
    for role,rolelabel in roles.items():
        for method in ['P','B4','B5','P-no-rank']:
            group=[r for r in freq if r['source']=='E5_raw_recorded_indices' and r['role']==role and r['method_seed'].rsplit('_s',1)[0]==method];fields=[]
            for index in [9,4]:
                rr=[r for r in group if int(r['candidate'])==index];n=sum(int(r['requests']) for r in rr);fields.append(f"{100*sum(float(r['frequency_lower'])*int(r['requests']) for r in rr)/n:.2f}")
            body.append(rolelabel+' & '+method.replace('P-no-rank','P-nr')+' & '+' & '.join(fields)+' \\\\')
    write('revision_new_heat_choices.tex',table('Nonnegative-initial-field heat benchmark: exact recorded B0-command and hold-candidate frequencies over all 200 ticks. This new parent population is not pooled with the original snapshot analysis.','tab:choices_new_heat','llll','Condition & Method & B0 candidate (\\%) & Hold candidate (\\%)',body),['E1_candidate_frequency.csv'])
if (D/'E2_ANALYSIS.json').exists():
    from build_revision_timing_tables import build as build_timing
    records.extend(build_timing())
if (D/'REVISION_TRAINING_EXPORT.json').exists() and (D/'REVISION_RAW_AUDIT.json').exists():
    assert json.loads((D/'REVISION_RAW_AUDIT.json').read_text())['status']=='PASS'
    from build_revision_training_tables import build as build_training
    records.extend(build_training(O))
elif (D/'E3_TRAINING_EXPORT.json').exists():
    from build_revision_training_tables import build_scaling
    records.extend(build_scaling(O, 'E3'))
(D/'table_provenance.json').write_text(json.dumps(records,indent=2));print('Revision table files',len(records))
