"""Render TeX tables from included, independently audited numerical summaries."""
import csv, json
from pathlib import Path
O=Path(__file__).resolve().parents[1];D=O/'data';T=O/'tables'
def rows(f):return list(csv.DictReader((D/f).open(encoding='utf-8-sig',newline='')))
def label(m):return m.replace('P-no-rank','P-nr')
methods=['B0','B1','B2','B3','B4','B5','P-no-rank','P']
control=rows('control_all_methods.csv')
out=[r'\begin{table*}[t]',r'\caption{Original nominal parents: mean cost, tracking RMSE and excess cost relative to B0. Learned seeds are averaged within parent. Oracle rows reuse the original parents and are separate post-hoc diagnostics; State has privileged current truth and Observed uses B0 reconstruction. Original heat is distinct from the new heat population in Fig.~\ref{fig:control}.}\label{tab:nominal}',r'\centering\small',r'\begin{tabular}{@{}lrrrrrr@{}}\toprule',r'& \multicolumn{3}{c}{Burgers} & \multicolumn{3}{c}{Original heat}\\',r'\cmidrule(lr){2-4}\cmidrule(l){5-7}',r'Method & Cost & RMSE & $\Delta$ (\%) & Cost & RMSE & $\Delta$ (\%)\\\midrule']
for m in methods:
 vals=[label(m)]
 for pde in ['burgers','heat']:
  r=next(x for x in control if (x['role'],x['pde'],x['method'])==('locked_nominal',pde,m))
  vals += [f"{float(r['mean_cost']):.6f}",f"{float(r['tracking_rmse']):.6f}",f"{100*float(r['relative_cost_difference_vs_B0']):+.2f}"]
 out.append(' & '.join(vals)+r'\\')
revision=rows('revision/REVISION_COST_SUMMARY.csv')
for method,shown in [('O-cand-state','Oracle: State'),('O-cand-obs','Oracle: Observed')]:
 vals=[shown]
 for pde in ['burgers','heat']:
  selected=[r for r in revision if (r['stage'],r['role'],r['pde'],r['method'])==('E1','locked_nominal',pde,method)]
  assert len(selected)==1
  r=selected[0]
  vals += [f"{float(r['mean_cost']):.6f}",f"{float(r['tracking_rmse_mean']):.6f}",f"{100*float(r['relative_excess']):+.2f}"]
 out.append(' & '.join(vals)+r'\\')
out += [r'\bottomrule\end{tabular}',r'\end{table*}']
(T/'nominal.tex').write_text('\n'.join(out)+'\n')
out=[]
roles=['locked_nominal','locked_coefficient_ood','locked_delay_dropout'];rn=['Nominal','Coefficient shift','Delay/dropout']
for ri,role in enumerate(roles):
 if ri:out.append(r'\clearpage')
 for pde in ['burgers','heat']:
  out += [r'\begin{table}[ht]',r'\centering\small',f'\\caption{{{pde.capitalize()}: {rn[ri]}. Primary cost differences and paired-parent intervals relative to B0.}}',r'\begin{tabular}{@{}lrrrrr@{}}\toprule',r'Method & Cost & RMSE & $\Delta$ (\%) & 95\% interval (\%) & Violation (\%)\\\midrule']
  for m in methods:
   r=next(x for x in control if (x['role'],x['pde'],x['method'])==(role,pde,m))
   vals=[label(m),f"{float(r['mean_cost']):.6f}",f"{float(r['tracking_rmse']):.6f}",f"{100*float(r['relative_cost_difference_vs_B0']):+.3f}",f"[{100*float(r['ci_low']):+.3f}, {100*float(r['ci_high']):+.3f}]",f"{100*float(r['violation_rate']):.3f}"]
   out.append(' & '.join(vals)+r'\\')
  out += [r'\bottomrule\end{tabular}',r'\end{table}']
(T/'supp_control.tex').write_text('\n'.join(out)+'\n')
out=[]
for pi,pde in enumerate(['burgers','heat']):
 if pi:out.append(r'\clearpage')
 out += [r'\begin{table}[ht]',r'\centering\small',f'\\caption{{{pde.capitalize()}: complete method/seed host-ready timing. Each row pools 60,000 requests across three sessions.}}',r'\begin{tabular}{@{}llrrrrrr@{}}\toprule',r'Method & Seed & p50 & p95 & p99 & p99.9 & $>5$ ms & Rate (\%)\\\midrule']
 for r in rows('latency_per_seed.csv'):
  if r['pde']!=pde:continue
  vals=[label(r['method']),r['seed'] or '--']+[f"{float(r[k]):.4f}" for k in ['p50_ms','p95_ms','p99_ms','p999_ms']]+[r['misses_5ms'],f"{100*float(r['deadline_miss_rate_5ms']):.4f}"]
  out.append(' & '.join(vals)+r'\\')
 out += [r'\bottomrule\end{tabular}',r'\end{table}']
(T/'supp_latency.tex').write_text('\n'.join(out)+'\n')
out=[r'\begin{table}[ht]',r'\centering\small',r'\caption{Frozen passive forecast margins. Every group uses 64 parents, 512 queries, and zero-based order index 61.}',r'\begin{tabular}{@{}lllr@{}}\toprule',r'PDE & Method & Seed & Absolute-error margin\\\midrule']
for r in rows('calibration_margins.csv'):
 out.append(' & '.join([r['pde'].capitalize(),label(r['method']),r['seed'] or '--',f"{float(r['margin']):.6f}"])+r'\\')
out += [r'\bottomrule\end{tabular}',r'\end{table}']
(T/'supp_calibration.tex').write_text('\n'.join(out)+'\n')
print(json.dumps({'nominal_rows':10,'supplement_control_rows':48,'latency_rows':40,'calibration_rows':26}))
import subprocess, sys
subprocess.run([sys.executable,str(O/'scripts/build_revision_tables.py')],check=True)
if (D/'number_bindings.json').exists():
 # The public code release intentionally excludes manuscript/template sources.
 # A partial manuscript set remains an error; only the absent pair skips scanning.
 args=[] if any((O/name).exists() for name in ['main.tex','supplement.tex']) else ['--numbers-only']
 subprocess.run([sys.executable,str(O/'scripts/build_evidence_numbers.py'),*args],check=True)
if (D/'revision/E2_ANALYSIS.json').exists():
 subprocess.run([sys.executable,str(O/'scripts/audit_revision_timing_tables.py')],check=True)
subprocess.run([sys.executable,str(O/'scripts/audit_completed_tables.py')],check=True)
