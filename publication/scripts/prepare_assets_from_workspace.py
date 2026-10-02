"""Prepare publication inputs and editable vector figures from frozen summaries."""
import csv, hashlib, json, shutil, zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

B=Path(__file__).resolve().parent
R=B.parents[1]/'experiments/pdno_jevLite_20260927_v3'
O=B/'overleaf'; D=O/'data'; F=O/'figures'; T=O/'tables'
for p in (D,F,T,O/'scripts'):p.mkdir(exist_ok=True)
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def rows(p):return list(csv.DictReader(p.open(encoding='utf-8-sig',newline='')))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
sources=['reports/analysis_20260928/'+x for x in ['control_all_methods.csv','latency_pooled_descriptive.csv','latency_per_seed.csv','selected_checkpoints.csv','heat_constraint_diagnostic.csv','P_B5_matched_actions.csv','calibration_margins.csv']]
sources+=['evidence/latency_analysis_v3.json','evidence/confirmatory_checkpoint_manifest_v3.json','evidence/locked_test_analysis_v3.json','config/final_spec_v3.json','reports/source_audit.json','evidence/raw_evidence_freeze_v3.json','AUTHOR_INPUT_REQUIRED.md']
provenance=[]
for name in sources:
 p=R/name; provenance.append({'source':name,'sha256':sha(p),'bytes':p.stat().st_size})
 if p.suffix=='.csv' or p.name in ['latency_analysis_v3.json','final_spec_v3.json']:shutil.copy2(p,D/p.name)
(B/'qa/source_hashes.json').write_text(json.dumps(provenance,indent=2)+'\n')
with zipfile.ZipFile(B/'vendor/ACCESS_latex_template.zip') as z:
 (O/'bullet.png').write_bytes(z.read('ACCESS_latex_template_20260513/bullet.png'))
history=[]
for row in js(R/'evidence/confirmatory_checkpoint_manifest_v3.json')['checkpoints']:
 if row['method'] not in ['P','P-no-rank','B4','B5']:continue
 for h in row['validation_summary']['history']:
  history.append({k:v for k,v in {'pde':row['pde'],'method':row['method'],'seed':row['seed'],'selected_update':row['selected_update'],**h}.items() if not isinstance(v,(list,dict))})
fields=list(dict.fromkeys(k for h in history for k in h))
with (D/'training_history.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fields);w.writeheader();w.writerows(history)
shutil.copy2(__file__,O/'scripts/prepare_assets_from_workspace.py')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'axes.titlesize':9,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,'lines.linewidth':1.15,'savefig.facecolor':'white'})
BLUE='#23699D'; ORANGE='#C76B25'; GREY='#727D89'; DARK='#243648'; PALE='#EDF3F7'
methods=['B0','B1','B2','B3','B4','B5','P-no-rank','P']
labels={m:('P-nr' if m=='P-no-rank' else m) for m in methods}
colors={m:(ORANGE if m=='P' else BLUE if m=='B4' else DARK if m=='B0' else GREY) for m in methods}
fig_records=[]
def save(fig,name,desc,sources):
 fig.savefig(F/(name+'.pdf'),bbox_inches='tight',pad_inches=.04)
 fig.savefig(F/(name+'.svg'),bbox_inches='tight',pad_inches=.04)
 fig.savefig(F/(name+'.png'),dpi=300,bbox_inches='tight',pad_inches=.04)
 fig_records.append({'id':name,'description':desc,'sources':sources,'pdf_sha256':sha(F/(name+'.pdf')),'vector':True})
 plt.close(fig)
def panel(ax,title):
 ax.set_title(title,loc='left',fontweight='bold',pad=9)
 ax.grid(axis='x',color='#E5E9ED',lw=.6,zorder=0)
 ax.set_axisbelow(True)

# 1. Precisely labeled information flow, not a simulated trajectory.
fig,ax=plt.subplots(figsize=(7.16,4.35));ax.set(xlim=(-1.5,101),ylim=(-2,100));ax.axis('off')
def box(x,y,w,h,text,fc=PALE,ec=BLUE,size=8):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.45,rounding_size=1.0',facecolor=fc,edgecolor=ec,lw=.8))
 ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size,color=DARK,linespacing=1.45)
def arrow(x1,y1,x2,y2,color=DARK):
 ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=9,lw=.9,color=color))
ax.text(0,98,'a  Causal control loop',weight='bold',fontsize=9,va='top')
box(0,77,19,14,'Burgers / heat\n256-state solver');box(26,77,22,14,'Causal observation\nsensors + image + ages');box(55,77,22,14,'Controller + shared\nbox / slew projection');box(84,77,15,14,'Applied action\n2 actuators')
for x1,x2 in [(19,26),(48,55),(77,84)]:arrow(x1,84,x2,84)
ax.plot([91.5,91.5,9.5,9.5],[77,71,71,77],color=DARK,lw=.8);arrow(9.5,72,9.5,77)
ax.text(50,72,'200 steps; 20 ms simulation interval',ha='center',va='bottom',fontsize=7)
ax.text(0,65,'b  Shared encoding; different action-response computation',weight='bold',fontsize=9,va='top')
box(0,37,20,16,'Observation encoder\n+ initial-field head\nshared in both paths',size=7.4)
box(28,47,27,13,'P: one response branch\n6 action-basis responses',fc='#FCF0E7',ec=ORANGE)
box(28,25,27,13,'B4: conditioned branch\n10 candidates in one batch',size=7.6)
box(64,47,19,13,'Polynomial combination\nfor 10 actions',fc='#FCF0E7',ec=ORANGE,size=7.5)
box(64,25,19,13,'Candidate forecasts\nshared trunk',size=7.5)
box(88,34,11,16,'Score\nselect\nproject',fc='#F1F3F5',ec=GREY)
arrow(20,47,28,53.5,ORANGE);arrow(20,41,28,31.5,BLUE)
arrow(55,53.5,64,53.5,ORANGE);arrow(55,31.5,64,31.5,BLUE)
arrow(83,53.5,88,46,ORANGE);arrow(83,31.5,88,39,BLUE)
ax.text(0,16,'c  Frozen evidence',weight='bold',fontsize=9,va='top')
box(0,0,31,11,'Split by parent\ntrain / validation / calibration',fc='white',ec=GREY,size=7.3)
box(35,0,31,11,'Locked control evaluation\n1,280 parents; 25,600 executions',fc='white',ec=GREY,size=7.0)
box(70,0,29,11,'Separate serialized replay\n2.4 million timed requests',fc='white',ec=GREY,size=7.3)
save(fig,'fig01_design','Executed information flow and comparison boundary',['final_spec_v3.json','src/pdno/models/operators.py','src/pdno/evaluation/closed_loop.py'])

# 2. Seed-level prediction gates, common quantitative scale.
cp=rows(D/'selected_checkpoints.csv')
fig,axs=plt.subplots(1,2,figsize=(7.16,2.35),sharey=True,layout='constrained')
pred=['P','P-no-rank','B4','B5']
for ax,pde,letter in zip(axs,['burgers','heat'],'ab'):
 for j,m in enumerate(pred):
  rr=[r for r in cp if r['pde']==pde and r['method']==m]
  for dx,r,mk in zip([-.12,0,.12],rr,['o','s','^']):ax.plot(j+dx,float(r['validation_field_nrmse']),marker=mk,ms=5,color=colors[m],linestyle='none',markerfacecolor='white' if m!='P' else colors[m])
 ax.axhline(.05,color=DARK,ls='--',lw=.8);ax.text(3.35,.052,'gate 0.05',fontsize=7,ha='right',va='bottom')
 ax.set_xticks(range(4),[labels[m] for m in pred]);ax.set_ylim(0,.18);ax.set_yticks([0,.05,.10,.15]);ax.grid(axis='y',color='#E5E9ED',lw=.6)
 ax.set_title(f'{letter}  {pde.capitalize()}',loc='left',weight='bold')
axs[0].set_ylabel('Validation field nRMSE')
axs[1].legend([Line2D([],[],marker=m,color=GREY,ls='none',ms=4) for m in ['o','s','^']],['Seed 11','Seed 23','Seed 37'],loc='upper right',frameon=False,ncol=1)
save(fig,'fig02_prediction','Selected field validation error for all 24 predictive checkpoints',['selected_checkpoints.csv'])

# 3. Full comparator family in every locked condition with primary intervals.
control=rows(D/'control_all_methods.csv')
roles=['locked_nominal','locked_coefficient_ood','locked_delay_dropout']; rolelabels=['Nominal','Coefficient shift','Delay / dropout']
fig,axs=plt.subplots(2,3,figsize=(7.16,4.75),sharey=True,layout='constrained')
for pi,pde in enumerate(['burgers','heat']):
 for ri,role in enumerate(roles):
  ax=axs[pi,ri]
  for j,m in enumerate(methods):
   r=next(r for r in control if (r['pde'],r['role'],r['method'])==(pde,role,m))
   v,lo,hi=[100*float(r[k]) for k in ['relative_cost_difference_vs_B0','ci_low','ci_high']]
   ax.errorbar(v,j,xerr=[[v-lo],[hi-v]],fmt='s' if m=='P' else 'o',color=colors[m],ms=4.2 if m=='P' else 3.5,capsize=2,lw=.9,markerfacecolor=colors[m] if m in ['P','B0'] else 'white')
  ax.axvline(0,color=DARK,lw=.8);ax.axvline(5,color=DARK,ls='--',lw=.75)
  ax.set_yticks(range(8),[labels[m] for m in methods]);ax.set_ylim(7.7,-.7)
  ax.set_xlim((-3,58) if pde=='burgers' else (-8,175));ax.set_xticks([0,20,40] if pde=='burgers' else [0,50,100,150])
  panel(ax,f'{chr(97+pi*3+ri)}  {pde.capitalize()}\n{rolelabels[ri]}')
  if pi==1:ax.set_xlabel('Cost difference vs. B0 (%)')
save(fig,'fig03_control','All eight controllers; primary paired-parent 95% intervals; dashed 5% margin',['control_all_methods.csv'])

# 4. All-method tail latency and exact deadline-miss proportions (linear scales).
lat=rows(D/'latency_pooled_descriptive.csv')
fig,axs=plt.subplots(2,2,figsize=(7.16,4.3),sharey=True,layout='constrained')
for pi,pde in enumerate(['burgers','heat']):
 for j,m in enumerate(methods):
  r=next(r for r in lat if (r['pde'],r['method'])==(pde,m))
  axs[pi,0].plot(float(r['p99_ms']),j,'s' if m=='P' else 'o',ms=4.6,color=colors[m])
  rate=100*float(r['miss_rate_5ms']);axs[pi,1].barh(j,rate,height=.53,color=colors[m],alpha=.92)
  txt='0' if rate==0 else f'{rate:.3f}' if rate<.1 else f'{rate:.2f}'
  axs[pi,1].text(rate+1.2,j,txt,fontsize=6.6,va='center')
 for ri in range(2):
  axs[pi,ri].set_yticks(range(8),[labels[m] for m in methods]);axs[pi,ri].set_ylim(7.7,-.7)
  panel(axs[pi,ri],f'{chr(97+pi*2+ri)}  {pde.capitalize()}: '+('99th percentile' if ri==0 else '>5 ms requests'))
 axs[pi,0].set_xlim(0,17);axs[pi,0].set_xticks([0,2,5,10,15]);axs[pi,0].axvline(2,color=DARK,ls='--',lw=.8);axs[pi,0].axvline(5,color=GREY,ls=':',lw=.8)
 axs[pi,1].set_xlim(0,112);axs[pi,1].set_xticks([0,25,50,75,100])
axs[1,0].set_xlabel('Host-ready latency (ms)');axs[1,1].set_xlabel('Deadline-miss proportion (%)')
save(fig,'fig04_latency','Pooled raw-request p99 and exact 5 ms miss proportions for all methods',['latency_pooled_descriptive.csv'])

# 5. Primary H1 interval, rather than a ratio of pooled percentiles.
h1=js(D/'latency_analysis_v3.json')['H1_p99_ratio_analysis']
(D/'h1_results.json').write_text(json.dumps(h1,indent=2))
print('H1 schema:',list(h1[0]))
fig,ax=plt.subplots(figsize=(3.45,1.8),layout='constrained')
# Exact field names are resolved from the primary artifact, never guessed values.
for i,r in enumerate(h1):
 mean=r['mean_per_seed_ratio']
 ci=r['hierarchical_seed_session_bootstrap_ci95']
 if mean is None or ci is None:
  print(json.dumps(r));plt.close(fig);break
 ax.errorbar(mean,i,xerr=[[mean-ci[0]],[ci[1]-mean]],fmt='o',color=ORANGE,ms=5,capsize=3)
 ax.text(.76,i+.18,f'{mean:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]',fontsize=6.8)
else:
 ax.axvline(.75,color=BLUE,ls='--',lw=.9);ax.axvline(1,color=DARK,lw=.8);ax.set_xlim(.73,1.03);ax.set_ylim(1.55,-.55)
 ax.set_yticks([0,1],['Burgers','Heat']);ax.set_xticks([.75,.85,.95,1]);ax.set_xlabel('Mean seed-specific p99 ratio, P / B4')
 save(fig,'fig05_h1','Hierarchical seed/session bootstrap intervals and 25% reduction target',['h1_results.json'])

# 6. Heat endpoint decomposition. Counts are identical across controller/seed variants.
heat=rows(D/'heat_constraint_diagnostic.csv')
fig,ax=plt.subplots(figsize=(3.45,2.25),layout='constrained')
for i,role in enumerate(roles):
 rr=[r for r in heat if r['role']==role]
 assert len({(r['n'],r['first_step_lower_count'],r['ever_lower_count'],r['ever_upper_count'],r['late_new_lower_count']) for r in rr})==1
 r=rr[0];n=int(r['n']);first=int(r['first_step_lower_count']);valid=n-first
 ax.barh(i,100*first/n,color=ORANGE,height=.52,label='First-step violation' if i==0 else None)
 ax.barh(i,100*valid/n,left=100*first/n,color='#DCE4EB',height=.52,label='No recorded violation' if i==0 else None)
 ax.text(2,i,f'{first}/{n}',color='white',weight='bold',fontsize=8,va='center')
ax.set_yticks(range(3),['Nominal','Coefficient shift','Delay / dropout']);ax.set_xlim(0,100);ax.invert_yaxis();ax.set_xlabel('Parents (%)');ax.set_xticks([0,25,50,75,100])
ax.legend(frameon=False,fontsize=6.2,loc='upper center',bbox_to_anchor=(.43,-.24),ncol=1)
save(fig,'fig06_heat','All observed heat episode events already occur at first advanced step; no later new or upper events',['heat_constraint_diagnostic.csv'])

# Supplement: validation histories, one panel per PDE/family, no new training.
fig,axs=plt.subplots(2,4,figsize=(7.16,3.6),sharex=True,sharey=True,layout='constrained')
for pi,pde in enumerate(['burgers','heat']):
 for mi,m in enumerate(pred):
  ax=axs[pi,mi]
  for seed,ls in zip([11,23,37],['-','--',':']):
   rr=[r for r in history if r['pde']==pde and r['method']==m and r['seed']==seed]
   ax.plot([r['update'] for r in rr],[r['validation_field_nrmse'] for r in rr],ls=ls,color=colors[m],lw=.9,label=str(seed))
  ax.axhline(.05,color=DARK,lw=.7,ls='--');ax.axvline(2000,color=GREY,lw=.6,ls=':')
  ax.set_title(pde.capitalize()+' / '+labels[m],fontsize=8,loc='left');ax.set_ylim(0,.3);ax.set_xticks([1000,3000,5000]);ax.tick_params(axis='x',labelsize=6)
  if mi==0:ax.set_ylabel('Validation nRMSE')
  if pi==1:ax.set_xlabel('Update')
axs[0,3].legend(title='Seed',frameon=False,fontsize=6,title_fontsize=6)
save(fig,'figS01_training','Recorded validation histories; phase boundary at update 2000',['training_history.csv'])
(D/'figure_provenance.json').write_text(json.dumps(fig_records,indent=2)+'\n')
print('Figures:',len(fig_records))

# Portable regeneration uses the included small tables, not the experiment tree.
portable='''import csv, hashlib, json\nfrom pathlib import Path\nimport numpy as np\nimport matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\nfrom matplotlib.patches import FancyBboxPatch, FancyArrowPatch\nfrom matplotlib.lines import Line2D\nO=Path(__file__).resolve().parents[1];D=O/'data';F=O/'figures'\ndef js(p):return json.loads(p.read_text(encoding='utf-8-sig'))\ndef rows(p):return list(csv.DictReader(p.open(encoding='utf-8-sig',newline='')))\ndef sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()\nhistory=rows(D/'training_history.csv')\nfor r in history:\n for k in ['seed','update','selected_update']:r[k]=int(r[k])\n r['validation_field_nrmse']=float(r['validation_field_nrmse'])\n'''
body=Path(__file__).read_text(encoding='utf-8').split('plt.rcParams.update',1)[1].split('# Portable regeneration',1)[0]
(O/'scripts/build_figures.py').write_text(portable+'plt.rcParams.update'+body,encoding='utf-8')
