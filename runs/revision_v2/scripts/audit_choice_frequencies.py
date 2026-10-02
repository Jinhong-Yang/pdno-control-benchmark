"""Action-equivalence bounds from recorded snapshots; no invented tie indices."""
from pathlib import Path
import json,csv
import numpy as np
R=Path(__file__).resolve().parents[1];OLD=R.parents[1]/'experiments/pdno_jevLite_20260927_v3';rows=[]
for path in sorted((OLD/'evidence/locked_test_raw_v3').glob('*/*/*.npz')):
    if not any(path.name.startswith(m+'_s') for m in ['P','P-no-rank','B4','B5']):continue
    with np.load(path,allow_pickle=False) as z:
        candidates=z['latency_candidate_action'];actual=z['action_applied'][:,::10];assert candidates.shape[:2]==actual.shape[:2]
        matches=np.all(np.abs(candidates-actual[:,:,None,:])<=1e-7,axis=-1);n=matches.shape[0]*matches.shape[1]
        assert matches.any(-1).all();ambiguous=matches.sum(-1)>1
        for candidate in range(10):
            compatible=matches[...,candidate];unique=compatible & ~ambiguous
            rows.append({'source':'original_recorded_snapshot_action_equivalence','role':path.parent.parent.name,'pde':path.parent.name,'method_seed':path.stem,'candidate':candidate,'requests':n,'frequency_lower':float(unique.mean()),'frequency_upper':float(compatible.mean()),'ambiguous_fraction':float(ambiguous.mean()),'sampling':'ticks 0,10,...,190 (20 of 200); indices unavailable in original raw; duplicate projected actions create bounds'})
for path in sorted(p for root in ['E1_raw','E5_raw'] for p in (R/'results'/root).glob('*/*/*.npz')):
    if path.parents[2].name=='E5_raw' and not any(path.name.startswith(m+'_s') for m in ['P','P-no-rank','B4','B5']):continue
    with np.load(path,allow_pickle=False) as z:choices=z['selected_candidate_index'];assert (choices>=0).all()
    for candidate in range(10):
        f=float(np.mean(choices==candidate));rows.append({'source':path.parents[2].name+'_recorded_indices','role':path.parent.parent.name,'pde':path.parent.name,'method_seed':path.stem,'candidate':candidate,'requests':int(choices.size),'frequency_lower':f,'frequency_upper':f,'ambiguous_fraction':0.,'sampling':'all 200 decision ticks; ties resolved by actual argmin'})
with (R/'results/E1_candidate_frequency.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(R/'results/E1_candidate_frequency_record.json').write_text(json.dumps({'rows':len(rows),'B0_candidate_index':9,'hold_candidate_index':4,'original_bound_interpretation':'An action may equal several projected candidates. Lower bound counts uniquely identifying actions; upper bound counts all compatible actions. Never interpreted as exact original argmin frequency.','revision_oracle_files_available':len(list((R/'results/E1_raw').glob('*/*/*.npz')))},indent=2))
print('Candidate-frequency rows',len(rows))
