"""CPU support check: archived controllers load from a relocated source/weight tree.

Uses one training observation per PDE and seed11 only. This does not evaluate
control quality or certify the future complete v1.1 release.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

R = Path(__file__).resolve().parents[1]
OLD = R.parents[1] / 'experiments/pdno_jevLite_20260927_v3'
METHODS = ['P', 'P-no-rank', 'B5', 'B4', 'B2', 'B3']
KEYS = ['sensor_value', 'sensor_mask', 'sensor_age', 'instrument_image',
        'image_mask', 'goal_coefficients', 'material_context',
        'previous_applied_action', 'applied_action_history', 'image_age',
        'image_valid', 'candidate_action']

CHILD = r'''
from pathlib import Path
import sys,json,hashlib
import numpy as np
import torch
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'src'))
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from pdno.evaluation.closed_loop import load_controller
from pdno.training.fit import _grid
original_load=torch.load
loaded=[]
def guarded_load(path,*args,**kwargs):
    resolved=Path(path).resolve()
    assert resolved.is_relative_to((root/'models').resolve()),str(resolved)
    loaded.append(resolved.name)
    return original_load(path,*args,**kwargs)
torch.load=guarded_load
rows=[]
for pde in ['burgers','heat']:
    with np.load(root/(pde+'.npz'),allow_pickle=False) as z:
        fixture={k:torch.as_tensor(z[k]) for k in z.files}
    actions=fixture.pop('candidate_action').float()
    x,tau=_grid(pde,torch.device('cpu'))
    for method in ['P','P-no-rank','B5','B4','B2','B3']:
        checkpoint=root/'models'/(pde+'_'+method+'.pt')
        model=load_controller(method,pde,11,checkpoint,torch.device('cpu'))
        with torch.no_grad():
            output=model(fixture) if method in ['B2','B3'] else model(fixture,actions,x,tau)
        expected=(1,2) if method in ['B2','B3'] else (1,10,8,128)
        assert tuple(output.shape)==expected and torch.isfinite(output).all()
        values=output.numpy()
        rows.append({'pde':pde,'method':method,'seed':11,'shape':list(output.shape),
                     'output_sha256':hashlib.sha256(values.tobytes()).hexdigest()})
assert len(loaded)==12 and len(set(loaded))==12
print(json.dumps({'rows':rows,'checkpoint_reads':loaded}))
'''


def main():
    hashes = {}
    with tempfile.TemporaryDirectory(prefix='pdno_checkpoint_relocation_') as td:
        root = Path(td)
        shutil.copytree(R / 'src', root / 'src', ignore=shutil.ignore_patterns('__pycache__'))
        (root / 'models').mkdir()
        for pde in ['burgers', 'heat']:
            source = OLD / f'data/queries_v3/train/{pde}/teacher_queries.npz'
            with np.load(source, allow_pickle=False) as archive:
                np.savez(root / (pde+'.npz'), **{key: archive[key][:1] for key in KEYS})
            for method in METHODS:
                source = OLD / f'runs/confirmatory_v3_training/{pde}/{method}_s11.pt'
                destination = root / 'models' / (pde+'_'+method+'.pt')
                shutil.copy2(source, destination)
                source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
                assert hashlib.sha256(destination.read_bytes()).hexdigest() == source_hash
                hashes[str(source.relative_to(R.parents[1]))] = source_hash
        (root / 'check.py').write_text(CHILD, encoding='utf-8')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', PYTHONPATH='')
        result = subprocess.run([sys.executable, '-B', '-X', 'utf8', str(root/'check.py')],
                                cwd=root, env=env, capture_output=True, text=True,
                                encoding='utf-8', timeout=120)
        if result.returncode:
            raise RuntimeError(result.stdout+'\n'+result.stderr)
        output = json.loads(result.stdout)
    receipt = {'status': 'PASS_RELOCATED_ARCHIVED_CONTROLLERS_CPU_SUPPORT',
               'scope': 'Twelve original seed11 controllers, one training observation per PDE. No evaluation of revision fitted checkpoints, full source-release layout, scientific utility or GPU latency.',
               'external_checkpoint_reads_allowed': False, 'input_sha256': hashes, **output}
    (R/'results/CHECKPOINT_RELOCATION_SUPPORT.json').write_text(json.dumps(receipt, indent=2))
    print('Relocated checkpoint support PASS:', len(output['rows']), 'CPU forward cases')


if __name__ == '__main__':
    main()
