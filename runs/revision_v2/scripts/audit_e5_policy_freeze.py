"""Read-only check of E5 seal linkage, current hashes, and actual parent splits.

Requires completed E5 training and new test generation, not completed control
evaluation. The recorded local execution order is not external preregistration
or independent timestamp certification. This script never generates test data.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np

R = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    freeze_path = R/'evidence/E5_POLICY_FREEZE.json'
    generation_path = R/'results/E5_TEST_GENERATION.json'
    training_path = R/'results/E5_GPU_TRAINING.json'
    freeze = json.loads(freeze_path.read_text())
    generation = json.loads(generation_path.read_text())
    training = json.loads(training_path.read_text())
    assert training['status'] == 'TRAINING_COMPLETE_EVALUATION_PENDING' and training['stage'] == 'E5'
    assert freeze['status'] == 'FROZEN_BEFORE_NEW_TEST_GENERATION'
    assert freeze['new_heat_test_generated'] is False
    assert generation['status'] == 'COMPLETE_AFTER_POLICY_FREEZE'
    assert generation['freeze_sha256'] == sha(freeze_path)
    expected = {f'checkpoints/E5/{m}_s{s}.pt' for m in ['P','B4','B5','P-no-rank','B2','B3'] for s in [11,23,37]}
    expected.add('checkpoints/E5/observer/heat_seed7.pt')
    actual = {row['path'].replace('\\','/') for row in freeze['checkpoints']}
    assert actual == expected and len(freeze['checkpoints']) == 19
    checked = {}
    for row in freeze['checkpoints'] + freeze['source_and_config']:
        path = (R / row['path']).resolve()
        assert path.is_relative_to(R.resolve())
        digest = sha(path)
        assert digest == row['sha256'], str(path)
        checked[str(path.relative_to(R))] = digest
    source_names = {row['path'].replace('\\','/') for row in freeze['source_and_config']}
    required_sources = {'config/PHASE1_SPEC.json','config/heat_parent_manifest.json','heat_ic_spec.md','scripts/evaluate_revision.py'}
    required_sources.update(p.relative_to(R).as_posix() for p in (R/'src').rglob('*.py'))
    assert source_names == required_sources
    assert len(training['rows']) == 19
    assert {r['condition'] for r in training['rows']} == {
        f'{m}_s{s}' for m in ['P','B4','B5','P-no-rank','B2','B3'] for s in [11,23,37]} | {'heat_observer'}

    manifest = json.loads((R/'config/heat_parent_manifest.json').read_text())
    block = manifest['roles']['heat']
    counts = {'train':256,'validation':64,'locked_nominal':384,'locked_coefficient_ood':128,'locked_delay_dropout':128}
    assert block['role_counts'] == counts
    assert len({p['parent_id'] for p in block['parents']}) == sum(counts.values())
    recorded = {r['split']:r for r in generation['records']}
    assert set(recorded) == {role for role in counts if role.startswith('locked_')}
    split_ids = {}
    for role, count in counts.items():
        expected_ids = {p['parent_id'] for p in block['parents'] if p['split'] == role}
        assert len(expected_ids) == count
        if role in ['train','validation']:
            path = R/f'data/positive_heat_queries/{role}/heat/teacher_queries.npz'
            digest = sha(path)
            assert digest == freeze[role+'_targets_sha256']
            with np.load(path, allow_pickle=False) as arrays:
                ids = set(arrays['parent_id'].astype(str).tolist())
        else:
            path = R/f'data/positive_heat/{role}/heat/trajectories.npz'
            digest = sha(path)
            record = recorded[role]
            assert record['pde'] == 'heat' and record['parent_count'] == count and record['outer_ticks'] == 200
            assert record['fields_shape'] == [count,201,256]
            with np.load(path, allow_pickle=False) as arrays:
                raw_ids = arrays['parent_id'].astype(str).tolist()
                assert len(raw_ids) == len(set(raw_ids)) == count
                assert np.all(arrays['role'].astype(str) == role)
                ids = set(raw_ids)
        assert ids == expected_ids, role
        assert all(not ids.intersection(other) for other in split_ids.values())
        split_ids[role] = ids
        checked[str(path.relative_to(R))] = digest
    for path in [freeze_path, generation_path, training_path]:
        checked[str(path.relative_to(R))] = sha(path)
    result = dict(status='PASS_E5_SEAL_LINKAGE_CURRENT_HASHES_AND_PARENT_SPLITS',
                  utc=datetime.now(timezone.utc).isoformat(), checkpoint_count=19,
                  source_config_count=len(source_names), parent_counts={r:len(ids) for r,ids in split_ids.items()},
                  freeze_created_utc=freeze['created_utc'], source_sha256=checked, auditor_sha256=sha(Path(__file__)),
                  limits='Confirms current files and local seal-to-generation linkage. Source control flow verifies then generates test data. Local recorded chronology is not external preregistration or independently certified timing. Does not audit control outcomes, model predictions or final scientific benefit.')
    (R/'results/E5_POLICY_FREEZE_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['status','checkpoint_count','source_config_count','parent_counts']}))


if __name__ == '__main__':
    main()
