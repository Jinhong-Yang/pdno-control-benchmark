"""Export the completed E3 histories without claiming E4/E5 completion."""
from pathlib import Path
import csv
import hashlib
import itertools
import json
from export_training_evidence import extract

R = Path(__file__).resolve().parents[1]
audit_path = R / 'results/E3_COMPLETION_AUDIT.json'
audit = json.loads(audit_path.read_text())
assert audit['status'] == 'PASS_COMPLETED_E3_SELECTION_AND_FREEZE_SCOPE'
freeze = json.loads((R / 'evidence/E3_POLICY_FREEZE.json').read_text())
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
assert len(freeze['checkpoints']) == 21
for item in freeze['checkpoints']:
    assert sha(R / item['path']) == item['sha256']
selected, curves = [], []
for path in sorted((R / 'checkpoints/E3').rglob('*.summary.json')):
    row, trace = extract(path, 'E3')
    selected.append(row)
    curves.extend(trace)
assert len(selected) == 21
operators = [row for row in selected if row['method'] != 'observer']
assert {(r['target_factor'], r['method'], r['seed']) for r in operators} == set(
    itertools.product([1, 4, 16], ['P', 'B4'], [11, 23, 37]))
assert sum(r['accuracy_gate_005'] for r in operators) == audit['eligible_operator_count']
outputs = {}
for suffix, rows in [('TRAINING_SELECTION', selected), ('TRAINING_CURVES', curves)]:
    path = R / f'results/E3_{suffix}.csv'
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    outputs[path.name] = sha(path)
(R / 'results/E3_TRAINING_EXPORT.json').write_text(json.dumps({
    'status': 'COMPLETE_E3_ONLY',
    'selected_checkpoint_rows': len(selected),
    'curve_metric_rows': len(curves),
    'input_sha256': {r['summary_path']: r['summary_sha256'] for r in selected},
    'output_sha256': outputs,
    'completion_audit_sha256': sha(audit_path),
    'scope': 'All E3 conditions only; E4/E5 completion is not asserted.',
    'selection': 'Recorded composite selection; no minimum-field-error reselection.',
}, indent=2) + '\n', encoding='utf-8')
print('Completed E3 export:', len(selected), 'selected checkpoints;', len(curves), 'metric records')
