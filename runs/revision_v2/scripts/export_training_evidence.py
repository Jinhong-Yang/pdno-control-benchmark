"""Export recorded selection and validation histories, without loading models or fitting."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import re

R = Path(__file__).resolve().parents[1]
OLD = R.parents[1] / 'experiments/pdno_jevLite_20260927_v3'


def extract(path, stage):
    report = json.loads(path.read_text(encoding='utf-8-sig'))
    history = report['history']
    chosen = int(report['best_step'])
    matches = [h for h in history if int(h.get('update', h.get('step'))) == chosen]
    assert len(matches) == 1, (path, chosen)
    selected = matches[0]
    actual = int(report.get('actual_updates', report.get('updates', 0)))
    assert actual >= chosen > 0
    assert max(int(h.get('update', h.get('step'))) for h in history) == actual
    factor_match = re.search(r'_x(1|4|16)(?:_|\.)', path.name)
    factor = int(factor_match.group(1)) if stage == 'E3' and factor_match else 1
    method = report['method']
    cap = report.get('max_updates', report.get('updates'))
    if cap is None:
        cap = {'B2': 3000, 'B3': 1000}[method]
    nrmse = selected.get('validation_field_nrmse')
    selection_metric = ('validation_nrmse' if method == 'observer' else
                        'validation_action_mse' if method == 'B2' else 'selection_score')
    assert selection_metric in selected, (path, selection_metric)
    row = {
        'stage': stage, 'pde': report['pde'], 'method': method, 'seed': report['seed'],
        'condition': path.name.removesuffix('.summary.json'), 'target_factor': factor,
        'update_cap': int(cap), 'actual_updates': actual, 'selected_update': chosen,
        'validation_field_nrmse': nrmse,
        'accuracy_gate_005': '' if nrmse is None else nrmse <= .05,
        'validation_observer_nrmse': selected.get('validation_nrmse'),
        'selection_score': report.get('selection_score'),
        'selection_metric': selection_metric,
        'selection_value': selected[selection_metric],
        'validation_teacher_regret': selected.get('validation_normalized_teacher_regret_mean'),
        'validation_teacher_best_agreement': selected.get('validation_teacher_best_agreement'),
        'process_wall_seconds': report['elapsed_seconds'],
        'summary_path': str(path.relative_to(R.parents[1])),
        'summary_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    assert actual <= row['update_cap']
    curves = []
    for entry in history:
        step = int(entry.get('update', entry.get('step')))
        common = {k: row[k] for k in ['stage', 'pde', 'method', 'seed', 'condition', 'target_factor']}
        for key, value in entry.items():
            if isinstance(value, (int, float)) and key not in ['update', 'step']:
                curves.append({**common, 'update': step, 'selected_update': chosen,
                               'phase': entry.get('phase', 'not_applicable'), 'metric': key, 'value': value})
    return row, curves


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original-only', action='store_true')
    args = parser.parse_args()
    selected, curves = [], []
    sources = [('original', OLD / 'runs/confirmatory_v3_training')]
    if not args.original_only:
        for stage, folder, expected in [('E3', 'E3', 21), ('E5', 'E5', 19), ('E4', 'E4_cuda', 10)]:
            completion = json.loads((R / f'results/{stage}_GPU_TRAINING.json').read_text())
            assert completion['status'] == 'TRAINING_COMPLETE_EVALUATION_PENDING'
            assert len(completion['rows']) == expected
            folder = R / 'checkpoints' / folder
            assert len(list(folder.rglob('*.summary.json'))) == expected
            sources.append((stage, folder))
    for stage, folder in sources:
        for path in sorted(folder.rglob('*.summary.json')):
            row, trace = extract(path, stage)
            selected.append(row)
            curves.extend(trace)
    if not args.original_only:
        decisions = json.loads((R / 'results/E3_GATE_DECISIONS.json').read_text())
        eligible = [r for r in selected if r['stage'] == 'E3' and r['method'] != 'observer']
        assert len(eligible) == 18
        assert sum(r['accuracy_gate_005'] for r in eligible) == decisions['selected_for_evaluation']
        assert len(decisions['skipped']) == sum(not r['accuracy_gate_005'] for r in eligible)
    prefix = 'ORIGINAL' if args.original_only else 'REVISION'
    for suffix, records in [('TRAINING_SELECTION', selected), ('TRAINING_CURVES', curves)]:
        with (R / f'results/{prefix}_{suffix}.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    (R / f'results/{prefix}_TRAINING_EXPORT.json').write_text(json.dumps({
        'status': 'COMPLETE_ORIGINAL_ONLY' if args.original_only else 'COMPLETE',
        'selected_checkpoint_rows': len(selected), 'curve_metric_rows': len(curves),
        'counts_by_stage': {stage: sum(r['stage'] == stage for r in selected) for stage, _ in sources},
        'selection': 'Recorded composite selection for operators, not post-hoc minimum field nRMSE.',
        'time': 'Process wall time; not directly measured GPU active time.',
        'E3_scope': 'Same 512 training parents; nested temporal targets. Observer refit per factor.',
        'E4_scope': 'Ten new CUDA cells; original P seed11 is the baseline for each PDE. Preliminary CPU cells excluded.',
        'input_sha256': {r['summary_path']: r['summary_sha256'] for r in selected},
    }, indent=2), encoding='utf-8')
    print(prefix, 'training selection rows:', len(selected), 'metric records:', len(curves))


if __name__ == '__main__':
    main()
