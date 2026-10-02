"""Complete revision training, control and loss-sensitivity tables."""
from pathlib import Path
import csv
import hashlib
import json
from statistics import median


def build_scaling(root, prefix='REVISION'):
    root = Path(root)
    data = root / 'data/revision'
    receipt = json.loads((data / f'{prefix}_TRAINING_EXPORT.json').read_text())
    assert receipt['status'] == ('COMPLETE' if prefix == 'REVISION' else 'COMPLETE_E3_ONLY')
    source = data / f'{prefix}_TRAINING_SELECTION.csv'
    if prefix == 'E3':
        assert hashlib.sha256(source.read_bytes()).hexdigest() == receipt['output_sha256'][source.name]
    with source.open(newline='', encoding='utf-8') as stream:
        selection = list(csv.DictReader(stream))
    def table(caption, label, columns, header, body):
        return ('\\begin{table}[!ht]\n\\centering\\small\n\\caption{' + caption + '}\\label{' + label
                + '}\n\\begin{tabular}{@{}' + columns + '@{}}\\toprule\n' + header
                + '\\\\\\midrule\n' + '\n'.join(body) + '\n\\bottomrule\\end{tabular}\n\\end{table}\n')

    sections = []
    for stage, factor in ([('original', 1)] if prefix == 'REVISION' else []) + [('E3', 1), ('E3', 4), ('E3', 16)]:
        rows = sorted([r for r in selection if (r['stage'], r['pde'], int(r['target_factor'])) == (stage, 'burgers', factor) and r['method'] in ['P', 'B4']], key=lambda r: (r['method'], int(r['seed'])))
        assert len(rows) == 6
        body = []
        for r in rows:
            error = float(r['validation_field_nrmse'])
            assert (r['accuracy_gate_005'] == 'True') == (error <= .05)
            body.append(' & '.join([r['method'], r['seed'], r['actual_updates'], r['selected_update'], f'{error:.4f}', 'Yes' if error <= .05 else 'No']) + r'\\')
        group = 'Original allocation' if stage == 'original' else f'E3, target factor {factor}'
        sections.append(table(f'{group}: Burgers, {2048*factor:,} reference-solver training targets from the same 512 parents. Selected update follows the recorded composite criterion. E3 refits one shared observer per target factor; it is a joint pipeline-budget comparison.',
                              f'tab:scaling_{stage}_{factor}', 'llrrrr', r'Method & Seed & Actual updates & Selected update & nRMSE & $\leq0.05$', body))
    output = root / 'tables/revision_scaling_selection.tex'
    output.write_text('\n'.join(sections), encoding='utf-8')
    return [{'table': str(output.relative_to(root)), 'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
             'inputs': [{'path': str(source.relative_to(root)), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}]}]


def build(root):
    root = Path(root)
    data, tables = root / 'data/revision', root / 'tables'
    assert json.loads((data / 'REVISION_TRAINING_EXPORT.json').read_text())['status'] == 'COMPLETE'
    records = []

    def read(name):
        with (data / name).open(newline='', encoding='utf-8') as stream:
            return list(csv.DictReader(stream))

    def table(caption, label, columns, header, body):
        return ('\\begin{table}[ht]\n\\centering\\small\n\\caption{' + caption + '}\\label{' + label
                + '}\n\\begin{tabular}{@{}' + columns + '@{}}\\toprule\n' + header
                + '\\\\\\midrule\n' + '\n'.join(body) + '\n\\bottomrule\\end{tabular}\n\\end{table}\n')

    def save(name, contents, sources):
        path = tables / name
        path.write_text('\n'.join(contents), encoding='utf-8')
        records.append({'table': str(path.relative_to(root)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'inputs': [{'path': str((data / s).relative_to(root)), 'sha256': hashlib.sha256((data / s).read_bytes()).hexdigest()} for s in sources]})

    def interval(row, prefix):
        return f"[{100*float(row[prefix+'_ci_low']):.2f}, {100*float(row[prefix+'_ci_high']):.2f}]"

    def display(method):
        return method.replace('P-no-rank', 'P-nr').replace('_', r'\_')

    selection = read('REVISION_TRAINING_SELECTION.csv')
    costs = read('REVISION_COST_SUMMARY.csv')
    records.extend(build_scaling(root))

    rows = sorted([r for r in selection if r['stage'] == 'E5'], key=lambda r: (r['method'], int(r['seed'])))
    assert len(rows) == 19
    body = []
    for r in rows:
        metric = {'validation_nrmse': 'Observer nRMSE', 'validation_action_mse': 'Action MSE', 'selection_score': 'Composite'}[r['selection_metric']]
        error = '--' if r['validation_field_nrmse'] == '' else f"{float(r['validation_field_nrmse']):.4f}"
        body.append(' & '.join([display(r['method']), r['seed'], r['actual_updates'], r['selected_update'], metric, f"{float(r['selection_value']):.6f}", error]) + r'\\')
    save('revision_heat_selection.tex', [table('New heat training: all six learned controller families and their shared observer. The selection metric is method-specific; a direct policy does not have a field-prediction nRMSE. Composite values for operator and B3 selection use different definitions and are not comparable across method families.',
                                             'tab:new_heat_selection', 'llrrlrr', 'Method & Seed & Actual & Selected & Selection metric & Value & Field nRMSE', body)], ['REVISION_TRAINING_SELECTION.csv'])

    gradients = read('E4_selected_gradient_diagnostics.csv')
    assert len(gradients) == 240
    sections = []
    for pde in ['burgers', 'heat']:
        groups = {}
        for row in [r for r in gradients if r['pde'] == pde]:
            groups.setdefault(row['checkpoint'], []).append(row)
        assert len(groups) == 6
        body = []
        ordered = sorted(groups.items(), key=lambda pair: (float(pair[1][0]['physics_weight']), float(pair[1][0]['ranking_weight'])))
        for checkpoint, values in ordered:
            assert len(values) == 20
            baseline = checkpoint == 'P_s11'
            rr = [r for r in selection if r['pde'] == pde and ((r['stage'] == 'original' and r['method'] == 'P' and r['seed'] == '11') if baseline else (r['stage'] == 'E4' and r['condition'] == checkpoint))]
            assert len(rr) == 1
            selected = rr[0]
            ratios = [median(float(v[key]) for v in values) for key in ['physics_data_ratio', 'balance_data_ratio']]
            body.append(' & '.join([values[0]['physics_weight'], values[0]['ranking_weight'], f"{float(selected['validation_field_nrmse']):.4f}", selected['actual_updates'], selected['selected_update']]+[f'{v:.3e}' for v in ratios]) + r'\\')
        sections.append(table(f'E4 {pde}, seed 11: equal physics/balance weights and ranking-weight sweep. The (0.01,0.1) cell reuses original P. Gradient ratios are medians over 20 common training batches at selected weights and full ramp; they are not historical per-update gradients.',
                              f'tab:weights_{pde}', 'rrrrrrr', r'$\lambda_{p,b}$ & $\lambda_r$ & nRMSE & Actual & Selected & Physics/data & Balance/data', body))
    save('revision_weight_selection.tex', sections, ['REVISION_TRAINING_SELECTION.csv', 'E4_selected_gradient_diagnostics.csv'])

    agreements = read('E4_ACTION_AGREEMENT.csv')
    roles = {'locked_nominal': 'Nominal', 'locked_coefficient_ood': 'Coefficient shift', 'locked_delay_dropout': 'Delay/dropout'}
    sections = []
    for pde in ['burgers', 'heat']:
        for role, label in roles.items():
            rows = [r for r in costs if (r['stage'], r['pde'], r['role']) == ('E4', pde, role)]
            assert len(rows) == 6
            body = []
            for r in sorted(rows, key=lambda r: r['method']):
                condition = r['method']
                weights = condition.split('_')[1:3]
                rr = [a for a in agreements if (a['pde'], a['role'], a['condition']) == (pde, role, condition)]
                assert len(rr) == 1
                body.append(' & '.join([weights[0].removeprefix('phys'), weights[1].removeprefix('rank'), f"{float(r['mean_cost']):.6f}", f"{100*float(r['relative_excess']):.2f}", interval(r, 'fixed'), f"{100*float(rr[0]['exact_history_agreement']):.2f}", f"{100*float(rr[0]['history_agreement_at_1e7']):.2f}"]) + r'\\')
            sections.append(table(f'E4 {pde}, {label.lower()}, seed 11. Excess cost is relative to B0, with a fixed-denominator paired-parent 95\\% interval. History agreement compares all 200 actions against original P seed 11. Exact and maximum absolute difference $\\leq10^{{-7}}$ columns are percentages of parent histories.',
                                  f'tab:weights_control_{pde}_{role}', 'rrrrlrr', r'$\lambda_{p,b}$ & $\lambda_r$ & Cost & Excess (\%) & Fixed 95\% & Exact (\%) & $10^{-7}$ (\%)', body))
    save('revision_weight_control.tex', sections, ['REVISION_COST_SUMMARY.csv', 'E4_ACTION_AGREEMENT.csv'])

    sections = []
    for stage in ['E1', 'E3', 'E4', 'E5']:
        for pde in ['burgers', 'heat']:
            for role, label in roles.items():
                rows = [r for r in costs if (r['stage'], r['pde'], r['role']) == (stage, pde, role)]
                if not rows:
                    continue
                body = []
                for r in rows:
                    method = display(r['method'])
                    if stage == 'E4':
                        parts = r['method'].split('_')
                        method = parts[1].removeprefix('phys')+' / '+parts[2].removeprefix('rank')
                    elif stage == 'E3':
                        parts = r['method'].split('_')
                        method = parts[2]+' ('+parts[1]+')'
                    body.append(' & '.join([method, r['training_seed_count'] if int(r['training_seed_count']) > 0 else '--', f"{float(r['mean_cost']):.6f}", f"{100*float(r['relative_excess']):.2f}", interval(r, 'fixed'), interval(r, 'joint')]) + r'\\')
                population = 'new heat parents' if stage == 'E5' else 'original parents'
                scope = (' Only gate-passing checkpoints enter E3 rows; the displayed seed count can be below three.' if stage == 'E3' else
                         ' Conditions list physics/balance weight and ranking weight, separated by a slash; all use seed 11 and the (0.01,0.1) cell reuses original P.' if stage == 'E4' else '')
                sections.append(table(f'{stage}, {pde}, {label.lower()}, {population}: mean cost and excess percent relative to the matched B0. Fixed-denominator primary and jointly resampled-denominator sensitivity intervals use the same paired parent draws.'+scope,
                                      f'tab:revision_cost_{stage}_{pde}_{role}', 'lrrrrl', r'Condition & Seeds ($n$) & Cost & Excess (\%) & Fixed 95\% & Joint 95\%', body))
    save('revision_all_cost_intervals.tex', sections, ['REVISION_COST_SUMMARY.csv'])

    decomposition = read('E6_new_heat_cost_decomposition.csv')
    assert len(decomposition) == 60
    sections = []
    for role, label in roles.items():
        body = []
        for method in ['B0', 'B1', 'B2', 'B3', 'B4', 'B5', 'P-no-rank', 'P']:
            rows = [r for r in decomposition if (r['role'], r['method']) == (role, method)]
            assert len(rows) == (1 if method in ['B0', 'B1'] else 3)
            values = [sum(float(r[key]) for r in rows)/len(rows) for key in ['cost', 'tracking', 'action', 'slew', 'violation', 'first_step_violation_cost']]
            body.append(display(method)+' & '+' & '.join(f'{v:.6f}' for v in values)+r'\\')
        sections.append(table(f'New heat, {label.lower()}: additive mean episode-cost components. First-step violation cost is a subset of the violation component, not an additional term.',
                              f'tab:new_heat_costparts_{role}', 'lrrrrrr', 'Method & Total & Tracking & Action & Slew & Violation & First step', body))
    save('revision_new_heat_cost.tex', sections, ['E6_new_heat_cost_decomposition.csv'])
    sections = []
    for role, label in roles.items():
        body = []
        for method in ['B0', 'B1', 'B2', 'B3', 'B4', 'B5', 'P-no-rank', 'P']:
            rows = [r for r in costs if (r['stage'], r['role'], r['method']) == ('E5', role, method)]
            assert len(rows) == 1
            row = rows[0]
            strict, tolerant = float(row['any_violation_rate']), float(row['violation_rate_tolerance_1e7'])
            assert 0 <= tolerant <= strict <= 1
            amounts = [float(row[key]) for key in ['violation_duration_seconds', 'violation_max', 'violation_integrated_mean']]
            body.append(display(method)+' & '+' & '.join([f'{100*strict:.2f}', f'{100*tolerant:.2f}']+[f'{value:.6g}' for value in amounts])+r'\\')
        sections.append(table(f'New heat, {label.lower()}: descriptive violation frequencies and parent/seed-averaged amounts. Strict means excess greater than zero; the separately reported tolerance diagnostic requires excess greater than $10^{{-7}}$. Duration is in seconds; integrated excess is spatial-mean normalized-field excess integrated in time. The original strict endpoint is unchanged.',
                              f'tab:new_heat_constraints_{role}', 'lrrrrr', r'Method & Strict (\%) & $>10^{-7}$ (\%) & Duration & Max excess & Integrated', body))
    save('revision_new_heat_constraints.tex', sections, ['REVISION_COST_SUMMARY.csv'])
    return records
