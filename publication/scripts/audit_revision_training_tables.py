"""Independent keyed transcription audit for E3/E4/E5 generated tables.

Default requires completed campaign inputs. --e3-only checks the completed E3
selection tables while other experiments run. Neither mode verifies raw
experiments, statistical assumptions, scientific claims or page layout.
"""
import argparse
import hashlib
import json
from collections import defaultdict
from statistics import mean, median

import audit_completed_tables as a


def stage_receipt(name, status):
    path = a.O / 'data/revision' / name
    a.paths.add(path)
    receipt = json.loads(path.read_text())
    assert receipt['status'] == status, (name, receipt['status'])
    return receipt


def unique_tables(name, labels):
    tables = a.parse(name)
    assert len(tables) == len(labels)
    assert {label for _, label, _ in tables} == set(labels)
    return tables


def display(method):
    return method.replace('P-no-rank', 'P-nr').replace('_', r'\_')


def selection_tables(selection, e3_only):
    groups = {('E3', f) for f in [1, 4, 16]}
    if not e3_only:
        groups.add(('original', 1))
    labels = {f'tab:scaling_{stage}_{factor}' for stage, factor in groups}
    seen = set()
    for _, label, rows in unique_tables('revision_scaling_selection.tex', labels):
        stage, factor = label.removeprefix('tab:scaling_').split('_')
        assert len(rows) == 6
        for row in rows:
            assert len(row) == 6
            key = (stage, factor, row[0], row[1])
            assert key not in seen
            seen.add(key)
            source = a.single(selection, stage=stage, pde='burgers', target_factor=factor,
                              method=row[0], seed=row[1])
            assert row[0] in ['P', 'B4'] and row[1] in ['11', '23', '37']
            a.eq(row[2], source['actual_updates'], str(key)+'/actual')
            a.eq(row[3], source['selected_update'], str(key)+'/selected')
            error = float(source['validation_field_nrmse'])
            a.num(row[4], error, 4, str(key)+'/field_nrmse')
            assert (source['accuracy_gate_005'] == 'True') == (error <= .05)
            a.eq(row[5], 'Yes' if error <= .05 else 'No', str(key)+'/gate')
    assert len(seen) == len(groups)*6


def completed_campaign(selection):
    cost = a.load('revision/REVISION_COST_SUMMARY.csv')
    gradients = a.load('revision/E4_selected_gradient_diagnostics.csv')
    agreement = a.load('revision/E4_ACTION_AGREEMENT.csv')
    assert len(gradients) == 240
    heat = [r for r in selection if r['stage'] == 'E5']
    assert len(heat) == 19
    rows = unique_tables('revision_heat_selection.tex', ['tab:new_heat_selection'])[0][2]
    assert len(rows) == 19
    a.distinct(rows, lambda r: tuple(r[:2]))
    for row in rows:
        assert len(row) == 7
        source = a.single(heat, method=a.method(row[0]), seed=row[1])
        a.eq(row[2], source['actual_updates'], str(row[:2])+'/actual')
        a.eq(row[3], source['selected_update'], str(row[:2])+'/selected')
        metric = {'validation_nrmse':'Observer nRMSE', 'validation_action_mse':'Action MSE',
                  'selection_score':'Composite'}[source['selection_metric']]
        a.eq(row[4], metric, str(row[:2])+'/metric')
        a.num(row[5], source['selection_value'], 6, str(row[:2])+'/selection_value')
        if source['validation_field_nrmse'] == '':
            a.eq(row[6], '--', str(row[:2])+'/no_field_metric')
        else:
            a.num(row[6], source['validation_field_nrmse'], 4, str(row[:2])+'/field_nrmse')

    for _, label, rows in unique_tables('revision_weight_selection.tex', ['tab:weights_burgers','tab:weights_heat']):
        pde = label.removeprefix('tab:weights_')
        assert len(rows) == 6
        a.distinct(rows, lambda r: tuple(r[:2]))
        for row in rows:
            assert len(row) == 7
            selected = [g for g in gradients if g['pde'] == pde
                        and float(g['physics_weight']) == float(row[0])
                        and float(g['ranking_weight']) == float(row[1])]
            assert len(selected) == 20 and len({g['checkpoint'] for g in selected}) == 1
            name = selected[0]['checkpoint']
            source = (a.single(selection, stage='original', pde=pde, method='P', seed='11') if name == 'P_s11'
                      else a.single(selection, stage='E4', pde=pde, condition=name))
            a.num(row[2], source['validation_field_nrmse'], 4, label+'/'+name+'/field')
            a.eq(row[3], source['actual_updates'], label+'/'+name+'/actual')
            a.eq(row[4], source['selected_update'], label+'/'+name+'/selected')
            for cell, metric in zip(row[5:], ['physics_data_ratio','balance_data_ratio']):
                a.eq(cell, f'{median(float(g[metric]) for g in selected):.3e}', label+'/'+name+'/'+metric)

    expected = [f'tab:weights_control_{pde}_{role}' for pde in ['burgers','heat'] for role in a.ROLES.values()]
    for _, label, rows in unique_tables('revision_weight_control.tex', expected):
        pde, role = label.removeprefix('tab:weights_control_').split('_', 1)
        assert len(rows) == 6
        a.distinct(rows, lambda r: tuple(r[:2]))
        for row in rows:
            assert len(row) == 7
            candidates = [r for r in cost if r['stage'] == 'E4' and r['pde'] == pde and r['role'] == role
                          and float(r['method'].split('_')[1].removeprefix('phys')) == float(row[0])
                          and float(r['method'].split('_')[2].removeprefix('rank')) == float(row[1])]
            assert len(candidates) == 1
            source = candidates[0]
            agree = a.single(agreement, pde=pde, role=role, condition=source['method'])
            loc = label+'/'+source['method']
            a.num(row[2], source['mean_cost'], 6, loc+'/cost')
            a.num(row[3], source['relative_excess'], 2, loc+'/excess', 100)
            a.bounds(row[4], source['fixed_ci_low'], source['fixed_ci_high'], 2, loc+'/fixed', 100)
            for cell, metric in zip(row[5:], ['exact_history_agreement','history_agreement_at_1e7']):
                a.num(cell, agree[metric], 2, loc+'/'+metric, 100)

    groups = defaultdict(list)
    for source in cost:
        groups[(source['stage'], source['pde'], source['role'])].append(source)
    labels = {'tab:revision_cost_'+'_'.join(key):key for key in groups}
    count = 0
    for _, label, rows in unique_tables('revision_all_cost_intervals.tex', labels):
        stage, pde, role = labels[label]
        candidates = {}
        for source in groups[(stage,pde,role)]:
            method = source['method']
            if stage == 'E4':
                fields = method.split('_')
                shown = fields[1].removeprefix('phys')+' / '+fields[2].removeprefix('rank')
            elif stage == 'E3':
                fields = method.split('_')
                shown = fields[2]+' ('+fields[1]+')'
            else:
                shown = display(method)
            assert shown not in candidates
            candidates[shown] = source
        assert len(rows) == len(candidates) and {r[0] for r in rows} == set(candidates)
        for row in rows:
            assert len(row) == 6
            source = candidates[row[0]]; loc=label+'/'+row[0]; count += 1
            seeds = source['training_seed_count'] if stage in ['E3','E5'] and source['method'] not in ['B0','B1'] else '--'
            a.eq(row[1], seeds, loc+'/seeds')
            a.num(row[2], source['mean_cost'], 6, loc+'/cost')
            a.num(row[3], source['relative_excess'], 2, loc+'/excess', 100)
            for cell, prefix in zip(row[4:], ['fixed','joint']):
                a.bounds(cell, source[prefix+'_ci_low'], source[prefix+'_ci_high'], 2, loc+'/'+prefix, 100)
    assert count == len(cost)

    parts = a.load('revision/E6_new_heat_cost_decomposition.csv')
    assert len(parts) == 60
    for _, label, rows in unique_tables('revision_new_heat_cost.tex', ['tab:new_heat_costparts_'+role for role in a.ROLES.values()]):
        role = label.removeprefix('tab:new_heat_costparts_')
        assert len(rows) == 8 and {a.method(row[0]) for row in rows} == a.METHODS
        for row in rows:
            assert len(row) == 7
            selected = [r for r in parts if r['role'] == role and r['method'] == a.method(row[0])]
            assert {r['seed'] for r in selected} == ({''} if a.method(row[0]) in ['B0','B1'] else {'11','23','37'})
            a.distinct(selected, lambda r:r['seed'])
            for cell, key in zip(row[1:], ['cost','tracking','action','slew','violation','first_step_violation_cost']):
                a.num(cell, mean(float(r[key]) for r in selected), 6, label+'/'+row[0]+'/'+key)

    for _, label, rows in unique_tables('revision_new_heat_constraints.tex', ['tab:new_heat_constraints_'+role for role in a.ROLES.values()]):
        role = label.removeprefix('tab:new_heat_constraints_')
        assert len(rows) == 8 and {a.method(row[0]) for row in rows} == a.METHODS
        for row in rows:
            assert len(row) == 6
            source = a.single(cost, stage='E5', pde='heat', role=role, method=a.method(row[0]))
            assert 0 <= float(source['violation_rate_tolerance_1e7']) <= float(source['any_violation_rate']) <= 1
            for cell, key in zip(row[1:3], ['any_violation_rate','violation_rate_tolerance_1e7']):
                a.num(cell, source[key], 2, label+'/'+row[0]+'/'+key, 100)
            for cell, key in zip(row[3:], ['violation_duration_seconds','violation_max','violation_integrated_mean']):
                a.eq(cell, format(float(source[key]), '.6g'), label+'/'+row[0]+'/'+key)

    frequencies = a.load('revision/E1_candidate_frequency.csv')
    rows = unique_tables('revision_new_heat_choices.tex', ['tab:choices_new_heat'])[0][2]
    assert len(rows) == 12
    a.distinct(rows, lambda r:tuple(r[:2]))
    for row in rows:
        assert len(row) == 4
        role, method = a.ROLES[row[0]], a.method(row[1])
        assert method in ['P','B4','B5','P-no-rank']
        for cell, candidate in zip(row[2:], [9,4]):
            selected = [r for r in frequencies if r['source'] == 'E5_raw_recorded_indices' and r['role'] == role
                        and r['method_seed'].rsplit('_s',1)[0] == method and int(r['candidate']) == candidate]
            assert len(selected) == 3
            assert {r['method_seed'].rsplit('_s',1)[1] for r in selected} == {'11','23','37'}
            assert all(float(r['frequency_lower']) == float(r['frequency_upper']) for r in selected)
            requests = sum(int(r['requests']) for r in selected)
            frequency = sum(float(r['frequency_lower'])*int(r['requests']) for r in selected)/requests
            a.num(cell, frequency, 2, str(row[:2])+'/'+str(candidate), 100)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--e3-only', action='store_true')
    args=parser.parse_args()
    prefix = 'E3' if args.e3_only else 'REVISION'
    receipt=stage_receipt(prefix+'_TRAINING_EXPORT.json', 'COMPLETE_E3_ONLY' if args.e3_only else 'COMPLETE')
    if not args.e3_only:
        stage_receipt('REVISION_RAW_AUDIT.json', 'PASS')
    else:
        for name, digest in receipt['output_sha256'].items():
            path=a.O/'data/revision'/name; a.paths.add(path)
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    selection=a.load('revision/'+prefix+'_TRAINING_SELECTION.csv')
    selection_tables(selection, args.e3_only)
    if not args.e3_only:
        completed_campaign(selection)
    scope='E3_ONLY' if args.e3_only else 'REVISION_TRAINING'
    result=dict(status='PASS_'+scope+'_TABLE_TRANSCRIPTION', tables=a.table_count, rows=a.row_count,
                displayed_values_checked=len(a.checks), checks=a.checks,
                sha256={str(p.relative_to(a.O)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(a.paths)},
                limits='Keyed summary-to-table identities, aggregation and rounding only. No raw recomputation, confidence-interval methodology, prose/main-paper numeric coverage or PDF review.')
    (a.O/'data/revision'/f'{scope}_TABLE_NUMERIC_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['status','tables','rows','displayed_values_checked']}))


if __name__ == '__main__':
    main()
