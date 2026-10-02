"""Revision scientific figures from staged, complete CSV evidence only."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BLUE, ORANGE, GREY, DARK = '#23699D', '#C76B25', '#727D89', '#243648'
COLORS = {'P': ORANGE, 'B4': BLUE, 'B0': DARK, 'B2': GREY}
ROLES = ['locked_nominal', 'locked_coefficient_ood', 'locked_delay_dropout']
ROLE_LABELS = ['Nominal', 'Coefficient shift', 'Delay / dropout']


def build(root):
    root = Path(root)
    data, figures = root / 'data', root / 'figures'
    revision = data / 'revision'
    records = []

    def read(name):
        with (data / name).open(encoding='utf-8-sig', newline='') as stream:
            return list(csv.DictReader(stream))

    def save(fig, name, description, sources):
        hashes = []
        for source in sources:
            path = data / source
            hashes.append({'path': str(path.relative_to(root)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        outputs = {}
        for suffix in ['pdf', 'svg', 'png']:
            path = figures / (name + '.' + suffix)
            options = {'dpi': 300} if suffix == 'png' else {}
            fig.savefig(path, bbox_inches='tight', pad_inches=.05, **options)
            outputs[suffix] = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append({'id': name, 'description': description, 'sources': sources,
                        'input_sha256': hashes, 'output_sha256': outputs,
                        'pdf_sha256': outputs['pdf'], 'vector': True})
        plt.close(fig)

    def interval_point(ax, x, y, low, high, color, marker='o'):
        # Percentile intervals need not contain the point estimate.
        assert np.isfinite([x, low, high]).all() and low <= high
        ax.hlines(y, low, high, color=color, lw=.9)
        ax.vlines([low, high], y-.065, y+.065, color=color, lw=.9)
        ax.plot(x, y, marker, color=color, ms=4.3, markerfacecolor='white' if marker == 'D' else color)

    if (revision / 'REVISION_COST_SUMMARY.csv').exists():
        new = read('revision/REVISION_COST_SUMMARY.csv')
        old = read('control_all_methods.csv')
        oracles = [r for r in new if r['stage'] == 'E1']
        assert len(oracles) == 12
        methods = ['B0', 'O-cand-state', 'O-cand-obs', 'B4', 'P']
        labels = ['B0', 'RS-true', 'RS-obs', 'B4', 'P']
        fig, axes = plt.subplots(2, 3, figsize=(7.16, 3.9), sharey=True, layout='constrained')
        for pi, pde in enumerate(['burgers', 'heat']):
            row_bounds = [0., 5.]
            for ri, role in enumerate(ROLES):
                ax = axes[pi, ri]
                for j, method in enumerate(methods):
                    if method.startswith('O-cand'):
                        selected = [r for r in oracles if (r['pde'], r['role'], r['method']) == (pde, role, method)]
                        assert len(selected) == 1
                        r = selected[0]
                        value, low, high = [100*float(r[k]) for k in ['relative_excess', 'fixed_ci_low', 'fixed_ci_high']]
                    else:
                        selected = [r for r in old if (r['pde'], r['role'], r['method']) == (pde, role, method)]
                        assert len(selected) == 1
                        r = selected[0]
                        value, low, high = [100*float(r[k]) for k in ['relative_cost_difference_vs_B0', 'ci_low', 'ci_high']]
                    interval_point(ax, value, j, low, high, COLORS.get(method, GREY), 'D' if method.startswith('O-') else 'o')
                    row_bounds.extend([low, high, value])
                ax.axvline(0, color=DARK, lw=.7)
                ax.axvline(5, color=DARK, lw=.7, ls='--')
                ax.set_yticks(range(len(methods)), labels)
                ax.set_ylim(len(methods)-.5, -.5)
                ax.grid(axis='x', color='#E5E9ED', lw=.6)
                population_label = 'Primary Burgers' if pde == 'burgers' else 'SH heat'
                ax.set_title(f'{chr(97+3*pi+ri)}  {population_label}\n{ROLE_LABELS[ri]}', loc='left', fontsize=8)
                if pi == 1:
                    ax.set_xlabel('Excess cost vs. B0 (%)')
            lower, upper = min(row_bounds), max(row_bounds)
            span = max(upper-lower, 1.)
            for ax in axes[pi]:
                ax.set_xlim(lower-.05*span, upper+.07*span)
        save(fig, 'figR01_oracles', 'Primary scenarios only: privileged-state and observed-state oracle comparisons. Primary fixed-denominator paired-scenario95% intervals; learned-controller seeds averaged within parent. Shared horizontal scales within each PDE; no new-heat oracle implied.',
             ['revision/REVISION_COST_SUMMARY.csv', 'control_all_methods.csv'])

        heat = [r for r in new if r['stage'] == 'E5']
        if heat:
            assert len(heat) == 24
            methods = ['B0', 'B1', 'B2', 'B3', 'B4', 'B5', 'P-no-rank', 'P']
            fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.8), layout='constrained')
            for pi, population in enumerate(['original_burgers', 'new_heat']):
                shown = methods + ['O-cand-state', 'O-cand-obs'] if pi == 0 else methods
                bounds = [0., 5.]
                for ri, role in enumerate(ROLES):
                    ax = axes[pi, ri]
                    for j, method in enumerate(shown):
                        if pi == 1:
                            rr = [r for r in heat if (r['role'], r['method']) == (role, method)]
                            fields = ['relative_excess', 'fixed_ci_low', 'fixed_ci_high']
                        elif method.startswith('O-'):
                            rr = [r for r in oracles if (r['pde'], r['role'], r['method']) == ('burgers', role, method)]
                            fields = ['relative_excess', 'fixed_ci_low', 'fixed_ci_high']
                        else:
                            rr = [r for r in old if (r['pde'], r['role'], r['method']) == ('burgers', role, method)]
                            fields = ['relative_cost_difference_vs_B0', 'ci_low', 'ci_high']
                        assert len(rr) == 1
                        value, low, high = [100*float(rr[0][key]) for key in fields]
                        interval_point(ax, value, j, low, high, COLORS.get(method, GREY), 'D' if method.startswith('O-') else 'o')
                        bounds.extend([value, low, high])
                    labels = [m.replace('P-no-rank', 'P-nr').replace('O-cand-state', 'RS-true').replace('O-cand-obs', 'RS-obs') for m in shown]
                    ax.set_yticks(range(len(shown)), labels if ri == 0 else [])
                    ax.set_ylim(len(shown)-.5, -.5)
                    ax.axvline(0, color=DARK, lw=.7)
                    ax.axvline(5, color=DARK, lw=.7, ls='--')
                    ax.set_xscale('symlog', linthresh=5, linscale=1)
                    ax.grid(axis='x', color='#E5E9ED', lw=.6)
                    title = 'Primary Burgers scenarios' if pi == 0 else 'NH heat scenarios'
                    ax.set_title(f'{chr(97+3*pi+ri)}  {title}\n{ROLE_LABELS[ri]}', loc='left', fontsize=8)
                    if pi == 1:
                        ax.set_xlabel('Excess cost (%)\nsymlog; linear within ±5%')
                low, high = min(bounds), max(bounds)
                margin = max((high-low)*.04, .5)
                low, high = low-margin, high+margin
                ticks = [x for x in [-100, -50, -20, -5, 0, 5, 20, 50, 100, 200, 500, 1000] if low <= x <= high]
                if 100 in ticks:
                    ticks = [x for x in ticks if x != 50]
                if high > 1000:
                    ticks.extend([10**i for i in range(4, int(np.ceil(np.log10(high)))+1) if 10**i <= high])
                for ax in axes[pi]:
                    ax.set_xlim(low, high)
                    ax.set_xticks(ticks, [str(x) for x in ticks])
            save(fig, 'figR05_main_control', 'Main control comparison: original Burgers parents including two oracles; separately generated positive-initial-field heat parents and retrained learned controllers. Fixed-denominator paired-scenario95% intervals. Symlog axis is linear between-5% and5%, logarithmic outside; common horizontal scales within each row. No original-heat oracle is compared with new-heat models.', ['revision/REVISION_COST_SUMMARY.csv', 'control_all_methods.csv'])
            fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.5), layout='constrained')
            metrics = [('any_violation_rate', 'Parents with any violation (%)', 100),
                       ('violation_duration_seconds', 'Mean violating duration (s)', 1),
                       ('violation_max', 'Mean episode maximum excess', 1),
                       ('violation_integrated_mean', 'Mean integrated spatial excess', 1)]
            for mi, (key, label, scale) in enumerate(metrics):
                ax = axes.flat[mi]
                plotted = []
                for ri, (role, marker, offset) in enumerate(zip(ROLES, ['o', 's', '^'], [-.2, 0, .2])):
                    values = []
                    for method in methods:
                        rr = [r for r in heat if (r['role'], r['method']) == (role, method)]
                        assert len(rr) == 1 and rr[0][key] != ''
                        values.append(scale*float(rr[0][key]))
                    assert np.isfinite(values).all() and min(values) >= 0
                    plotted.extend(values)
                    ax.plot(np.arange(len(methods))+offset, values, marker, ls='none', color=[DARK, BLUE, ORANGE][ri], ms=4,
                            markerfacecolor='white' if ri == 0 else [DARK, BLUE, ORANGE][ri], label=ROLE_LABELS[ri])
                ax.set_xticks(range(len(methods)), [m.replace('P-no-rank', 'P-nr') for m in methods])
                ax.set_title(chr(97+mi)+'  '+label, loc='left', fontsize=8)
                ax.grid(axis='y', color='#E5E9ED', lw=.6)
                if max(plotted) == 0:
                    ax.set_ylim(-.05, .3)
                    ax.set_yticks([0])
                    ax.text(.5, .64, 'All recorded values = 0', ha='center', transform=ax.transAxes, fontsize=8)
                else:
                    ax.set_ylim(-.025*max(plotted), max(plotted)*1.16)
            axes[0, 0].legend(frameon=False, fontsize=6.5, loc='upper right')
            save(fig, 'figR02_new_heat_constraints', 'New nonnegative-initial-field heat population only. Descriptive parent-and-seed means; duration, maximum and integrated excess accompany binary frequency. Zero observations do not establish a risk guarantee or discriminatory endpoint.', ['revision/REVISION_COST_SUMMARY.csv'])

    if (revision / 'E2_ANALYSIS.json').exists():
        receipt = json.loads((revision / 'E2_ANALYSIS.json').read_text())
        assert receipt['status'] == 'COMPLETE' and receipt['ratio_cells'] == 20
        ratios = read('revision/E2_ratio_bootstrap.csv')
        assert len(ratios) == 20
        fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.55), sharey=True, layout='constrained')
        all_bounds = [.75, 1.]
        for pi, pde in enumerate(['burgers', 'heat']):
            ax = axes[pi]
            for cache, color, marker, ls, offset in [('False', DARK, 'o', '-', -.02), ('True', GREY, 's', '--', .02)]:
                rr = sorted([r for r in ratios if r['pde'] == pde and r['cache'] == cache], key=lambda r: int(r['K']))
                assert [int(r['K']) for r in rr] == [10, 25, 50, 100, 200]
                x = np.array([int(r['K']) for r in rr])*(1+offset)
                value, low, high = [np.array([float(r[k]) for r in rr]) for k in ['mean_per_seed_p99_ratio', 'ci_low', 'ci_high']]
                assert (low <= high).all()
                ax.plot(x, value, marker=marker, ls=ls, color=color, ms=4, label='Cache '+('on' if cache == 'True' else 'off'))
                ax.vlines(x, low, high, color=color, lw=.9)
                for xx, lo, hi in zip(x, low, high):
                    ax.hlines([lo, hi], xx*.98, xx*1.02, color=color, lw=.9)
                all_bounds.extend(low.tolist()+high.tolist()+value.tolist())
            ax.set_xscale('log')
            ax.set_xticks([10, 25, 50, 100, 200], ['10', '25', '50', '100', '200'])
            ax.axhline(1, color=DARK, lw=.7)
            ax.axhline(.75, color=BLUE, lw=.7, ls=':')
            ax.set_xlabel('Candidate count K')
            ax.set_title(chr(97+pi)+'  '+pde.capitalize(), loc='left')
            ax.grid(axis='y', color='#E5E9ED', lw=.6)
        axes[0].set_ylabel('Mean seed-specific p99 ratio, P / B4')
        axes[1].legend(frameon=False, fontsize=7)
        margin = max(max(all_bounds)-min(all_bounds), .1)*.08
        axes[0].set_ylim(min(all_bounds)-margin, max(all_bounds)+margin)
        save(fig, 'figR03_K_ratio', 'Complete20-cell K/cache sweep, mean seed-specific p99 ratio and pointwise95% matched seed/session intervals. Slight horizontal offsets distinguish cache conditions; reference lines1 and0.75. No simultaneous coverage.', ['revision/E2_ratio_bootstrap.csv'])

        pooled = read('revision/E2_pooled.csv')
        assert len(pooled) == 40
        fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.05), sharex=True, sharey=True, layout='constrained')
        for pi, pde in enumerate(['burgers', 'heat']):
            for ci, cache in enumerate(['False', 'True']):
                ax = axes[pi, ci]
                for method, marker in [('P', 'o'), ('B4', 's')]:
                    rr = sorted([r for r in pooled if (r['pde'], r['cache'], r['method']) == (pde, cache, method)], key=lambda r: int(r['K']))
                    assert len(rr) == 5 and all(int(r['requests']) == 45000 for r in rr)
                    ax.plot([int(r['K']) for r in rr], [float(r['p99_ms']) for r in rr], marker=marker, color=COLORS[method], ms=4, label=method)
                ax.axhline(5, color=DARK, lw=.7, ls=':')
                ax.set_xscale('log')
                ax.set_xticks([10, 25, 50, 100, 200], ['10', '25', '50', '100', '200'])
                ax.set_title(f'{chr(97+pi*2+ci)}  {pde.capitalize()}, cache '+('on' if cache == 'True' else 'off'), loc='left', fontsize=8)
                ax.grid(axis='y', color='#E5E9ED', lw=.6)
                if ci == 0:
                    ax.set_ylabel('Pooled request p99 (ms)')
                if pi == 1:
                    ax.set_xlabel('Candidate count K')
        axes[0, 1].legend(frameon=False)
        axes[0, 0].set_ylim(bottom=0)
        save(fig, 'figR04_K_p99', 'Descriptive pooled request p99,45,000 requests per point, shared scales. These pooled percentiles are not the seed-specific ratio statistic.', ['revision/E2_pooled.csv'])

        histogram = read('revision/E2_latency_histograms.csv')
        fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.45), sharex=True, sharey=True, layout='constrained')
        for pi, pde in enumerate(['burgers', 'heat']):
            for ci, campaign in enumerate(['original', 'revision_K10']):
                ax = axes[pi, ci]
                groups = [('P', 'False'), ('B4', 'False'), ('B2', 'False')] if campaign == 'original' else [(m, cache) for m in ['P', 'B4'] for cache in ['False', 'True']]
                for method, cache in groups:
                    rr = sorted([r for r in histogram if (r['campaign'], r['pde'], r['method'], r['cache']) == (campaign, pde, method, cache)], key=lambda r: float(r['bin_left_ms']))
                    assert len(rr) == 100
                    assert sum(int(r['count']) for r in rr) == int(rr[0]['total_requests'])
                    edges = [float(r['bin_left_ms']) for r in rr]+[float(rr[-1]['bin_right_ms'])]
                    values = [float(r['density_per_log10_ms']) for r in rr]
                    label = method if campaign == 'original' else method+' cache '+('on' if cache == 'True' else 'off')
                    ax.stairs(values, edges, color=COLORS[method], ls='--' if cache == 'True' else '-', lw=1, label=label)
                ax.set_xscale('log')
                ax.axvline(5, color=DARK, lw=.7, ls=':')
                ax.set_title(f'{chr(97+pi*2+ci)}  {pde.capitalize()}: '+('primary' if ci == 0 else 'follow-up K=10'), loc='left', fontsize=8)
                ax.legend(frameon=False, fontsize=6.3)
                if ci == 0:
                    ax.set_ylabel('Density per log10(ms)')
                if pi == 1:
                    ax.set_xlabel('Request latency (ms, log scale)')
        save(fig, 'figRS02_latency_distributions', 'All observed timing values retained in shared log-spaced bins. Original campaign P/B4/B2 is separated from revisionK10 P/B4 cache conditions. Density is per log10 latency, not per linear millisecond.', ['revision/E2_latency_histograms.csv'])

    training_prefix = ('REVISION' if (revision / 'REVISION_TRAINING_EXPORT.json').exists()
                       else 'E3' if (revision / 'E3_TRAINING_EXPORT.json').exists() else None)
    if training_prefix:
        receipt = json.loads((revision / f'{training_prefix}_TRAINING_EXPORT.json').read_text())
        assert receipt['status'] == ('COMPLETE' if training_prefix == 'REVISION' else 'COMPLETE_E3_ONLY')
        if training_prefix == 'E3':
            for name, expected in receipt['output_sha256'].items():
                assert hashlib.sha256((revision / name).read_bytes()).hexdigest() == expected
        training_sources = [f'revision/{training_prefix}_TRAINING_CURVES.csv',
                            f'revision/{training_prefix}_TRAINING_SELECTION.csv']
        curves = read(training_sources[0])
        selected = read(training_sources[1])
        e3_field_values = [float(r['value']) for r in curves
                           if r['stage'] == 'E3' and r['method'] in ['P', 'B4']
                           and r['metric'] == 'validation_field_nrmse']
        common_ymax = float(np.ceil(max(e3_field_values) * 1.04 / .05) * .05)
        fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.15), sharex=True, sharey=True, layout='constrained')
        for mi, method in enumerate(['P', 'B4']):
            for fi, factor in enumerate([1, 4, 16]):
                ax = axes[mi, fi]
                for seed, style in zip([11, 23, 37], ['-', '--', ':']):
                    rr = [r for r in curves if (r['stage'], r['method'], int(r['target_factor']), int(r['seed']), r['metric']) == ('E3', method, factor, seed, 'validation_field_nrmse')]
                    rr.sort(key=lambda r: int(r['update']))
                    assert rr
                    selection = [r for r in selected if (r['stage'], r['method'], int(r['target_factor']), int(r['seed'])) == ('E3', method, factor, seed)]
                    assert len(selection) == 1
                    ax.plot([int(r['update']) for r in rr], [float(r['value']) for r in rr], ls=style, color=COLORS[method], lw=.9)
                    ax.plot(int(selection[0]['selected_update']), float(selection[0]['validation_field_nrmse']), 'o', color=COLORS[method], ms=3)
                ax.axhline(.05, color=DARK, lw=.7, ls='--')
                ax.set_title(f'{chr(97+mi*3+fi)}  {method}, {2048*factor:,} targets', loc='left', fontsize=8)
                ax.set_xlim(0, 20500)
                ax.set_xticks([0, 10000, 20000], ['0', '10,000', '20,000'])
                ax.set_ylim(0, common_ymax)
                ax.grid(axis='y', color='#E5E9ED', lw=.6)
                if fi == 0:
                    ax.set_ylabel('Validation field nRMSE')
                if mi == 1:
                    ax.set_xlabel('Actual update')
        axes[0, 2].legend([Line2D([], [], color=GREY, ls=ls) for ls in ['-', '--', ':']], ['Seed 11', 'Seed 23', 'Seed 37'], frameon=False, fontsize=6)
        save(fig, 'figRS03_scaling_curves', 'E3 Burgers recorded validation curves. Same512parents with nested temporal target density; one refitted observer per factor shared across methods/seeds. Lines stop at actual termination; points mark composite-validation-selected updates, not minimum field-error reselection. Dashed horizontal gate0.05.', training_sources)
    return records
