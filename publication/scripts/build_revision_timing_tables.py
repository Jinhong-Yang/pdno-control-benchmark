"""Portable E2 tables; called only after the complete timing analysis is staged."""
from pathlib import Path
import csv
import hashlib
import json

O = Path(__file__).resolve().parents[1]
D = O / 'data/revision'
T = O / 'tables'


def build():
    receipt = json.loads((D / 'E2_ANALYSIS.json').read_text())
    assert receipt['status'] == 'COMPLETE' and receipt['ratio_cells'] == 20
    records = []

    def read(name):
        with (D / name).open(newline='', encoding='utf-8') as stream:
            return list(csv.DictReader(stream))

    def table(caption, label, columns, header, body):
        return ('\\begin{table}[ht]\n\\centering\\small\n\\caption{' + caption + '}\\label{' + label
                + '}\n\\begin{tabular}{@{}' + columns + '@{}}\\toprule\n' + header
                + '\\\\\\midrule\n' + '\n'.join(body) + '\n\\bottomrule\\end{tabular}\n\\end{table}\n')

    def save(name, contents, source):
        path = T / name
        path.write_text('\n'.join(contents), encoding='utf-8')
        records.append({'table': str(path.relative_to(O)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'inputs': [{'path': str((D / source).relative_to(O)),
                                    'sha256': hashlib.sha256((D / source).read_bytes()).hexdigest()}]})

    seed = read('E2_per_seed.csv')
    assert len(seed) == 120
    tables = []
    for pde in ['burgers', 'heat']:
        for cache in ['False', 'True']:
            rows = sorted([r for r in seed if r['pde'] == pde and r['cache'] == cache],
                          key=lambda r: (int(r['K']), r['method'], int(r['seed'])))
            assert len(rows) == 30 and all(int(r['requests']) == 15000 for r in rows)
            body = [' & '.join([r['K'], r['method'], r['seed']]
                               + [f"{float(r[k]):.4f}" for k in ['p50_ms', 'p99_ms', 'p999_ms']]
                               + [f"{100*float(r['miss_rate_5ms']):.2f}"]) + r'\\' for r in rows]
            mode = 'on' if cache == 'True' else 'off'
            tables.append(table(f'Revision {pde}, cache {mode}: host-ready latency in ms. Each training-seed row pools 15,000 requests across three sessions; warmup is excluded.',
                                f'tab:e2_seed_{pde}_{mode}', 'rlrrrrr', r'$K$ & Method & Seed & p50 & p99 & p99.9 & $>5$ ms (\%)', body))
    save('revision_latency_per_seed.tex', tables, 'E2_per_seed.csv')

    ratios = read('E2_ratio_bootstrap.csv')
    assert len(ratios) == 20
    tables = []
    for pde in ['burgers', 'heat']:
        body = []
        for r in sorted([r for r in ratios if r['pde'] == pde], key=lambda r: (int(r['K']), r['cache'])):
            body.append(' & '.join([r['K'], 'On' if r['cache'] == 'True' else 'Off',
                                    f"{float(r['mean_per_seed_p99_ratio']):.4f}",
                                    f"[{float(r['ci_low']):.4f}, {float(r['ci_high']):.4f}]"]) + r'\\')
        tables.append(table(f'Revision {pde}: mean training-seed P/B4 p99 ratio with pointwise 95\\% matched seed/session bootstrap interval. These exploratory intervals do not provide simultaneous coverage across conditions.',
                            f'tab:e2_ratio_{pde}', 'rlrl', r'$K$ & Cache & Ratio & Pointwise 95\% interval', body))
    save('revision_latency_ratios.tex', tables, 'E2_ratio_bootstrap.csv')

    profiles = read('E2_stage_profile.csv')
    assert len(profiles) == 160
    stage_names = {'request_preparation': 'Request preparation', 'H2D_and_grid': 'Host-to-device / grid',
                   'encoder': 'Encoder', 'branch': 'Branch', 'trunk': 'Trunk',
                   'field_assembly_and_other': 'Field assembly / remainder',
                   'scoring_including_goal_transfer': 'Scoring / goal transfer',
                   'D2H_selected_index_and_forecast': 'Device-to-host return',
                   'projection': 'Projection', 'verification': 'Finite-value verification'}
    tables = []
    for pde in ['burgers', 'heat']:
        for cache in ['False', 'True']:
            for K in [10, 200]:
                body = []
                for stage, label in stage_names.items():
                    values = []
                    for method in ['P', 'B4']:
                        rr = [r for r in profiles if (r['pde'], r['cache'], int(r['K']), r['stage'], r['method']) == (pde, cache, K, stage, method)]
                        assert len(rr) == 1 and int(rr[0]['requests']) == 500
                        values.extend(f"{float(rr[0][key]):.4f}" for key in ['p50_ms', 'p99_ms'])
                    body.append(label + ' & ' + ' & '.join(values) + r'\\')
                mode = 'on' if cache == 'True' else 'off'
                tables.append(table(f'Revision {pde}, $K={K}$, cache {mode}: separately synchronized stage profiling, 500 requests per method, seed 11, session 1. Times are in ms. Instrumentation perturbs execution; marginal stage quantiles must not be summed.',
                                    f'tab:e2_profile_{pde}_{K}_{mode}', 'lrrrr', 'Stage & P p50 & P p99 & B4 p50 & B4 p99', body))
    save('revision_latency_profiles.tex', tables, 'E2_stage_profile.csv')
    return records


if __name__ == '__main__':
    print(json.dumps(build(), indent=2))
