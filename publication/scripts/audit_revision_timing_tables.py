"""Check every displayed E2 table value and condition key against staged CSVs.

This validates numerical transcription, not the statistical method, causal
interpretation, PDF rendering, or complete manuscript coverage.
"""
from pathlib import Path
import csv
import hashlib
import json
import re

O = Path(__file__).resolve().parents[1]
FILES = ['revision_latency_per_seed.tex', 'revision_latency_ratios.tex',
         'revision_latency_profiles.tex']


def read_csv(name):
    with (O/'data/revision'/name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def tables(name):
    text = (O/'tables'/name).read_text(encoding='utf-8')
    found = {}
    for block in text.split(r'\begin{table}')[1:]:
        label = block.split(r'\label{', 1)[1].split('}', 1)[0]
        assert label not in found
        content = block.split(r'\midrule', 1)[1].split(r'\bottomrule', 1)[0]
        rows = [[cell.strip() for cell in row.strip().split('&')]
                for row in content.split(r'\\') if row.strip()]
        found[label] = rows
    return found


def main():
    checks = []

    def value(actual, expected, location):
        assert actual == expected, (location, actual, expected)
        checks.append({'location': location, 'displayed': actual})

    seed = read_csv('E2_per_seed.csv')
    seed_tables = tables(FILES[0])
    assert len(seed) == 120 and len(seed_tables) == 4
    seen = set()
    for pde in ['burgers', 'heat']:
        for cache, mode in [('False', 'off'), ('True', 'on')]:
            label = f'tab:e2_seed_{pde}_{mode}'
            assert len(seed_tables[label]) == 30
            for row in seed_tables[label]:
                assert len(row) == 7
                key = (pde, cache, row[0], row[1], row[2])
                assert key not in seen
                seen.add(key)
                matches = [r for r in seed if (r['pde'], r['cache'], r['K'], r['method'], r['seed']) == key]
                assert len(matches) == 1 and int(matches[0]['requests']) == 15000
                source = matches[0]
                for cell, field in zip(row[3:6], ['p50_ms', 'p99_ms', 'p999_ms']):
                    value(cell, format(float(source[field]), '.4f'), str(key)+'/'+field)
                value(row[6], format(100*float(source['miss_rate_5ms']), '.2f'), str(key)+'/miss_percent')
    assert len(seen) == 120

    ratios = read_csv('E2_ratio_bootstrap.csv')
    ratio_tables = tables(FILES[1])
    assert len(ratios) == 20 and len(ratio_tables) == 2
    seen = set()
    for pde in ['burgers', 'heat']:
        label = f'tab:e2_ratio_{pde}'
        assert len(ratio_tables[label]) == 10
        for row in ratio_tables[label]:
            assert len(row) == 4 and row[1] in ['Off', 'On']
            key = (pde, row[0], str(row[1] == 'On'))
            assert key not in seen
            seen.add(key)
            matches = [r for r in ratios if (r['pde'], r['K'], r['cache']) == key]
            assert len(matches) == 1
            source = matches[0]
            value(row[2], format(float(source['mean_per_seed_p99_ratio']), '.4f'), str(key)+'/ratio')
            bounds = re.fullmatch(r'\[([^,]+),\s*([^\]]+)\]', row[3])
            assert bounds
            for cell, field in zip(bounds.groups(), ['ci_low', 'ci_high']):
                value(cell, format(float(source[field]), '.4f'), str(key)+'/'+field)
    assert len(seen) == 20

    profiles = read_csv('E2_stage_profile.csv')
    profile_tables = tables(FILES[2])
    assert len(profiles) == 160 and len(profile_tables) == 8
    stage_map = {
        'Request preparation': 'request_preparation',
        'Host-to-device / grid': 'H2D_and_grid', 'Encoder': 'encoder',
        'Branch': 'branch', 'Trunk': 'trunk',
        'Field assembly / remainder': 'field_assembly_and_other',
        'Scoring / goal transfer': 'scoring_including_goal_transfer',
        'Device-to-host return': 'D2H_selected_index_and_forecast',
        'Projection': 'projection', 'Finite-value verification': 'verification',
    }
    seen = set()
    for pde in ['burgers', 'heat']:
        for cache, mode in [('False', 'off'), ('True', 'on')]:
            for k in ['10', '200']:
                label = f'tab:e2_profile_{pde}_{k}_{mode}'
                rows = profile_tables[label]
                assert len(rows) == 10 and {r[0] for r in rows} == set(stage_map)
                for row in rows:
                    assert len(row) == 5
                    stage = stage_map[row[0]]
                    for method, cells in [('P', row[1:3]), ('B4', row[3:5])]:
                        key = (pde, k, cache, stage, method)
                        assert key not in seen
                        seen.add(key)
                        matches = [r for r in profiles if (r['pde'], r['K'], r['cache'], r['stage'], r['method']) == key]
                        assert len(matches) == 1 and int(matches[0]['requests']) == 500
                        for cell, field in zip(cells, ['p50_ms', 'p99_ms']):
                            value(cell, format(float(matches[0][field]), '.4f'), str(key)+'/'+field)
    assert len(seen) == 160 and len(checks) == 860
    paths = [O/'tables'/name for name in FILES] + [O/'data/revision'/name for name in
            ['E2_per_seed.csv', 'E2_ratio_bootstrap.csv', 'E2_stage_profile.csv']]
    result = {'status': 'PASS_E2_TABLE_TRANSCRIPTION', 'tables': 14,
              'displayed_numeric_values_checked': len(checks),
              'condition_records': 300,
              'sha256': {str(p.relative_to(O)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              'scope': 'All E2 per-seed, ratio and stage table cells, unique keys, counts and displayed rounding. Does not certify interpretation, PDF layout, or other manuscript numbers.',
              'checks': checks}
    (O/'data/revision/E2_TABLE_NUMERIC_AUDIT.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('E2 table transcription PASS:14tables,300condition records,860displayed values.')


if __name__ == '__main__':
    main()
