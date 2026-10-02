"""Render source-bound manuscript numbers with explicit population/metric locators.

This resolves values and hashes; it does not validate the scientific meaning of
the surrounding sentence or establish complete manuscript coverage by itself.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import re
import statistics

O = Path(__file__).resolve().parents[1]


def json_value(value, path):
    for key in path:
        value = value[key]
    return value


def resolve(root, binding):
    source = (root / binding['source']).resolve(strict=True)
    assert source.is_relative_to(root.resolve())
    assert binding['population'] and binding['metric'] and binding['unit']
    if binding['kind'] == 'csv':
        with source.open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        selected = [r for r in rows if all(r[key] == str(value) for key, value in binding.get('filter', {}).items())]
        assert len(selected) == binding['expected_rows'], (binding, len(selected))
        operation = binding.get('reduce', 'single')
        if operation == 'count':
            value = len(selected)
        else:
            values = [float(r[binding['column']]) for r in selected]
            assert values and all(math.isfinite(v) for v in values)
            if operation == 'single':
                assert len(values) == 1
                value = values[0]
            else:
                value = {'sum': sum, 'mean': statistics.mean, 'median': statistics.median,
                         'min': min, 'max': max}[operation](values)
    elif binding['kind'] == 'json':
        document = json.loads(source.read_text(encoding='utf-8-sig'))
        for condition in binding.get('assertions', []):
            assert json_value(document, condition['path']) == condition['equals']
        value = float(json_value(document, binding['path']))
    else:
        raise ValueError('Unknown source kind')
    value = value * binding.get('scale', 1) + binding.get('offset', 0)
    assert math.isfinite(value)
    formatted = format(value, binding['format'])
    if 'e' in formatted.lower():
        mantissa, exponent = formatted.lower().split('e')
        tex = r'\ensuremath{' + mantissa + r'\times10^{' + str(int(exponent)) + '}}'
    else:
        tex = formatted
    return {'value': value, 'formatted': formatted, 'tex': tex,
            'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}


def build(root=O, scan_manuscript=True):
    source = root / 'data/number_bindings.json'
    registry = json.loads(source.read_text(encoding='utf-8'))
    assert registry['bindings'], 'No number bindings supplied'
    resolved = {}
    tex = [r'\providecommand{\EvidenceNumber}[1]{\ifcsname evidence@#1\endcsname\csname evidence@#1\endcsname\else\PackageError{evidence}{Unknown numeric binding #1}{Regenerate tables/evidence_numbers.tex before compiling.}\fi}']
    for identifier, binding in registry['bindings'].items():
        assert re.fullmatch('[A-Za-z][A-Za-z0-9]*', identifier)
        result = resolve(root, binding)
        resolved[identifier] = {**binding, **result}
        tex.append(r'\expandafter\def\csname evidence@'+identifier+r'\endcsname{'+result['tex']+'}')
    target = root / 'tables/evidence_numbers.tex'
    target.write_text('\n'.join(tex)+'\n', encoding='utf-8')
    uses = []
    documents = ['main.tex', 'supplement.tex'] if scan_manuscript else []
    for filename in documents:
        for line, content in enumerate((root / filename).read_text(encoding='utf-8').splitlines(), 1):
            for identifier in re.findall(r'\\EvidenceNumber\{([A-Za-z][A-Za-z0-9]*)\}', content):
                assert identifier in resolved, (filename, line, identifier)
                uses.append({'file': filename, 'line': line, 'id': identifier})
    receipt = {'status': 'BOUND_VALUES_RESOLVED_ONLY', 'registry_status': registry['status'],
               'manuscript_scan': 'PERFORMED' if scan_manuscript else 'NOT_REQUESTED_SOURCE_ONLY_PACKAGE',
               'scope': 'Each registered numeric locator and transformation resolved. Literal-number coverage, claim context, table/caption alignment and final rendered QA require separate final verification.',
               'registry_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
               'tex_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
               'bindings': resolved, 'manuscript_macro_uses': uses}
    (root / 'data/number_bindings_resolved.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print('Resolved numeric bindings:', len(resolved), 'current manuscript uses:', len(uses))
    return receipt


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--numbers-only', action='store_true',
                        help='Resolve numeric assets without scanning manuscript sources excluded from the public code package.')
    args = parser.parse_args()
    build(scan_manuscript=not args.numbers_only)
