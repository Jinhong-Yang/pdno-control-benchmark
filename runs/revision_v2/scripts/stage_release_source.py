"""Allowlisted v1.1 source staging; default mode only inventories relative paths.

Use --build with final reviewed metadata after experiments and figure builds.
The scientific snapshots retain their byte hashes and original relative layout.
This script does not mutate a Git repository, tag a version, or upload files.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

R = Path(__file__).resolve().parents[1]
ROOT = R.parents[1]
OLD = ROOT / 'experiments/pdno_jevLite_20260927_v3'
PUB = ROOT / 'publications/ieee_access_pdno_revision_v2_20261002'
LEGACY = ROOT / 'publications/ieee_access_pdno_v3_20261001/public_release/pdno-control-benchmark'
TEXT = {'.py', '.json', '.jsonl', '.csv', '.yaml', '.yml', '.txt', '.sha256'}


def mapping():
    result = {}

    def add(source, target):
        assert source.is_file() and not source.is_symlink(), source
        assert not any(part.startswith('.') for part in target.parts if part not in ['.gitattributes'])
        assert source.suffix not in ['.pt', '.npz', '.pth', '.pkl']
        assert target not in result, target
        result[target] = source

    def tree(source, target, suffixes):
        for path in sorted(source.rglob('*')):
            if path.is_file() and path.suffix in suffixes and not any(part.startswith('.') or part == '__pycache__' for part in path.relative_to(source).parts):
                add(path, target / path.relative_to(source))

    for name in ['LICENSE', 'NOTICE', '.gitattributes']:
        add(LEGACY / name, Path(name))
    add(OLD / 'pyproject.toml', Path('pyproject.toml'))
    tree(R / 'src', Path('src'), {'.py'})
    tree(OLD / 'tests', Path('tests'), {'.py'})
    old_target = OLD.relative_to(ROOT)
    for folder in ['src', 'scripts', 'tests', 'config', 'contracts', 'evidence', 'data', 'state']:
        tree(OLD / folder, old_target / folder, TEXT)
    for path in sorted((OLD / 'runs').rglob('*.summary.json')):
        add(path, path.relative_to(ROOT))
    tree(OLD / 'reports', old_target / 'reports', TEXT | {'.md'})
    add(OLD / 'pyproject.toml', old_target / 'pyproject.toml')
    for folder in ['src', 'scripts', 'tests', 'config', 'results', 'evidence', 'data', 'release_support']:
        tree(R / folder, R.relative_to(ROOT) / folder, TEXT)
    # Training histories accompany the separately archived checkpoint binaries.
    # They are needed to regenerate selection/curve CSVs from the released files.
    for folder in ['E3', 'E4_cuda', 'E5']:
        for path in sorted((R / 'checkpoints' / folder).rglob('*.summary.json')):
            add(path, path.relative_to(ROOT))
    add(R / 'heat_ic_spec.md', R.relative_to(ROOT) / 'heat_ic_spec.md')
    # These publication subfolders contain only authored figures/tables and their
    # quantitative inputs. Author portraits and IEEE template assets are elsewhere.
    for folder, suffixes in [('scripts', {'.py'}), ('data', TEXT), ('tables', {'.tex'}),
                             ('figures', {'.tex', '.pdf', '.svg', '.png'})]:
        tree(PUB / 'overleaf' / folder, Path('publication') / folder, suffixes)
    add(PUB / 'overleaf/requirements-figures.txt', Path('publication/requirements-figures.txt'))
    add(PUB / 'overleaf/FIGURE_GUIDE.md', Path('publication/FIGURE_GUIDE.md'))
    return result


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--metadata-dir', type=Path)
    args = parser.parse_args()
    files = mapping()
    if not args.build:
        report = {'status': 'SOURCE_INVENTORY_ONLY_NOT_A_RELEASE', 'file_count': len(files),
                  'bytes': sum(p.stat().st_size for p in files.values()),
                  'paths': [{'source': str(source.relative_to(ROOT)), 'target': target.as_posix()} for target, source in files.items()],
                  'limits': 'Figures and metadata remain unfinished. No files copied, hashed as final, committed or uploaded.'}
        (R / 'evidence/RELEASE_SOURCE_INVENTORY.json').write_text(json.dumps(report, indent=2))
        print(json.dumps({k: v for k, v in report.items() if k != 'paths'}))
        return
    assert args.output is not None and args.metadata_dir is not None
    assert json.loads((R / 'results/REVISION_RAW_AUDIT.json').read_text())['status'] == 'PASS'
    assert json.loads((R / 'results/REVISION_TRAINING_EXPORT.json').read_text())['status'] == 'COMPLETE'
    intervals = json.loads((R / 'results/REVISION_INTERVAL_AUDIT.json').read_text())
    assert intervals['status'] == 'PASS_ALL_REVISION_INTERVAL_AUDIT' and intervals['summary_rows'] >= 72
    interval_sources = {k.replace('\\', '/'): v for k, v in intervals['source_sha256'].items()}
    assert interval_sources['results/REVISION_COST_SUMMARY.csv'] == sha(R / 'results/REVISION_COST_SUMMARY.csv')
    assert intervals['auditor_sha256'] == sha(R / 'scripts/audit_revision_intervals.py')
    relocation = json.loads((R / 'results/REVISION_CHECKPOINT_RELOCATION.json').read_text())
    assert relocation['status'] == 'PASS_ALL_50_REVISION_CHECKPOINTS_CPU_RELOCATION'
    assert relocation['complete_checkpoint_inventory'] is True and len(relocation['rows']) == 50
    assert relocation['test_sha256'] == sha(R / 'tests/check_revision_checkpoint_relocation.py')
    for row in relocation['rows']:
        checkpoint = ROOT / row['source']
        assert sha(checkpoint) == relocation['input_sha256'][row['source']]
    for folder, expected in [('E3', 21), ('E4_cuda', 10), ('E5', 19)]:
        summaries = list((R / 'checkpoints' / folder).rglob('*.summary.json'))
        assert len(summaries) == expected, (folder, len(summaries), expected)
        assert all(path.relative_to(ROOT) in files for path in summaries)
    figure_records = json.loads((PUB / 'overleaf/data/figure_provenance.json').read_text())
    required_figures = {'fig01_design', 'figR01_oracles', 'figR02_new_heat_constraints', 'figR03_K_ratio',
                        'figR04_K_p99', 'figR05_main_control', 'figRS02_latency_distributions', 'figRS03_scaling_curves'}
    assert required_figures <= {record['id'] for record in figure_records}
    for record in figure_records:
        for source in record['input_sha256']:
            assert sha(PUB / 'overleaf' / source['path']) == source['sha256']
        assert sha(PUB / 'overleaf/figures' / (record['id']+'.pdf')) == record['pdf_sha256']
    # Reviewed user-facing metadata is authored after Phase2, never invented here.
    for name in ['README.md', 'REPRODUCE.md', 'CITATION.cff', '.zenodo.json']:
        source = (args.metadata_dir / name).resolve()
        assert source.is_file(), source
        assert Path(name) not in files
        files[Path(name)] = source
    destination = args.output.resolve()
    # Deep historical evidence names can exceed Windows MAX_PATH when nested
    # under the long publication folder. A dedicated short workspace staging
    # root permits the same preserved relative layout without system changes.
    allowed = [PUB.resolve(), (ROOT / 'release_staging').resolve()]
    assert any(destination.is_relative_to(base) for base in allowed)
    assert not destination.exists()
    destination.mkdir(parents=True)
    records = []
    for target, source in files.items():
        output = destination / target
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, output)
        assert sha(output) == sha(source)
        records.append({'path': target.as_posix(), 'sha256': sha(output), 'source': str(source.relative_to(ROOT))})
    project = destination / 'pyproject.toml'
    original_text = project.read_text(encoding='utf-8')
    assert original_text.count('version = "0.1.0"') == 1
    project.write_text(original_text.replace('version = "0.1.0"', 'version = "1.1.0"'), encoding='utf-8')
    entry = next(row for row in records if row['path'] == 'pyproject.toml')
    entry.update(source_sha256=entry['sha256'], sha256=sha(project),
                 transformation='Top-level installable distribution version1.1.0; historical snapshot pyproject unchanged.')
    (destination / '.gitignore').write_text('__pycache__/\n*.py[cod]\n.pytest_cache/\n.venv/\n*.egg-info/\n*.npz\n*.pt\n*.pth\n*.aux\n*.log\n*.out\n', encoding='utf-8')
    provenance = destination / 'provenance'
    provenance.mkdir(exist_ok=True)
    (provenance / 'SOURCE_FILES.json').write_text(json.dumps({'status': 'STAGED_BYTE_VERIFIED_NOT_PUBLISHED', 'files': records}, indent=2))
    print(json.dumps({'status': 'STAGED_BYTE_VERIFIED_NOT_PUBLISHED', 'files': len(records), 'output': str(destination)}))


if __name__ == '__main__':
    main()
