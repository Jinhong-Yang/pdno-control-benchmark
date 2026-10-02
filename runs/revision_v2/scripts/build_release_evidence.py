"""Package explicitly selected scientific binary evidence after final raw verification.

Default mode inventories file names and sizes only. --build requires completed
experiments, an independent raw audit, and a new empty destination directory.
Archives preserve workspace-relative paths for the existing read-only auditors.
No upload, Git mutation, or extraction is performed by this script.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import zipfile

R = Path(__file__).resolve().parents[1]
ROOT = R.parents[1]
OLD = ROOT / 'experiments/pdno_jevLite_20260927_v3'
MAX_PART_BYTES = 1_500_000_000


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def inventory():
    paths = set()
    for name in ['ORIGINAL_240_ARRAYS.json', 'ORIGINAL_PROTECTION.json']:
        for item in json.loads((R / 'evidence' / name).read_text()):
            path = ROOT / item['path']
            if path.suffix in ['.pt', '.npz']:
                paths.add(path)
    paths.update((OLD / 'data').rglob('*.npz'))
    paths.update((R / 'data').rglob('*.npz'))
    for stage in ['E1', 'E2', 'E3', 'E4', 'E5']:
        paths.update((R / f'results/{stage}_raw').rglob('*.npz'))
    for folder in ['E3', 'E4_cuda', 'E5']:
        paths.update((R / 'checkpoints' / folder).rglob('*.pt'))
    allowed_roots = [OLD.resolve(), R.resolve()]
    records = []
    for path in sorted(paths):
        resolved = path.resolve(strict=True)
        assert any(resolved.is_relative_to(base) for base in allowed_roots)
        relative = path.relative_to(ROOT).as_posix()
        assert not any(part.startswith('.') for part in Path(relative).parts)
        assert path.suffix in ['.npz', '.pt'] and not path.is_symlink()
        records.append({'path': relative, 'bytes': path.stat().st_size})
    return records


def require_complete():
    requirements = {
        'E1_heat_COMPLETE.json': 'COMPLETE', 'E1_burgers_cuda_COMPLETE.json': 'COMPLETE',
        'E2_TIMING.json': 'COMPLETE', 'E2_ANALYSIS.json': 'COMPLETE',
        'E3_EVALUATION.json': 'COMPLETE', 'E4_EVALUATION.json': 'COMPLETE',
        'E5_EVALUATION.json': 'COMPLETE', 'REVISION_TRAINING_EXPORT.json': 'COMPLETE',
        'REVISION_RAW_AUDIT.json': 'PASS',
    }
    for filename, status in requirements.items():
        receipt = json.loads((R / 'results' / filename).read_text())
        assert receipt['status'] == status, (filename, receipt['status'])
    for folder, expected in [('E3', 21), ('E4_cuda', 10), ('E5', 19)]:
        assert len(list((R / 'checkpoints' / folder).rglob('*.pt'))) == expected
    return {name: sha(R / 'results' / name) for name in requirements}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    records = inventory()
    manifest = {
        'status': 'INVENTORY_ONLY_NOT_BUILT',
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'layout': 'Extract all parts into the source repository root, retaining member paths.',
        'scope': 'Original protected checkpoints and raw outcomes/timings, original synthetic data, revision data, all new raw outcomes/timings, and E3/E4-CUDA/E5 selected checkpoints.',
        'excluded': ['preliminary E4 CPU fits', 'virtual environments', 'credentials',
                     'portraits', 'IEEE template and manuscript assets', 'unrelated research projects'],
        'max_uncompressed_part_bytes': MAX_PART_BYTES,
        'files': records,
        'source_limits': {
            'zenodo': 'https://help.zenodo.org/docs/deposit/manage-files/',
            'github': 'https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases',
            'checked_date': '2026-10-02',
        },
    }
    if not args.build:
        target = R / 'evidence/RELEASE_EVIDENCE_INVENTORY.json'
        target.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        print(json.dumps({'status': manifest['status'], 'files': len(records),
                          'bytes': sum(r['bytes'] for r in records), 'manifest': str(target)}))
        return
    assert args.output is not None, '--output required for build'
    manifest['completion_receipt_sha256'] = require_complete()
    destination = args.output.resolve()
    assert destination.is_relative_to(ROOT / 'publications/ieee_access_pdno_revision_v2_20261002')
    assert not destination.exists(), 'Use a new output directory; existing releases are never overwritten.'
    # Individual compressed NPZs are retained byte-for-byte inside ZIP_STORED parts.
    # Reserve ample ZIP header room below the documented GitHub per-asset limit.
    assert sum(r['bytes'] for r in records) < 49_000_000_000
    destination.mkdir(parents=True)
    parts, group, size = [], [], 0
    for row in records:
        assert row['bytes'] < MAX_PART_BYTES, ('Single source too large for planned part', row)
        if group and size + row['bytes'] > MAX_PART_BYTES:
            parts.append(group)
            group, size = [], 0
        group.append(row)
        size += row['bytes']
    if group:
        parts.append(group)
    assert len(parts) < 95
    archive_records = []
    for index, group in enumerate(parts, 1):
        archive = destination / f'pdno-v1.1.0-evidence-{index:02d}.zip'
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as bundle:
            for row in group:
                source = ROOT / row['path']
                before = source.stat()
                row['sha256'] = sha(source)
                row['archive'] = archive.name
                bundle.write(source, arcname=row['path'])
                after = source.stat()
                assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        assert archive.stat().st_size < 2 * 1024**3
        # Verify the archive payload independently, not just its directory listing.
        with zipfile.ZipFile(archive) as bundle:
            assert len(bundle.infolist()) == len(group)
            for row in group:
                digest = hashlib.sha256()
                with bundle.open(row['path']) as stream:
                    for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                        digest.update(block)
                assert digest.hexdigest() == row['sha256']
        archive_records.append({'name': archive.name, 'bytes': archive.stat().st_size,
                                'sha256': sha(archive), 'members': len(group)})
        print(json.dumps(archive_records[-1]), flush=True)
    manifest.update(status='BUILT_PAYLOAD_HASHES_VERIFIED_NOT_PUBLISHED', archives=archive_records)
    (destination / 'EVIDENCE_MANIFEST.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Evidence archives built and verified; no upload performed.')


if __name__ == '__main__':
    main()
