"""Read-only validation of downloaded archive parts or extracted evidence.

Examples:
  python verify_evidence.py EVIDENCE_MANIFEST.json --archives downloads
  python verify_evidence.py EVIDENCE_MANIFEST.json --root pdno-control-benchmark
"""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def validate(manifest, base, archive_mode=False):
    if manifest.get('status') != 'BUILT_PAYLOAD_HASHES_VERIFIED_NOT_PUBLISHED':
        raise ValueError('Manifest is an incomplete inventory, not a verified evidence build.')
    base = Path(base).resolve(strict=True)
    rows = manifest['archives'] if archive_mode else manifest['files']
    if not rows:
        raise ValueError('Empty inventory cannot establish evidence completeness.')
    issues, seen = [], set()
    for row in rows:
        name = row['name'] if archive_mode else row['path']
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or '\\' in name or ':' in name:
            raise ValueError('Unsafe manifest member: ' + name)
        if name in seen:
            raise ValueError('Duplicate manifest member: ' + name)
        seen.add(name)
        path = (base / name).resolve()
        if not path.is_relative_to(base):
            raise ValueError('Manifest member leaves selected root: ' + name)
        if not path.is_file():
            issues.append({'path': name, 'error': 'missing'})
        elif path.stat().st_size != row['bytes']:
            issues.append({'path': name, 'error': 'size_mismatch'})
        elif digest(path) != row['sha256']:
            issues.append({'path': name, 'error': 'sha256_mismatch'})
    return {'status': 'PASS' if not issues else 'FAIL', 'checked_files': len(rows),
            'scope': 'archive bytes' if archive_mode else 'extracted evidence bytes', 'issues': issues}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest', type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--root', type=Path)
    group.add_argument('--archives', type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    result = validate(manifest, args.archives or args.root, archive_mode=args.archives is not None)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
