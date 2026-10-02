"""Small file-integrity and path-safety checks; no scientific or GPU claim."""
from pathlib import Path
import importlib.util
import json
import tempfile

R = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('verifier', R / 'release_support/verify_evidence.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
checks = []
with tempfile.TemporaryDirectory(prefix='pdno-release-check-') as folder:
    base = Path(folder)
    payload = base / 'sample.bin'
    payload.write_bytes(b'original')
    manifest = {'status': 'BUILT_PAYLOAD_HASHES_VERIFIED_NOT_PUBLISHED',
                'files': [{'path': 'sample.bin', 'bytes': 8, 'sha256': module.digest(payload)}]}
    assert module.validate(manifest, base)['status'] == 'PASS'
    checks.append('matching payload passes')
    payload.write_bytes(b'tampered')
    assert module.validate(manifest, base)['issues'][0]['error'] == 'sha256_mismatch'
    checks.append('same-size tamper detected')
    payload.unlink()
    assert module.validate(manifest, base)['issues'][0]['error'] == 'missing'
    checks.append('missing payload detected')
    for unsafe in ['../outside.bin', '/absolute.bin', 'C:/outside.bin', 'folder\\file.bin']:
        manifest['files'][0]['path'] = unsafe
        try:
            module.validate(manifest, base)
        except ValueError:
            checks.append('unsafe path rejected: ' + unsafe)
        else:
            raise AssertionError(unsafe)
    manifest['status'] = 'INVENTORY_ONLY_INCOMPLETE_CAMPAIGN'
    try:
        module.validate(manifest, base)
    except ValueError:
        checks.append('incomplete inventory rejected')
    else:
        raise AssertionError('incomplete manifest accepted')
record = {'status': 'PASS', 'checks': checks,
          'scope': 'Tiny integrity fixtures only; final archive download and full hash validation remain pending.'}
(R / 'results/RELEASE_VERIFIER_SUPPORT_TESTS.json').write_text(json.dumps(record, indent=2))
print(json.dumps(record))
