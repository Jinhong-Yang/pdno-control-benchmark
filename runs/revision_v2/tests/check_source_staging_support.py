"""Relocate current allowlisted source files and run documented CPU commands.

This deliberately does not build a release, install fresh dependencies, copy
binary evidence, train, or claim that the incomplete campaign is reproducible.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile

R = Path(__file__).resolve().parents[1]
ROOT = R.parents[1]
PUB = ROOT/"publications/ieee_access_pdno_revision_v2_20261002"
stager = R/"scripts/stage_release_source.py"
spec = importlib.util.spec_from_file_location("release_source_mapping", stager)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
mapping = module.mapping()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
hashes = {}
commands = []
temporary_base = (ROOT/"tmp"/"release_checks").resolve()
assert temporary_base.is_relative_to(ROOT.resolve())
temporary_base.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(prefix="src-", dir=temporary_base) as folder:
    root = Path(folder).resolve()
    assert root.is_relative_to(temporary_base)
    for target, source in mapping.items():
        destination = root/target
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        hashes[target.as_posix()] = sha(destination)
        assert sha(source) == hashes[target.as_posix()]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", PYTHONPATH=str(root/"src"), PYTHONDONTWRITEBYTECODE="1")
    for args in [
        ["-m", "pytest", "-q", "tests"],
        ["publication/scripts/build_tables.py"],
    ]:
        result = subprocess.run([sys.executable, "-B", "-X", "utf8", *args],
                                cwd=root, env=env, capture_output=True,
                                text=True, encoding="utf-8", timeout=180)
        command = dict(command=["python", *args], exit_code=result.returncode,
                       stdout=result.stdout, stderr=result.stderr)
        commands.append(command)
        if result.returncode:
            failure = dict(status="FAIL_CURRENT_SOURCE_LAYOUT_SUPPORT", commands=commands,
                           stager_sha256=sha(stager))
            (R/"results/SOURCE_LAYOUT_SUPPORT_FAILED.json").write_text(json.dumps(failure, indent=2)+"\n")
            raise RuntimeError(result.stdout+"\n"+result.stderr)
    assert not (root/"publication/main.tex").exists()
    binding = json.loads((root/"publication/data/number_bindings_resolved.json").read_text())
    audits = [json.loads((root/"publication/data/revision"/name).read_text())
              for name in ["E2_TABLE_NUMERIC_AUDIT.json", "COMPLETED_TABLE_NUMERIC_AUDIT.json"]]
    assert all(a["status"].startswith("PASS") for a in audits)
    audit_counts = [dict(status=a["status"], tables=a["tables"],
                        values=a["displayed_numeric_values_checked"]) for a in audits]

receipt = dict(status="PASS_CURRENT_SOURCE_LAYOUT_CPU_AND_TABLE_SUPPORT",
               time_utc=datetime.now(timezone.utc).isoformat(),
               stager_sha256=sha(stager), files_copied=len(mapping), copied_sha256=hashes,
               commands=commands, table_audits=audit_counts,
               manuscript_sources_excluded=True,
               scope="Current source allowlist in isolated directory; existing dependency environment, CPU-only tests and summary-to-table commands. No final metadata, archives, binary extraction, fresh installation, new checkpoint test, GPU benchmark or complete-release certification.")
(R/"results/SOURCE_LAYOUT_SUPPORT.json").write_text(json.dumps(receipt, indent=2)+"\n")
print(json.dumps(dict(status=receipt["status"], files_copied=len(mapping),
                      table_audits=audit_counts, test_stdout=commands[0]["stdout"])))
