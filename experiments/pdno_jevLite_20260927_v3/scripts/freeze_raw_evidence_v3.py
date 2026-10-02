"""Freeze completed v3 locked-test and host-latency evidence by content hash.

This is a post-analysis utility. It refuses partial runs and never changes raw
artifacts. Invoke only after locked aggregation and host E2E latency analysis.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return value


def checked_inventory(expected: dict[str, str], label: str) -> list[dict[str, str]]:
    rows = []
    for rel, wanted in sorted(expected.items()):
        path = ROOT / rel
        if not path.is_file():
            raise SystemExit(f"{label} artifact missing: {path}")
        observed = sha256(path)
        if observed != wanted:
            raise SystemExit(f"{label} hash mismatch: {path}")
        rows.append({"path": rel, "sha256": observed})
    return rows


def add_file(rows: list[dict[str, str]], path: Path) -> None:
    rows.append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path)})


def main() -> int:
    output = EVIDENCE / "raw_evidence_freeze_v3.json"
    sidecar = EVIDENCE / "raw_evidence_freeze_v3.sha256"
    if output.exists() or sidecar.exists():
        raise SystemExit("raw evidence freeze already exists; refusing overwrite")

    marker_path = ROOT / "state" / "LOCKED_TEST_ONCE_V3.json"
    marker = load_json(marker_path)
    if marker.get("status") != "LOCKED_TEST_EVALUATION_COMPLETE" or marker.get("test_opened") is not True:
        raise SystemExit("locked one-shot test is not complete")

    locked_root = EVIDENCE / "locked_test_raw_v3"
    locked_files = sorted([*locked_root.glob("*/**/*.npz"), *locked_root.glob("*/**/*.record.json")])
    locked_npz = [p for p in locked_files if p.suffix == ".npz"]
    locked_records = [p for p in locked_files if p.name.endswith(".record.json")]
    if len(locked_npz) != 120 or len(locked_records) != 120:
        raise SystemExit(f"locked raw inventory incomplete: {len(locked_npz)} NPZ, {len(locked_records)} records")
    locked_expected = {str(p.relative_to(ROOT)): marker.get("raw_result_sha256", {}).get(str(p.relative_to(ROOT)))
                       for p in locked_npz}
    if any(value is None for value in locked_expected.values()):
        raise SystemExit("locked marker lacks one or more raw-result hashes")
    record_expected = {str(p.relative_to(ROOT)): marker.get("raw_artifact_sha256", {}).get(str(p.relative_to(ROOT)))
                       for p in locked_records}
    if any(value is None for value in record_expected.values()):
        raise SystemExit("locked marker lacks one or more raw-record hashes")
    locked_inventory = checked_inventory({**locked_expected, **record_expected}, "locked raw")

    aggregate_path = EVIDENCE / "locked_test_analysis_v3.json"
    aggregate = load_json(aggregate_path)
    if aggregate.get("status") != "LOCKED_TEST_AGGREGATED" or aggregate.get("test_opened") is not True:
        raise SystemExit("registered locked aggregation is missing or incomplete")

    latency_summary_path = EVIDENCE / "latency_summary_v3.json"
    latency_summary = load_json(latency_summary_path)
    if latency_summary.get("status") != "HOST_READY_E2E_THREE_SESSION_COMPLETE" or latency_summary.get("test_opened") is not True:
        raise SystemExit("host-ready E2E latency run is missing or incomplete")
    latency_root = EVIDENCE / "latency_raw_v3"
    latency_npz = sorted(latency_root.glob("*.npz"))
    latency_expected = latency_summary.get("raw_files_sha256", {})
    if len(latency_npz) != 120 or len(latency_expected) != 120:
        raise SystemExit(f"latency raw inventory incomplete: {len(latency_npz)} files, {len(latency_expected)} hashes")
    latency_inventory = checked_inventory(latency_expected, "host E2E latency")
    if {row["path"] for row in latency_inventory} != {str(p.relative_to(ROOT)) for p in latency_npz}:
        raise SystemExit("latency raw files differ from summary inventory")

    latency_analysis_path = EVIDENCE / "latency_analysis_v3.json"
    latency_analysis = load_json(latency_analysis_path)
    if latency_analysis.get("status") != "LOCKED_LATENCY_ANALYSIS_COMPLETE" or latency_analysis.get("test_opened") is not True:
        raise SystemExit("registered latency analysis is missing or incomplete")

    required = [
        marker_path,
        ROOT / "state" / "LOCKED_TEST_PROCESS_V3.json",
        ROOT / "state" / "RUN_STATE.json",
        ROOT / "state" / "STATE.md",
        ROOT / "state" / "RUN_LEDGER.md",
        ROOT / "AUTHOR_INPUT_REQUIRED.md",
        ROOT / "evidence" / "failures_and_modifications_v3.jsonl",
        ROOT / "logs" / "locked_test_v3" / "stdout.log",
        ROOT / "logs" / "locked_test_v3" / "stderr.log",
        aggregate_path,
        latency_summary_path,
        latency_analysis_path,
        ROOT / "scripts" / "aggregate_locked_test_v3.py",
        ROOT / "scripts" / "benchmark_host_e2e_v3.py",
        ROOT / "scripts" / "analyze_latency_v3.py",
        Path(__file__),
    ]
    other_inventory = []
    for path in required:
        if not path.is_file():
            raise SystemExit(f"required evidence/context file missing: {path}")
        add_file(other_inventory, path)

    # Capture the complete experiment evidence and reproduction tree while
    # excluding the venv, caches, and transient state that changes after freeze.
    tree_inventory = []
    for area in ("data", "evidence", "logs", "reports", "runs", "config", "contracts", "src", "scripts"):
        area_root = ROOT / area
        if not area_root.is_dir():
            raise SystemExit(f"required experiment area missing: {area_root}")
        for path in sorted(p for p in area_root.rglob("*") if p.is_file()):
            if "__pycache__" in path.parts or path.suffix in {".pyc", ".tmp"}:
                continue
            add_file(tree_inventory, path)

    payload = {
        "schema": "pdno_jevLite_raw_evidence_freeze_v3",
        "status": "RAW_EVIDENCE_FROZEN",
        "test_opened": True,
        "locked_test_status": marker["status"],
        "locked_test_marker_sha256": sha256(marker_path),
        "locked_raw_artifacts_count": len(locked_inventory),
        "locked_raw_artifacts": locked_inventory,
        "locked_aggregate_sha256": sha256(aggregate_path),
        "host_latency_status": latency_summary["status"],
        "host_latency_raw_count": len(latency_inventory),
        "host_latency_raw_artifacts": latency_inventory,
        "host_latency_summary_sha256": sha256(latency_summary_path),
        "latency_analysis_sha256": sha256(latency_analysis_path),
        "context_and_procedure_files": sorted(other_inventory, key=lambda row: row["path"]),
        "experiment_tree_file_count": len(tree_inventory),
        "experiment_tree_inventory": sorted(tree_inventory, key=lambda row: row["path"]),
    }
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    temp = output.with_name(output.name + ".tmp")
    with temp.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, output)
    digest = hashlib.sha256(encoded).hexdigest()
    with sidecar.open("xb") as stream:
        stream.write(f"{digest}  {output.name}\n".encode("ascii"))
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"status": payload["status"], "manifest": str(output),
                      "manifest_sha256": digest, "locked_artifacts": len(locked_inventory),
                      "latency_artifacts": len(latency_inventory)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
