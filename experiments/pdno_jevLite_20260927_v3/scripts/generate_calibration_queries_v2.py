"""Generate eight teacher continuation snapshots per v2 calibration parent."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pdno.data.teacher_queries import build_teacher_queries  # noqa: E402


def main() -> int:
    receipt_path = ROOT / "evidence" / "v2_calibration_generation_receipt.json"
    if not receipt_path.is_file():
        raise SystemExit("v2 calibration trajectory receipt missing")
    if (ROOT / "data" / "queries_calibration_v2").exists():
        raise SystemExit("calibration teacher query output already exists")
    records = []
    for pde in ("burgers", "heat"):
        qmax = None
        if pde == "heat":
            maxima = []
            for role in ("train", "validation"):
                with np.load(ROOT / "data" / role / pde / "trajectories.npz", allow_pickle=False) as z:
                    maxima.append(float(z["goal"].max()))
            qmax = 1.25 * max(maxima)
        records.append(build_teacher_queries(
            ROOT / "data" / "calibration_v2" / "calibration" / pde / "trajectories.npz",
            pde, "calibration",
            ROOT / "data" / "queries_calibration_v2" / "calibration" / pde / "teacher_queries.npz",
            8, None, qmax))
    if len(records) != 2 or any(record.get("parent_count") != 64 for record in records):
        raise RuntimeError(f"calibration query coverage mismatch: {records}")
    result = {"status": "V2_CALIBRATION_QUERIES_GENERATED_TEST_SEALED", "test_opened": False,
              "snapshots_per_parent": 8, "records": records}
    out = ROOT / "evidence" / "v2_calibration_query_manifest.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "records": records, "output": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
