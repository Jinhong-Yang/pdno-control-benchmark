"""Generate calibration-only teacher continuations after model freeze."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pdno.data.teacher_queries import build_teacher_queries  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=ROOT / "data/evaluation_v1")
    ap.add_argument("--out-root", type=Path, default=ROOT / "data/queries_calibration_v1")
    ap.add_argument("--snapshots-per-parent", type=int, default=8)
    args = ap.parse_args()
    if args.snapshots_per_parent != 8:
        ap.error("frozen calibration protocol requires exactly 8 snapshots per parent")
    records = []
    for pde in ("burgers", "heat"):
        qmax = None
        if pde == "heat":
            maxima = []
            for role in ("train", "validation"):
                with np.load(ROOT / "data/v2" / role / pde / "trajectories.npz", allow_pickle=False) as z:
                    maxima.append(float(z["goal"].max()))
            qmax = 1.25 * max(maxima)
        records.append(build_teacher_queries(
            args.data_root / "calibration" / pde / "trajectories.npz", pde, "calibration",
            args.out_root / "calibration" / pde / "teacher_queries.npz", 8, None, qmax))
    print(json.dumps(records, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
