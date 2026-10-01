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
    parser = argparse.ArgumentParser(description="Generate privileged teacher queries for train/validation only.")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/v2")
    parser.add_argument("--out-root", type=Path, default=ROOT / "data/queries_v2")
    parser.add_argument("--train-snapshots-per-parent", type=int, default=4)
    parser.add_argument("--validation-snapshots-per-parent", type=int, default=1)
    parser.add_argument("--limit-parents", type=int, default=None)
    args = parser.parse_args()
    records = []
    for pde in ("burgers", "heat"):
        shared_heat_qmax = None
        if pde == "heat":
            maxima = []
            for role in ("train", "validation"):
                with np.load(args.data_root / role / pde / "trajectories.npz", allow_pickle=False) as archive:
                    maxima.append(float(archive["goal"].max()))
            shared_heat_qmax = 1.25 * max(maxima)
        for split, count in (("train", args.train_snapshots_per_parent), ("validation", args.validation_snapshots_per_parent)):
            records.append(build_teacher_queries(args.data_root / split / pde / "trajectories.npz", pde, split,
                                                 args.out_root / split / pde / "teacher_queries.npz", count, args.limit_parents,
                                                 shared_heat_qmax))
    print(json.dumps(records, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
