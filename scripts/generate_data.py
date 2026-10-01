from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pdno.data.generate import generate_roles  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate train, validation, calibration, or explicitly unlocked locked-role shards.")
    parser.add_argument("--limit-per-role", type=int, default=None, help="small pre-data runtime pilot; omitted means full train/validation counts")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/manifests/parent_roles_metadata_only.json")
    parser.add_argument("--out", type=Path, default=ROOT / "data")
    parser.add_argument("--roles", nargs="+", choices=("train", "validation", "calibration", "locked_nominal",
                        "locked_coefficient_ood", "locked_delay_dropout"), default=("train", "validation"))
    parser.add_argument("--unlock-locked-after-calibration", action="store_true")
    parser.add_argument("--outer-ticks", type=int, default=128)
    args = parser.parse_args()
    if args.limit_per_role is not None and args.limit_per_role < 1:
        parser.error("--limit-per-role must be positive")
    if any(role.startswith("locked_") for role in args.roles):
        if not args.unlock_locked_after_calibration:
            parser.error("locked roles require --unlock-locked-after-calibration after calibration lock")
        lock = ROOT / "evidence/calibration_lock.json"
        if not lock.is_file():
            parser.error("evidence/calibration_lock.json is missing; locked targets remain sealed")
        if args.outer_ticks != 200:
            parser.error("locked evaluation episodes must use the source-specified --outer-ticks 200")
    records = generate_roles(args.manifest, args.out, args.limit_per_role, tuple(args.roles), args.unlock_locked_after_calibration, args.outer_ticks)
    print(json.dumps(records, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
