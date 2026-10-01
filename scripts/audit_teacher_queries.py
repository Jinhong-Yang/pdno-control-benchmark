from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pdno.data.audit_teacher import audit_teacher_tree  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--role-manifest", type=Path, default=ROOT / "data/manifests/parent_roles_metadata_only.json")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    result = audit_teacher_tree(args.root, args.role_manifest)
    output = json.dumps(result, indent=2)
    print(output)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output + "\n", encoding="utf-8")
    return 0 if result["passed"] and not result["calibration_or_test_query_shards_found"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
