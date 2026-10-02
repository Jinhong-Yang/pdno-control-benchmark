from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pdno.data.manifests import audit_role_manifest, build_role_manifest  # noqa: E402


def main() -> int:
    manifest = build_role_manifest()
    audit = audit_role_manifest(manifest)
    if not audit["passed"] or audit["test_opened"]:
        raise SystemExit(f"role manifest audit failed: {audit}")
    outdir = ROOT / "data" / "manifests"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "parent_roles_metadata_only.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (outdir / "parent_roles_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
