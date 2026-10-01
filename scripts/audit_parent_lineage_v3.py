"""Verify v3 identities, role membership and cross-version freshness from metadata."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _rows(manifest: dict) -> list[tuple[str, str, int, int, int, str]]:
    return [
        (row["pde"], row["parent_id"], int(row["initial_condition_seed"]),
         int(row["physical_parameter_seed"]), int(row["disturbance_seed"]), row["split"])
        for group in manifest["roles"].values() for row in group["parents"]
    ]


def main() -> int:
    current = json.loads((ROOT / "data/manifests/parent_roles_metadata_only.json").read_text(encoding="utf-8"))
    previous = {
        name: json.loads((ROOT / relative / "data/manifests/parent_roles_metadata_only.json").read_text(encoding="utf-8"))
        for name, relative in (("v1", "../pdno_jevLite_20260925"), ("v2", "../pdno_jevLite_20260926_v2"))
    }
    current_rows = _rows(current)
    current_keys = {(pde, parent, ic, parameter, disturbance) for pde, parent, ic, parameter, disturbance, _ in current_rows}
    overlap = {}
    for name, manifest in previous.items():
        old_keys = {(pde, parent, ic, parameter, disturbance) for pde, parent, ic, parameter, disturbance, _ in _rows(manifest)}
        overlap[name] = len(current_keys & old_keys)

    role_expected = {(pde, role): {row["parent_id"] for row in group["parents"] if row["split"] == role}
                     for pde, group in current["roles"].items()
                     for role in {row["split"] for row in group["parents"]}}
    role_actual, role_errors = {}, []
    for (pde, role), expected in role_expected.items():
        path = ROOT / "data" / "train_validation_v3" / role / pde / "trajectories.npz"
        if role not in {"train", "validation"}:
            if path.exists():
                role_errors.append(f"pre-calibration generation unexpectedly includes {role}/{pde}")
            continue
        if not path.is_file():
            role_errors.append(f"missing generated shard {role}/{pde}")
            continue
        with np.load(path, allow_pickle=False) as archive:
            actual = set(archive["parent_id"].astype(str).tolist())
        if actual != expected:
            role_errors.append(f"actual generated parent IDs do not equal manifest role IDs for {role}/{pde}")
        role_actual[f"{role}/{pde}"] = {"expected": len(expected), "actual": len(actual), "match": actual == expected}

    role_sets = {}
    for pde, parent, ic, parameter, disturbance, role in current_rows:
        role_sets.setdefault((pde, role), set()).add((parent, ic, parameter, disturbance))
    within_overlaps = {}
    for pde in current["roles"]:
        names = sorted(role for pd, role in role_sets if pd == pde)
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                n = len(role_sets[(pde, left)] & role_sets[(pde, right)])
                within_overlaps[f"{pde}:{left}__{right}"] = n
                if n:
                    role_errors.append(f"v3 cross-role identity overlap: {pde} {left} vs {right}: {n}")

    result = {
        "namespace": current["namespace"], "metadata_only_for_predecessors": True,
        "test_opened": False, "v3_parent_tuple_count": len(current_keys),
        "cross_version_parent_seed_tuple_overlap": overlap,
        "v3_within_pde_role_overlap_counts": within_overlaps,
        "generated_train_validation_parent_check": role_actual,
        "calibration_or_locked_shards_found": any(
            (ROOT / "data" / "train_validation_v3" / role / pde / "trajectories.npz").exists()
            for pde, role in role_expected if role not in {"train", "validation"}),
        "errors": role_errors,
        "passed": not role_errors and all(n == 0 for n in overlap.values()),
    }
    out = ROOT / "evidence" / "parent_lineage_audit_v3.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
