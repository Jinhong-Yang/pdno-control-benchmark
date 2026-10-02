"""Metadata-only parent pools; split roles are frozen before target generation."""

from __future__ import annotations

import hashlib


COUNTS = {
    "burgers": {
        "train": 512,
        "validation": 64,
        "calibration": 64,
        "locked_nominal": 384,
        "locked_coefficient_ood": 128,
        "locked_delay_dropout": 128,
    },
    "heat": {
        "train": 256,
        "validation": 64,
        "calibration": 64,
        "locked_nominal": 384,
        "locked_coefficient_ood": 128,
        "locked_delay_dropout": 128,
    },
}


def _ordered_ids(pde: str, count: int, namespace: str) -> list[str]:
    # Put the namespace in the identity itself as well as the ordering hash.
    # Hashing only the order reuses the same physical parents across versions.
    ids = [f"{namespace}:{pde}:parent:{i:06d}" for i in range(count)]
    return sorted(ids, key=lambda item: hashlib.sha256(f"{namespace}|{item}".encode()).digest())


def build_role_manifest(namespace: str = "pdno-jevLite-parent-pool-v3-fresh-20260927") -> dict:
    pdes = {}
    for pde, role_counts in COUNTS.items():
        ordered = _ordered_ids(pde, sum(role_counts.values()), namespace)
        cursor = 0
        rows = []
        for role, count in role_counts.items():
            for pid in ordered[cursor : cursor + count]:
                index = int(pid.rsplit(":", 1)[1])
                rows.append({
                    "parent_id": pid,
                    "split": role,
                    "pde": pde,
                    "initial_condition_seed": 100000 + index,
                    "physical_parameter_seed": 200000 + index,
                    "disturbance_seed": 300000 + index,
                    "target_status": "WITHHELD" if role.startswith("locked_") or role == "calibration" else "NOT_GENERATED",
                })
            cursor += count
        pdes[pde] = {"role_counts": role_counts, "parents": rows}
    return {
        "manifest_version": 3,
        "namespace": namespace,
        "split_unit": "parent_episode",
        "metadata_only": True,
        "test_opened": False,
        "roles": pdes,
    }


def audit_role_manifest(manifest: dict) -> dict:
    errors = []
    summaries = {}
    for pde, block in manifest["roles"].items():
        rows = block["parents"]
        ids = [row["parent_id"] for row in rows]
        counts: dict[str, int] = {}
        for row in rows:
            counts[row["split"]] = counts.get(row["split"], 0) + 1
        if len(ids) != len(set(ids)):
            errors.append(f"{pde}: duplicate parent id")
        if counts != block["role_counts"]:
            errors.append(f"{pde}: role counts differ from contract")
        seeds = [tuple(row[k] for k in ("initial_condition_seed", "physical_parameter_seed", "disturbance_seed")) for row in rows]
        if len(seeds) != len(set(seeds)):
            errors.append(f"{pde}: duplicate seed tuple")
        if any(row["target_status"] != "WITHHELD" for row in rows if row["split"].startswith("locked_") or row["split"] == "calibration"):
            errors.append(f"{pde}: locked/calibration target was not withheld")
        summaries[pde] = {"parent_count": len(rows), "role_counts": counts, "unique_parent_ids": len(set(ids))}
    return {"passed": not errors, "errors": errors, "pdes": summaries, "test_opened": manifest.get("test_opened") is True}
