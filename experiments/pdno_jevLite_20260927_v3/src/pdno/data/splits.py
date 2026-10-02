"""Parent-level deterministic split allocation and overlap auditing."""

from __future__ import annotations

from collections import defaultdict
import hashlib
from typing import Iterable


def parent_id(pde: str, initial_seed: int, parameter_seed: int, disturbance_seed: int) -> str:
    return f"{pde}:ic{initial_seed:08d}:p{parameter_seed:08d}:d{disturbance_seed:08d}"


def assign_role(parent: str, namespace: str, weights: dict[str, int]) -> str:
    """Stable weighted bucket assignment; each physical parent gets exactly one role."""
    if not weights or any(v <= 0 for v in weights.values()):
        raise ValueError("weights must be a nonempty mapping of positive integers")
    total = sum(weights.values())
    digest = hashlib.sha256(f"{namespace}|{parent}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % total
    cumulative = 0
    for role, count in weights.items():
        cumulative += count
        if bucket < cumulative:
            return role
    raise AssertionError("unreachable bucket")


def audit_parent_roles(rows: Iterable[dict]) -> dict:
    roles = defaultdict(set)
    seen_rows = set()
    duplicates = []
    for row in rows:
        pid, role = row["parent_id"], row["split"]
        key = (pid, role)
        if key in seen_rows:
            duplicates.append(key)
        seen_rows.add(key)
        roles[pid].add(role)
    cross_role = {pid: sorted(value) for pid, value in roles.items() if len(value) > 1}
    return {
        "row_count": len(seen_rows),
        "parent_count": len(roles),
        "within_role_duplicate_rows": len(duplicates),
        "cross_role_parent_count": len(cross_role),
        "cross_role_parents": cross_role,
        "passed": not duplicates and not cross_role,
    }
