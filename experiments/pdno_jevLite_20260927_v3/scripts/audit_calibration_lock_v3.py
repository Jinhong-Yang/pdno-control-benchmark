"""Independently audit v3 calibration margins, groups, finite-sample ranks, and hashes."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PREDICTIVE = ("P", "P-no-rank", "B4", "B5")
SEEDS = (11, 23, 37)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    lock_path = ROOT / "evidence" / "calibration_lock_v3.json"
    query_manifest_path = ROOT / "evidence" / "v3_calibration_query_manifest.json"
    query_audit_path = ROOT / "evidence" / "v3_calibration_query_audit_receipt.json"
    data_receipt_path = ROOT / "evidence" / "v3_calibration_generation_receipt.json"
    checkpoint_path = ROOT / "evidence" / "confirmatory_checkpoint_manifest_v3.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    query_manifest = json.loads(query_manifest_path.read_text(encoding="utf-8"))
    query_audit = json.loads(query_audit_path.read_text(encoding="utf-8"))
    checkpoint_manifest = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    data_receipt = json.loads(data_receipt_path.read_text(encoding="utf-8"))
    if lock.get("status") != "CALIBRATION_LOCKED_TEST_SEALED" or lock.get("test_opened") is not False:
        raise SystemExit("calibration lock missing, invalid, or test-opened")
    if query_manifest.get("test_opened") is not False or query_audit.get("status") != "CALIBRATION_QUERY_AUDIT_PASSED":
        raise SystemExit("calibration query artifacts are not audited and test-sealed")
    if checkpoint_manifest.get("status") != "FROZEN_CONFIRMATORY_CHECKPOINTS" or checkpoint_manifest.get("test_opened") is not False:
        raise SystemExit("confirmatory checkpoints are not frozen/test-sealed")
    if data_receipt.get("test_opened") is not False:
        raise SystemExit("calibration parent data is not test-sealed")

    expected = {(pde, "B1", None) for pde in ("burgers", "heat")}
    expected |= {(pde, method, seed) for pde in ("burgers", "heat")
                 for method in PREDICTIVE for seed in SEEDS}
    rows = {(row["pde"], row["method"], row["seed"]): row for row in lock.get("groups", [])}
    if len(rows) != 26 or rows.keys() != expected:
        raise SystemExit(f"calibration group coverage mismatch: missing={expected-rows.keys()}, extra={rows.keys()-expected}")

    query_by_pde = {row["pde"]: row for row in query_audit["shards"]}
    checkpoints = {(r["pde"], r["method"], int(r["seed"])): r for r in checkpoint_manifest["checkpoints"]}
    expected_inputs = {
        "final_spec": ROOT / "config" / "final_spec_v3.json",
        "evaluation_addendum": ROOT / "config" / "evaluation_addendum_v3.yaml",
        "checkpoint_manifest": checkpoint_path,
        "calibration_query_manifest": query_manifest_path,
        "calibration_generation_receipt": data_receipt_path,
    }
    for key, path in expected_inputs.items():
        if not path.is_file() or lock.get("inputs_sha256", {}).get(key) != sha(path):
            raise SystemExit(f"calibration input hash mismatch: {key}")

    group_rows = []
    for key, group in rows.items():
        pde, method, seed = key
        query = query_by_pde[pde]
        scores = group.get("parent_scores", {})
        values = np.asarray(list(scores.values()), dtype=np.float64)
        order_index = math.ceil((len(values) + 1) * float(lock["quantile"])) - 1
        if (group.get("parent_count") != 64 or group.get("query_count") != 512
                or len(scores) != 64 or not np.isfinite(values).all()):
            raise SystemExit(f"calibration parent score/count invalid: {key}")
        if group.get("finite_sample_order_index_zero_based") != order_index:
            raise SystemExit(f"finite sample order index mismatch: {key}")
        if float(group["margin"]) != float(np.sort(values)[order_index]):
            raise SystemExit(f"stored calibration margin does not match declared order statistic: {key}")
        if group.get("action_selection_uses_margin") is not False:
            raise SystemExit(f"calibration margin feeds action selection: {key}")
        if group.get("calibration_query_sha256") != query["query_sha256"]:
            raise SystemExit(f"query hash mismatch in group: {key}")
        if method == "B1":
            if group.get("checkpoint_sha256") is not None:
                raise SystemExit(f"analytic B1 has unexpected checkpoint hash: {key}")
        else:
            checkpoint = checkpoints[(pde, method, int(seed))]
            if group.get("checkpoint_sha256") != checkpoint["sha256"]:
                raise SystemExit(f"checkpoint hash mismatch in group: {key}")
        group_rows.append({"pde": pde, "method": method, "seed": seed, "parent_count": 64,
                           "query_count": 512, "margin": float(group["margin"]),
                           "order_index_zero_based": order_index,
                           "finite_parent_scores": True, "action_selection_uses_margin": False})

    receipt = {"status": "CALIBRATION_LOCK_AUDIT_PASSED", "test_opened": False,
        "group_count": len(group_rows), "groups": group_rows,
        "quantile": lock["quantile"], "exchangeability_scope": lock["exchangeability_scope"],
        "limitations": lock["limitations"], "calibration_lock_sha256": sha(lock_path),
        "query_audit_sha256": sha(query_audit_path), "query_manifest_sha256": sha(query_manifest_path),
        "checkpoint_manifest_sha256": sha(checkpoint_path), "generation_receipt_sha256": sha(data_receipt_path),
        "all_checks_passed": True}
    out = ROOT / "evidence" / "v3_calibration_lock_audit_receipt.json"
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "group_count": len(group_rows),
                      "output": str(out), "sha256": sha(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
