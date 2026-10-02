"""Independent arithmetic and source-hash audit of completed X3 scalar outputs."""
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[1]
RESULTS = RUN / "results" / "X3"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
    summary_path = RESULTS / "X3_OPERATOR_SUMMARIES.json"
    vectors_path = RESULTS / "X3_PAIRED_VECTORS.csv"
    d = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = d["rows"]
    with vectors_path.open(newline="", encoding="utf-8-sig") as f:
        vectors = list(csv.DictReader(f))
    key = lambda r: (r["stage"], r["pde"], r["method"], int(r["seed"]), "" if r["target_factor"] is None else str(r["target_factor"]))
    by = defaultdict(list)
    for v in vectors:
        by[(v["stage"], v["pde"], v["method"], int(v["seed"]), v["target_factor"])].append(v)
    if len(rows) != 54 or len(vectors) != 3456 or len(by) != 54:
        raise RuntimeError("X3 row inventory mismatch")
    max_metric_error = 0.0
    max_identity_error = 0.0
    target_hash_mismatches = []
    branch_mismatches = []
    for r in rows:
        vr = by[key(r)]
        if len(vr) != 64 or len({x["parent_id"] for x in vr}) != 64:
            raise RuntimeError(f"Expected 64 distinct parent rows for {r['checkpoint']}")
        sumc = lambda col: sum(float(x[col]) for x in vr)
        recomputed = {
            "E_obs": math.sqrt(sumc("initial_error_sq") / sumc("target_initial_sq")),
            "E_prop": math.sqrt(sumc("prop_error_sq") / sumc("target_future_sq")),
            "E_op": math.sqrt(sumc("op_error_sq") / sumc("target_future_sq")),
            "E_total": math.sqrt(sumc("total_error_sq") / sumc("target_future_sq")),
            "persistence": math.sqrt(sumc("persistence_error_sq") / sumc("target_future_sq")),
        }
        for name, value in recomputed.items():
            max_metric_error = max(max_metric_error, abs(value - r[name]))
        ratio = recomputed["E_prop"] / recomputed["E_total"]
        branch = "observer_dominant" if ratio >= .8 else "operator_error_substantial" if ratio < .5 else "both_contributions_present"
        if branch != r["decision_branch"]:
            branch_mismatches.append(r["checkpoint"])
        identity = abs(sumc("total_error_sq") - sumc("prop_error_sq") - sumc("op_error_sq") - sumc("cross_term_2prop_dot_op"))
        max_identity_error = max(max_identity_error, identity)
        query = ROOT / r["queries"]
        with np.load(query, allow_pickle=False) as z:
            truth = np.ascontiguousarray(z["future_field"].astype("<f4"))
        raw_hash = hashlib.sha256(truth.tobytes()).hexdigest()
        if raw_hash != r["raw_vector_sha256"]["future_target_float32"]:
            target_hash_mismatches.append(r["checkpoint"])
    if max_metric_error > 1e-6 or max_identity_error > 1e-5 or branch_mismatches or target_hash_mismatches:
        raise RuntimeError("X3 paired arithmetic, branch, or target-hash audit failed")
    failures = [r for r in rows if not r["gate_reproduction_four_decimals"]]
    groups = defaultdict(list)
    for r in rows:
        groups[(r["stage"], r["pde"])].append(r)
    out = {
        "status": "PASS_PAIRED_ARITHMETIC_WITH_GATE_REPRODUCTION_MISMATCH" if failures else "PASS_ALL_X3_AUDITS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "summary_sha256": sha(summary_path), "paired_vectors_sha256": sha(vectors_path),
        "operator_count": len(rows), "paired_parent_rows": len(vectors),
        "reaggregated_scalar_metrics_max_abs_error": max_metric_error,
        "error_vector_identity_max_abs_error": max_identity_error,
        "prediction_target_reference_hashes_present": all(len(r["raw_vector_sha256"]) == 4 for r in rows),
        "raw_target_hash_mismatches": target_hash_mismatches,
        "branch_mismatches": branch_mismatches,
        "four_decimal_gate_mismatch_count": len(failures),
        "max_abs_Etotal_gate_difference": max(r["E_total_abs_difference_from_saved_gate"] for r in rows),
        "gate_mismatches": [{"stage":r["stage"],"pde":r["pde"],"method":r["method"],"seed":r["seed"],"target_factor":r["target_factor"],"cpu_E_total":r["E_total"],"frozen_gate_E_total":r["saved_gate_E_total"],"absolute_difference":r["E_total_abs_difference_from_saved_gate"],"cpu_4dp":format(r["E_total"],".4f"),"gate_4dp":format(r["saved_gate_E_total"],".4f")} for r in failures],
        "group_summaries": [{"stage":s,"pde":p,"operators":len(g),"ratio_min":min(r["E_prop_over_E_total"] for r in g),"ratio_max":max(r["E_prop_over_E_total"] for r in g),"branches":{b:sum(r["decision_branch"]==b for r in g) for b in sorted({r["decision_branch"] for r in g})}} for (s,p),g in groups.items()],
        "interpretation": "Gate values are preserved. Any four-decimal mismatches remain acceptance failures; no saved gate or recomputed result was substituted.",
        "limits": ["Validation-only, parent-clustered paired diagnostics.", "Norm ratios are descriptive and are not explained-variance or causal shares.", "Reference future targets follow the zero-future-disturbance teacher-query counterfactual contract."],
    }
    target = RESULTS / "X3_INDEPENDENT_AUDIT.json"
    target.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    receipt = {"status": out["status"], "audit_sha256": sha(target),
               "summary_sha256": out["summary_sha256"], "paired_vectors_sha256": out["paired_vectors_sha256"],
               "operator_count": len(rows), "four_decimal_gate_mismatch_count": len(failures),
               "created_utc": out["created_utc"]}
    (RUN / "evidence" / "X3_AUDIT_RECEIPT.json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
