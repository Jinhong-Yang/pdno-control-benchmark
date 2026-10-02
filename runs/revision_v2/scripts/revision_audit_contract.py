"""Pure inventory/status checks for the final revision auditor."""
from itertools import product

ROLES = ("locked_nominal", "locked_coefficient_ood", "locked_delay_dropout")
STATUS = {
    "E1_heat_COMPLETE.json": "COMPLETE", "E1_burgers_cuda_COMPLETE.json": "COMPLETE",
    "E2_TIMING.json": "COMPLETE", "E3_GPU_TRAINING.json": "TRAINING_COMPLETE_EVALUATION_PENDING",
    "E4_GPU_TRAINING.json": "TRAINING_COMPLETE_EVALUATION_PENDING",
    "E5_GPU_TRAINING.json": "TRAINING_COMPLETE_EVALUATION_PENDING",
    "E3_EVALUATION.json": "COMPLETE", "E4_EVALUATION.json": "COMPLETE",
    "E5_EVALUATION.json": "COMPLETE",
}

def validate_completion(receipts):
    assert set(receipts) == set(STATUS), "completion receipt inventory"
    for name, expected in STATUS.items():
        assert receipts[name].get("status") == expected, (name, receipts[name].get("status"), expected)
    for stage, count in (("E3", 21), ("E4", 10), ("E5", 19)):
        r = receipts[f"{stage}_GPU_TRAINING.json"]
        assert r["stage"] == stage and r["device"] == "cuda" and len(r["rows"]) == count, stage
        labels = [x["condition"] for x in r["rows"]]
        assert len(set(labels)) == count, ("duplicate training condition", stage)
    for name in ("E1_heat_COMPLETE.json", "E1_burgers_cuda_COMPLETE.json"):
        assert receipts[name]["executions"] == 1280, name

def expected_outcomes(stage, e3_scores=None):
    if stage == "E1":
        conditions = list(product(("burgers", "heat"), ("O-cand-state", "O-cand-obs")))
    elif stage == "E4":
        conditions = [(pde, f"{pde}_phys{phys:g}_rank{rank:g}_s11")
                      for pde, phys, rank in product(("burgers", "heat"), (.01, .1, 1.), (.1, 1.))
                      if (phys, rank) != (.01, .1)]
    elif stage == "E5":
        conditions = [("heat", f"{m}_sna") for m in ("B0", "B1")]
        conditions += [("heat", f"{m}_s{s}") for m, s in
                       product(("P", "B4", "B5", "P-no-rank", "B2", "B3"), (11, 23, 37))]
    elif stage == "E3":
        import math
        expected = {f"burgers_x{f}_{m}_s{s}" for f, m, s in product((1, 4, 16), ("P", "B4"), (11, 23, 37))}
        assert set(e3_scores) == expected, "all eighteen operator selections required"
        assert all(math.isfinite(v) and v >= 0 for v in e3_scores.values()), "invalid selected field nRMSE"
        conditions = [("burgers", name) for name, score in e3_scores.items() if score <= .05]
    else:
        raise ValueError(stage)
    return {(role, pde, label) for role in ROLES for pde, label in conditions}

def validate_outcome_inventory(stage, actual, e3_scores=None):
    expected = expected_outcomes(stage, e3_scores)
    assert len(actual) == len(set(actual)), ("duplicate outcome", stage)
    assert set(actual) == expected, (stage, "outcome inventory mismatch",
                                     sorted(expected-set(actual)), sorted(set(actual)-expected))
    return len(expected)
