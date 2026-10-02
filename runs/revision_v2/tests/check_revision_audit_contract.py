"""CPU synthetic support cases, never a substitute for the final raw audit."""
from pathlib import Path
import copy
import hashlib
import itertools
import json
import sys
R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "scripts"))
from revision_audit_contract import STATUS, validate_completion, expected_outcomes, validate_outcome_inventory

cases = []
def passed(name):
    cases.append({"case": name, "pass": True})
def rejected(name, fn):
    try:
        fn()
    except AssertionError:
        passed(name)
    else:
        raise AssertionError("Invalid fixture accepted: " + name)

receipts = {name: {"status": status} for name, status in STATUS.items()}
for stage, count in (("E3", 21), ("E4", 10), ("E5", 19)):
    receipts[f"{stage}_GPU_TRAINING.json"].update(stage=stage, device="cuda", rows=[{"condition": str(i)} for i in range(count)])
for name in ("E1_heat_COMPLETE.json", "E1_burgers_cuda_COMPLETE.json"):
    receipts[name]["executions"] = 1280
validate_completion(receipts); passed("complete synthetic status set accepted")
bad = copy.deepcopy(receipts); bad["E5_EVALUATION.json"]["status"] = "RUNNING"
rejected("existing but incomplete receipt rejected", lambda: validate_completion(bad))
bad = copy.deepcopy(receipts); bad["E3_GPU_TRAINING.json"]["rows"][1]["condition"] = "0"
rejected("duplicate training condition rejected", lambda: validate_completion(bad))

# Construct E5 independently from the manuscript's two classical plus six
# learned families, three seeds, three populations.
e5 = []
for role in ("locked_nominal", "locked_coefficient_ood", "locked_delay_dropout"):
    e5.extend((role, "heat", m) for m in ("B0_sna", "B1_sna"))
    for m in ("P", "B4", "B5", "P-no-rank", "B2", "B3"):
        for seed in (11, 23, 37):
            e5.append((role, "heat", f"{m}_s{seed}"))
assert validate_outcome_inventory("E5", e5) == 60
passed("all sixty independent E5 fixture keys accepted")
bad_keys = e5.copy(); bad_keys[0] = ("locked_nominal", "burgers", "B0_sna")
rejected("same count with wrong PDE rejected", lambda: validate_outcome_inventory("E5", bad_keys))
bad_keys = e5.copy(); bad_keys[1] = bad_keys[0]
rejected("same count with duplicated missing condition rejected", lambda: validate_outcome_inventory("E5", bad_keys))

scores = {f"burgers_x{f}_{m}_s{s}": .06 for f, m, s in
          itertools.product((1, 4, 16), ("P", "B4"), (11, 23, 37))}
assert validate_outcome_inventory("E3", [], scores) == 0
passed("all failed field gates legitimately yield zero E3 outcomes")
scores["burgers_x4_P_s23"] = .05
accepted = [(role, "burgers", "burgers_x4_P_s23") for role in
            ("locked_nominal", "locked_coefficient_ood", "locked_delay_dropout")]
assert validate_outcome_inventory("E3", accepted, scores) == 3
passed("exact field gate boundary yields three matching role arrays")
rejected("gate passing model with missing evaluation rejected", lambda: validate_outcome_inventory("E3", [], scores))
assert len(expected_outcomes("E1")) == 12 and len(expected_outcomes("E4")) == 30
assert not any("_phys0.01_rank0.1_" in key[2] for key in expected_outcomes("E4"))
passed("E1 factorial and ten new E4 cells exclude original baseline")

output = {"status": "PASS_SYNTHETIC_CPU_SUPPORT_ONLY", "cases": cases,
          "case_count": len(cases), "gpu_operations": 0,
          "source_sha256": hashlib.sha256((R/"scripts/revision_audit_contract.py").read_bytes()).hexdigest(),
          "limits": "Synthetic keys/statuses only. Final scientific arrays, full checkpoint seals and outcome arithmetic remain subject to the complete revision audit."}
(R/"results/REVISION_AUDIT_CONTRACT_SUPPORT.json").write_text(json.dumps(output, indent=2)+"\n")
print(json.dumps({"status": output["status"], "cases": len(cases)}))
