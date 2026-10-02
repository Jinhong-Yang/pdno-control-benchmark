"""Check completed non-E2 supplement tables against portable source summaries.

Independent table parsing and keyed source lookup validate transcription,
aggregation and rounding. They do not validate the original raw experiments,
confidence-interval methodology, causal interpretation or PDF layout.
"""
from pathlib import Path
import csv
import hashlib
import json
import re

O = Path(__file__).resolve().parents[1]
paths = set()
checks = []
table_count = 0
row_count = 0
ROLES = {"Nominal": "locked_nominal", "Coefficient shift": "locked_coefficient_ood",
         "Delay/dropout": "locked_delay_dropout"}
METHODS = {"B0", "B1", "B2", "B3", "B4", "B5", "P", "P-no-rank"}

def load(name):
    p = O / "data" / name
    paths.add(p)
    with p.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def parse(name):
    global table_count, row_count
    p = O / "tables" / name
    paths.add(p)
    result = []
    for block in p.read_text(encoding="utf-8").split(r"\begin{table}")[1:]:
        content = block.split(r"\midrule", 1)[1].split(r"\bottomrule", 1)[0]
        rows = [[v.strip() for v in row.strip().split("&")]
                for row in content.split(r"\\") if row.strip()]
        label = block.split(r"\label{", 1)[1].split("}", 1)[0] if r"\label{" in block else None
        result.append((block, label, rows))
        table_count += 1
        row_count += len(rows)
    return result

def single(rows, **key):
    found = [r for r in rows if all(r[k] == v for k, v in key.items())]
    assert len(found) == 1, (key, len(found))
    return found[0]

def eq(actual, expected, location):
    assert actual == expected, (location, actual, expected)
    checks.append(dict(location=location, displayed=actual, expected=expected))

def num(actual, source, precision, location, scale=1, signed=False):
    expected = format(scale * float(source), ("+" if signed else "") + f".{precision}f")
    eq(actual, expected, location)

def bounds(actual, low, high, precision, location, scale=1, signed=False):
    m = re.fullmatch(r"\[([^,]+),\s*([^\]]+)\]", actual)
    assert m, (location, actual)
    for v, source, suffix in zip(m.groups(), [low, high], ["low", "high"]):
        num(v, source, precision, location+"/"+suffix, scale, signed)

def method(label):
    return "P-no-rank" if label == "P-nr" else label

def distinct(rows, key):
    values = [key(r) for r in rows]
    assert len(values) == len(set(values)), values

def main():
    global table_count, row_count
    control = load("control_all_methods.csv")
    seen = set()
    for block, _, rows in parse("supp_control.tex"):
        pde, condition = re.search(r"\\caption\{(Burgers|Heat): ([^.]+)\.", block).groups()
        role = ROLES[condition]
        pde = pde.lower()
        assert len(rows) == 8 and {method(r[0]) for r in rows} == METHODS
        for row in rows:
            assert len(row) == 6
            key = (pde, role, method(row[0])); assert key not in seen; seen.add(key)
            s = single(control, pde=pde, role=role, method=key[2])
            loc = str(key)
            for cell, field in zip(row[1:3], ["mean_cost", "tracking_rmse"]):
                num(cell, s[field], 6, loc+"/"+field)
            num(row[3], s["relative_cost_difference_vs_B0"], 3, loc+"/excess", 100, True)
            bounds(row[4], s["ci_low"], s["ci_high"], 3, loc+"/ci", 100, True)
            num(row[5], s["violation_rate"], 3, loc+"/violation", 100)
    assert len(seen) == len(control) == 48

    latency = load("latency_per_seed.csv"); seen = set()
    for block, _, rows in parse("supp_latency.tex"):
        pde = re.search(r"\\caption\{(Burgers|Heat):", block)[1].lower()
        assert len(rows) == 20
        for row in rows:
            assert len(row) == 8
            key = (pde, method(row[0]), "" if row[1] == "--" else row[1])
            assert key not in seen; seen.add(key)
            s = single(latency, pde=key[0], method=key[1], seed=key[2])
            assert int(s["requests"]) == 60000
            for cell, field in zip(row[2:6], ["p50_ms", "p95_ms", "p99_ms", "p999_ms"]):
                num(cell, s[field], 4, str(key)+"/"+field)
            eq(row[6], str(int(s["misses_5ms"])), str(key)+"/miss_count")
            num(row[7], s["deadline_miss_rate_5ms"], 4, str(key)+"/miss_percent", 100)
    assert len(seen) == len(latency) == 40

    calibration = load("calibration_margins.csv")
    tabs = parse("supp_calibration.tex"); assert len(tabs) == 1
    rows = tabs[0][2]; assert len(rows) == len(calibration) == 26
    distinct(rows, lambda r: tuple(r[:3]))
    for row in rows:
        assert len(row) == 4
        s = single(calibration, pde=row[0].lower(), method=method(row[1]),
                   seed="" if row[2] == "--" else row[2])
        assert (int(s["parent_count"]), int(s["query_count"]),
                int(s["finite_sample_order_index_zero_based"])) == (64, 512, 61)
        num(row[3], s["margin"], 6, str(row[:3])+"/margin")

    p0path = O / "data/revision/P0_1_gradient_audit.json"; paths.add(p0path)
    p0 = json.loads(p0path.read_text())["reports"]
    tabs = parse("revision_gradients.tex"); assert len(tabs) == 1
    rows = tabs[0][2]; assert len(rows) == 2
    distinct(rows, lambda r: r[0])
    for row in rows:
        assert len(row) == 5
        s = single(p0, pde=row[0].lower())
        for cell, field in zip(row[1:4], ["median_weighted_physics_data_gradient_ratio",
                                        "median_balance_data_gradient_ratio", "final_parameter_difference"]):
            mantissa, exponent = format(float(s[field]), ".3e").split("e")
            eq(cell, "$"+mantissa+r"\times10^{"+str(int(exponent))+"}$", row[0]+"/"+field)
        eq(row[4], str(s["physics_gradient_nonzero_updates"])+"/"+str(s["updates"]), row[0]+"/nonzero")

    joint = load("revision/E7_joint_denominator_bootstrap.csv"); seen = set()
    for _, label, rows in parse("revision_bootstrap.tex"):
        pde, role = label.removeprefix("tab:joint_").split("_", 1)
        assert pde in ["burgers", "heat"] and role in ROLES.values()
        assert len(rows) == 8 and {method(r[0]) for r in rows} == METHODS
        for row in rows:
            assert len(row) == 4
            key = (pde, role, method(row[0])); assert key not in seen; seen.add(key)
            s = single(joint, pde=pde, role=role, method=key[2])
            old = single(control, pde=pde, role=role, method=key[2])
            num(row[1], s["estimate_percent"], 2, str(key)+"/estimate")
            bounds(row[2], old["ci_low"], old["ci_high"], 2, str(key)+"/archived", 100)
            bounds(row[3], s["joint_low"], s["joint_high"], 2, str(key)+"/joint")
    assert len(seen) == len(joint) == 48

    oracle = [r for r in load("revision/REVISION_COST_SUMMARY.csv") if r["stage"] == "E1"]
    tabs = parse("revision_oracle.tex"); assert len(tabs) == 1
    rows = tabs[0][2]; assert len(rows) == len(oracle) == 12
    distinct(rows, lambda r: tuple(r[:3]))
    for row in rows:
        assert len(row) == 6
        key = dict(stage="E1", pde=row[0].lower(), role=ROLES[row[1]],
                   method={"Observed": "O-cand-obs", "State": "O-cand-state"}[row[2]])
        s = single(oracle, **key)
        num(row[3], s["mean_cost"], 6, str(key)+"/cost")
        num(row[4], s["relative_excess"], 2, str(key)+"/excess", 100)
        bounds(row[5], s["fixed_ci_low"], s["fixed_ci_high"], 2, str(key)+"/ci", 100)

    costs = load("revision/E6_original_heat_cost_decomposition.csv"); seen = set()
    for _, label, rows in parse("revision_heat_cost.tex"):
        role = label.removeprefix("tab:costparts_")
        assert role in ROLES.values() and len(rows) == 8
        assert {method(r[0]) for r in rows} == METHODS
        for row in rows:
            assert len(row) == 7
            m = method(row[0]); key = (role, m); assert key not in seen; seen.add(key)
            group = [r for r in costs if r["role"] == role and r["method"] == m]
            assert {r["seed"] for r in group} == ({""} if m in {"B0", "B1"} else {"11", "23", "37"})
            assert len(group) == (1 if m in {"B0", "B1"} else 3)
            for cell, field in zip(row[1:], ["cost", "tracking", "action", "slew", "violation", "first_step_violation_cost"]):
                mean = sum(float(r[field]) for r in group)/len(group)
                num(cell, mean, 6, str(key)+"/"+field)
    assert len(seen) == 24

    freq = load("revision/E1_candidate_frequency.csv"); seen = set()
    for _, label, rows in parse("revision_choices.tex"):
        pde = label.removeprefix("tab:choices_")
        assert pde in ["burgers", "heat"] and len(rows) == 18
        for row in rows:
            assert len(row) == 4
            role, m = ROLES[row[0]], method(row[1])
            assert m in {"P", "P-no-rank", "B4", "B5", "O-cand-state", "O-cand-obs"}
            key = (pde, role, m); assert key not in seen; seen.add(key)
            group = [r for r in freq if r["source"] != "E5_raw_recorded_indices" and
                     r["pde"] == pde and r["role"] == role and
                     (r["method_seed"] == m or r["method_seed"].rsplit("_s", 1)[0] == m)]
            for cell, candidate in zip(row[2:], [9, 4]):
                selected = [r for r in group if int(r["candidate"]) == candidate]
                assert len(selected) == (1 if m.startswith("O-cand") else 3)
                distinct(selected, lambda r: r["method_seed"])
                count = sum(int(r["requests"]) for r in selected)
                low, high = [sum(float(r[f])*int(r["requests"]) for r in selected)/count
                             for f in ["frequency_lower", "frequency_upper"]]
                assert 0 <= low <= high <= 1
                loc = str(key)+"/candidate"+str(candidate)
                if low == high:
                    num(cell, low, 2, loc, 100)
                else:
                    bounds(cell, low, high, 2, loc, 100)
    assert len(seen) == 36 and table_count == 22 and row_count == 236

    # Main Table 2 combines the original population with post-hoc oracle rows.
    # Parse its two-PDE cells independently of the table generator.
    nominal_path = O / "tables/nominal.tex"; paths.add(nominal_path)
    nominal_text = nominal_path.read_text(encoding="utf-8")
    nominal_body = nominal_text.split(r"\midrule", 1)[1].split(r"\bottomrule", 1)[0]
    nominal_rows = [[v.strip() for v in row.strip().split("&")]
                    for row in nominal_body.split(r"\\") if row.strip()]
    expected_labels = {"B0", "B1", "B2", "B3", "B4", "B5", "P-nr", "P",
                       "Oracle: State", "Oracle: Observed"}
    assert len(nominal_rows) == 10 and {r[0] for r in nominal_rows} == expected_labels
    for row in nominal_rows:
        assert len(row) == 7
        for pde, cells in zip(["burgers", "heat"], [row[1:4], row[4:7]]):
            if row[0].startswith("Oracle:"):
                m = {"Oracle: State": "O-cand-state", "Oracle: Observed": "O-cand-obs"}[row[0]]
                source = single(oracle, pde=pde, role="locked_nominal", method=m)
                fields = ["mean_cost", "tracking_rmse_mean", "relative_excess"]
            else:
                source = single(control, pde=pde, role="locked_nominal", method=method(row[0]))
                fields = ["mean_cost", "tracking_rmse", "relative_cost_difference_vs_B0"]
            for i, (cell, field) in enumerate(zip(cells, fields)):
                num(cell, source[field], 2 if i == 2 else 6,
                    "main_nominal/"+pde+"/"+row[0]+"/"+field,
                    scale=100 if i == 2 else 1, signed=i == 2)
    table_count += 1; row_count += len(nominal_rows)
    result = dict(status="PASS_COMPLETED_NON_E2_TABLE_TRANSCRIPTION", tables=table_count,
                  table_rows=row_count, displayed_numeric_values_checked=len(checks),
                  sha256={str(p.relative_to(O)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)},
                  scope="22 completed supplement tables and main nominal table: keyed identities, counts, seed averaging, frequency weighting, rounding, and displayed bounds. Excludes other main/revision tables, raw recomputation, methodology, claims and PDF layout.",
                  checks=checks)
    (O/"data/revision/COMPLETED_TABLE_NUMERIC_AUDIT.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k in
                     ["status", "tables", "table_rows", "displayed_numeric_values_checked"]}))

if __name__ == "__main__":
    main()
