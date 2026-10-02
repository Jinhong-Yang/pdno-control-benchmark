"""Reproduce X2a/X2b summaries from immutable saved timing arrays and profiles."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
PRIMARY = REPO / "experiments/pdno_jevLite_20260927_v3"
FOLLOWUP = REPO / "runs/revision_v2"
PROFILE_DIR = FOLLOWUP / "results/E2_profiles"
STAGE_MAP = {
    "request_preparation": "request_preparation",
    "H2D_and_grid": "transfer_grid",
    "encoder": "encoder",
    "branch": "branch",
    "trunk": "trunk",
    "field_assembly_and_other": "field_assembly",
    "scoring_including_goal_transfer": "scoring",
    "D2H_selected_index_and_forecast": "return",
    "projection": "projection",
    "verification": "verification",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def quantile_summary(values: Iterable[float]) -> dict[str, float | int | bool]:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or not x.size or not np.isfinite(x).all() or np.any(x < 0):
        raise ValueError("latency values must be a non-empty finite nonnegative vector")
    median = float(np.quantile(x, .5))
    p99 = float(np.quantile(x, .99))
    threshold = 2.0 * median
    above = x > threshold
    return {
        "n": int(x.size), "median_ms": median, "p99_ms": p99,
        "threshold_2x_median_ms": threshold,
        "above_count": int(above.sum()), "above_share": float(above.mean()),
        "p99_above_threshold": bool(p99 > threshold),
    }


def profile_rows() -> tuple[list[dict], list[dict]]:
    paths = sorted(PROFILE_DIR.glob("session1_*.json"))
    if len(paths) != 16:
        raise RuntimeError(f"expected 16 stage profiles, found {len(paths)}")
    tidy: list[dict] = []
    totals: list[dict] = []
    expected_keys = set(STAGE_MAP)
    for path in paths:
        doc = json.loads(path.read_text(encoding="utf-8"))
        records = doc.get("stage_ms")
        if len(records) != 500 or doc.get("requests") != 500:
            raise RuntimeError(f"profile request count mismatch: {path}")
        if doc.get("pde") not in {"burgers", "heat"} or doc.get("method") not in {"P", "B4"}:
            raise RuntimeError(f"unexpected profile identity: {path}")
        means = {}
        for source_key, stage_name in STAGE_MAP.items():
            values = np.asarray([row[source_key] for row in records], dtype=np.float64)
            if values.shape != (500,) or not np.isfinite(values).all() or np.any(values < 0):
                raise RuntimeError(f"invalid stage values: {path}/{source_key}")
            means[stage_name] = float(values.mean())
        if any(set(record) != expected_keys for record in records):
            raise RuntimeError(f"stage schema mismatch: {path}")
        instrumented_sum = float(sum(means.values()))
        if instrumented_sum <= 0:
            raise RuntimeError(f"nonpositive instrumented stage sum: {path}")
        row = {
            "pde": doc["pde"], "method": doc["method"], "K": int(doc["K"]),
            "cache": bool(doc["cache"]), "requests": 500,
            "mean_summed_instrumented_stages_ms": instrumented_sum,
            "branch_plus_trunk_ms": means["branch"] + means["trunk"],
            "branch_plus_trunk_fraction_of_summed_instrumented_stages":
                (means["branch"] + means["trunk"]) / instrumented_sum,
            "profile_file": str(path.relative_to(REPO)),
        }
        for stage_name, mean in means.items():
            row[f"{stage_name}_mean_ms"] = mean
            row[f"{stage_name}_share_of_summed_instrumented_stages"] = mean / instrumented_sum
        totals.append(row)
        for stage_name, mean in means.items():
            tidy.append({
                "pde": doc["pde"], "method": doc["method"], "K": int(doc["K"]),
                "cache": bool(doc["cache"]), "requests": 500,
                "stage": stage_name, "mean_ms": mean,
                "share_of_summed_instrumented_stages": mean / instrumented_sum,
                "instrumented_stage_sum_ms": instrumented_sum,
                "profile_file": str(path.relative_to(REPO)),
            })
    return tidy, totals


def _latency_row(row: dict, archive: str, raw_path: Path, recorded_sha: str,
                 identity: dict) -> dict:
    if not raw_path.is_file():
        raise RuntimeError(f"missing saved array: {raw_path}")
    actual_sha = sha256(raw_path)
    if actual_sha != recorded_sha:
        raise RuntimeError(f"raw SHA-256 mismatch: {raw_path}")
    with np.load(raw_path, allow_pickle=False) as z:
        required = {"latency_ms"}
        if not required.issubset(z.files):
            raise RuntimeError(f"missing latency_ms array: {raw_path}")
        values = np.array(z["latency_ms"], dtype=np.float64)
        for field in ("pde", "method", "seed", "session"):
            if field in z.files and field in identity:
                actual = z[field].item()
                expected = identity[field]
                if field == "seed" and expected is None:
                    expected = -1
                if str(actual) != str(expected):
                    raise RuntimeError(f"array identity mismatch {field}: {raw_path}")
    summary = quantile_summary(values)
    return {
        "archive": archive, **identity, **summary,
        "raw_file": str(raw_path.relative_to(REPO)), "raw_sha256": actual_sha,
    }


def latency_rows() -> tuple[list[dict], list[dict]]:
    primary_manifest = json.loads((PRIMARY / "evidence/latency_summary_v3.json").read_text(encoding="utf-8"))
    if primary_manifest.get("status") != "HOST_READY_E2E_THREE_SESSION_COMPLETE":
        raise RuntimeError("primary timing archive is not marked complete")
    primary_sha_map = primary_manifest.get("raw_files_sha256", {})
    primary_summary = json.loads((PRIMARY / "evidence/latency_analysis_v3.json").read_text(encoding="utf-8"))
    primary_rows: list[dict] = []
    for rel, expected_sha in sorted(primary_sha_map.items()):
        raw_path = PRIMARY / rel
        with np.load(raw_path, allow_pickle=False) as z:
            identity = {k: z[k].item() for k in ("pde", "method", "seed", "session")}
            identity["seed"] = None if int(identity["seed"]) < 0 else int(identity["seed"])
            identity.update({"K": 10, "cache": False, "requests": len(z["latency_ms"]),
                             "warmup": 50, "boundary": "host-ready policy request to projected action"})
        primary_rows.append(_latency_row(identity.copy(), "primary", raw_path, expected_sha, identity))
    if len(primary_rows) != 120:
        raise RuntimeError(f"expected 120 primary raw rows, found {len(primary_rows)}")

    followup_doc = json.loads((FOLLOWUP / "results/E2_TIMING.json").read_text(encoding="utf-8"))
    if followup_doc.get("status") != "COMPLETE":
        raise RuntimeError("follow-up timing archive is not marked complete")
    followup_rows: list[dict] = []
    for row in followup_doc["rows"]:
        raw_path = FOLLOWUP / "results/E2_raw" / (
            f"session{row['session']}_{row['pde']}_{row['method']}_s{row['seed']}"
            f"_K{row['K']}_cache{int(row['cache'])}.npz")
        identity = {k: row[k] for k in ("session", "pde", "method", "seed", "K", "cache")}
        identity.update({"requests": row["requests"], "warmup": row["warmup"],
                         "boundary": followup_doc.get("boundary", "unspecified")})
        actual = _latency_row(identity.copy(), "followup", raw_path, row["sha256"], identity)
        if actual["n"] != row["requests"]:
            raise RuntimeError(f"sample count mismatch: {raw_path}")
        followup_rows.append(actual)
    if len(followup_rows) != 360:
        raise RuntimeError(f"expected 360 follow-up raw rows, found {len(followup_rows)}")
    return primary_rows, followup_rows


def matched_comparison(primary: list[dict], followup: list[dict]) -> list[dict]:
    # Use the session-level seed rows with K=10/cache-off, retaining all 18 pairs.
    pmap = {(r["pde"], r["method"], r["seed"], r["session"]): r
            for r in primary if r["method"] in {"P", "B4"}}
    fmap = {(r["pde"], r["method"], r["seed"], r["session"]): r
            for r in followup if r["method"] in {"P", "B4"} and r["K"] == 10 and not r["cache"]}
    result = []
    for pde in ("burgers", "heat"):
        for seed in (11, 23, 37):
            for session in (1, 2, 3):
                for method in ("P", "B4"):
                    key = (pde, method, seed, session)
                    if key not in pmap or key not in fmap:
                        raise RuntimeError(f"unmatched primary/follow-up row: {key}")
                    p, f = pmap[key], fmap[key]
                    result.append({
                        "pde": pde, "seed": seed, "session": session, "method": method,
                        "primary_p99_ms": p["p99_ms"], "followup_p99_ms": f["p99_ms"],
                        "primary_above_share": p["above_share"],
                        "followup_above_share": f["above_share"],
                        "primary_p99_above_threshold": p["p99_above_threshold"],
                        "followup_p99_above_threshold": f["p99_above_threshold"],
                    })
    # Pair P/B4 within each archive, PDE, seed and session. The comparison is descriptive.
    paired = []
    for archive in ("primary", "followup"):
        for pde in ("burgers", "heat"):
            for seed in (11, 23, 37):
                for session in (1, 2, 3):
                    ix = {(r["method"]): r for r in result if r["pde"] == pde and r["seed"] == seed
                          and r["session"] == session}
                    if archive == "primary":
                        p99p, p99b = ix["P"]["primary_p99_ms"], ix["B4"]["primary_p99_ms"]
                        flagp, flagb = ix["P"]["primary_p99_above_threshold"], ix["B4"]["primary_p99_above_threshold"]
                        sp, sb = ix["P"]["primary_above_share"], ix["B4"]["primary_above_share"]
                    else:
                        p99p, p99b = ix["P"]["followup_p99_ms"], ix["B4"]["followup_p99_ms"]
                        flagp, flagb = ix["P"]["followup_p99_above_threshold"], ix["B4"]["followup_p99_above_threshold"]
                        sp, sb = ix["P"]["followup_above_share"], ix["B4"]["followup_above_share"]
                    paired.append({"archive": archive, "pde": pde, "seed": seed, "session": session,
                                   "P_p99_ms": p99p, "B4_p99_ms": p99b, "P_over_B4_p99_ratio": p99p / p99b,
                                   "P_above_share": sp, "B4_above_share": sb,
                                   "P_p99_above_threshold": flagp, "B4_p99_above_threshold": flagb})
    return paired


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"cannot write empty CSV: {path}")
    fields = list(dict.fromkeys(k for row in rows for k in row.keys()))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    out = ROOT / "results/X2"
    evidence = ROOT / "evidence/X2"
    out.mkdir(parents=True, exist_ok=True)
    evidence.mkdir(parents=True, exist_ok=True)
    tidy, profile_totals = profile_rows()
    primary, followup = latency_rows()
    pairs = matched_comparison(primary, followup)
    write_csv(out / "X2a_stage_shares.csv", tidy)
    write_csv(out / "X2a_profile_totals.csv", profile_totals)
    all_rows = primary + followup
    write_csv(out / "X2b_row_statistics.csv", all_rows)
    write_csv(out / "X2b_matched_p99_ratios.csv", pairs)
    (out / "X2_summary.json").write_text(json.dumps({
        "status": "ANALYSIS_COMPLETE", "analysis_plan": "config/X2_ANALYSIS_PLAN.md",
        "x2a_profiles": len(profile_totals), "x2a_requests": sum(r["requests"] for r in profile_totals),
        "x2a_note": "means and shares use summed separately synchronized instrumented stages; no separately recorded profiled-request total",
        "primary_rows": len(primary), "followup_rows": len(followup), "total_rows": len(all_rows),
        "matched_method_rows": 72, "matched_P_B4_ratio_rows": len(pairs),
        "threshold_definition": "latency_ms > 2 * row median (strict); secondary-mode proxy only",
        "threshold_statement_assessment": assess_modes(all_rows, pairs),
        "burgers_matched_seed_p99": matched_seed_p99(primary, followup),
        "inputs": input_hashes(),
        "analysis_script_sha256": sha256(Path(__file__)),
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ANALYSIS_COMPLETE", "output": str(out),
                      "profiles": len(profile_totals), "primary": len(primary),
                      "followup": len(followup), "matched": len(pairs)}, indent=2))
    return 0


def assess_modes(rows: list[dict], pairs: list[dict]) -> dict:
    flags = [r["p99_above_threshold"] for r in rows]
    shares = np.asarray([r["above_share"] for r in rows], dtype=float)
    near_one_percent = bool(np.mean((shares >= .005) & (shares <= .02)) >= .5)
    variable_flags = bool(any(flags) and not all(flags))
    ratio_by_archive_flag = {}
    for archive in ("primary", "followup"):
        for flag in (False, True):
            selected = [r["P_over_B4_p99_ratio"] for r in pairs
                        if r["archive"] == archive and
                        (r["P_p99_above_threshold"] or r["B4_p99_above_threshold"]) == flag]
            ratio_by_archive_flag[f"{archive}_any_p99_above_{str(flag).lower()}"] = {
                "n": len(selected), "mean_P_over_B4_p99_ratio": float(np.mean(selected)) if selected else None}
    ratio_varies = any(v["n"] for v in ratio_by_archive_flag.values())
    support = near_one_percent and variable_flags and ratio_varies
    return {"rows_with_share_0.5_to_2_percent": int(np.sum((shares >= .005) & (shares <= .02))),
            "row_count": len(rows), "median_above_share": float(np.median(shares)),
            "rows_p99_above_threshold": int(np.sum(flags)), "rows_p99_at_or_below_threshold": int(len(flags)-np.sum(flags)),
            "shares_near_1_percent_in_majority": near_one_percent,
            "p99_flag_varies": variable_flags, "matched_ratio_by_flag": ratio_by_archive_flag,
            "work_order_expectation_supported": bool(support),
            "interpretation": "descriptive proxy association only; does not establish physical modes or causal mechanism"}


def matched_seed_p99(primary: list[dict], followup: list[dict]) -> dict:
    """Aggregate each seed across its three sessions, then average the three seed ratios."""
    result = {}
    for archive, rows in (("primary", primary), ("followup", followup)):
        ratios = {}
        p99_by_method = {"P": {}, "B4": {}}
        for seed in (11, 23, 37):
            for method in ("P", "B4"):
                selected = [r for r in rows if r["pde"] == "burgers" and r["method"] == method
                            and r["seed"] == seed and
                            (archive == "primary" or (r["K"] == 10 and not r["cache"]))]
                if len(selected) != 3:
                    raise RuntimeError(f"expected 3 matched session rows for {archive}/{method}/{seed}, found {len(selected)}")
                values = []
                for row in selected:
                    with np.load(REPO / row["raw_file"], allow_pickle=False) as z:
                        values.append(np.asarray(z["latency_ms"], dtype=np.float64))
                p99_by_method[method][str(seed)] = float(np.quantile(np.concatenate(values), .99))
            ratios[str(seed)] = p99_by_method["P"][str(seed)] / p99_by_method["B4"][str(seed)]
        result[archive] = {
            "P_p99_ms_by_seed": p99_by_method["P"], "B4_p99_ms_by_seed": p99_by_method["B4"],
            "P_over_B4_p99_ratio_by_seed": ratios,
            "mean_of_three_seed_ratios": float(np.mean(list(ratios.values()))),
            "aggregation": "pool three sessions within each seed, compute each seed P/B4 p99 ratio, then arithmetic mean across seeds",
        }
    return result


def input_hashes() -> dict:
    paths = [ROOT / "config/X2_ANALYSIS_PLAN.md",
             FOLLOWUP / "results/E2_TIMING.json",
             PRIMARY / "evidence/latency_summary_v3.json",
             PRIMARY / "evidence/confirmatory_checkpoint_manifest_v3.json",
             PRIMARY / "scripts/benchmark_host_e2e_v3.py",
             PRIMARY / "src/pdno/evaluation/closed_loop.py",
             PRIMARY / "src/pdno/controllers/linear.py",
             PRIMARY / "src/pdno/data/teacher_queries.py",
             FOLLOWUP / "scripts/run_e2_timing.py"]
    paths += [FOLLOWUP / "src/pdno/evaluation/closed_loop.py",
              FOLLOWUP / "src/pdno/controllers/linear.py",
              FOLLOWUP / "src/pdno/data/teacher_queries.py"]
    paths += sorted(PROFILE_DIR.glob("session1_*.json"))
    return {str(p.relative_to(REPO)): sha256(p) for p in paths}


if __name__ == "__main__":
    raise SystemExit(main())
