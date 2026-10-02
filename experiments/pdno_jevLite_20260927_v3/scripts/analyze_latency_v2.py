"""Summarize frozen three-session v2 latency evidence and H1 P/B4 p99 ratios."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    manifest_path = ROOT / "evidence" / "latency_summary_v2.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "HOST_READY_E2E_THREE_SESSION_COMPLETE" or manifest.get("test_opened") is not True:
        raise SystemExit("latency runs are incomplete or no completed locked test is recorded")
    raw_root = ROOT / "evidence" / "latency_raw_v2"
    arrays = {}
    for rel, expected in manifest.get("raw_files_sha256", {}).items():
        path = ROOT / rel
        if not path.is_file() or sha(path) != expected:
            raise SystemExit(f"latency raw hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as z:
            key = (str(z["pde"].item()), str(z["method"].item()),
                   int(z["seed"].item()), int(z["session"].item()))
            values = z["latency_ms"].astype(np.float64)
            if len(values) != 20000 or not np.isfinite(values).all():
                raise SystemExit(f"invalid latency samples: {path}")
            arrays[key] = values
    expected = {(pde, method, seed, session) for pde in ("burgers", "heat")
                for method in ("B0", "B1", "P", "P-no-rank", "B4", "B5", "B2", "B3")
                for seed in ((-1,) if method in {"B0", "B1"} else (11, 23, 37))
                for session in (1, 2, 3)}
    if arrays.keys() != expected:
        raise SystemExit(f"latency variant inventory mismatch: missing={expected-arrays.keys()}, extra={arrays.keys()-expected}")
    groups = []
    group_keys = sorted({(pde, method, seed) for pde, method, seed, _ in expected})
    for pde, method, seed in group_keys:
        sessions = [arrays[(pde, method, seed, s)] for s in (1, 2, 3)]
        values = np.concatenate(sessions)
        groups.append({"pde": pde, "method": method, "seed": None if seed < 0 else seed,
            "requests": int(len(values)), "p50_ms": float(np.quantile(values, .50)),
            "p95_ms": float(np.quantile(values, .95)), "p99_ms": float(np.quantile(values, .99)),
            "p999_ms": float(np.quantile(values, .999)),
            "deadline_misses": {f"{ms}ms": int(np.sum(values > ms)) for ms in (1, 2, 5, 10)},
            "deadline_miss_rate_5ms": float(np.mean(values > 5))})
    rng = np.random.default_rng(20260926)
    h1 = []
    for pde in ("burgers", "heat"):
        per_seed = {}
        for seed in (11, 23, 37):
            p = np.concatenate([arrays[(pde, "P", seed, s)] for s in (1, 2, 3)])
            b = np.concatenate([arrays[(pde, "B4", seed, s)] for s in (1, 2, 3)])
            per_seed[seed] = float(np.quantile(p, .99) / max(np.quantile(b, .99), 1e-12))
        boots = np.empty(10000, dtype=np.float64)
        seed_values = np.asarray((11, 23, 37))
        for i in range(len(boots)):
            chosen_seeds = rng.choice(seed_values, size=3, replace=True)
            ratios = []
            for seed in chosen_seeds:
                chosen_sessions = rng.integers(1, 4, size=3)
                pvals = np.concatenate([arrays[(pde, "P", int(seed), int(s))] for s in chosen_sessions])
                bvals = np.concatenate([arrays[(pde, "B4", int(seed), int(s))] for s in chosen_sessions])
                ratios.append(float(np.quantile(pvals, .99) / max(np.quantile(bvals, .99), 1e-12)))
            boots[i] = float(np.mean(ratios))
        h1.append({"pde": pde, "comparison": "P/B4 host-ready p99 ratio; <1 favors P",
            "per_seed_p99_ratio": {str(seed): value for seed, value in per_seed.items()},
            "mean_per_seed_ratio": float(np.mean(list(per_seed.values()))),
            "hierarchical_seed_session_bootstrap_ci95": [float(np.quantile(boots, .025)),
                                                            float(np.quantile(boots, .975))],
            "bootstrap_replicates": len(boots), "sessions": 3, "learned_seeds": 3,
            "interpretation": "descriptive with seed/session uncertainty; sessions are not independent environments"})
    result = {"status": "LOCKED_LATENCY_ANALYSIS_COMPLETE", "test_opened": True,
        "target_p99_ms": 2, "hard_deadline_ms": 5, "sessions": 3,
        "requests_per_variant": 60000, "boundary": manifest["boundary"],
        "controller_groups": groups, "H1_p99_ratio_analysis": h1,
        "latency_summary_sha256": sha(manifest_path)}
    output = ROOT / "evidence" / "latency_analysis_v2.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "groups": len(groups), "H1": h1,
                      "output": str(output), "sha256": sha(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
