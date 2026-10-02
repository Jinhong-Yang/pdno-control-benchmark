from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
NEW = ROOT / "results/X1/burgers_nominal/K10_H8/n384.npz"
OLD = REPO / "runs/revision_v2/results/E1_raw/locked_nominal/burgers/O-cand-state.npz"
TOL = 0.0000005


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    with np.load(NEW, allow_pickle=False) as z:
        current = {k: z[k] for k in z.files}
    with np.load(OLD, allow_pickle=False) as z:
        frozen = {k: z[k] for k in z.files}
    if len(current["parent_id"]) != 384 or len(frozen["parent_id"]) != 384:
        raise ValueError("full384 Burgers baseline required")
    frozen_idx = {str(v): i for i, v in enumerate(frozen["parent_id"].astype(str))}
    if set(current["parent_id"].astype(str)) != set(frozen_idx):
        raise ValueError("scenario-ID sets differ from the archived full-precision result")
    aligned = np.asarray([frozen_idx[str(v)] for v in current["parent_id"].astype(str)], dtype=int)
    c_new = current["episode_control_cost"].astype(np.float64)
    c_old = frozen["episode_control_cost"][aligned].astype(np.float64)
    mean_new = float(c_new.mean())
    mean_old = float(c_old.mean())
    report = {
        "status": "PASS" if abs(mean_new - 0.341698) <= TOL and abs(mean_new - mean_old) <= TOL else "STOP_SCIENTIFIC_MISMATCH",
        "required_mean": 0.341698,
        "absolute_mean_tolerance": TOL,
        "mean_new": mean_new,
        "mean_archived_full_precision": mean_old,
        "difference_new_vs_archived": mean_new - mean_old,
        "max_paired_absolute_cost_difference": float(np.max(np.abs(c_new - c_old))),
        "choice_exact_agreement": float(np.mean(current["selected_candidate_index"] == frozen["selected_candidate_index"][aligned])),
        "action_exact_agreement_fraction": float(np.mean(np.all(np.isclose(current["action_applied"], frozen["action_applied"][aligned], rtol=0, atol=1e-7), axis=-1))),
        "state_max_absolute_difference": float(np.max(np.abs(current["state_true"].astype(float) - frozen["state_true"][aligned].astype(float)))),
        "new_sha256": sha(NEW),
        "archived_sha256": sha(OLD),
        "new_scenario_ids_sha256": hashlib.sha256("\n".join(current["parent_id"].astype(str)).encode()).hexdigest(),
        "frozen_scenario_ids_sha256": hashlib.sha256("\n".join(frozen["parent_id"].astype(str)[aligned]).encode()).hexdigest(),
        "interpretation": "Acceptance uses only the preregistered six-decimal rounding tolerance for both the target mean and archived raw mean; per-scenario/action/choice differences are diagnostics and cannot be tuned post hoc.",
    }
    out = ROOT / "results/X1_BASELINE_VERIFICATION.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
