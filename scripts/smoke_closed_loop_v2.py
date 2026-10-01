"""Short validation-only controller smoke for v2; never accesses locked inputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pdno.evaluation.closed_loop import load_controller, run_episode  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pde = "burgers"
    data_path = ROOT / "data" / "validation" / pde / "trajectories.npz"
    run_path = ROOT / "runs" / "smoke_confirmatory_cuda"
    with np.load(data_path, allow_pickle=False) as archive:
        data = {key: archive[key][:1] for key in archive.files}
    data["role"] = np.asarray(["validation"])
    data["disturbance_seed"] = np.asarray([300001], dtype=np.int64)
    methods = ("B0", "B1", "P", "P-no-rank", "B4", "B5", "B2", "B3")
    rows = []
    for method in methods:
        checkpoint = None if method in {"B0", "B1"} else run_path / f"{method}.pt"
        model = load_controller(method, pde, 11, checkpoint, device)
        result = run_episode(data, 0, pde, method, model, device, 1.2, query_tick=100, ticks=2)
        actions = result["action_applied"]
        proposed, projected = result["action_proposed"], result["action_projected"]
        if actions.shape != (2, 2) or not np.isfinite(actions).all():
            raise RuntimeError(f"invalid applied action for {method}")
        if not np.allclose(projected, actions, rtol=0.0, atol=1e-7):
            raise RuntimeError(f"projected/applied action accounting mismatch for {method}")
        if not np.isfinite(proposed).all() or result["fallback_count"]:
            raise RuntimeError(f"non-finite proposal or fallback for {method}: {result['fallback_errors']}")
        if np.any(np.abs(actions) > 1.0 + 1e-7) or np.any(np.abs(np.diff(np.vstack([np.zeros((1, 2)), actions]), axis=0)) > 0.150001):
            raise RuntimeError(f"box/slew contract violation for {method}")
        rows.append({"method": method, "device": str(device), "finite": True,
                     "action_projection_accounting": "pass", "fallback_count": result["fallback_count"],
                     "episode_control_cost_finite": bool(np.isfinite(result["episode_control_cost"]))})
    out = {"status": "VALIDATION_ONLY_SMOKE_PASS", "pde": pde, "ticks": 2,
           "parent_id": str(data["parent_id"][0]), "data_path": str(data_path),
           "data_sha256": sha(data_path), "controller_checkpoint_dir": str(run_path),
           "methods": rows, "locked_data_accessed": False, "experiment_or_effect_claim": False}
    out_path = ROOT / "evidence" / "v2_closed_loop_smoke.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
