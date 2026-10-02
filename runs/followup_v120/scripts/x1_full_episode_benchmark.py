from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
OLD = REPO / "experiments/pdno_jevLite_20260927_v3"
sys.path.insert(0, str(OLD / "src"))
sys.path.insert(0, str(ROOT.parent / "revision_v2" / "src"))
sys.path.insert(0, str(ROOT / "src"))

from x1_run import CELL_MAP, evaluate_chunk


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    runs = []
    for pde in ("burgers", "heat"):
        path = OLD / f"data/train_validation_v3/validation/{pde}/trajectories.npz"
        with np.load(path, allow_pickle=False) as z:
            data = {k: z[k] for k in z.files}
        order = np.argsort(data["parent_id"].astype(str), kind="stable")
        data = {k: v[order][:4] for k, v in data.items()}
        for name in ("K50_H16", "K10_H8_FB"):
            started = time.perf_counter()
            result = evaluate_chunk(data, np.arange(4), pde, CELL_MAP[name])
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            assert result["selected_candidate_index"].shape == (4, 200)
            assert np.isfinite(result["episode_control_cost"]).all()
            runs.append({"pde": pde, "cell": name, "n_scenarios": 4, "ticks_per_scenario": 200,
                         "seconds": elapsed, "seconds_per_scenario": elapsed / 4,
                         "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                         "finite_costs": True})
    rec = {"status": "TRAIN_VALIDATION_FULL_EPISODE_BENCHMARK_ONLY",
           "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "input_data_sha256": {str(p): sha(p) for p in sorted((OLD / "data/train_validation_v3/validation").glob("*/trajectories.npz"))},
           "runs": runs,
           "estimate_method": "For the 4 worst-cell (K50/H16) and feedback episode timing, project candidate forecast cells from the 64-parent synchronized microbenchmark and add measured per-episode orchestration/plant time. Root-approved locked scope is 128 per cell on each PDE plus a separate 384-scenario Burgers baseline.",
           "scope": "Synthetic validation scenarios only; no locked or test arrays opened."}
    out = ROOT / "evidence/X1_FULL_EPISODE_BENCHMARK.json"
    out.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
