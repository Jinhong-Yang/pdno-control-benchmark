from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
REV = ROOT.parent / "revision_v2"
OLD = ROOT.parents[1] / "experiments" / "pdno_jevLite_20260927_v3"
sys.path.insert(0, str(REV / "src"))
sys.path.insert(0, str(ROOT / "src"))

from x1.design import candidates
from x1.solvers import BurgersHorizonCUDA, heat_forecast
from pdno.data.teacher_queries import restrict_field, _future_burgers
from pdno.controllers.revision_oracle_cuda import BurgersCandidateCUDA


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; benchmark did not run")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    records = []
    sources = {}
    for pde in ("burgers", "heat"):
        for split in ("train", "validation"):
            path = OLD / f"data/train_validation_v3/{split}/{pde}/trajectories.npz"
            sources[str(path)] = sha(path)
    for pde in ("burgers", "heat"):
        src = OLD / f"data/train_validation_v3/validation/{pde}/trajectories.npz"
        with np.load(src, allow_pickle=False) as z:
            count = min(64, len(z["parent_id"]))
            states = z["state_true"][:count, 16].astype(np.float32)
            previous = np.zeros((count, 2), dtype=np.float32) if pde == "burgers" else np.full((count, 2), 0.4, dtype=np.float32)
            b0_actions = previous.copy()
            material = (np.array([float(z["nu"][0]), -1, 1, .15, .02, .16, 1.2, 128], np.float32)
                        if pde == "burgers" else
                        np.array([float(z["kappa"][0]), float(z["decay"][0]), 0, 1, .1, .02, .16, 128], np.float32))
            goal_full = np.zeros(256, np.float32) if pde == "burgers" else z["goal"][0].astype(np.float32)
            nu_vec = z["nu"][:count].astype(np.float64) if pde == "burgers" else None
            nu = float(nu_vec[0]) if pde == "burgers" else None
            kappa, decay = (float(z["kappa"][0]), float(z["decay"][0])) if pde == "heat" else (None, None)
            qmax = 1.2 if pde == "burgers" else 1.25 * float(np.max(z["goal"]))
        goal_score = np.zeros(128, np.float32) if pde == "burgers" else restrict_field(goal_full, "heat")
        for K, H in ((10, 1), (10, 4), (10, 8), (10, 16), (50, 1), (50, 4), (50, 8), (50, 16)):
            acts = np.stack([candidates(pde, previous[j], b0_actions[j], K) for j in range(count)])
            if pde == "burgers":
                solver = BurgersHorizonCUDA(nu_vec, K)
                # Warm captures and allocations on validation data, then time five synchronized replays.
                solver.forecast(states, acts, H)
                torch.cuda.synchronize()
                samples = []
                for _ in range(5):
                    start = time.perf_counter()
                    out = solver.forecast(states, acts, H)
                    torch.cuda.synchronize()
                    samples.append(time.perf_counter() - start)
                elapsed = float(np.median(samples))
                if K == 10 and H == 8:
                    frozen = BurgersCandidateCUDA(nu_vec, K=10)
                    ref = frozen(states, acts)
                    np.testing.assert_allclose(out, ref[:, :, :H].cpu().numpy(), rtol=0, atol=2e-6)
                    # Also check one validation scenario against the frozen CPU solver.
                    cpu_ref = _future_burgers(states[0], nu, acts[0, 0])
                    np.testing.assert_allclose(out[0, 0], cpu_ref, rtol=0, atol=2e-6)
            else:
                start = time.perf_counter()
                out = np.stack([heat_forecast(np.broadcast_to(states[j], (K, 256)), acts[j], kappa, decay, H)
                                for j in range(count)])
                elapsed = (time.perf_counter() - start) / count
            records.append({"pde": pde, "K": K, "H": H, "batch_scenarios": count,
                            "median_elapsed_s_per_batch": elapsed,
                            "elapsed_s_per_scenario_equivalent": elapsed / count,
                            "forecast_shape": list(out.shape), "finite": bool(np.isfinite(out).all())})
            if not np.isfinite(out).all():
                raise AssertionError("forecast emitted nonfinite values")
    payload = {
        "status": "TRAIN_VALIDATION_MICROBENCHMARK_ONLY",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "source_hashes": sources,
        "records": records,
        "scale_note": "Burgers timings are five synchronized forecasts on a batch of up to 64 validation scenarios (median); they exclude 200-tick observation/reconstruction/plant work, while heat is CPU batched-by-scenario timing. K10/H8 Burgers output is compared against both frozen CUDA and CPU reference solvers. Combine with archived per-scenario overhead; these data do not include locked scenarios.",
    }
    out = ROOT / "evidence" / "X1_TRAIN_VALIDATION_BENCHMARK.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
