"""Tiny CUDA smoke for train-only shared-observer input normalization."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pdno.training.confirmatory import train_observer, train_operator_staged  # noqa: E402


def main() -> int:
    pde = "heat"
    train = ROOT / "data" / "queries_v2" / "train" / pde / "teacher_queries.npz"
    validation = ROOT / "data" / "queries_v2" / "validation" / pde / "teacher_queries.npz"
    out = ROOT / "runs" / "smoke_normalizer_v2"
    observer_path = out / "observer.pt"
    observer = train_observer(train, validation, pde, observer_path, seed=991,
                              max_updates=3, eval_interval=1, batch_size=8, device_name="cuda")
    model_path = out / "P.pt"
    model = train_operator_staged(train, validation, observer_path, pde, "P", 991, model_path,
                                  field_updates=2, physics_updates=2, eval_interval=1,
                                  batch_size=2, device_name="cuda")
    observer_state = torch.load(observer_path, map_location="cpu", weights_only=False)["state_dict"]
    model_state = torch.load(model_path, map_location="cpu", weights_only=False)["state_dict"]
    keys = ("encoder.sensor_value_mean", "encoder.sensor_value_std",
            "encoder.material_context_mean", "encoder.material_context_std")
    for key in keys:
        if key not in observer_state or key not in model_state:
            raise RuntimeError(f"normalizer buffer missing from shared checkpoints: {key}")
        if not torch.equal(observer_state[key], model_state[key]):
            raise RuntimeError(f"operator did not inherit shared observer normalizer: {key}")
    result = {"status": "TRAIN_ONLY_NORMALIZER_CUDA_SMOKE_PASS", "device": "cuda",
              "fit_split": "train", "pde": pde, "observer_updates": observer["updates"],
              "operator_updates": model["actual_updates"], "shared_buffer_keys": list(keys),
              "locked_data_accessed": False}
    dest = ROOT / "evidence" / "v2_normalizer_training_smoke.json"
    dest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
