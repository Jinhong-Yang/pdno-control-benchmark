"""Validation-only field and finite-candidate regret evaluation for a saved pilot model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pdno.controllers.objectives import candidate_cost  # noqa: E402
from pdno.models.operators import ActionFactorizedOperator, CandidateConditionedOperator  # noqa: E402
from pdno.models.policies import DirectPolicy  # noqa: E402
from pdno.models.encoders import load_state_with_normalizer_defaults  # noqa: E402
from pdno.training.fit import _grid, _load, _obs  # noqa: E402


def evaluate(checkpoint: Path, query: Path, device_name: str = "cuda", batch_size: int = 8) -> dict:
    split = next((part for part in query.parts if part in {"train", "validation"}), None)
    if split is None:
        raise ValueError("pilot evaluation accepts train or validation queries only; calibration/locked roles remain sealed")
    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    meta = saved["metadata"]
    method, pde = meta["method"], meta["pde"]
    if method in {"P", "B5", "P-no-rank"}:
        model = ActionFactorizedOperator(pde, 33 if pde == "burgers" else 32, rank=meta.get("rank", 32))
    elif method == "B4":
        model = CandidateConditionedOperator(pde, 33 if pde == "burgers" else 32, rank=meta.get("rank", 32))
    elif method in {"B2", "B3"}:
        model = DirectPolicy(33 if pde == "burgers" else 32, pde)
    else:
        raise ValueError(f"unsupported learned method {method}")
    load_state_with_normalizer_defaults(model, saved["state_dict"])
    model.to(device).eval()
    data = _load(query, device)
    x, tau = _grid(pde, device)
    all_field_sq, all_target_sq, all_regret, all_rank_match, all_action_mse = [], [], [], [], []
    train_path = Path(meta.get("train_query_path", meta.get("query_path",
        ROOT / "data" / "queries_v2" / "train" / pde / "teacher_queries.npz")))
    train = _load(train_path, device)
    cost_iqr = torch.quantile(train["teacher_cost"].float().flatten(), 0.75) - torch.quantile(train["teacher_cost"].float().flatten(), 0.25)
    for start in range(0, len(data["candidate_action"]), batch_size):
        ids = torch.arange(start, min(start + batch_size, len(data["candidate_action"])), device=device)
        obs = _obs(data, ids)
        actions = data["candidate_action"][ids].float()
        with torch.no_grad():
            if method in {"P", "B4", "B5", "P-no-rank"}:
                predicted = model(obs, actions, x, tau)
                target = data["future_field"][ids].float()
                all_field_sq.append((predicted - target).square().sum().cpu())
                all_target_sq.append(target.square().sum().cpu())
                predicted_cost = candidate_cost(predicted, data["goal"][ids].float(), actions,
                                               obs["previous_applied_action"].float(),
                                               float(data["q_max"][ids].max()),
                                               q_min=0.0 if pde == "heat" else None)
                pred_choice = predicted_cost.argmin(dim=1)
                teacher = data["teacher_cost"][ids].float()
                chosen = teacher.gather(1, pred_choice[:, None]).squeeze(1)
                regret = (chosen - teacher.min(dim=1).values) / cost_iqr.clamp_min(1e-4)
                all_regret.extend(regret.cpu().tolist())
                all_rank_match.extend((pred_choice == teacher.argmin(dim=1)).float().cpu().tolist())
            else:
                target_idx = data["teacher_best_index"][ids].long()
                target_action = actions.gather(1, target_idx[:, None, None].expand(-1, 1, 2)).squeeze(1)
                prediction = model(obs)
                all_action_mse.extend((prediction - target_action).square().mean(dim=-1).cpu().tolist())
    result = {"method": method, "pde": pde, "checkpoint": str(checkpoint), "validation_queries": len(data["candidate_action"]),
             "validation_parents": len(set(np.load(query, allow_pickle=False)["parent_id"].tolist())),
             "split": split, "test_opened": False}
    if all_field_sq:
        result["field_nrmse"] = float(torch.sqrt(torch.stack(all_field_sq).sum() / torch.stack(all_target_sq).sum().clamp_min(1e-12)))
        result["normalized_teacher_regret_mean"] = float(np.mean(all_regret))
        result["normalized_teacher_regret_median"] = float(np.median(all_regret))
        result["teacher_best_agreement"] = float(np.mean(all_rank_match))
    else:
        result["teacher_action_mse"] = float(np.mean(all_action_mse))
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--query", type=Path, required=True)
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = ap.parse_args()
    print(json.dumps(evaluate(args.checkpoint, args.query, args.device), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
