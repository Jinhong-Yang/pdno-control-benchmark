from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
REV = ROOT.parent / "revision_v2"
OLD = REPO / "experiments/pdno_jevLite_20260927_v3"
sys.path.insert(0, str(OLD / "src"))
sys.path.insert(0, str(REV / "src"))
sys.path.insert(0, str(ROOT / "src"))

from x1.design import candidates, full_state_b0_action, objective, summarize_indices
from x1.solvers import BurgersHorizonCUDA, heat_advance, heat_forecast
from pdno.controllers.actions import project_box_slew
from pdno.controllers.linear import nominal_lqr_action
from pdno.data.teacher_queries import restrict_field
from pdno.evaluation.closed_loop import _schedule, _sensor_at, _image_at, _obs_for_tick


SPEC = json.loads((ROOT / "config/X1_SPEC.json").read_text(encoding="utf-8"))
CELL_MAP = {c["id"]: c for c in SPEC["cells"]}
QMAX_HEAT = 2.1322593092918396


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_for(pde: str) -> Path:
    if pde == "burgers":
        return OLD / "data/locked_v3/locked_nominal/burgers/trajectories.npz"
    return REV / "data/positive_heat/locked_nominal/heat/trajectories.npz"


def truth_pulse(seed: int, n: int = 256) -> np.ndarray | None:
    if seed % 4:
        return None
    x = np.arange(n) / n
    center = np.random.default_rng(seed).random()
    d = np.minimum(abs(x - center), 1 - abs(x - center))
    pulse = 0.05 * np.exp(-0.5 * (d / 0.025) ** 2)
    pulse -= pulse.mean()
    return pulse


def load_population(pde: str) -> tuple[dict, Path]:
    source = source_for(pde)
    with np.load(source, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    ids = np.argsort(data["parent_id"].astype(str), kind="stable")
    data = {k: v[ids] for k, v in data.items()}
    return data, source


def next_truth(states: np.ndarray, actions: np.ndarray, pde: str, material: np.ndarray,
               tick: int, parent_ids: np.ndarray, pulses: list[np.ndarray | None],
               plant_solver: BurgersHorizonCUDA | None) -> np.ndarray:
    if pde == "burgers":
        extras = np.stack([p if p is not None else np.zeros(256) for p in pulses]) if tick == 100 else None
        return plant_solver.advance(states, actions, extras)
    return np.stack([heat_advance(states[b], actions[b], float(material[b, 0]), float(material[b, 1]))
                     for b in range(len(states))]).astype(np.float32)


def fixed_forecasts(states: np.ndarray, actions: np.ndarray, pde: str, material: np.ndarray,
                    H: int, forecast_solver: BurgersHorizonCUDA | None) -> np.ndarray:
    if pde == "burgers":
        return forecast_solver.forecast(states, actions, H)
    return np.stack([heat_forecast(np.broadcast_to(states[b], (len(actions[b]), 256)),
                                   actions[b], float(material[b, 0]), float(material[b, 1]), H)
                     for b in range(len(states))])


def feedback_forecasts(states: np.ndarray, b0: np.ndarray, pde: str, materials: np.ndarray,
                       goals: np.ndarray, H: int, previous: np.ndarray,
                       qmax: float, plant_solver: BurgersHorizonCUDA | None) -> tuple[np.ndarray, np.ndarray]:
    current = states.copy()
    actions = np.empty((len(states), H, 2), dtype=np.float32)
    fields = np.empty((len(states), H, 128), dtype=np.float32)
    first = b0.copy()
    for h in range(H):
        if h == 0:
            proposed = first
        else:
            proposed = np.stack([full_state_b0_action(current[b], pde, materials[b], goals[b], actions[b, h-1])
                                 for b in range(len(states))])
            low, high, slew = (-1, 1, 0.15) if pde == "burgers" else (0, 1, 0.10)
            proposed = np.stack([project_box_slew(proposed[b], actions[b, h-1], low, high, slew)
                                 for b in range(len(states))])
        actions[:, h] = proposed
        current = next_truth(current, proposed, pde, materials, h, np.arange(len(states)),
                             [None] * len(states), plant_solver)
        fields[:, h] = current[:, ::2] if pde == "burgers" else np.stack(
            [restrict_field(v, "heat") for v in current])
    return fields, actions


def evaluate_chunk(data: dict, indices: np.ndarray, pde: str, cell: dict,
                   full384_baseline: bool = False) -> dict:
    K, H = int(cell["K"]), int(cell["H"])
    B = len(indices)
    parent_ids = data["parent_id"][indices].astype(str)
    states = data["state_true"][indices, 0].astype(np.float32).copy()
    if pde == "burgers":
        materials = np.stack([np.array([float(data["nu"][i]), -1, 1, .15, .02, .16, 1.2, 128], np.float32)
                              for i in indices])
        goals = np.zeros((B, 256), dtype=np.float32)
        qmax = 1.2
        low, high, slew = -1., 1., .15
        nudata = np.array([float(v) for v in materials[:, 0]], dtype=np.float64)
        forecast_solver = BurgersHorizonCUDA(nudata, K)
        plant_solver = BurgersHorizonCUDA(nudata, 1)
    else:
        materials = np.stack([np.array([float(data["kappa"][i]), float(data["decay"][i]), 0, 1, .1, .02, .16, 128], np.float32)
                              for i in indices])
        goals = data["goal"][indices].astype(np.float32)
        qmax = QMAX_HEAT
        low, high, slew = 0., 1., .10
        forecast_solver = None
        plant_solver = None
    goals_after = data.get("goal_step_field", data.get("goal"))[indices].astype(np.float32) if pde == "heat" else goals
    step_ticks = data.get("goal_step_tick", np.full(len(data["parent_id"]), -1))[indices] if pde == "heat" else np.full(B, -1)
    schedules = [_schedule(str(pid), 200, str(data.get("role", np.array(["locked_nominal"]))[idx]) == "locked_delay_dropout")
                 for pid, idx in zip(parent_ids, indices)]
    sensors = [np.zeros((200, 16), np.float32) for _ in range(B)]
    images = [[None] * 200 for _ in range(B)]
    actions_history = [[] for _ in range(B)]
    pulses = [truth_pulse(int(data["disturbance_seed"][i])) for i in indices]
    state_history = np.empty((B, 201, 256), np.float32)
    state_history[:, 0] = states
    applied_history = np.empty((B, 200, 2), np.float32)
    goal_history = np.empty((B, 200, 256), np.float32)
    choice_history = np.full((B, 200), -1, np.int16)
    costs = np.zeros(B, dtype=np.float64)
    sum_sq = np.zeros(B, dtype=np.float64)
    action_change = np.zeros(B, dtype=np.float64)
    prev = np.stack([np.zeros(2, np.float32) if pde == "burgers" else np.full(2, .4, np.float32) for _ in range(B)])
    for tick in range(200):
        observations, nominal, candidates_by_b = [], [], []
        targets = np.where(((step_ticks >= 0) & (tick >= step_ticks))[:, None], goals_after, goals)
        for b in range(B):
            schedules_b = schedules[b]
            sensors[b][tick] = _sensor_at(states[b], pde, schedules_b["sensor_noise"][tick])
            if not schedules_b["image_drop"][tick]:
                images[b][tick] = _image_at(states[b], pde, schedules_b, tick)
            obs, previous = _obs_for_tick(actions_history[b], tick, pde, materials[b], targets[b],
                                          schedules_b, sensors[b], images[b])
            observations.append(obs)
            nominal.append(nominal_lqr_action(obs, pde, previous))
        nominal = np.asarray(nominal, dtype=np.float32)
        for b in range(B):
            candidates_by_b.append(candidates(pde, prev[b], nominal[b], K))
        candidate_actions = np.stack(candidates_by_b)
        previous = prev.copy()
        futures = fixed_forecasts(states, candidate_actions, pde, materials, H, forecast_solver)
        goal_score = np.zeros((B, 128), np.float32) if pde == "burgers" else np.stack(
            [restrict_field(row, "heat") for row in targets])
        candidate_costs = np.empty((B, K + int(cell["feedback_rollout"])), dtype=np.float64)
        for b in range(B):
            for k in range(K):
                candidate_costs[b, k] = objective(futures[b, k], goal_score[b], candidate_actions[b, k],
                                                  previous[b], qmax, pde)
        feedback_actions = None
        if cell["feedback_rollout"]:
            ff, feedback_actions = feedback_forecasts(states, nominal, pde, materials, targets, H,
                                                      previous, qmax, plant_solver)
            for b in range(B):
                candidate_costs[b, K] = objective(ff[b], goal_score[b], feedback_actions[b, 0],
                                                   previous[b], qmax, pde)
        chosen = np.argmin(candidate_costs, axis=1)
        proposal = np.stack([feedback_actions[b, 0] if chosen[b] == K else candidate_actions[b, chosen[b]]
                             for b in range(B)])
        applied = np.stack([project_box_slew(proposal[b], previous[b], low, high, slew) for b in range(B)])
        nxt = next_truth(states, applied, pde, materials, tick, parent_ids, pulses, plant_solver)
        error = (nxt - targets).astype(np.float64)
        effort = .01 * np.sum(applied.astype(float) ** 2, axis=1)
        slew_term = .05 * np.sum((applied.astype(float) - previous.astype(float)) ** 2, axis=1)
        if pde == "burgers":
            violation = np.maximum(np.abs(nxt) - qmax, 0) ** 2
        else:
            violation = np.maximum(-nxt, 0) ** 2 + np.maximum(nxt - qmax, 0) ** 2
        costs += np.mean(error ** 2, axis=1) + effort + slew_term + 10 * np.mean(violation, axis=1)
        sum_sq += np.sum(error ** 2, axis=1)
        action_change += np.mean(np.abs(applied - previous), axis=1)
        for b in range(B):
            actions_history[b].append(applied[b].copy())
        states = nxt
        state_history[:, tick + 1] = states
        applied_history[:, tick] = applied
        goal_history[:, tick] = targets
        choice_history[:, tick] = chosen
        prev = applied
    return {
        "parent_id": parent_ids,
        "episode_control_cost": costs,
        "tracking_rmse": np.sqrt(sum_sq / (200 * 256)),
        "selected_candidate_index": choice_history,
        "action_applied": applied_history,
        "state_true": state_history,
        "goal": goal_history,
        "mean_abs_action_change": action_change / 200,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pde", choices=["burgers", "heat"], required=True)
    parser.add_argument("--cell", choices=tuple(CELL_MAP), required=True)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--chunk", type=int, default=64)
    parser.add_argument("--baseline-full384", action="store_true")
    args = parser.parse_args()
    if args.n not in (128, 384):
        raise ValueError("X1 run supports the sealed n=128 or baseline n=384 only")
    if args.cell == "K10_H8" and args.pde == "burgers" and args.n == 384 and not args.baseline_full384:
        raise ValueError("Burgers full-384 baseline requires explicit --baseline-full384")
    if args.baseline_full384 and (args.cell != "K10_H8" or args.pde != "burgers" or args.n != 384):
        raise ValueError("--baseline-full384 is reserved for the Burgers K10/H8 anchor")
    is_anchor = args.cell == "K10_H8" and args.pde == "burgers" and args.n == 384 and args.baseline_full384
    baseline_receipt = ROOT / "results/X1_BASELINE_VERIFICATION.json"
    if not is_anchor:
        if not baseline_receipt.exists():
            raise RuntimeError("X1 is sealed to run and verify the full384 Burgers K10/H8 anchor first")
        baseline_status = json.loads(baseline_receipt.read_text(encoding="utf-8"))
        if baseline_status.get("status") != "PASS":
            raise RuntimeError("full384 Burgers baseline did not pass; X1 cells are halted")
    data, source = load_population(args.pde)
    count = min(args.n, len(data["parent_id"]))
    idx = np.arange(count, dtype=int)
    if count != len(data["parent_id"]):
        pass  # load_population already sorted by string scenario ID
    cell = CELL_MAP[args.cell]
    outdir = ROOT / "results/X1" / ("burgers_nominal" if args.pde == "burgers" else "heat_nonnegative_nominal") / args.cell
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"n{count}.npz"
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    started = time.perf_counter()
    blocks = []
    for first in range(0, count, args.chunk):
        end = min(first + args.chunk, count)
        blocks.append(evaluate_chunk(data, idx[first:end], args.pde, cell))
        print(json.dumps({"pde": args.pde, "cell": args.cell, "completed_scenarios": end,
                          "total": count, "elapsed_s": time.perf_counter() - started}), flush=True)
    arrays = {k: np.concatenate([b[k] for b in blocks], axis=0) for k in blocks[0]}
    np.savez_compressed(out, **arrays)
    rec = {
        "status": "COMPLETE",
        "pde": args.pde,
        "cell": cell,
        "parent_count": count,
        "scenario_id_order": "ascending lexical scenario ID; first n only",
        "source_data": str(source),
        "source_data_sha256": sha(source),
        "config_sha256": sha(ROOT / "config/X1_SPEC.json"),
        "analysis_plan_sha256": sha(ROOT / "config/X1_ANALYSIS_PLAN.md"),
        "baseline_verification_sha256": None if is_anchor else sha(baseline_receipt),
        "x1_source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / "src/x1").glob("*.py"))},
        "runner_sha256": sha(Path(__file__)),
        "git_commit": __import__("subprocess").check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": time.perf_counter() - started,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "result_sha256": sha(out),
        "metrics": {"mean_episode_cost": float(arrays["episode_control_cost"].mean()),
                    "mean_tracking_rmse": float(arrays["tracking_rmse"].mean()),
                    **summarize_indices(arrays["selected_candidate_index"], int(cell["K"]), bool(cell["feedback_rollout"])),
                    "mean_abs_action_change": float(arrays["mean_abs_action_change"].mean())},
    }
    out.with_suffix(".record.json").write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
