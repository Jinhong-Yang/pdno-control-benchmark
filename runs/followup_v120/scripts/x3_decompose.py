"""Frozen CPU evaluation for the X3 observer/operator error decomposition."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
RUN = HERE.parents[1]
ROOT = RUN.parents[1]
REV = ROOT / "runs" / "revision_v2"
ORIGINAL = ROOT / "experiments" / "pdno_jevLite_20260927_v3"
sys.path.insert(0, str(REV / "src"))
from pdno.models.operators import ActionFactorizedOperator, CandidateConditionedOperator, spatial_basis
from pdno.models.encoders import load_state_with_normalizer_defaults
from pdno.data.generate import _periodic_actuators, _dirichlet_actuators

SPEC_PATH = RUN / "config" / "X3_SPEC.json"
FREEZE_PATH = RUN / "evidence" / "X3_FREEZE.json"
AMENDMENT_PATH = RUN / "evidence" / "X3_FREEZE_AMENDMENT.json"
OBS_KEYS = (
    "sensor_value", "sensor_mask", "sensor_age", "instrument_image", "image_mask",
    "goal_coefficients", "material_context", "previous_applied_action",
    "applied_action_history", "image_age", "image_valid",
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _primary_roster() -> list[dict]:
    path = REV / "results" / "ORIGINAL_TRAINING_SELECTION.csv"
    rows = _read_csv(path)
    selected = [r for r in rows if r["method"] in {"P", "P-no-rank", "B4", "B5"}]
    if len(selected) != 24:
        raise RuntimeError(f"Expected 24 primary operators; found {len(selected)}")
    out = []
    for r in selected:
        summary = ROOT / Path(r["summary_path"])
        checkpoint = summary.with_name(summary.name.removesuffix(".summary.json") + ".pt")
        out.append(_record("primary", r["pde"], r["method"], int(r["seed"]), None,
                           checkpoint, float(r["validation_field_nrmse"]),
                           ORIGINAL / "data" / "queries_v3" / "validation" / r["pde"] / "teacher_queries.npz",
                           summary, int(r["selected_update"])))
    return out


def _e3_roster() -> list[dict]:
    path = REV / "results" / "E3_TRAINING_SELECTION.csv"
    rows = _read_csv(path)
    selected = [r for r in rows if r["method"] in {"P", "B4"}]
    if len(selected) != 18:
        raise RuntimeError(f"Expected 18 expanded Burgers operators; found {len(selected)}")
    out = []
    for r in selected:
        summary = ROOT / Path(r["summary_path"])
        checkpoint = summary.with_name(summary.name.removesuffix(".summary.json") + ".pt")
        out.append(_record("expanded_burgers", "burgers", r["method"], int(r["seed"]),
                           int(r["target_factor"]), checkpoint,
                           float(r["validation_field_nrmse"]),
                           ORIGINAL / "data" / "queries_v3" / "validation" / "burgers" / "teacher_queries.npz",
                           summary, int(r["selected_update"])))
    return out


def _e5_gate(summary: Path) -> tuple[float, int]:
    record = json.loads(summary.read_text(encoding="utf-8"))
    selected_update = int(record["best_step"])
    selected = [row for row in record["history"] if int(row["update"]) == selected_update]
    if len(selected) != 1:
        raise RuntimeError(f"Cannot identify E5 selected validation row in {summary}")
    return float(selected[0]["validation_field_nrmse"]), selected_update


def _e5_roster() -> list[dict]:
    out = []
    val = REV / "data" / "positive_heat_queries" / "validation" / "heat" / "teacher_queries.npz"
    for method in ("P", "P-no-rank", "B4", "B5"):
        for seed in (11, 23, 37):
            stem = f"{method}_s{seed}"
            summary = REV / "checkpoints" / "E5" / f"{stem}.summary.json"
            gate, update = _e5_gate(summary)
            checkpoint = summary.with_name(summary.name.removesuffix(".summary.json") + ".pt")
            out.append(_record("nonnegative_heat", "heat", method, seed, None,
                               checkpoint, gate, val, summary, update))
    if len(out) != 12:
        raise RuntimeError("E5 roster count mismatch")
    return out


def _record(stage, pde, method, seed, factor, checkpoint, gate, query, summary, update):
    return {
        "stage": stage, "pde": pde, "method": method, "seed": seed,
        "target_factor": factor, "selected_update": update,
        "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": sha(checkpoint),
        "summary": str(summary.relative_to(ROOT)), "summary_sha256": sha(summary),
        "gate_validation_field_nrmse": gate,
        "queries": str(query.relative_to(ROOT)), "queries_sha256": sha(query),
    }


def roster() -> list[dict]:
    rows = _primary_roster() + _e3_roster() + _e5_roster()
    if len(rows) != 54 or len({r["checkpoint"] for r in rows}) != 54:
        raise RuntimeError("X3 frozen roster must contain 54 unique selected checkpoints")
    return rows


def freeze() -> dict:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    inputs = {}
    for path in [SPEC_PATH, HERE, REV / "src/pdno/models/operators.py",
                 REV / "src/pdno/models/encoders.py", REV / "src/pdno/data/teacher_queries.py",
                 REV / "src/pdno/physics/burgers.py", REV / "src/pdno/physics/heat.py",
                 REV / "src/pdno/controllers/linear.py",
                 REV / "results/ORIGINAL_TRAINING_SELECTION.csv",
                 REV / "results/E3_TRAINING_SELECTION.csv"]:
        inputs[str(path.relative_to(ROOT))] = sha(path)
    return {
        "status": "FROZEN_BEFORE_X3_EVALUATION",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pre_evaluation_commit": subprocess.check_output(["git", "-C", str(RUN), "rev-parse", "HEAD"], text=True).strip(),
        "spec_sha256": sha(SPEC_PATH), "script_sha256": sha(HERE),
        "input_source_hashes": inputs, "roster": roster(),
        "thresholds": spec["frozen_decision_rules"],
        "evaluation_contract": spec["evaluation_contract"],
        "expected_operator_count": 54,
    }


def verify_frozen() -> dict:
    f = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if f["status"] != "FROZEN_BEFORE_X3_EVALUATION":
        raise RuntimeError("X3 freeze receipt has unexpected status")
    if f["spec_sha256"] != sha(SPEC_PATH):
        raise RuntimeError("X3 spec changed after freeze")
    expected_script = f["script_sha256"]
    amended_script = False
    if sha(HERE) != expected_script:
        if not AMENDMENT_PATH.exists():
            raise RuntimeError("X3 script changed after freeze without a recorded amendment")
        amendment = json.loads(AMENDMENT_PATH.read_text(encoding="utf-8"))
        if (amendment["original_freeze_sha256"] != sha(FREEZE_PATH)
                or amendment["old_script_sha256"] != expected_script
                or amendment["new_script_sha256"] != sha(HERE)
                or amendment["thresholds_changed"]):
            raise RuntimeError("X3 amendment does not validate the current code and unchanged thresholds")
        amended_script = True
    for rel, expected in f["input_source_hashes"].items():
        if amended_script and rel.replace("\\", "/").endswith("runs/followup_v120/scripts/x3_decompose.py"):
            continue
        if sha(ROOT / rel) != expected:
            raise RuntimeError(f"Frozen source changed: {rel}")
    for row in f["roster"]:
        for key in ("checkpoint", "summary", "queries"):
            if sha(ROOT / row[key]) != row[f"{key}_sha256"]:
                raise RuntimeError(f"Frozen {key} changed: {row[key]}")
    return f


def _model(row: dict):
    pde = row["pde"]
    cls = ActionFactorizedOperator if row["method"] in {"P", "P-no-rank", "B5"} else CandidateConditionedOperator
    model = cls(pde, 33 if pde == "burgers" else 32, rank=32)
    saved = torch.load(ROOT / row["checkpoint"], map_location="cpu", weights_only=False)
    load_state_with_normalizer_defaults(model, saved["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _observation(arrays: dict[str, np.ndarray], start: int, end: int) -> dict[str, torch.Tensor]:
    return {k: torch.as_tensor(arrays[k][start:end]) for k in OBS_KEYS}


def _grid(pde: str):
    x = torch.arange(128, dtype=torch.float32) / 128 if pde == "burgers" else (torch.arange(128, dtype=torch.float32) + 1) / 129
    tau = torch.linspace(1 / 8, 1, 8, dtype=torch.float32)
    return x, tau


def _coeff_hash(model) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(model.encoder.state_dict().items()):
        h.update(name.encode()); h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    for name, tensor in sorted(model.initial_head.state_dict().items()):
        h.update(name.encode()); h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _qhat(model, arrays: dict[str, np.ndarray], batch: int = 64) -> tuple[np.ndarray, np.ndarray, list[str]]:
    x, _ = _grid(model.pde)
    coarse, coefficients = [], []
    with torch.inference_mode():
        for start in range(0, len(arrays["initial_field"]), batch):
            end = min(start + batch, len(arrays["initial_field"]))
            obs = _observation(arrays, start, end)
            context = model.encoder(obs)
            coeff = model.initial_head(context)
            coefficients.append(coeff.numpy())
            coarse.append(torch.einsum("bq,qn->bn", coeff, spatial_basis(model.pde, x)).numpy())
    return np.concatenate(coarse), np.concatenate(coefficients), [str(x) for x in arrays["parent_id"]]


def _burgers_rollout(coeff: np.ndarray, actions: np.ndarray, nu: np.ndarray) -> np.ndarray:
    """Vectorized exact-discrete counterpart of teacher_queries._future_burgers."""
    n = 256
    x = np.arange(n, dtype=np.float64) / n
    basis = np.stack([np.ones_like(x)] + [fn(k, x) for k in range(1, 17) for fn in (lambda k, x: np.sin(2*np.pi*k*x), lambda k, x: np.cos(2*np.pi*k*x))])
    qcoeff = coeff.astype(np.float64)
    modes = np.arange(1, 17, dtype=np.float64)
    # The checkpoint's initial head is the same learned Fourier coefficient vector used by spatial_basis.
    q0 = qcoeff @ basis
    force_basis = _periodic_actuators(n)
    force = actions.astype(np.float64) @ force_basis
    k = 2 * np.pi * np.fft.fftfreq(n, d=1/n)
    mode = np.fft.fftfreq(n) * n
    dk = 1j * k
    dk[np.abs(mode) > n/3] = 0
    mul = np.exp(-nu[:, None] * (k[None, :] ** 2) * (2.5e-4 / 2))
    q = q0.copy()
    outputs = []
    dt = 2.5e-4
    def rhs(v):
        flux = np.fft.fft(.5 * v * v, axis=-1)
        flux[..., np.abs(mode) > n/3] = 0
        return -np.fft.ifft(dk * flux, axis=-1).real + force
    for _ in range(8):
        for _ in range(80):
            q = np.fft.ifft(mul * np.fft.fft(q, axis=-1), axis=-1).real
            k1 = rhs(q); k2 = rhs(q + .5*dt*k1); k3 = rhs(q + .5*dt*k2); k4 = rhs(q + dt*k3)
            q = q + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            q = np.fft.ifft(mul * np.fft.fft(q, axis=-1), axis=-1).real
        outputs.append(q[:, ::2].astype(np.float32))
    return np.stack(outputs, axis=1)


def _heat_rollout(coeff: np.ndarray, actions: np.ndarray, kappa: np.ndarray, decay: np.ndarray) -> np.ndarray:
    from scipy.linalg import solve_banded
    n = 256; dx = 1 / 257; dt = .02
    x = (np.arange(n) + 1) / 257
    basis = np.stack([np.sin(j * np.pi * x) - x * np.sin(j * np.pi) for j in range(1, 33)])
    q = coeff.astype(np.float64) @ basis
    fbase = _dirichlet_actuators(n)
    forcing = actions.astype(np.float64) @ fbase
    out = []
    for i in range(len(q)):
        r = float(kappa[i]) * dt / (2 * dx**2); c = float(decay[i]) * dt / 2
        ab = np.zeros((3, n)); ab[0, 1:] = -r; ab[1] = 1 + 2*r + c; ab[2, :-1] = -r
        qi = q[i:i+1].copy(); fi = forcing[i:i+1]; rows = []
        for _ in range(8):
            rhs = (1 - 2*r-c) * qi
            rhs[:, 1:] += r * qi[:, :-1]
            rhs[:, :-1] += r * qi[:, 1:]
            rhs += dt * fi
            qi = solve_banded((1, 1), ab, rhs.T).T
            # Match restrict_field's cell-center interpolation and homogeneous endpoints.
            xx = (np.arange(n) + 1) / (n + 1)
            xm = (np.arange(128) + 1) / 129
            rows.append(np.stack([np.interp(xm, np.r_[0., xx, 1.], np.r_[0., v, 0.]) for v in qi]))
        out.append(rows)
    return np.asarray(out, dtype=np.float32).squeeze(2)


def _target_true(arrays: dict[str, np.ndarray]) -> np.ndarray:
    return arrays["future_field"].astype(np.float32, copy=False)


def evaluate(freeze_record: dict, row: dict, cache: dict) -> tuple[dict, list[dict]]:
    qpath = ROOT / row["queries"]
    with np.load(qpath, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    model = _model(row)
    observer_key = row["queries_sha256"] + ":" + _coeff_hash(model)
    if observer_key not in cache:
        qhat, coeff, parents = _qhat(model, arrays)
        # Future counterfactual rows are candidate-action major, which shares the same observation/qhat.
        acts = arrays["candidate_action"].reshape(-1, 2)
        rep_coeff = np.repeat(coeff, arrays["candidate_action"].shape[1], axis=0)
        if row["pde"] == "burgers":
            nus = np.repeat(arrays["material_context"][:, 0], arrays["candidate_action"].shape[1])
            prop = _burgers_rollout(rep_coeff, acts, nus)
        else:
            kappa = np.repeat(arrays["material_context"][:, 0], arrays["candidate_action"].shape[1])
            decay = np.repeat(arrays["material_context"][:, 1], arrays["candidate_action"].shape[1])
            prop = _heat_rollout(rep_coeff, acts, kappa, decay)
        prop = prop.reshape(len(qhat), arrays["candidate_action"].shape[1], 8, 128)
        cache[observer_key] = {"qhat": qhat, "coeff": coeff, "prop": prop, "parents": parents}
    cached = cache[observer_key]
    qhat, prop = cached["qhat"], cached["prop"]
    target = _target_true(arrays)
    pred_parts, pred_sq_acc, target_sq_acc = [], torch.zeros((), dtype=torch.float32), torch.zeros((), dtype=torch.float32)
    residual_prop_parts, residual_op_parts = [], []
    # Match confirmatory._evaluate_operator's 32-row batches and FP32 pooled accumulation.
    with torch.inference_mode():
        x, tau = _grid(row["pde"])
        for start in range(0, len(target), 32):
            end = min(start + 32, len(target))
            obs = _observation(arrays, start, end)
            actions = torch.as_tensor(arrays["candidate_action"][start:end], dtype=torch.float32)
            pred = model(obs, actions, x, tau).cpu()
            tgt = torch.as_tensor(target[start:end], dtype=torch.float32)
            prop_t = torch.as_tensor(prop[start:end], dtype=torch.float32)
            diff = pred - tgt
            pred_sq_acc += diff.square().sum()
            target_sq_acc += tgt.square().sum()
            residual_prop_parts.append((prop_t - tgt).numpy())
            residual_op_parts.append((pred - prop_t).numpy())
            pred_parts.append(pred.numpy())
    pred = np.concatenate(pred_parts, axis=0)
    ep = np.concatenate(residual_prop_parts, axis=0).astype(np.float64)
    eo = np.concatenate(residual_op_parts, axis=0).astype(np.float64)
    truth = target.astype(np.float64)
    denom = float(np.square(truth).sum())
    total_numer = float(np.square(pred.astype(np.float64) - truth).sum())
    prop_numer = float(np.square(ep).sum())
    op_numer = float(np.square(eo).sum())
    cross2 = float(2 * np.sum(ep * eo))
    e_total = float(torch.sqrt(pred_sq_acc / target_sq_acc).item())
    e_prop = math.sqrt(prop_numer / denom)
    e_op = math.sqrt(op_numer / denom)
    # Observer error is computed once per query row; no artificial candidate replication.
    init_target = arrays["initial_field"].astype(np.float64)
    obs_numer = float(np.square(qhat.astype(np.float64) - init_target).sum())
    obs_denom = float(np.square(init_target).sum())
    e_obs = math.sqrt(obs_numer / obs_denom)
    persistence = np.repeat(qhat[:, None, None, :], 8, axis=2)
    persistence_numer = float(np.square(persistence.astype(np.float64) - truth).sum())
    e_persist = math.sqrt(persistence_numer / denom)
    pred_raw_sha = hashlib.sha256(np.ascontiguousarray(pred.astype("<f4")).tobytes()).hexdigest()
    target_raw_sha = hashlib.sha256(np.ascontiguousarray(target.astype("<f4")).tobytes()).hexdigest()
    prop_raw_sha = hashlib.sha256(np.ascontiguousarray(prop.astype("<f4")).tobytes()).hexdigest()
    qhat_raw_sha = hashlib.sha256(np.ascontiguousarray(qhat.astype("<f4")).tobytes()).hexdigest()
    saved = float(row["gate_validation_field_nrmse"])
    delta = e_total - saved
    gate_match_4dp = round(e_total, 4) == round(saved, 4)
    ratio = e_prop / e_total
    if ratio >= .8:
        branch = "observer_dominant"
    elif ratio < .5:
        branch = "operator_error_substantial"
    else:
        branch = "both_contributions_present"
    row_vectors = []
    for i in range(len(qhat)):
        tr = truth[i]
        erp = ep[i]; ero = eo[i]
        row_target_sq = float(np.square(tr).sum())
        row_prop_sq = float(np.square(erp).sum())
        row_op_sq = float(np.square(ero).sum())
        row_cross = float(2 * np.sum(erp * ero))
        row_total_sq = float(np.square(pred[i].astype(np.float64) - tr).sum())
        row_persist_sq = float(np.square(persistence[i].astype(np.float64) - tr).sum())
        row_initial_sq = float(np.square(init_target[i]).sum())
        row_obs_sq = float(np.square(qhat[i].astype(np.float64) - init_target[i]).sum())
        row_vectors.append({
            "stage": row["stage"], "pde": row["pde"], "method": row["method"], "seed": row["seed"],
            "target_factor": row["target_factor"], "query_index": i, "parent_id": str(arrays["parent_id"][i]),
            "decision_tick": int(arrays["decision_tick"][i]),
            "Eobs_query_nrmse": math.sqrt(row_obs_sq / row_initial_sq),
            "Eprop_query_nrmse": math.sqrt(row_prop_sq / row_target_sq),
            "Eop_query_nrmse": math.sqrt(row_op_sq / row_target_sq),
            "Etotal_query_nrmse": math.sqrt(row_total_sq / row_target_sq),
            "persistence_query_nrmse": math.sqrt(row_persist_sq / row_target_sq),
            "initial_error_sq": row_obs_sq, "target_initial_sq": row_initial_sq,
            "prop_error_sq": row_prop_sq, "op_error_sq": row_op_sq,
            "cross_term_2prop_dot_op": row_cross, "total_error_sq": row_total_sq,
            "target_future_sq": row_target_sq, "persistence_error_sq": row_persist_sq,
        })
    summary = {
        **{k: row[k] for k in ("stage", "pde", "method", "seed", "target_factor", "selected_update", "checkpoint", "checkpoint_sha256", "queries", "queries_sha256")},
        "parent_count": len(set(cached["parents"])), "query_count": len(target),
        "candidate_query_count": int(target.shape[0] * target.shape[1]), "candidate_count_per_query": int(target.shape[1]),
        "E_obs": e_obs, "E_prop": e_prop, "E_total": e_total, "E_op": e_op,
        "persistence": e_persist, "E_prop_over_E_total": ratio, "decision_branch": branch,
        "saved_gate_E_total": saved, "E_total_abs_difference_from_saved_gate": abs(delta),
        "gate_reproduction_four_decimals": gate_match_4dp,
        "raw_sums": {"obs_error_sq": obs_numer, "initial_target_sq": obs_denom,
                     "prop_error_sq": prop_numer, "operator_error_sq": op_numer,
                     "cross_term_2prop_dot_op": cross2, "total_error_sq_fp64_reconstructed": total_numer,
                     "total_error_sq_fp32_gate_accumulator": float(pred_sq_acc), "future_target_sq": denom,
                     "persistence_error_sq": persistence_numer,
                     "identity_reconstruction_abs_error": abs(total_numer - (prop_numer + op_numer + cross2))},
        "raw_vector_sha256": {"operator_prediction_float32": pred_raw_sha,
                               "future_target_float32": target_raw_sha,
                               "observer_propagated_reference_float32": prop_raw_sha,
                               "observer_initial_float32": qhat_raw_sha},
        "claim_scope": "Validation decomposition only; descriptive norm ratio, not additive explained variance or causal attribution.",
    }
    return summary, row_vectors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true", help="write the pre-evaluation X3 roster/hash/threshold seal")
    parser.add_argument("--evaluate", action="store_true", help="run the frozen X3 CPU evaluation")
    parser.add_argument("--limit", type=int, help="development smoke only; results remain non-evidence")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.freeze:
        if FREEZE_PATH.exists():
            raise RuntimeError("X3 freeze already exists; refusing to replace it")
        rec = freeze()
        FREEZE_PATH.parent.mkdir(parents=True, exist_ok=True)
        FREEZE_PATH.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": rec["status"], "operators": len(rec["roster"]), "commit": rec["pre_evaluation_commit"]}))
    if args.evaluate:
        frozen = verify_frozen()
        rows = frozen["roster"][:args.limit] if args.limit else frozen["roster"]
        cache = {}
        summaries, vectors = [], []
        t0 = time.perf_counter()
        for i, row in enumerate(rows, 1):
            print(f"[{i}/{len(rows)}] {row['stage']} {row['pde']} {row['method']} seed={row['seed']} factor={row['target_factor']}", flush=True)
            summary, paired = evaluate(frozen, row, cache)
            summaries.append(summary); vectors.extend(paired)
            print(json.dumps({k: summary[k] for k in ("stage", "pde", "method", "seed", "target_factor", "E_obs", "E_prop", "E_total", "E_op", "E_prop_over_E_total", "decision_branch")}), flush=True)
        out = RUN / "results" / "X3"
        out.mkdir(parents=True, exist_ok=True)
        failed_gate_rows = [r for r in summaries if not r["gate_reproduction_four_decimals"]]
        result_status = ("COMPLETE_GATE_REPRODUCED" if not failed_gate_rows else "COMPLETE_WITH_GATE_REPRODUCTION_MISMATCH") if len(rows) == 54 else "DEVELOPMENT_SMOKE_ONLY"
        (out / "X3_OPERATOR_SUMMARIES.json").write_text(json.dumps({
            "status": result_status,
            "study": "X3 observer/operator error decomposition", "freeze_sha256": sha(FREEZE_PATH),
            "spec_sha256": sha(SPEC_PATH), "script_sha256": sha(HERE), "operator_count": len(summaries),
            "elapsed_wall_seconds": time.perf_counter() - t0, "device": "CPU", "rows": summaries,
            "gate_reproduction_failures": [{"checkpoint":r["checkpoint"],"recomputed":r["E_total"],"frozen_gate":r["saved_gate_E_total"],"absolute_difference":r["E_total_abs_difference_from_saved_gate"]} for r in failed_gate_rows],
            "limits": ["Validation-only decomposition.", "Query rows are clustered within parent scenarios.",
                       "E_prop/E_total is a descriptive norm ratio, not an additive or causal share.",
                       "Reference counterfactual futures use zero future disturbance as specified by teacher_queries.py."]
        }, indent=2) + "\n", encoding="utf-8")
        with (out / "X3_PAIRED_VECTORS.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(vectors[0]))
            w.writeheader(); w.writerows(vectors)
        receipt = {"status": result_status,
                   "operator_count": len(rows), "paired_vector_rows": len(vectors),
                   "gate_reproduction_mismatch_count": len(failed_gate_rows),
                   "summary_sha256": sha(out / "X3_OPERATOR_SUMMARIES.json"),
                   "paired_vectors_sha256": sha(out / "X3_PAIRED_VECTORS.csv"),
                   "freeze_sha256": sha(FREEZE_PATH), "script_sha256": sha(HERE),
                   "completed_utc": datetime.now(timezone.utc).isoformat(), "device": "CPU"}
        (RUN / "evidence" / "X3_COMPLETION.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(receipt, indent=2))
    if not args.freeze and not args.evaluate:
        parser.error("Choose --freeze or --evaluate")


if __name__ == "__main__":
    main()
