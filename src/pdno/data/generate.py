"""Deterministic PDE trajectory and causal observation generation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.linalg import solve_banded

from pdno.controllers.actions import project_box_slew
from pdno.data.causal import ObservationEvent, latest_available
from pdno.physics.burgers import burgers_imex_split_step, periodic_grid
from pdno.physics.heat import heat_cn_step


def _rng_for(parent_id: str) -> np.random.Generator:
    seed = int.from_bytes(hashlib.sha256(parent_id.encode()).digest()[:8], "big")
    return np.random.default_rng(seed)


def _periodic_actuators(n: int) -> np.ndarray:
    x = periodic_grid(n)
    rows = []
    for center in (0.25, 0.75):
        d = np.minimum(np.abs(x - center), 1.0 - np.abs(x - center))
        b = np.exp(-0.5 * (d / 0.08) ** 2)
        b -= b.mean()
        b /= np.max(np.abs(b))
        rows.append(b)
    return np.asarray(rows)


def _dirichlet_actuators(n: int) -> np.ndarray:
    x = np.arange(1, n + 1, dtype=np.float64) / (n + 1)
    return np.asarray([np.exp(-0.5 * ((x - c) / 0.1) ** 2) for c in (0.3, 0.7)])


def _burgers_initial(rng: np.random.Generator, n: int) -> np.ndarray:
    x = periodic_grid(n)
    q = np.zeros(n)
    for k in range(1, 9):
        q += rng.normal() * np.sin(2 * np.pi * k * x) + rng.normal() * np.cos(2 * np.pi * k * x)
    q -= q.mean()
    amplitude = rng.uniform(0.2, 0.8)
    return q * (amplitude / np.max(np.abs(q)))


def _heat_initial(rng: np.random.Generator, n: int) -> np.ndarray:
    x = np.arange(1, n + 1, dtype=np.float64) / (n + 1)
    u = sum(rng.normal() / k * np.sin(k * np.pi * x) for k in range(1, 9))
    return u * (rng.uniform(0.1, 0.5) / np.max(np.abs(u)))


def _make_observations(fields: np.ndarray, pde: str, rng: np.random.Generator,
                      stress_delay_dropout: bool = False) -> dict[str, np.ndarray]:
    # Sensor values are sampled from each contemporaneous field; availability is then
    # selected using only event capture/receive times at each decision tick.
    ntime, ngrid = fields.shape
    positions = np.linspace(0, ngrid, 16, endpoint=False) if pde == "burgers" else np.linspace(0, ngrid - 1, 16)
    if pde == "burgers":
        lo = np.floor(positions).astype(int)
        frac = positions - lo
        sensor_raw = fields[:, lo] * (1 - frac) + fields[:, (lo + 1) % ngrid] * frac
    else:
        sensor_raw = np.stack([np.interp(positions, np.arange(ngrid), row) for row in fields])
    sensor_raw = sensor_raw + rng.normal(0, 0.01, sensor_raw.shape)
    sensor_mask = np.zeros(sensor_raw.shape, dtype=bool)
    sensor_values = np.zeros(sensor_raw.shape, dtype=np.float32)
    sensor_age = np.zeros(sensor_raw.shape, dtype=np.int16)
    sensor_capture = np.full(sensor_raw.shape, -1, dtype=np.int32)
    sensor_receive = np.full(sensor_raw.shape, -1, dtype=np.int32)
    for sensor in range(16):
        events = []
        for tick in range(ntime):
            if rng.random() < (0.20 if stress_delay_dropout else 0.03):
                continue
            delay = int(rng.choice([0, 0, 0, 1]))
            events.append((ObservationEvent(tick, tick + delay, str(tick)), float(sensor_raw[tick, sensor])))
        for decision in range(ntime):
            available = [event for event, _ in events if event.capture_tick <= decision and event.receive_tick <= decision]
            if available:
                latest = max(available, key=lambda e: (e.capture_tick, e.receive_tick))
                sensor_values[decision, sensor] = sensor_raw[latest.capture_tick, sensor]
                sensor_age[decision, sensor] = decision - latest.capture_tick
                sensor_mask[decision, sensor] = True
                sensor_capture[decision, sensor] = latest.capture_tick
                sensor_receive[decision, sensor] = latest.receive_tick
    image = np.empty((ntime, 1, 16, 64), dtype=np.float16)
    image_mask = np.ones(image.shape, dtype=bool)
    image_events = []
    for tick, field in enumerate(fields):
        # Blur/interpolate one synthetic metrology strip and duplicate it across rows
        # with nuisance gain/noise. Mapping bounds are benchmark-fixed, not per-image.
        if pde == "burgers":
            x_old = np.arange(ngrid, dtype=np.float64) / ngrid
            x_new = np.arange(64, dtype=np.float64) / 64
        else:
            # Heat states contain cell-interior samples; do not pretend the first
            # and last cells are Dirichlet boundary values at x=0 and x=1.
            x_old = np.arange(1, ngrid + 1, dtype=np.float64) / (ngrid + 1)
            x_new = np.arange(1, 65, dtype=np.float64) / 65
        line = np.interp(x_new, x_old, field)
        line = gaussian_filter1d(line, sigma=0.8, mode="wrap" if pde == "burgers" else "nearest")
        scale = 1.5 if pde == "burgers" else 1.0
        normalized = np.clip(0.5 + 0.5 * line / scale, 0, 1)
        row_gain = rng.normal(1.0, 0.03, size=(16, 1))
        image_events.append((tick, normalized[None, :] * row_gain + rng.normal(0, 0.01, (16, 64))))
    delays = (rng.choice([0, 2, 4, 8], size=ntime) if stress_delay_dropout
              else rng.choice([0, 0, 1, 2], size=ntime))
    dropped = rng.random(ntime) < 0.03
    events = [ObservationEvent(t, t + int(delays[t]), str(t)) for t in range(ntime) if not dropped[t]]
    image_age = np.zeros(ntime, dtype=np.int16)
    image_valid = np.zeros(ntime, dtype=bool)
    image_capture = np.full(ntime, -1, dtype=np.int32)
    image_receive = np.full(ntime, -1, dtype=np.int32)
    image_values = np.zeros_like(image)
    for decision in range(ntime):
        event = latest_available(events, decision)
        if event is None:
            continue
        tick = int(event.payload_ref)
        image_values[decision, 0] = image_events[tick][1]
        image_age[decision] = decision - tick
        image_valid[decision] = True
        image_capture[decision] = event.capture_tick
        image_receive[decision] = event.receive_tick
        if rng.random() < 0.1:
            # Synthetic occlusion, explicitly carried as a mask.
            col = int(rng.integers(0, 56))
            image_values[decision, 0, :, col : col + 8] = 0
            image_mask[decision, 0, :, col : col + 8] = False
    return {
        "sensor_value": sensor_values,
        "sensor_mask": sensor_mask,
        "sensor_age": sensor_age,
        "sensor_capture_time": sensor_capture,
        "sensor_receive_time": sensor_receive,
        "instrument_image": image_values,
        "image_mask": image_mask,
        "image_age": image_age,
        "image_valid": image_valid,
        "image_capture_time": image_capture,
        "image_receive_time": image_receive,
    }


def draw_burgers_nu(meta: dict, rng: np.random.Generator) -> float:
    if meta.get("split") == "locked_coefficient_ood":
        index = int(meta.get("_role_index", meta.get("role_index", 0)))
        if index % 2 == 0:
            return float(rng.uniform(0.005, 0.008))
        return float(rng.uniform(0.035, 0.045))
    return float(rng.uniform(0.01, 0.03))


def draw_heat_kappa(meta: dict, rng: np.random.Generator) -> float:
    if meta.get("split") == "locked_coefficient_ood":
        return float(rng.uniform(0.024, 0.032))
    return float(rng.uniform(0.005, 0.02))


def generate_burgers_parent(meta: dict, ntruth: int = 256, outer_ticks: int = 128, dt: float = 2.5e-4) -> dict:
    rng = _rng_for(meta["parent_id"])
    nu = draw_burgers_nu(meta, rng)
    q = _burgers_initial(rng, ntruth)
    basis = _periodic_actuators(ntruth)
    truth = np.empty((outer_ticks + 1, ntruth), dtype=np.float32)
    actions = np.empty((outer_ticks, 2), dtype=np.float32)
    projected = np.empty_like(actions)
    truth[0] = q
    previous = np.zeros(2)
    inner_steps = round(0.02 / dt)
    for tick in range(outer_ticks):
        proposed = previous + rng.uniform(-0.15, 0.15, size=2)
        applied = project_box_slew(proposed, previous, -1.0, 1.0, 0.15)
        disturbance = np.zeros(ntruth)
        if tick == outer_ticks // 2 and int(meta["disturbance_seed"]) % 4 == 0:
            x = periodic_grid(ntruth)
            center = rng.random()
            d = np.minimum(np.abs(x - center), 1 - np.abs(x - center))
            disturbance = 0.05 * np.exp(-0.5 * (d / 0.025) ** 2)
            disturbance -= disturbance.mean()
        force = applied @ basis + disturbance
        for _ in range(inner_steps):
            q = burgers_imex_split_step(q, nu, dt, force)
        truth[tick + 1] = q
        actions[tick] = proposed
        projected[tick] = applied
        previous = applied
    fields = truth
    obs = _make_observations(fields, "burgers", rng, meta.get("split") == "locked_delay_dropout")
    return {"state_true": truth, "action_proposed": actions, "action_projected": projected,
            "action_applied": projected.copy(), "nu": np.float32(nu), "dx": np.float32(1 / ntruth), **obs}


def generate_heat_parent(meta: dict, ntruth: int = 256, outer_ticks: int = 128) -> dict:
    rng = _rng_for(meta["parent_id"])
    kappa, decay = draw_heat_kappa(meta, rng), rng.uniform(0.2, 0.5)
    dx, dt = 1.0 / (ntruth + 1), 0.02
    u = _heat_initial(rng, ntruth)
    basis = _dirichlet_actuators(ntruth)
    truth = np.empty((outer_ticks + 1, ntruth), dtype=np.float32)
    actions = np.empty((outer_ticks, 2), dtype=np.float32)
    projected = np.empty_like(actions)
    truth[0] = u
    previous = np.full(2, 0.4)
    for tick in range(outer_ticks):
        proposed = previous + rng.uniform(-0.1, 0.1, size=2)
        applied = project_box_slew(proposed, previous, 0.0, 1.0, 0.1)
        force = applied @ basis
        u = heat_cn_step(u, dx, dt, kappa, decay, force)
        truth[tick + 1] = u
        actions[tick] = proposed
        projected[tick] = applied
        previous = applied
    # Reachable steady target from an admissible nominal heater action.
    nominal = rng.uniform(0.25, 0.65, size=2)
    source = nominal @ basis
    r = kappa / dx**2
    ab = np.zeros((3, ntruth))
    ab[0, 1:] = -r
    ab[1] = 2 * r + decay
    ab[2, :-1] = -r
    goal = solve_banded((1, 1), ab, source)
    goal_step = goal.copy()
    goal_step_tick = -1
    # Half of locked nominal heat parents carry one reachable mid-episode setpoint step.
    role_index = int(meta.get("_role_index", meta.get("role_index", 1)))
    if meta.get("split") == "locked_nominal" and role_index % 2 == 0:
        nominal_after = rng.uniform(0.25, 0.65, size=2)
        source_after = nominal_after @ basis
        goal_step = solve_banded((1, 1), ab, source_after)
        goal_step_tick = outer_ticks // 2
    obs = _make_observations(truth, "heat", rng, meta.get("split") == "locked_delay_dropout")
    return {"state_true": truth, "action_proposed": actions, "action_projected": projected,
            "action_applied": projected.copy(), "goal": goal.astype(np.float32),
            "goal_step_field": goal_step.astype(np.float32),
            "goal_step_tick": np.int16(goal_step_tick),
            "kappa": np.float32(kappa), "decay": np.float32(decay), "dx": np.float32(dx), **obs}


def generate_roles(manifest_path: Path, out_root: Path, limit_per_role: int | None = None,
                   roles: tuple[str, ...] = ("train", "validation"),
                   allow_locked: bool = False, outer_ticks: int = 128) -> list[dict]:
    if any(role.startswith("locked_") for role in roles) and not allow_locked:
        raise ValueError("locked target generation requires the explicit post-calibration unlock")
    if any(role.startswith("locked_") for role in roles) and outer_ticks != 200:
        raise ValueError("locked evaluation episodes must use the source-specified 200 ticks")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    generated = []
    for pde, block in manifest["roles"].items():
        by_role: dict[str, list[dict]] = {}
        for row in block["parents"]:
            by_role.setdefault(row["split"], []).append(row)
        for role in roles:
            if role not in by_role:
                raise ValueError(f"role missing from frozen manifest: {role}")
            parents = by_role[role]
            if limit_per_role is not None:
                parents = parents[:limit_per_role]
            data = []
            start = time.perf_counter()
            for role_index, source_meta in enumerate(parents):
                meta = {**source_meta, "_role_index": role_index}
                row = (generate_burgers_parent(meta, outer_ticks=outer_ticks) if pde == "burgers"
                       else generate_heat_parent(meta, outer_ticks=outer_ticks))
                row["parent_id"] = np.asarray(meta["parent_id"])
                data.append(row)
            elapsed = time.perf_counter() - start
            arrays = {key: np.stack([row[key] for row in data]) for key in data[0]}
            arrays.update({
                "role": np.asarray([role] * len(parents)),
                "role_index": np.arange(len(parents), dtype=np.int16),
                "initial_condition_seed": np.asarray([m["initial_condition_seed"] for m in parents], dtype=np.int64),
                "physical_parameter_seed": np.asarray([m["physical_parameter_seed"] for m in parents], dtype=np.int64),
                "disturbance_seed": np.asarray([m["disturbance_seed"] for m in parents], dtype=np.int64),
            })
            target = out_root / role / pde
            target.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target / "trajectories.npz", **arrays)
            record = {"generator_version": "causal-observation-v5-200tick-locked-balanced-roles",
                      "observation_mapping_version": 2,
                      "outer_ticks": outer_ticks,
                      "role_assignment": "frozen manifest order; OOD low/high and heat step/no-step alternate exactly by role index",
                      "pde": pde, "split": role, "parent_count": len(parents), "elapsed_seconds": elapsed,
                      "seconds_per_parent": elapsed / len(parents), "file": str(target / "trajectories.npz"),
                      "fields_shape": list(arrays["state_true"].shape), "targets_generated": True}
            (target / "generation_record.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            generated.append(record)
    return generated
