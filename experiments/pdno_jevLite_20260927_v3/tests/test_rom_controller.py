import numpy as np

from pdno.controllers.rom import rom_candidate_costs, rom_select_action


def _heat_observation():
    x = (np.arange(16) + 1) / 17
    values = np.sin(np.pi * x)[None, :].astype(np.float32)
    return {
        "sensor_value": values,
        "sensor_mask": np.ones_like(values, dtype=bool),
        "sensor_age": np.zeros_like(values),
        "instrument_image": np.zeros((1, 16, 64), dtype=np.float32),
        "image_mask": np.zeros((1, 16, 64), dtype=bool),
        "image_valid": False,
        "material_context": np.asarray([0.01, 0.05, 0, 1, 0.10, 0.02, 0.16, 128], dtype=np.float32),
        "previous_applied_action": np.asarray([0.4, 0.4], dtype=np.float32),
    }


def test_b1_heat_predicts_same_ten_candidates_and_selects_feasible_action():
    obs = _heat_observation()
    candidates = np.asarray([[0.4 + a * 0.1, 0.4 + b * 0.1]
                             for a, b in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0),
                                          (0, 1), (1, -1), (1, 0), (1, 1), (0, 0))])
    goal = np.sin(np.pi * (np.arange(1, 129) / 129))
    costs, fields = rom_candidate_costs(obs, "heat", candidates, goal, 1.0)
    action, selected_costs = rom_select_action(obs, "heat", candidates, goal, 1.0)
    assert costs.shape == selected_costs.shape == (10,)
    assert fields.shape == (10, 8, 128)
    assert np.isfinite(costs).all() and np.isfinite(fields).all()
    assert np.array_equal(action, candidates[int(np.argmin(costs))])
    assert np.all(np.abs(action - obs["previous_applied_action"]) <= 0.100001)


def test_b1_burgers_galerkin_candidate_rollout_is_finite():
    obs = _heat_observation()
    x = np.arange(16) / 16
    obs["sensor_value"] = np.sin(2 * np.pi * x)[None, :].astype(np.float32)
    obs["previous_applied_action"] = np.zeros(2, dtype=np.float32)
    obs["material_context"] = np.asarray([0.02, -1, 1, 0.15, 0.02, 0.16, 1.2, 128], dtype=np.float32)
    candidates = np.asarray([[a, b] for a, b in ((-0.15, -0.15), (-0.15, 0), (-0.15, 0.15),
                                                   (0, -0.15), (0, 0), (0, 0.15),
                                                   (0.15, -0.15), (0.15, 0), (0.15, 0.15), (0, 0))])
    costs, fields = rom_candidate_costs(obs, "burgers", candidates, np.zeros(128), 1.2)
    assert costs.shape == (10,)
    assert fields.shape == (10, 8, 128)
    assert np.isfinite(costs).all() and np.isfinite(fields).all()
