import numpy as np

from pdno.data.audit_teacher import audit_teacher_shard


def _teacher_rows(path):
    n = 2
    ticks = np.asarray([8, 9], dtype=np.int16)
    history_ticks = ticks[:, None, None] - 7 + np.arange(8, dtype=np.int32)[None, :, None]
    capture = np.broadcast_to(history_ticks - 1, (n, 8, 16)).copy()
    receive = capture.copy()
    sensor_mask = np.ones((n, 8, 16), dtype=bool)
    sensor_age = np.ones((n, 8, 16), dtype=np.int16)
    image_capture = ticks - 1
    arrays = {
        "parent_id": np.asarray(["p0", "p0"]),
        "decision_tick": ticks,
        "sensor_mask": sensor_mask,
        "sensor_age": sensor_age,
        "sensor_capture_time": capture,
        "sensor_receive_time": receive,
        "image_valid": np.ones(n, dtype=bool),
        "image_age": np.ones(n, dtype=np.int16),
        "image_capture_time": image_capture,
        "image_receive_time": ticks.copy(),
        "candidate_action": np.zeros((n, 10, 2), dtype=np.float32),
        "previous_applied_action": np.zeros((n, 2), dtype=np.float32),
        "applied_action_history": np.zeros((n, 8, 2), dtype=np.float32),
        "future_field": np.zeros((n, 10, 8, 128), dtype=np.float32),
        "teacher_cost": np.zeros((n, 10), dtype=np.float32),
        "teacher_peak": np.zeros((n, 10), dtype=np.float32),
        "teacher_best_index": np.zeros(n, dtype=np.int8),
        "goal": np.zeros((n, 128), dtype=np.float32),
        "q_min": np.full(n, -1.2, dtype=np.float32),
        "q_max": np.full(n, 1.2, dtype=np.float32),
    }
    np.savez_compressed(path, **arrays)


def test_teacher_audit_checks_causal_history_and_parent_role(tmp_path):
    path = tmp_path / "teacher_queries.npz"
    _teacher_rows(path)
    result = audit_teacher_shard(path, "burgers", {"p0"})
    assert result["passed"]
    assert result["parent_count"] == 1


def test_teacher_audit_rejects_misaligned_applied_action_history(tmp_path):
    path = tmp_path / "teacher_queries.npz"
    _teacher_rows(path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["previous_applied_action"][0, 0] = 0.5
    np.savez_compressed(path, **arrays)
    result = audit_teacher_shard(path, "burgers", {"p0"})
    assert not result["passed"]
    assert any("history" in error for error in result["errors"])
