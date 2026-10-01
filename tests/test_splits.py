from pdno.data.splits import assign_role, audit_parent_roles, parent_id
from pdno.data.manifests import audit_role_manifest, build_role_manifest
from pdno.data.generate import draw_burgers_nu, draw_heat_kappa
from pdno.data.generate import generate_roles
from pdno.controllers.linear import _measurements
import numpy as np
import pytest


def test_parent_identity_includes_all_independence_seeds():
    a = parent_id("burgers", 1, 2, 3)
    assert a != parent_id("burgers", 1, 2, 4)
    assert a != parent_id("heat", 1, 2, 3)


def test_role_assignment_is_stable_and_has_only_declared_roles():
    weights = {"train": 80, "validation": 10, "calibration": 10}
    roles = {assign_role(f"parent-{i}", "split-v1", weights) for i in range(1000)}
    assert roles == set(weights)
    assert assign_role("parent-a", "split-v1", weights) == assign_role("parent-a", "split-v1", weights)


def test_parent_role_audit_detects_cross_role_leakage_and_duplicates():
    rows = [
        {"parent_id": "p0", "split": "train"},
        {"parent_id": "p1", "split": "validation"},
        {"parent_id": "p1", "split": "test"},
        {"parent_id": "p0", "split": "train"},
    ]
    result = audit_parent_roles(rows)
    assert result["cross_role_parent_count"] == 1
    assert result["within_role_duplicate_rows"] == 1
    assert not result["passed"]


def test_protocol_parent_manifest_has_exact_disjoint_roles_and_withheld_targets():
    manifest = build_role_manifest()
    audit = audit_role_manifest(manifest)
    assert audit["passed"]
    assert not audit["test_opened"]
    assert audit["pdes"]["burgers"]["parent_count"] == 1280
    assert audit["pdes"]["heat"]["parent_count"] == 1024
    assert all(row["target_status"] == "WITHHELD" for p in manifest["roles"].values() for row in p["parents"] if row["split"].startswith("locked_"))


def test_v3_parent_namespace_changes_physical_identity_from_v1_and_v2():
    v2 = build_role_manifest(namespace="pdno-jevLite-parent-pool-v1")
    v3 = build_role_manifest()
    old_ids = {row["parent_id"] for block in v2["roles"].values() for row in block["parents"]}
    new_ids = {row["parent_id"] for block in v3["roles"].values() for row in block["parents"]}
    assert v3["namespace"] == "pdno-jevLite-parent-pool-v3-fresh-20260927"
    assert old_ids.isdisjoint(new_ids)
    assert v3["manifest_version"] == 3


def test_heat_classical_observer_uses_actual_sensor_cell_center_coordinates():
    obs = {
        "sensor_value": np.zeros((1, 16)),
        "sensor_mask": np.ones((1, 16), dtype=bool),
        "image_valid": False,
    }
    x, y = _measurements(obs, "heat")
    expected = np.linspace(1, 256, 16) / 257
    np.testing.assert_allclose(x, expected, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(y, np.zeros(16))


def test_locked_coefficient_ood_ranges_are_separate_from_training_ranges():
    rng = np.random.default_rng(9)
    for role_index in (0, 2, 126):
        nu = draw_burgers_nu({"split":"locked_coefficient_ood","_role_index":role_index}, rng)
        assert 0.005 <= nu <= 0.008
    for role_index in (1, 3, 127):
        nu = draw_burgers_nu({"split":"locked_coefficient_ood","_role_index":role_index}, rng)
        assert 0.035 <= nu <= 0.045
    kappa = draw_heat_kappa({"split":"locked_coefficient_ood"}, rng)
    assert 0.024 <= kappa <= 0.032
    assert 0.01 <= draw_burgers_nu({"split":"train"}, rng) <= 0.03
    assert 0.005 <= draw_heat_kappa({"split":"train"}, rng) <= 0.02


def test_locked_role_generation_requires_explicit_unlock(tmp_path):
    with pytest.raises(ValueError, match="explicit post-calibration unlock"):
        generate_roles(tmp_path / "manifest.json", tmp_path / "out", roles=("locked_nominal",))


def test_locked_role_generation_requires_source_episode_length(tmp_path):
    with pytest.raises(ValueError, match="source-specified 200 ticks"):
        generate_roles(tmp_path / "manifest.json", tmp_path / "out", roles=("locked_nominal",),
                       allow_locked=True, outer_ticks=128)
