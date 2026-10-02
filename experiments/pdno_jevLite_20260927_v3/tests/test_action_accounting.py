import numpy as np

from pdno.controllers.actions import account_action


def test_proposed_projected_applied_are_explicit_and_consistent():
    result = account_action(np.array([0.8, -0.7]), np.array([0.0, 0.0]), -1.0, 1.0, 0.15)
    np.testing.assert_allclose(result.proposed, [0.8, -0.7])
    np.testing.assert_allclose(result.projected, [0.15, -0.15])
    np.testing.assert_array_equal(result.applied, result.projected)


def test_action_box_and_slew_respected_for_nonzero_previous_action():
    prev = np.array([0.95, -0.95])
    result = account_action(np.array([-5.0, 5.0]), prev, -1.0, 1.0, 0.15)
    assert np.all(result.applied <= 1.0) and np.all(result.applied >= -1.0)
    assert np.all(np.abs(result.applied - prev) <= 0.15 + 1e-12)
