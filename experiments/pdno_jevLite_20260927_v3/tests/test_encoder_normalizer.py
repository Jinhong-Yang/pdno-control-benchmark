import torch

from pdno.models.encoders import ObservationEncoder, load_state_with_normalizer_defaults


def test_train_fitted_sensor_and_material_normalizers_are_serialized_buffers():
    encoder = ObservationEncoder(goal_dim=32)
    sensor = torch.arange(2 * 8 * 16, dtype=torch.float32).reshape(2, 8, 16)
    mask = torch.ones_like(sensor, dtype=torch.bool)
    mask[0, 0, 0] = False
    sensor[0, 0, 0] = 1e6  # masked values must not affect fitted statistics
    material = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    encoder.fit_input_normalizer(sensor, mask, material)
    observed = sensor[mask]
    assert torch.allclose(encoder.sensor_value_mean, observed.mean().reshape(1))
    assert torch.allclose(encoder.sensor_value_std, observed.std(unbiased=False).reshape(1))
    assert torch.allclose(encoder.material_context_mean, material.mean(dim=0))
    assert torch.allclose(encoder.material_context_std, material.std(dim=0, unbiased=False))
    assert "sensor_value_mean" in encoder.state_dict()
    assert "material_context_std" in encoder.state_dict()


def test_fitted_encoder_produces_finite_context_and_zeroes_missing_normalized_sensors():
    torch.manual_seed(8)
    encoder = ObservationEncoder(goal_dim=32)
    sensor = torch.randn(3, 8, 16)
    mask = torch.rand(3, 8, 16) > 0.2
    material = torch.randn(3, 8)
    encoder.fit_input_normalizer(sensor, mask, material)
    obs = {
        "sensor_value": sensor,
        "sensor_mask": mask,
        "sensor_age": torch.zeros_like(sensor),
        "instrument_image": torch.rand(3, 1, 16, 64),
        "image_mask": torch.ones(3, 1, 16, 64, dtype=torch.bool),
        "goal_coefficients": torch.zeros(3, 32),
        "material_context": material,
        "previous_applied_action": torch.zeros(3, 2),
        "applied_action_history": torch.zeros(3, 8, 2),
        "image_age": torch.zeros(3),
        "image_valid": torch.ones(3, dtype=torch.bool),
    }
    context = encoder(obs)
    assert context.shape == (3, 128)
    assert torch.isfinite(context).all()


def test_legacy_encoder_state_defaults_only_the_new_normalizer_buffers():
    source = ObservationEncoder(goal_dim=32)
    legacy = {key: value for key, value in source.state_dict().items()
              if not key.endswith(("sensor_value_mean", "sensor_value_std",
                                   "material_context_mean", "material_context_std"))}
    target = ObservationEncoder(goal_dim=32)
    load_state_with_normalizer_defaults(target, legacy)
    assert torch.equal(target.sensor_value_mean, torch.zeros(1))
    assert torch.equal(target.sensor_value_std, torch.ones(1))
    assert torch.equal(target.material_context_mean, torch.zeros(8))
    assert torch.equal(target.material_context_std, torch.ones(8))
