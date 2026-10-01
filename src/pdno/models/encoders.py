from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def load_state_with_normalizer_defaults(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    """Load legacy raw-input weights while defaulting only newly added scaler buffers."""
    current = model.state_dict()
    missing = set(current) - set(state_dict)
    allowed = {key for key in missing if key.endswith(("sensor_value_mean", "sensor_value_std",
                                                        "material_context_mean", "material_context_std"))}
    unexpected = set(state_dict) - set(current)
    if missing - allowed or unexpected:
        raise RuntimeError(f"checkpoint state mismatch; missing={sorted(missing-allowed)}, unexpected={sorted(unexpected)}")
    current.update(state_dict)
    model.load_state_dict(current)


class CausalConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int, dilation: int):
        super().__init__()
        self.left_padding = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel, dilation=dilation)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv(F.pad(x, (self.left_padding, 0))))


class ObservationEncoder(nn.Module):
    """Shared causal sensor TCN + small image CNN, returning one 128-D context."""

    def __init__(self, goal_dim: int):
        super().__init__()
        self.sensor_tcn = nn.Sequential(
            CausalConvBlock(16 * 3, 64, kernel=3, dilation=1),
            CausalConvBlock(64, 64, kernel=3, dilation=2),
        )
        self.action_history = nn.Sequential(nn.Flatten(start_dim=1), nn.Linear(8 * 2, 32), nn.SiLU())
        # Statistics are fitted from train parents by the shared observer and are
        # serialized as buffers so every learned controller and replay uses the
        # exact same transform. Defaults preserve unit-test/smoke compatibility.
        self.register_buffer("sensor_value_mean", torch.zeros(1))
        self.register_buffer("sensor_value_std", torch.ones(1))
        self.register_buffer("material_context_mean", torch.zeros(8))
        self.register_buffer("material_context_std", torch.ones(8))
        self.image_cnn = nn.Sequential(
            nn.Conv2d(2, 16, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 8)), nn.Flatten(), nn.Linear(64 * 8, 64), nn.SiLU(),
        )
        fusion_in = 64 + 32 + 64 + goal_dim + 8 + 2 + 2
        self.fusion = nn.Sequential(nn.Linear(fusion_in, 256), nn.SiLU(), nn.Linear(256, 128), nn.SiLU())

    @torch.no_grad()
    def fit_input_normalizer(self, sensor_value: torch.Tensor, sensor_mask: torch.Tensor,
                             material_context: torch.Tensor) -> None:
        """Fit train-only scalar sensor and per-feature material standardization."""
        values = sensor_value.float()[sensor_mask.bool()]
        if values.numel() == 0:
            raise ValueError("cannot fit sensor normalizer without any observed sensor values")
        self.sensor_value_mean.copy_(values.mean().reshape_as(self.sensor_value_mean))
        self.sensor_value_std.copy_(values.std(unbiased=False).clamp_min(1e-6).reshape_as(self.sensor_value_std))
        material = material_context.float().reshape(-1, 8)
        self.material_context_mean.copy_(material.mean(dim=0))
        self.material_context_std.copy_(material.std(dim=0, unbiased=False).clamp_min(1e-6))

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        # NumPy least-squares goal fits can arrive as float64 while convolution
        # weights are float32. Normalize all continuous features at this boundary.
        sensor_value = ((obs["sensor_value"].float() - self.sensor_value_mean)
                        / self.sensor_value_std.clamp_min(1e-6))
        sensor_value = sensor_value * obs["sensor_mask"].float()
        sensor = torch.cat((sensor_value, obs["sensor_mask"].float(),
                            obs["sensor_age"].float() / 8.0), dim=-1)
        sensor = self.sensor_tcn(sensor.transpose(1, 2))[..., -1]
        action_history = obs.get("applied_action_history")
        if action_history is None:
            # Retain compatibility for unit fixtures; all v2 generated/replayed
            # production observations are required to provide this causal field.
            action_history = sensor.new_zeros((sensor.shape[0], 8, 2))
        action_features = self.action_history(action_history.float())
        image_mask = obs["image_mask"].float()
        image = self.image_cnn(torch.cat((obs["instrument_image"].float() * image_mask, image_mask), dim=1))
        material = ((obs["material_context"].float() - self.material_context_mean)
                    / self.material_context_std.clamp_min(1e-6))
        features = torch.cat((sensor, action_features, image, obs["goal_coefficients"].float(), material,
                              obs["previous_applied_action"].float(), obs["image_age"].float().reshape(-1, 1) / 8.0,
                              obs["image_valid"].float().reshape(-1, 1)), dim=-1)
        return self.fusion(features)
