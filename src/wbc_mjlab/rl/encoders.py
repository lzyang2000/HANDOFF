"""Temporal encoders for HANDOFF-style flat observation layouts."""

from __future__ import annotations

import torch
from torch import nn

from rsl_rl.utils import resolve_nn_activation


def _build_temporal_conv(
  activation: nn.Module,
  tsteps: int,
  channel_size: int,
) -> nn.Module:
  if tsteps == 1:
    return nn.Flatten()
  if tsteps in (10, 11):
    return nn.Sequential(
      nn.Conv1d(3 * channel_size, 2 * channel_size, kernel_size=4, stride=2),
      activation,
      nn.Conv1d(2 * channel_size, channel_size, kernel_size=2, stride=1),
      activation,
      nn.Flatten(),
    )
  if tsteps == 20:
    return nn.Sequential(
      nn.Conv1d(3 * channel_size, 2 * channel_size, kernel_size=6, stride=2),
      activation,
      nn.Conv1d(2 * channel_size, channel_size, kernel_size=4, stride=2),
      activation,
      nn.Flatten(),
    )
  if tsteps == 50:
    return nn.Sequential(
      nn.Conv1d(3 * channel_size, 2 * channel_size, kernel_size=8, stride=4),
      activation,
      nn.Conv1d(2 * channel_size, channel_size, kernel_size=5, stride=1),
      activation,
      nn.Conv1d(channel_size, channel_size, kernel_size=5, stride=1),
      activation,
      nn.Flatten(),
    )
  raise ValueError(f"Unsupported temporal encoder steps: {tsteps}")


class TemporalConvEncoder(nn.Module):
  """1D-conv temporal encoder for flat or sequence-shaped temporal observations."""

  def __init__(
    self,
    input_size: int,
    tsteps: int,
    output_size: int,
    activation: str,
  ) -> None:
    super().__init__()
    self.tsteps = tsteps
    self.input_size = input_size
    act = resolve_nn_activation(activation)
    channel_size = 20

    self.proj = nn.Sequential(nn.Linear(input_size, 3 * channel_size), act)
    self.temporal = _build_temporal_conv(act, tsteps, channel_size)
    self.out = nn.Linear(channel_size * 3, output_size)

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    if obs.dim() == 2:
      batch = obs.shape[0]
      obs = obs.reshape(batch, self.tsteps, self.input_size)
    elif obs.dim() == 3:
      batch = obs.shape[0]
      if obs.shape[1] != self.tsteps or obs.shape[2] != self.input_size:
        raise ValueError(
          "TemporalConvEncoder received a sequence with unexpected shape: "
          f"expected [B, {self.tsteps}, {self.input_size}], got {tuple(obs.shape)}"
        )
    else:
      raise ValueError(
        "TemporalConvEncoder expects a [B, T*D] or [B, T, D] tensor, "
        f"got {tuple(obs.shape)}"
      )

    projected = self.proj(obs)
    latent = self.temporal(projected.transpose(1, 2))
    return self.out(latent)


class DepthCNN2D(nn.Module):
  """Small 2D-conv encoder for pooled depth maps.

  Expects input shape [B, C, H, W] (or flat [B, C*H*W] auto-reshaped). Default
  conv stack is tuned for the stair teacher's 18x32 front_depth after 2x2 pool:
  three conv layers with stride 2 on the latter two, then a linear projection.
  """

  def __init__(
    self,
    input_shape: tuple[int, int, int] = (1, 18, 32),
    output_size: int = 64,
    activation: str = "elu",
    channels: tuple[int, int, int] = (8, 16, 32),
  ) -> None:
    super().__init__()
    c_in, h, w = input_shape
    self.input_shape = input_shape
    act = resolve_nn_activation(activation)
    c1, c2, c3 = channels
    self.conv = nn.Sequential(
      nn.Conv2d(c_in, c1, kernel_size=3, stride=1, padding=1),
      act,
      nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
      resolve_nn_activation(activation),
      nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1),
      resolve_nn_activation(activation),
      nn.Flatten(),
    )
    # Derive the post-conv flat size with a dummy forward pass so
    # stride/padding changes don't break the linear shape.
    with torch.no_grad():
      flat_dim = self.conv(torch.zeros(1, *input_shape)).shape[1]
    self.proj = nn.Sequential(
      nn.Linear(flat_dim, output_size),
      resolve_nn_activation(activation),
    )

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    if obs.dim() == 2:
      obs = obs.reshape(obs.shape[0], *self.input_shape)
    elif obs.dim() != 4:
      raise ValueError(
        "DepthCNN2D expects a [B, C*H*W] or [B, C, H, W] tensor, "
        f"got {tuple(obs.shape)}"
      )
    return self.proj(self.conv(obs))


class FutureMotionEncoder(nn.Module):
  """Simple MLP encoder over flattened future motion observations."""

  def __init__(
    self,
    input_size: int,
    output_size: int,
    activation: str,
    hidden_dims: tuple[int, ...] | list[int] = (256, 128),
    dropout: float = 0.1,
  ) -> None:
    super().__init__()
    act_name = activation
    layers: list[nn.Module] = []
    in_dim = input_size
    for hidden_dim in hidden_dims:
      layers.append(nn.Linear(in_dim, hidden_dim))
      layers.append(resolve_nn_activation(act_name))
      if dropout > 0:
        layers.append(nn.Dropout(dropout))
      in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, output_size))
    self.encoder = nn.Sequential(*layers)

    for layer in self.encoder:
      if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight, gain=0.5)
        nn.init.zeros_(layer.bias)

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    return self.encoder(obs)
