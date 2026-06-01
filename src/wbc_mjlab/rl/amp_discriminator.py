"""AMP discriminator: classifies (state, next_state) pairs as expert vs. policy.

Ported from AMP_mjlab/rsl_rl/modules/discriminator.py.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class Discriminator(nn.Module):
  """MLP that scores motion-likeness of consecutive AMP-obs pairs."""

  def __init__(
    self,
    input_dim: int,
    amp_reward_coef: float = 0.1,
    hidden_layer_sizes: tuple[int, ...] = (1024, 512, 256),
    device: str = "cpu",
    task_reward_lerp: float = 0.75,
  ) -> None:
    super().__init__()
    self.device = device
    self.input_dim = input_dim
    self.amp_reward_coef = amp_reward_coef
    self.task_reward_lerp = task_reward_lerp

    layers: list[nn.Module] = []
    prev = input_dim
    for size in hidden_layer_sizes:
      layers.append(nn.Linear(prev, size))
      layers.append(nn.ReLU())
      prev = size
    self.trunk = nn.Sequential(*layers).to(device)
    self.amp_linear = nn.Linear(hidden_layer_sizes[-1], 1).to(device)

    self.trunk.train()
    self.amp_linear.train()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    h = self.trunk(x)
    return self.amp_linear(h)

  def compute_grad_pen(
    self,
    expert_state: torch.Tensor,
    expert_next_state: torch.Tensor,
    lambda_: float = 10.0,
  ) -> torch.Tensor:
    expert_data = torch.cat([expert_state, expert_next_state], dim=-1)
    expert_data.requires_grad = True

    disc = self.amp_linear(self.trunk(expert_data))
    ones = torch.ones(disc.size(), device=disc.device)
    grad = torch.autograd.grad(
      outputs=disc,
      inputs=expert_data,
      grad_outputs=ones,
      create_graph=True,
      retain_graph=True,
      only_inputs=True,
    )[0]

    grad_pen = lambda_ * (grad.norm(2, dim=1).pow(2)).mean()
    return grad_pen

  def predict_amp_reward(
    self,
    state: torch.Tensor,
    next_state: torch.Tensor,
    task_reward: torch.Tensor,
    normalizer=None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
      self.eval()
      if normalizer is not None:
        state = normalizer.normalize_torch(state, self.device)
        next_state = normalizer.normalize_torch(next_state, self.device)

      d = self.amp_linear(self.trunk(torch.cat([state, next_state], dim=-1)))
      reward = self.amp_reward_coef * torch.clamp(
        1 - (1 / 4) * torch.square(d - 1), min=0
      )
      if self.task_reward_lerp > 0:
        reward = self._lerp_reward(reward.squeeze(), task_reward)
      self.train()
    return reward.squeeze(), d

  def _lerp_reward(
    self, disc_r: torch.Tensor, task_r: torch.Tensor
  ) -> torch.Tensor:
    return (1.0 - self.task_reward_lerp) * disc_r + self.task_reward_lerp * task_r
