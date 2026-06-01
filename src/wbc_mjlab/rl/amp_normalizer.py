"""Running mean/std normalizer for AMP observations.

Ported from AMP_mjlab/rsl_rl/utils/utils.py (RunningMeanStd + Normalizer).
"""

from __future__ import annotations

import numpy as np
import torch


class RunningMeanStd:
  """Welford-style streaming mean/var.

  https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
  """

  def __init__(self, epsilon: float = 1e-4, shape: tuple[int, ...] = ()) -> None:
    self.mean = np.zeros(shape, np.float64)
    self.var = np.ones(shape, np.float64)
    self.count = epsilon

  def update(self, arr: np.ndarray) -> None:
    batch_mean = np.mean(arr, axis=0)
    batch_var = np.var(arr, axis=0)
    batch_count = arr.shape[0]
    self.update_from_moments(batch_mean, batch_var, batch_count)

  def update_from_moments(
    self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int
  ) -> None:
    delta = batch_mean - self.mean
    tot_count = self.count + batch_count

    new_mean = self.mean + delta * batch_count / tot_count
    m_a = self.var * self.count
    m_b = batch_var * batch_count
    m_2 = (
      m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
    )
    new_var = m_2 / (self.count + batch_count)

    self.mean = new_mean
    self.var = new_var
    self.count = batch_count + self.count


class Normalizer(RunningMeanStd):
  """RunningMeanStd with torch-friendly normalization for AMP obs."""

  def __init__(
    self, input_dim: int | tuple[int, ...], epsilon: float = 1e-4, clip_obs: float = 10.0
  ) -> None:
    super().__init__(shape=input_dim if isinstance(input_dim, tuple) else (input_dim,))
    self.epsilon = epsilon
    self.clip_obs = clip_obs

  def normalize(self, x: np.ndarray) -> np.ndarray:
    return np.clip(
      (x - self.mean) / np.sqrt(self.var + self.epsilon),
      -self.clip_obs,
      self.clip_obs,
    )

  def normalize_torch(self, x: torch.Tensor, device: str | torch.device) -> torch.Tensor:
    mean = torch.tensor(self.mean, device=device, dtype=torch.float32)
    std = torch.sqrt(
      torch.tensor(self.var + self.epsilon, device=device, dtype=torch.float32)
    )
    return torch.clamp((x - mean) / std, -self.clip_obs, self.clip_obs)

  def update_normalizer(self, rollouts, expert_loader) -> None:
    """Update from a mix of policy + expert AMP observations."""
    policy_gen = rollouts.feed_forward_generator_amp(
      None, mini_batch_size=expert_loader.batch_size
    )
    expert_gen = expert_loader.dataset.feed_forward_generator_amp(
      expert_loader.batch_size
    )
    for expert_batch, policy_batch in zip(expert_gen, policy_gen):
      self.update(
        torch.vstack(tuple(policy_batch) + tuple(expert_batch)).cpu().numpy()
      )
