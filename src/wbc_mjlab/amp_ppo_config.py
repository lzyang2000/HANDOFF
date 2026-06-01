"""Dataclass configs for AMP-teacher tasks.

Mirrors the existing ``RslRlOnPolicyRunnerCfg`` / ``RslRlPpoAlgorithmCfg``
shapes from mjlab so the runner can consume them via the standard
``register_mjlab_task`` flow, but exposes:

  - ``AmpAlgorithmCfg``: PPO algorithm cfg whose ``class_name`` resolves to
    ``wbc_mjlab.rl.amp_algorithm:AmpPPO``, plus AMP-specific loss coefficients.
  - ``AmpRunnerCfg``: runner cfg whose ``class_name`` resolves to
    ``wbc_mjlab.rl.amp_runner:AmpOnPolicyRunner`` and which carries top-level
    AMP knobs (motion paths, tracked body names, anchor body, etc.).
"""


from dataclasses import dataclass, field
from typing import Tuple

from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@dataclass
class AmpAlgorithmCfg(RslRlPpoAlgorithmCfg):
  """PPO + discriminator loss coefficients."""

  amp_loss_coef: float = 1.0
  """Weight for the discriminator loss (and grad-pen) in the total loss."""
  amp_grad_pen_lambda: float = 10.0
  """Gradient penalty coefficient for WGAN-style stabilization."""
  amp_trunk_weight_decay: float = 1.0e-3
  """Weight decay applied to the discriminator trunk (matches AMP_mjlab)."""
  amp_head_weight_decay: float = 1.0e-2
  """Weight decay applied to the discriminator head (matches AMP_mjlab)."""
  class_name: str = "wbc_mjlab.rl.amp_algorithm:AmpPPO"


@dataclass
class AmpRunnerCfg(RslRlOnPolicyRunnerCfg):
  """Runner cfg for AMP-teacher training."""

  class_name: str = "wbc_mjlab.rl.amp_runner:AmpOnPolicyRunner"

  # AMP-specific runner inputs (read by AmpPPO.construct_algorithm).
  amp_motion_files: str = ""
  """Path to a single .npz or directory of .npz expert motion clips."""
  amp_body_names: Tuple[str, ...] = ()
  """Tracked body names for the AMP discriminator obs."""
  amp_anchor_name: str = ""
  """Anchor body for the body-local frame used by AMP obs."""
  amp_reward_coef: float = 0.1
  """Scale on the discriminator-derived reward (matches AMP_mjlab)."""
  amp_task_reward_lerp: float = 0.75
  """Blend ``(1-l) * disc_reward + l * task_reward`` per step (matches AMP_mjlab)."""
  amp_discr_hidden_dims: Tuple[int, ...] = (1024, 512, 256)
  """Hidden layer sizes for the discriminator MLP trunk."""
  amp_replay_buffer_size: int = 100_000
  """Capacity of the on-policy AMP transition replay buffer."""

  algorithm: AmpAlgorithmCfg = field(default_factory=AmpAlgorithmCfg)
