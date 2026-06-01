"""Custom action terms for the locomotion-teacher port."""

from __future__ import annotations

from dataclasses import dataclass
import re

import torch

from mjlab.asset_zoo.robots import G1_ACTION_SCALE
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand

from wbc_mjlab.commands import LocoArmMotionCommand
from wbc_mjlab.g1_constants_custom import (
  CBF_ALPHA_TRAINING,
  CBF_H_TARGET,
  CBF_VEL_MAX_ANKLE_TRAINING,
  CBF_VEL_MAX_WAIST_TRAINING,
)
from wbc_mjlab.pkl_motion_lib import PklMotionLib


@dataclass(kw_only=True)
class G1LocoTeacherActionCfg(ActionTermCfg):
  """15-DoF locomotion teacher action term with motion-driven arms."""

  body_joint_names: tuple[str, ...]
  arm_joint_names: tuple[str, ...]
  motion_file: str | None = None
  motion_command_name: str = "motion"
  motion_body_names: tuple[str, ...] = ("torso_link",)
  scale: float | dict[str, float] = 1.0
  use_default_offset: bool = True
  init_blend: float = 0.0
  curriculum_start_step: int = 12500 * 24
  curriculum_step: float = 0.002
  curriculum_threshold_lin: float = 0.7
  curriculum_threshold_ang: float = 0.45
  standing_command_name: str = "twist"
  tracking_lin_reward_name: str = "track_linear_velocity"
  tracking_ang_reward_name: str = "track_angular_velocity"
  # When True, arm targets come from the motion command term's current
  # joint_pos (aligned with the pose the motion reference expects). When
  # False, arm targets are replayed from a random motion per env (the
  # original loco-teacher training behavior).
  arms_from_motion_command: bool = False

  def build(self, env) -> "G1LocoTeacherAction":
    return G1LocoTeacherAction(self, env)


class G1LocoTeacherAction(ActionTerm):
  """Expands a 15-DoF body policy action into full 29-joint position targets."""

  cfg: G1LocoTeacherActionCfg

  def __init__(self, cfg: G1LocoTeacherActionCfg, env):
    super().__init__(cfg=cfg, env=env)

    body_ids, body_names = self._entity.find_joints(
      cfg.body_joint_names, preserve_order=True
    )
    arm_ids, arm_names = self._entity.find_joints(
      cfg.arm_joint_names, preserve_order=True
    )
    self._body_joint_ids = torch.tensor(body_ids, device=self.device, dtype=torch.long)
    self._arm_joint_ids = torch.tensor(arm_ids, device=self.device, dtype=torch.long)
    self._joint_ids = torch.cat((self._body_joint_ids, self._arm_joint_ids))
    self._body_joint_names = body_names
    self._arm_joint_names = arm_names
    self._action_dim = len(body_ids)

    self._raw_actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)
    self._processed_body_targets = torch.zeros_like(self._raw_actions)

    default_joint_pos = self._entity.data.default_joint_pos
    self._body_default_pos = default_joint_pos[:, self._body_joint_ids].clone()
    self._arm_default_pos = default_joint_pos[:, self._arm_joint_ids].clone()
    self._arm_target_pos = self._arm_default_pos.clone()
    self._joint_target_pos = default_joint_pos[:, self._joint_ids].clone()

    self._scale = self._resolve_scale(cfg.scale, self._body_joint_names)
    self._clip = self._resolve_clip(cfg.clip, self._body_joint_names)

    self._arm_blend_factor = float(cfg.init_blend)
    if cfg.arms_from_motion_command:
      # Arms are read from the motion command each step — no need for an
      # independent motion library / random motion tracking per env.
      self._motion_lib = None
      self._motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
      self._motion_times = torch.zeros(self.num_envs, device=self.device)
    else:
      self._motion_lib = self._resolve_motion_lib()
      self._motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
      self._motion_times = torch.zeros(self.num_envs, device=self.device)
      self._resample_motion(torch.arange(self.num_envs, device=self.device))
    self._refresh_arm_targets(env_ids=slice(None))

  @property
  def action_dim(self) -> int:
    return self._action_dim

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  @property
  def arm_blend_factor(self) -> float:
    return self._arm_blend_factor

  def set_arm_blend_factor(self, value: float) -> None:
    self._arm_blend_factor = min(max(float(value), 0.0), 1.0)

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw_actions[:] = actions
    self._processed_body_targets = self._body_default_pos + self._raw_actions * self._scale
    if self._clip is not None:
      self._processed_body_targets = torch.clamp(
        self._processed_body_targets,
        min=self._clip[:, :, 0],
        max=self._clip[:, :, 1],
      )
    self._refresh_arm_targets(env_ids=slice(None))
    self._joint_target_pos = torch.cat(
      (self._processed_body_targets, self._arm_target_pos), dim=-1
    )

  def apply_actions(self) -> None:
    encoder_bias = self._entity.data.encoder_bias[:, self._joint_ids]
    target = self._joint_target_pos - encoder_bias
    self._entity.set_joint_position_target(target, joint_ids=self._joint_ids)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)

    env_ids_tensor = self._slice_to_env_ids(env_ids)
    if env_ids_tensor.numel() > 0:
      if not self.cfg.arms_from_motion_command:
        self._resample_motion(env_ids_tensor)
      self._raw_actions[env_ids] = 0.0
      self._processed_body_targets[env_ids] = self._body_default_pos[env_ids]
      self._refresh_arm_targets(env_ids)
      self._joint_target_pos[env_ids] = torch.cat(
        (self._processed_body_targets[env_ids], self._arm_target_pos[env_ids]), dim=-1
      )

  def _resolve_scale(
    self, scale: float | dict[str, float], joint_names: list[str]
  ) -> torch.Tensor:
    if isinstance(scale, (float, int)):
      return torch.full(
        (self.num_envs, self._action_dim), float(scale), device=self.device
      )
    values = torch.ones(self._action_dim, device=self.device)
    for pattern, value in scale.items():
      for idx, joint_name in enumerate(joint_names):
        if re.fullmatch(pattern, joint_name):
          values[idx] = float(value)
    return values.unsqueeze(0).repeat(self.num_envs, 1)

  def _resolve_clip(
    self, clip: dict[str, tuple] | None, joint_names: list[str]
  ) -> torch.Tensor | None:
    if clip is None:
      return None
    bounds = torch.tensor(
      [[-float("inf"), float("inf")]], device=self.device
    ).repeat(self.num_envs, self._action_dim, 1)
    for pattern, value in clip.items():
      for idx, joint_name in enumerate(joint_names):
        if re.fullmatch(pattern, joint_name):
          bounds[:, idx] = torch.tensor(value, device=self.device)
    return bounds

  def _slice_to_env_ids(self, env_ids: torch.Tensor | slice) -> torch.Tensor:
    if isinstance(env_ids, slice):
      return torch.arange(self.num_envs, device=self.device)[env_ids]
    return env_ids

  def _resample_motion(self, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
      return
    self._motion_ids[env_ids] = self._motion_lib.sample_motions(env_ids.numel())
    lengths = self._motion_lib.get_motion_length(self._motion_ids[env_ids])
    episode_len = float(self._env.cfg.episode_length_s)
    margin = (lengths - episode_len).clamp(min=0.0)
    self._motion_times[env_ids] = torch.rand(env_ids.numel(), device=self.device) * margin

  def _refresh_arm_targets(self, env_ids: torch.Tensor | slice | None) -> None:
    twist_cmd = self._get_twist_command()
    active_mask = (~twist_cmd.is_standing_env).float().unsqueeze(-1)

    if self.cfg.arms_from_motion_command:
      # Read arm targets from the motion command term's current joint_pos
      # so arms stay aligned with the pose that pose_fail termination checks.
      motion_cmd = self._env.command_manager.get_term(self.cfg.motion_command_name)
      motion_joint_pos = motion_cmd.joint_pos  # type: ignore[attr-defined]
      motion_arm_pos = motion_joint_pos[:, self._arm_joint_ids]
    else:
      lengths = self._motion_lib.get_motion_length(self._motion_ids)
      max_times = torch.clamp(lengths - 1.0e-6, min=0.0)
      self._motion_times = torch.minimum(
        self._motion_times + self._env.step_dt * active_mask.squeeze(-1), max_times
      )
      frame = self._motion_lib.get_frame(self._motion_ids, self._motion_times)
      motion_arm_pos = frame.joint_pos[:, self._arm_joint_ids]

    blended = self._arm_default_pos + self._arm_blend_factor * (
      motion_arm_pos - self._arm_default_pos
    )
    targets = self._arm_default_pos + active_mask * (blended - self._arm_default_pos)
    if env_ids is None:
      self._arm_target_pos[:] = targets
    else:
      self._arm_target_pos[env_ids] = targets[env_ids]

  def _get_twist_command(self) -> UniformVelocityCommand:
    return self._env.command_manager.get_term(self.cfg.standing_command_name)

  def _resolve_motion_lib(self) -> PklMotionLib:
    motion_term = self._env.command_manager.get_term(self.cfg.motion_command_name)
    if isinstance(motion_term, LocoArmMotionCommand):
      return motion_term.motion_lib
    if self.cfg.motion_file:
      return PklMotionLib(
        motion_file=self.cfg.motion_file,
        body_names=self.cfg.motion_body_names,
        device=self.device,
      )
    raise ValueError(
      "Locomotion teacher action requires either a compatible motion command term "
      f"named '{self.cfg.motion_command_name}' or cfg.motion_file to be set."
    )


DEFAULT_G1_LOCO_TEACHER_ACTION_SCALE = G1_ACTION_SCALE


# ---------------------------------------------------------------------------
# CBF-filtered joint position action (CBF task variant only)
# ---------------------------------------------------------------------------

from dataclasses import field as dataclass_field
from typing import TYPE_CHECKING

from mjlab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg
from wbc_mjlab.observations import motion_loco_command




