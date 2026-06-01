"""Termination helpers for HANDOFF tracking and locomotion-teacher tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import euler_xyz_from_quat, quat_apply_inverse

from wbc_mjlab.observations import FEET_BODY_NAMES, KEY_BODY_NAMES, get_motion_command

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _get_robot(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> Entity:
  return env.scene[asset_cfg.name]

def handoff_motion_end(
  env: ManagerBasedRlEnv, command_name: str = "motion"
) -> torch.Tensor:
  command = get_motion_command(env, command_name)
  motion_lengths = command.motion_lib.get_motion_length(command.motion_ids)
  motion_times = env.episode_length_buf.to(dtype=torch.float32) * env.step_dt
  return motion_times >= motion_lengths


def handoff_root_height_diff(
  env: ManagerBasedRlEnv,
  command_name: str = "motion",
  threshold: float = 0.3,
) -> torch.Tensor:
  command = get_motion_command(env, command_name)
  ref_root_z = command.body_pos_w[:, 0, 2]
  robot_root_z = command.robot.data.root_link_pos_w[:, 2]
  return torch.abs(ref_root_z - robot_root_z) > threshold


def handoff_roll_limit(env: ManagerBasedRlEnv, threshold: float = 4.0) -> torch.Tensor:
  command = get_motion_command(env, "motion")
  roll, _, _ = euler_xyz_from_quat(command.robot.data.root_link_quat_w)
  return torch.abs(roll) > threshold


def handoff_pitch_limit(
  env: ManagerBasedRlEnv, threshold: float = 4.0
) -> torch.Tensor:
  command = get_motion_command(env, "motion")
  _, pitch, _ = euler_xyz_from_quat(command.robot.data.root_link_quat_w)
  return torch.abs(pitch) > threshold


def handoff_velocity_too_large(
  env: ManagerBasedRlEnv, threshold: float = 5.0
) -> torch.Tensor:
  command = get_motion_command(env, "motion")
  return torch.norm(command.robot.data.root_link_lin_vel_w, dim=-1) > threshold


def nan_state(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Detect non-finite robot state per-env (root pos / root quat / qpos / qvel).

  All other ``handoff_*`` terminations use absolute-threshold checks
  (``abs(x) > thr``) which return ``False`` when ``x`` is NaN — so a sim-
  physics divergence leaves the env alive with NaN state until the
  runner's NaN guard crashes training. This termination fires per-env on
  any non-finite component so the env resets cleanly at the next step
  boundary, allowing training to continue past sporadic divergence
  (e.g. stair-fall edge cases on a cold-start policy).

  Pair with ``torch.nan_to_num`` on rewards in the runner to mask the
  reward-bus contamination from terms computed against the NaN state
  before reset (mjlab's reward_manager runs BEFORE _reset_idx).
  """
  robot = _get_robot(env, asset_cfg)
  root_pos = robot.data.root_link_pos_w
  root_quat = robot.data.root_link_quat_w
  joint_pos = robot.data.joint_pos
  joint_vel = robot.data.joint_vel
  return ~(
    torch.isfinite(root_pos).all(dim=-1)
    & torch.isfinite(root_quat).all(dim=-1)
    & torch.isfinite(joint_pos).all(dim=-1)
    & torch.isfinite(joint_vel).all(dim=-1)
  )


def handoff_pose_fail(
  env: ManagerBasedRlEnv,
  command_name: str = "motion",
  threshold: float = 0.7,
  track_root: bool = False,
  root_tracking_threshold: float = 2.0,
) -> torch.Tensor:
  command = get_motion_command(env, command_name)
  key_body_ids = command.robot.find_bodies(KEY_BODY_NAMES, preserve_order=True)[0]
  key_body_ids = torch.tensor(key_body_ids, device=command.device, dtype=torch.long)

  robot_root_pos = command.robot.data.root_link_pos_w
  robot_root_quat = command.robot.data.root_link_quat_w
  robot_body_pos = command.robot.data.body_link_pos_w[:, key_body_ids]
  robot_body_delta = robot_body_pos - robot_root_pos[:, None, :]
  robot_body_local = quat_apply_inverse(
    robot_root_quat[:, None, :].expand(-1, len(KEY_BODY_NAMES), -1).reshape(-1, 4),
    robot_body_delta.reshape(-1, 3),
  ).reshape(env.num_envs, len(KEY_BODY_NAMES), 3)

  ref_root_pos = command.body_pos_w[:, 0]
  ref_root_quat = command.body_quat_w[:, 0]
  body_name_to_idx = {name: i for i, name in enumerate(command.cfg.body_names)}
  ref_key_ids = torch.tensor(
    [body_name_to_idx[name] for name in KEY_BODY_NAMES],
    device=command.device,
    dtype=torch.long,
  )
  ref_body_pos = command.body_pos_w[:, ref_key_ids]
  ref_body_delta = ref_body_pos - ref_root_pos[:, None, :]
  ref_body_local = quat_apply_inverse(
    ref_root_quat[:, None, :].expand(-1, len(KEY_BODY_NAMES), -1).reshape(-1, 4),
    ref_body_delta.reshape(-1, 3),
  ).reshape(env.num_envs, len(KEY_BODY_NAMES), 3)

  body_pos_diff = ref_body_local - robot_body_local
  body_pos_dist = torch.sum(torch.square(body_pos_diff), dim=-1)
  pose_fail = torch.max(body_pos_dist, dim=-1).values > threshold**2

  if track_root:
    root_pos_diff = ref_root_pos[:, :2] - robot_root_pos[:, :2]
    root_pos_dist = torch.sum(torch.square(root_pos_diff), dim=-1)
    pose_fail |= root_pos_dist > root_tracking_threshold**2

  # Match HANDOFF behavior: do not terminate on the first environment step.
  return pose_fail & (env.episode_length_buf > 0)


def foot_overlap(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold: float = 0.05,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  contact_force_threshold: float = 1.0,
  vertical_threshold: float | None = None,
) -> torch.Tensor:
  asset = _get_robot(env, asset_cfg)
  sensor: ContactSensor = env.scene[sensor_name]
  force = sensor.data.force
  assert force is not None
  contact = torch.norm(force[..., :3], dim=-1) > contact_force_threshold
  both_in_contact = torch.all(contact, dim=-1)

  if not asset_cfg.body_ids:
    feet_ids, _ = asset.find_bodies(FEET_BODY_NAMES, preserve_order=True)
    body_ids = feet_ids
  else:
    body_ids = asset_cfg.body_ids

  root_pos = asset.data.root_link_pos_w
  root_quat = asset.data.root_link_quat_w
  foot_pos = asset.data.body_link_pos_w[:, body_ids, :]
  delta = foot_pos - root_pos.unsqueeze(1)
  foot_pos_b = quat_apply_inverse(
    root_quat.unsqueeze(1).expand(-1, len(body_ids), -1).reshape(-1, 4),
    delta.reshape(-1, 3),
  ).reshape(env.num_envs, len(body_ids), 3)
  lateral = torch.abs(foot_pos_b[:, 0, 1] - foot_pos_b[:, 1, 1])
  overlap = (lateral < threshold) & both_in_contact
  if vertical_threshold is not None:
    # On stairs a valid stance can have one foot on a lower tread and the
    # other on a higher one; their body-frame Y distance is small but
    # their world Z differs by the riser. Only terminate when the feet
    # are also close in Z (i.e. on the same tread).
    vertical = torch.abs(foot_pos[:, 0, 2] - foot_pos[:, 1, 2])
    overlap = overlap & (vertical < vertical_threshold)
  return overlap
