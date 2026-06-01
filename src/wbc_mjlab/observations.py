"""Observation helpers for tracking and locomotion-teacher tasks."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, cast

import torch

from mjlab.envs import mdp as env_mdp
from mjlab.utils.lab_api.math import (
  axis_angle_from_quat,
  euler_xyz_from_quat,
  quat_apply_inverse,
  quat_inv,
  quat_mul,
  yaw_quat,
)
from mjlab.utils.lab_api.math import wrap_to_pi
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand

from wbc_mjlab.commands import PklMotionCommand
from wbc_mjlab.events import HAND_FORCE_EVENT_NAME, HAND_FORCE_OBS_DIM, get_event_term

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


PRIV_FUTURE_STEP_OFFSETS: tuple[int, ...] = (
  1,
  5,
  10,
  15,
  20,
  25,
  30,
  35,
  40,
  45,
  50,
  55,
  60,
  65,
  70,
  75,
  80,
  85,
  90,
  95,
)

KEY_BODY_NAMES: tuple[str, ...] = (
  "left_wrist_yaw_link",
  "right_wrist_yaw_link",
  "left_ankle_roll_link",
  "right_ankle_roll_link",
  "left_knee_link",
  "right_knee_link",
  "left_elbow_link",
  "right_elbow_link",
  "torso_link",
)
HAND_BODY_NAMES: tuple[str, ...] = ("left_wrist_yaw_link", "right_wrist_yaw_link")

FOOT_GEOM_PATTERN = r"^(left|right)_foot[1-7]_collision$"
NUM_G1_JOINTS = 29
NUM_LOCO_ACTIONS = 15
LOCO_OBS_DIM = 89
LOCO_BASE_LIN_VEL_SCALE = 2.0
LOCO_BASE_ANG_VEL_SCALE = 0.25
LOCO_JOINT_POS_SCALE = 1.0
LOCO_JOINT_VEL_SCALE = 0.05
LOCO_HEIGHT_CMD_CENTER = 0.64
LOCO_HEIGHT_CMD_SCALE = 1.0 / 0.14  # maps [0.5, 0.78] → [-1, 1]
LOCO_GAIT_PERIOD = float(os.environ.get("WBC_LOCO_GAIT_PERIOD", 1.0))
LOCO_GAIT_OFFSET = 0.5
LOCO_STANCE_RATIO = 0.55
BODY_JOINT_NAMES: tuple[str, ...] = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
)
ARM_JOINT_NAMES: tuple[str, ...] = (
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)
FEET_BODY_NAMES: tuple[str, ...] = ("left_ankle_roll_link", "right_ankle_roll_link")
KNEE_BODY_NAMES: tuple[str, ...] = (
  "left_knee_link",
  "left_hip_yaw_link",
  "right_knee_link",
  "right_hip_yaw_link",
)
ACTOR_MIMIC_DIM = 35
ACTOR_PROPRIO_DIM = 92
ACTOR_HISTORY_FEATURE_DIM = ACTOR_MIMIC_DIM + ACTOR_PROPRIO_DIM
ACTOR_HISTORY_LENGTH = 11
STAIR_DEPTH_FRAME_DIM = 18 * 32  # pooled front_depth: 18x32 = 576 dims/frame
HAND_STUDENT_MIMIC_DIM = 14
HAND_STUDENT_ACTOR_CURRENT_DIM = HAND_STUDENT_MIMIC_DIM + ACTOR_PROPRIO_DIM
HAND_STUDENT_HISTORY_FEATURE_DIM = HAND_STUDENT_ACTOR_CURRENT_DIM
LOCO15_PROPRIO_DIM = ACTOR_PROPRIO_DIM - (NUM_G1_JOINTS - NUM_LOCO_ACTIONS)  # last_action is 15 not 29
LOCO15_ACTOR_CURRENT_DIM = HAND_STUDENT_MIMIC_DIM + LOCO15_PROPRIO_DIM
LOCO15_HISTORY_FEATURE_DIM = LOCO15_ACTOR_CURRENT_DIM
CRITIC_PRIV_STEP_DIM = 21 + NUM_G1_JOINTS + 3 * len(KEY_BODY_NAMES)
CRITIC_PRIV_STEPS = len(PRIV_FUTURE_STEP_OFFSETS)


def get_motion_command(
  env: ManagerBasedRlEnv, command_name: str
) -> PklMotionCommand:
  return cast(PklMotionCommand, env.command_manager.get_term(command_name))


def tracked_body_indices(command: PklMotionCommand) -> torch.Tensor:
  return torch.tensor(
    [command.cfg.body_names.index(name) for name in KEY_BODY_NAMES],
    device=command.device,
    dtype=torch.long,
  )


def _command_body_indices(
  command: PklMotionCommand,
  body_names: tuple[str, ...],
) -> torch.Tensor:
  return torch.tensor(
    [command.cfg.body_names.index(name) for name in body_names],
    device=command.device,
    dtype=torch.long,
  )


def _robot_body_indices(
  env: ManagerBasedRlEnv,
  body_names: tuple[str, ...],
  command_name: str = "motion",
) -> torch.Tensor:
  robot = _get_robot(env, command_name)
  body_ids, _ = robot.find_bodies(body_names, preserve_order=True)
  return torch.tensor(body_ids, device=env.device, dtype=torch.long)


def _reference_root_state(
  command: PklMotionCommand,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  return (
    command.body_pos_w[:, 0],
    command.body_quat_w[:, 0],
    command.body_lin_vel_w[:, 0],
    command.body_ang_vel_w[:, 0],
  )


def _robot_root_state(
  command: PklMotionCommand,
) -> tuple[torch.Tensor, torch.Tensor]:
  return command.robot_body_pos_w[:, 0], command.robot_body_quat_w[:, 0]


def _body_positions_in_root_frame(
  root_pos_w: torch.Tensor,
  root_quat_w: torch.Tensor,
  body_pos_w: torch.Tensor,
) -> torch.Tensor:
  delta_w = body_pos_w - root_pos_w[:, None, :]
  root_yaw = yaw_quat(root_quat_w)[:, None, :].expand(-1, body_pos_w.shape[1], -1)
  return quat_apply_inverse(
    root_yaw.reshape(-1, 4),
    delta_w.reshape(-1, 3),
  ).reshape(body_pos_w.shape[0], body_pos_w.shape[1], 3)


def _get_robot(env: ManagerBasedRlEnv, command_name: str = "motion"):
  return get_motion_command(env, command_name).robot


def _torso_body_index(env: ManagerBasedRlEnv, command_name: str = "motion") -> int:
  robot = _get_robot(env, command_name)
  torso_body_ids, _ = robot.find_bodies(("torso_link",), preserve_order=True)
  return int(robot.indexing.body_ids[torso_body_ids[0]].item())


def _foot_geom_indices(
  env: ManagerBasedRlEnv, command_name: str = "motion"
) -> torch.Tensor:
  robot = _get_robot(env, command_name)
  foot_geom_ids, _ = robot.find_geoms(FOOT_GEOM_PATTERN)
  return robot.indexing.geom_ids[foot_geom_ids]


def _ctrl_ids(env: ManagerBasedRlEnv, command_name: str = "motion") -> torch.Tensor:
  return _get_robot(env, command_name).indexing.ctrl_ids


def get_velocity_command(
  env: ManagerBasedRlEnv, command_name: str = "twist"
) -> UniformVelocityCommand:
  return cast(UniformVelocityCommand, env.command_manager.get_term(command_name))




def loco_leg_phase(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  gait_period: float = LOCO_GAIT_PERIOD,
  gait_offset: float = LOCO_GAIT_OFFSET,
) -> torch.Tensor:
  walk_phase = (env.episode_length_buf.to(torch.float32) * env.step_dt) % gait_period
  walk_phase = walk_phase / gait_period
  twist_cmd = get_velocity_command(env, command_name)
  left = torch.where(twist_cmd.is_standing_env, torch.zeros_like(walk_phase), walk_phase)
  right = torch.where(
    twist_cmd.is_standing_env,
    torch.zeros_like(walk_phase),
    (walk_phase + gait_offset) % 1.0,
  )
  return torch.stack((left, right), dim=-1)


def loco_phase_features(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  gait_period: float = LOCO_GAIT_PERIOD,
  gait_offset: float = LOCO_GAIT_OFFSET,
) -> torch.Tensor:
  phase = loco_leg_phase(
    env,
    command_name=command_name,
    gait_period=gait_period,
    gait_offset=gait_offset,
  )
  two_pi_phase = 2.0 * torch.pi * phase
  return torch.cat(
    (
      torch.sin(two_pi_phase[:, 0:1]),
      torch.cos(two_pi_phase[:, 0:1]),
      torch.sin(two_pi_phase[:, 1:2]),
      torch.cos(two_pi_phase[:, 1:2]),
    ),
    dim=-1,
  )




def motion_root_vel_xy_b(
  env: ManagerBasedRlEnv, command_name: str = "motion"
) -> torch.Tensor:
  command = get_motion_command(env, command_name)
  _, root_quat_w, root_lin_vel_w, _ = _reference_root_state(command)
  root_lin_vel_b = quat_apply_inverse(root_quat_w, root_lin_vel_w)
  return root_lin_vel_b[:, :2]


def motion_root_z(
  env: ManagerBasedRlEnv, command_name: str = "motion"
) -> torch.Tensor:
  command = get_motion_command(env, command_name)
  root_pos_w, _, _, _ = _reference_root_state(command)
  return root_pos_w[:, 2:3]


def motion_root_roll_pitch(
  env: ManagerBasedRlEnv, command_name: str = "motion"
) -> torch.Tensor:
  command = get_motion_command(env, command_name)
  _, root_quat_w, _, _ = _reference_root_state(command)
  roll, pitch, _ = euler_xyz_from_quat(root_quat_w)
  return torch.stack((roll, pitch), dim=-1)




def motion_root_yaw_ang_vel_b(
  env: ManagerBasedRlEnv, command_name: str = "motion"
) -> torch.Tensor:
  command = get_motion_command(env, command_name)
  _, root_quat_w, _, root_ang_vel_w = _reference_root_state(command)
  root_ang_vel_b = quat_apply_inverse(root_quat_w, root_ang_vel_w)
  return root_ang_vel_b[:, 2:3]


def motion_joint_pos(
  env: ManagerBasedRlEnv, command_name: str = "motion"
) -> torch.Tensor:
  return get_motion_command(env, command_name).joint_pos


def motion_hand_pos_b_tensor(
  env: ManagerBasedRlEnv,
  command_name: str = "motion",
  body_names: tuple[str, ...] = HAND_BODY_NAMES,
) -> torch.Tensor:
  command = get_motion_command(env, command_name)
  body_ids = _command_body_indices(command, body_names)
  root_pos_w, root_quat_w, _, _ = _reference_root_state(command)
  body_pos_w = command.body_pos_w[:, body_ids]
  return _body_positions_in_root_frame(root_pos_w, root_quat_w, body_pos_w)


def motion_hand_pos_b(
  env: ManagerBasedRlEnv,
  command_name: str = "motion",
  body_names: tuple[str, ...] = HAND_BODY_NAMES,
) -> torch.Tensor:
  return motion_hand_pos_b_tensor(
    env, command_name=command_name, body_names=body_names
  ).reshape(env.num_envs, -1)


def robot_hand_pos_b_tensor(
  env: ManagerBasedRlEnv,
  command_name: str = "motion",
  body_names: tuple[str, ...] = HAND_BODY_NAMES,
) -> torch.Tensor:
  command = get_motion_command(env, command_name)
  body_ids = _robot_body_indices(env, body_names, command_name)
  root_pos_w, root_quat_w = _robot_root_state(command)
  body_pos_w = command.robot.data.body_link_pos_w[:, body_ids]
  return _body_positions_in_root_frame(root_pos_w, root_quat_w, body_pos_w)




def motion_loco_command(
  env: ManagerBasedRlEnv,
  command_name: str = "motion",
  cmd_limits: tuple[float, float, float] = (1.0, 1.0, 1.0),
  uniform_cmd_name: str | None = None,
) -> torch.Tensor:
  """Body-frame (vx, vy, wz) command.

  Default: derive from motion pkl anchor (clamped to ``cmd_limits``).
  If ``uniform_cmd_name`` is given, read directly from that
  UniformVelocityCommand term (already in body-frame, no clamp applied).
  All downstream helpers that forward ``uniform_cmd_name`` end up reading
  from this single source of truth.
  """
  if uniform_cmd_name is not None:
    return env_mdp.generated_commands(env, command_name=uniform_cmd_name)[:, :3]
  cmd = torch.cat(
    (
      motion_root_vel_xy_b(env, command_name=command_name),
      motion_root_yaw_ang_vel_b(env, command_name=command_name),
    ),
    dim=-1,
  )
  limits = torch.tensor(cmd_limits, device=cmd.device, dtype=cmd.dtype)
  return torch.clamp(cmd, min=-limits, max=limits)


def motion_loco_leg_phase(
  env: ManagerBasedRlEnv,
  command_name: str = "motion",
  gait_period: float = LOCO_GAIT_PERIOD,
  gait_offset: float = LOCO_GAIT_OFFSET,
  stand_vel_threshold: float = 0.1,
  cmd_limits: tuple[float, float, float] = (1.0, 1.0, 1.0),
  uniform_cmd_name: str | None = None,
) -> torch.Tensor:
  walk_phase = (env.episode_length_buf.to(torch.float32) * env.step_dt) % gait_period
  walk_phase = walk_phase / gait_period
  cmd = motion_loco_command(
    env,
    command_name=command_name,
    cmd_limits=cmd_limits,
    uniform_cmd_name=uniform_cmd_name,
  )
  stationary = torch.norm(cmd, dim=-1) < stand_vel_threshold
  left = torch.where(stationary, torch.zeros_like(walk_phase), walk_phase)
  right = torch.where(
    stationary, torch.zeros_like(walk_phase), (walk_phase + gait_offset) % 1.0
  )
  return torch.stack((left, right), dim=-1)


def motion_loco_phase_features(
  env: ManagerBasedRlEnv,
  command_name: str = "motion",
  gait_period: float = LOCO_GAIT_PERIOD,
  gait_offset: float = LOCO_GAIT_OFFSET,
  stand_vel_threshold: float = 0.1,
  cmd_limits: tuple[float, float, float] = (1.0, 1.0, 1.0),
  uniform_cmd_name: str | None = None,
) -> torch.Tensor:
  phase = motion_loco_leg_phase(
    env,
    command_name=command_name,
    gait_period=gait_period,
    gait_offset=gait_offset,
    stand_vel_threshold=stand_vel_threshold,
    cmd_limits=cmd_limits,
    uniform_cmd_name=uniform_cmd_name,
  )
  two_pi_phase = 2.0 * torch.pi * phase
  return torch.cat(
    (
      torch.sin(two_pi_phase[:, 0:1]),
      torch.cos(two_pi_phase[:, 0:1]),
      torch.sin(two_pi_phase[:, 1:2]),
      torch.cos(two_pi_phase[:, 1:2]),
    ),
    dim=-1,
  )


def hand_student_mimic_observation(
  env: ManagerBasedRlEnv,
  command_name: str = "motion",
  gait_period: float = LOCO_GAIT_PERIOD,
  gait_offset: float = LOCO_GAIT_OFFSET,
  stand_vel_threshold: float = 0.1,
  cmd_limits: tuple[float, float, float] = (1.0, 1.0, 1.0),
  uniform_cmd_name: str | None = None,
) -> torch.Tensor:
  """Hand-student "mimic" observation: [vx, vy, z, yaw_rate, hand6, phase4].

  When ``uniform_cmd_name`` is set, the vx/vy/yaw_rate slots (and the
  stationary gate that zeroes them) are sourced from the uniform command;
  z and hand-position targets stay motion-derived (they are tracking
  targets, not commanded velocity).
  """
  # In motion mode, the obs values come from the raw (unclamped) motion root
  # vel while the stationary gate uses the clamped canonical cmd, matching
  # every other downstream user of motion_loco_command. In uniform mode both
  # sources collapse to the same tensor (short-circuit in motion_loco_command).
  if uniform_cmd_name is not None:
    cmd = motion_loco_command(env, uniform_cmd_name=uniform_cmd_name)
    vel_xy = cmd[:, :2]
    yaw_rate = cmd[:, 2:3]
  else:
    vel_xy = motion_root_vel_xy_b(env, command_name=command_name)
    yaw_rate = motion_root_yaw_ang_vel_b(env, command_name=command_name)
    cmd = motion_loco_command(
      env, command_name=command_name, cmd_limits=cmd_limits
    )
  stationary = (torch.norm(cmd, dim=-1) < stand_vel_threshold).unsqueeze(-1)
  vel_xy = torch.where(stationary, torch.zeros_like(vel_xy), vel_xy)
  yaw_rate = torch.where(stationary, torch.zeros_like(yaw_rate), yaw_rate)

  return torch.cat(
    (
      vel_xy,
      motion_root_z(env, command_name=command_name),
      # motion_torso_pitch(env, command_name=command_name),
      yaw_rate,
      motion_hand_pos_b(env, command_name=command_name),
      motion_loco_phase_features(
        env,
        command_name=command_name,
        gait_period=gait_period,
        gait_offset=gait_offset,
        stand_vel_threshold=stand_vel_threshold,
        cmd_limits=cmd_limits,
        uniform_cmd_name=uniform_cmd_name,
      ),
    ),
    dim=-1,
  )


def motion_loco_blend(
  env: ManagerBasedRlEnv,
  command_name: str = "motion",
  threshold: float = 0.1,
  width: float = 0.02,
  cmd_limits: tuple[float, float, float] = (1.0, 1.0, 1.0),
  uniform_cmd_name: str | None = None,
) -> torch.Tensor:
  cmd = motion_loco_command(
    env,
    command_name=command_name,
    cmd_limits=cmd_limits,
    uniform_cmd_name=uniform_cmd_name,
  )
  vel_mag = torch.norm(cmd, dim=-1, keepdim=True)
  return torch.sigmoid((vel_mag - threshold) / width)


def motion_loco_teacher_observation(
  env: ManagerBasedRlEnv,
  command_name: str = "motion",
  gait_period: float = LOCO_GAIT_PERIOD,
  gait_offset: float = LOCO_GAIT_OFFSET,
  stand_vel_threshold: float = 0.1,
  cmd_limits: tuple[float, float, float] = (1.0, 1.0, 1.0),
  lin_vel_scale: float = LOCO_BASE_LIN_VEL_SCALE,
  ang_vel_scale: float = LOCO_BASE_ANG_VEL_SCALE,
  joint_pos_scale: float = LOCO_JOINT_POS_SCALE,
  joint_vel_scale: float = LOCO_JOINT_VEL_SCALE,
  uniform_cmd_name: str | None = None,
  include_base_lin_vel: bool = True,
) -> torch.Tensor:
  parts = []
  if include_base_lin_vel:
    parts.append(env_mdp.base_lin_vel(env) * lin_vel_scale)
  parts.extend([
    env_mdp.base_ang_vel(env) * ang_vel_scale,
    env_mdp.projected_gravity(env),
    motion_loco_command(
      env,
      command_name=command_name,
      cmd_limits=cmd_limits,
      uniform_cmd_name=uniform_cmd_name,
    ),
    env_mdp.joint_pos_rel(env, biased=True) * joint_pos_scale,
    env_mdp.joint_vel_rel(env) * joint_vel_scale,
    env_mdp.last_action(env)[:, :NUM_LOCO_ACTIONS],
    motion_loco_phase_features(
      env,
      command_name=command_name,
      gait_period=gait_period,
      gait_offset=gait_offset,
      stand_vel_threshold=stand_vel_threshold,
      cmd_limits=cmd_limits,
      uniform_cmd_name=uniform_cmd_name,
    ),
  ])
  return torch.cat(parts, dim=-1)


def imu_roll_pitch(env: ManagerBasedRlEnv) -> torch.Tensor:
  robot = get_motion_command(env, "motion").robot
  roll, pitch, _ = euler_xyz_from_quat(robot.data.root_link_quat_w)
  return torch.stack((roll, pitch), dim=-1)




def critic_root_pos_w(
  env: ManagerBasedRlEnv, command_name: str = "motion"
) -> torch.Tensor:
  command = get_motion_command(env, command_name)
  root_pos_w, _ = _robot_root_state(command)
  return root_pos_w


def critic_root_quat_w(
  env: ManagerBasedRlEnv, command_name: str = "motion"
) -> torch.Tensor:
  command = get_motion_command(env, command_name)
  _, root_quat_w = _robot_root_state(command)
  return root_quat_w


def critic_key_body_pos_b(
  env: ManagerBasedRlEnv, command_name: str = "motion"
) -> torch.Tensor:
  command = get_motion_command(env, command_name)
  key_body_indices = tracked_body_indices(command)
  root_pos_w, root_quat_w = _robot_root_state(command)
  key_body_pos_w = command.robot_body_pos_w[:, key_body_indices]
  delta_w = key_body_pos_w - root_pos_w[:, None, :]
  root_quat_repeat = root_quat_w[:, None, :].expand(-1, len(KEY_BODY_NAMES), -1)
  key_body_pos_b = quat_apply_inverse(
    root_quat_repeat.reshape(-1, 4),
    delta_w.reshape(-1, 3),
  ).reshape(env.num_envs, len(KEY_BODY_NAMES), 3)
  return key_body_pos_b.reshape(env.num_envs, -1)


def critic_foot_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str = "feet_ground_contact",
) -> torch.Tensor:
  return velocity_mdp.foot_contact(env, sensor_name)




def critic_base_com_offset(
  env: ManagerBasedRlEnv, command_name: str = "motion"
) -> torch.Tensor:
  torso_body_idx = _torso_body_index(env, command_name)
  current = env.sim.model.body_ipos[:, torso_body_idx, :]
  default = env.sim.get_default_field("body_ipos")[torso_body_idx].unsqueeze(0)
  return current - default


def critic_foot_friction(
  env: ManagerBasedRlEnv, command_name: str = "motion"
) -> torch.Tensor:
  foot_geom_indices = _foot_geom_indices(env, command_name)
  friction = env.sim.model.geom_friction[:, foot_geom_indices, 0]
  return friction.mean(dim=1, keepdim=True)


def critic_added_mass(
  env: ManagerBasedRlEnv, command_name: str = "motion"
) -> torch.Tensor:
  torso_body_idx = _torso_body_index(env, command_name)
  current = env.sim.model.body_mass[:, torso_body_idx].unsqueeze(-1)
  default = env.sim.get_default_field("body_mass")[torso_body_idx].view(1, 1)
  return current - default


def critic_motor_scales(
  env: ManagerBasedRlEnv, command_name: str = "motion"
) -> torch.Tensor:
  ctrl_ids = _ctrl_ids(env, command_name)
  current_kp = env.sim.model.actuator_gainprm[:, ctrl_ids, 0]
  current_kd = -env.sim.model.actuator_biasprm[:, ctrl_ids, 2]
  default_kp = env.sim.get_default_field("actuator_gainprm")[ctrl_ids, 0].unsqueeze(0)
  default_kd = -env.sim.get_default_field("actuator_biasprm")[ctrl_ids, 2].unsqueeze(0)
  return torch.cat((current_kp / default_kp - 1.0, current_kd / default_kd - 1.0), dim=-1)


def critic_encoder_bias(
  env: ManagerBasedRlEnv, command_name: str = "motion"
) -> torch.Tensor:
  del command_name
  robot = _get_robot(env)
  return robot.data.encoder_bias


def critic_hand_force(
  env: ManagerBasedRlEnv,
  event_name: str = HAND_FORCE_EVENT_NAME,
) -> torch.Tensor:
  return get_event_term(env, event_name).force_observation()


def privileged_future_sequence(
  env: ManagerBasedRlEnv,
  command_name: str = "motion",
  step_offsets: tuple[int, ...] = PRIV_FUTURE_STEP_OFFSETS,
) -> torch.Tensor:
  command = get_motion_command(env, command_name)
  offsets = torch.tensor(step_offsets, device=command.device, dtype=torch.float32)
  num_envs = env.num_envs
  num_steps = len(step_offsets)

  motion_ids = command.motion_ids[:, None].expand(-1, num_steps).reshape(-1)
  motion_times = command.motion_times[:, None] + offsets[None, :] * env.step_dt
  frame = command.motion_lib.get_frame(motion_ids, motion_times.reshape(-1))

  body_pos_w = frame.body_pos_w.reshape(num_envs, num_steps, -1, 3)
  body_quat_w = frame.body_quat_w.reshape(num_envs, num_steps, -1, 4)
  body_lin_vel_w = frame.body_lin_vel_w.reshape(num_envs, num_steps, -1, 3)
  body_ang_vel_w = frame.body_ang_vel_w.reshape(num_envs, num_steps, -1, 3)
  joint_pos = frame.joint_pos.reshape(num_envs, num_steps, -1)

  root_pos_w = body_pos_w[:, :, 0]
  root_quat_w = body_quat_w[:, :, 0]
  root_lin_vel_w = body_lin_vel_w[:, :, 0]
  root_ang_vel_w = body_ang_vel_w[:, :, 0]

  flat_root_quat = root_quat_w.reshape(-1, 4)
  root_lin_vel_b = quat_apply_inverse(
    flat_root_quat, root_lin_vel_w.reshape(-1, 3)
  ).reshape(num_envs, num_steps, 3)
  root_ang_vel_b = quat_apply_inverse(
    flat_root_quat, root_ang_vel_w.reshape(-1, 3)
  ).reshape(num_envs, num_steps, 3)

  roll, pitch, yaw = euler_xyz_from_quat(flat_root_quat)
  root_rpy = torch.stack((roll, pitch, wrap_to_pi(yaw)), dim=-1).reshape(
    num_envs, num_steps, 3
  )

  robot_root_pos_w, _ = _robot_root_state(command)
  root_pos_distance_to_target = root_pos_w - robot_root_pos_w[:, None, :]

  current_ref_root_pos_w, current_ref_root_quat_w, _, _ = _reference_root_state(command)
  current_ref_root_pos_w = current_ref_root_pos_w[:, None, :]
  current_ref_root_quat_w = current_ref_root_quat_w[:, None, :]

  root_pos_delta_w = root_pos_w - current_ref_root_pos_w
  root_pos_delta_b = quat_apply_inverse(
    current_ref_root_quat_w.expand(-1, num_steps, -1).reshape(-1, 4),
    root_pos_delta_w.reshape(-1, 3),
  ).reshape(num_envs, num_steps, 3)

  root_rot_delta = quat_mul(
    quat_inv(current_ref_root_quat_w.expand(-1, num_steps, -1).reshape(-1, 4)),
    flat_root_quat,
  )
  root_rot_delta_b = axis_angle_from_quat(root_rot_delta).reshape(num_envs, num_steps, 3)

  key_body_indices = tracked_body_indices(command)
  key_body_pos_w = body_pos_w[:, :, key_body_indices]
  key_body_delta_w = key_body_pos_w - root_pos_w[:, :, None, :]
  key_body_quat = root_quat_w[:, :, None, :].expand(-1, -1, len(KEY_BODY_NAMES), -1)
  key_body_pos_b = quat_apply_inverse(
    key_body_quat.reshape(-1, 4),
    key_body_delta_w.reshape(-1, 3),
  ).reshape(num_envs, num_steps, len(KEY_BODY_NAMES) * 3)

  return torch.cat(
    (
      root_pos_w,
      root_pos_distance_to_target,
      root_rpy,
      root_lin_vel_b,
      root_ang_vel_b,
      root_pos_delta_b,
      root_rot_delta_b,
      joint_pos,
      key_body_pos_b,
    ),
    dim=-1,
  )






def critic_dr_dim() -> int:
  return 3 + 1 + 1 + 2 * NUM_G1_JOINTS + NUM_G1_JOINTS


def critic_extras_dim(include_hand_force: bool = False) -> int:
  dim = 3 + 3 + 4 + 3 * len(KEY_BODY_NAMES) + 2 + critic_dr_dim()
  if include_hand_force:
    dim += HAND_FORCE_OBS_DIM
  return dim


def critic_priv_step_dim() -> int:
  return CRITIC_PRIV_STEP_DIM


import torch as _torch
from torch.nn import functional as _F

from mjlab.managers.manager_base import ManagerTermBase as _ManagerTermBase










class RecoveryActiveTerm(_ManagerTermBase):
  """Pose-based recovery flag, sourced from
  :class:`wbc_mjlab.amp_mdp._PoseRecoveryStateTracker`.

  Thin wrapper over the shared singleton: triggers a pose-based state
  update if not already done this step, then emits the resulting per-env
  [B, 1] float mask AND-ed with the delay-env subset. The per-env state
  is shared with the pose-based reward wrappers
  (:func:`wbc_mjlab.amp_mdp.wbc_reward_with_pose_recovery_mask` and
  :func:`wbc_mjlab.amp_mdp.track_root_height_pose`) so KL-side and
  reward-side gating read the same mask within a step.

  Threshold params (override via cfg.params):
    - ``fall_z_threshold``   (default ``POSE_RECOVERY_FALL_Z`` = 0.45 m)
    - ``fall_tilt_threshold_rad`` (default ``POSE_RECOVERY_FALL_TILT_RAD`` = 1.0)
    - ``recover_z_threshold`` (default ``POSE_RECOVERY_RECOVER_Z`` = 0.70 m)
    - ``recover_tilt_threshold_rad`` (default ``POSE_RECOVERY_RECOVER_TILT_RAD`` = 0.3)
    - ``asset_name``         (default "robot")

  Defaults are tuned to G1's commanded squat range — squat (low pelvis,
  upright torso) never enters recovery. State is sticky: enters on
  (low pelvis + tilted), holds until (high pelvis + upright). See
  ``_PoseRecoveryStateTracker`` for the state-machine details.
  """

  def __init__(self, cfg, env) -> None:
    super().__init__(env)
    from wbc_mjlab.amp_mdp import (
      POSE_RECOVERY_FALL_Z,
      POSE_RECOVERY_FALL_TILT_RAD,
      POSE_RECOVERY_RECOVER_Z,
      POSE_RECOVERY_RECOVER_TILT_RAD,
    )

    self._fall_z = float(cfg.params.get("fall_z_threshold", POSE_RECOVERY_FALL_Z))
    self._fall_tilt = float(cfg.params.get("fall_tilt_threshold_rad", POSE_RECOVERY_FALL_TILT_RAD))
    self._recover_z = float(cfg.params.get("recover_z_threshold", POSE_RECOVERY_RECOVER_Z))
    self._recover_tilt = float(cfg.params.get("recover_tilt_threshold_rad", POSE_RECOVERY_RECOVER_TILT_RAD))
    self._asset_name = str(cfg.params.get("asset_name", "robot"))

  def reset(self, env_ids):
    from wbc_mjlab.amp_mdp import _PoseRecoveryStateTracker
    _PoseRecoveryStateTracker.get().reset(env_ids)

  def __call__(self, env, **_kwargs) -> _torch.Tensor:
    from wbc_mjlab.amp_mdp import _PoseRecoveryStateTracker

    tracker = _PoseRecoveryStateTracker.get()
    tracker.ensure_updated(
      env,
      fall_z=self._fall_z,
      fall_tilt=self._fall_tilt,
      recover_z=self._recover_z,
      recover_tilt=self._recover_tilt,
      asset_name=self._asset_name,
    )
    mask = tracker.get_mask(env)
    if mask is None:
      return _torch.zeros(env.num_envs, 1, device=env.device)
    return mask.float().unsqueeze(-1)




