"""Environment configuration for PKL-based motion tracking."""

from __future__ import annotations

import copy
import inspect
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, cast

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as env_mdp
from mjlab.envs.mdp import dr
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sensor import TerrainHeightSensorCfg
from mjlab.sensor import ObjRef
from mjlab.sensor import CameraSensorCfg
from mjlab.sensor import RingPatternCfg
from mjlab.sensor import RayCastSensorCfg, GridPatternCfg
from mjlab.utils.spec_config import CameraCfg as SpecCameraCfg
from mjlab.tasks.tracking import mdp as tracking_mdp
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_rough_env_cfg
from mjlab.terrains.config import STAIRS_TERRAINS_CFG
from mjlab.terrains import BoxInvertedPyramidStairsTerrainCfg
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from wbc_mjlab.actions import (
  DEFAULT_G1_LOCO_TEACHER_ACTION_SCALE,
  G1LocoTeacherActionCfg,
)
from wbc_mjlab.commands import LocoArmMotionCommandCfg
from wbc_mjlab.commands import PklMotionCommandCfg
from wbc_mjlab import curriculums as wbc_curriculums
from wbc_mjlab import events as wbc_events
from wbc_mjlab import observations as wbc_obs
from wbc_mjlab import rewards as wbc_rewards
from wbc_mjlab import terminations as wbc_terms
from wbc_mjlab.g1_constants_custom import _ANKLE_DOF_INDICES, _BASE_ANG_VEL_SCALE, _DUAL_BASELINE_ADDED_LOCO_REWARDS, _DUAL_BASELINE_MOTION_GATE_THRESHOLD, _JOINT_POS_SCALE, _JOINT_VEL_SCALE, _JOINT_VEL_SCALE_WITH_ANKLE_MASK, _LOCO_BODY_STD_RUNNING, _LOCO_BODY_STD_WALKING, _LOCO_FIXED_ANG_VEL_RANGE, _LOCO_FIXED_BASE_HEIGHT_RANGE, _LOCO_FIXED_LIN_VEL_RANGE, _LOCO_HEIGHT_CAP, _LOCO_WARMUP_SCALE, _HANDOFF_BASE_MASS_RANGE, _HANDOFF_LOCO_MOTION_FILE, _HANDOFF_MOTOR_STRENGTH_RANGE, _WBC_DEFAULT_NUM_ENVS

_DUAL_BASELINE_REWARD_FUNC_OVERRIDES = {
  "foot_clearance": wbc_rewards.motion_feet_clearance,
  "foot_swing_height": wbc_rewards.motion_feet_swing_height,
  "air_time": wbc_rewards.motion_air_time,
  "soft_landing": wbc_rewards.motion_soft_landing,
}


def _scale_range(range_pair: tuple[float, float], scale: float) -> tuple[float, float]:
  lo, hi = range_pair
  center = 0.5 * (lo + hi)
  half = 0.5 * (hi - lo) * scale
  return (center - half, center + half)


def _feet_ground_sensor_cfg(*, track_air_time: bool = False) -> ContactSensorCfg:
  return ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=track_air_time,
  )


def _handoff_tracking_reward_cfg() -> dict[str, RewardTermCfg]:
  return {
    "tracking_joint_dof": RewardTermCfg(
      func=wbc_rewards.tracking_joint_dof,
      weight=2.0,
      params={"command_name": "motion"},
    ),
    "tracking_joint_vel": RewardTermCfg(
      func=wbc_rewards.tracking_joint_vel,
      weight=0.2,
      params={"command_name": "motion"},
    ),
    "tracking_root_translation_z": RewardTermCfg(
      func=wbc_rewards.tracking_root_translation_z,
      weight=1.0,
      params={"command_name": "motion"},
    ),
    "tracking_root_rotation": RewardTermCfg(
      func=wbc_rewards.tracking_root_rotation,
      weight=1.0,
      params={"command_name": "motion"},
    ),
    "tracking_root_linear_vel": RewardTermCfg(
      func=wbc_rewards.tracking_root_linear_vel,
      weight=1.0,
      params={"command_name": "motion"},
    ),
    "tracking_root_angular_vel": RewardTermCfg(
      func=wbc_rewards.tracking_root_angular_vel,
      weight=1.0,
      params={"command_name": "motion"},
    ),
    "tracking_keybody_pos": RewardTermCfg(
      func=wbc_rewards.tracking_keybody_pos,
      weight=2.0,
      params={"command_name": "motion"},
    ),
    "tracking_keybody_pos_global": RewardTermCfg(
      func=wbc_rewards.tracking_keybody_pos_global,
      weight=2.0,
      params={"command_name": "motion"},
    ),
  }


def _stability_reward_cfg() -> dict[str, RewardTermCfg]:
  """Whole-body stability rewards (CoM-in-polygon, capture-point-in-polygon,
  ankle/hip/step strategy, linear & angular momentum change penalties).

  Ported from handoff_mjlab_stable (originally adapted from IHMC IsaacLab).
  All terms key off ``feet_ground_contact``; weights match the upstream port.
  """
  return {
    "com_in_support_polygon": RewardTermCfg(
      func=wbc_rewards.simplified_com_in_support_polygon,
      weight=0.2,
      params={"feet_sensor_cfg": SceneEntityCfg("feet_ground_contact")},
    ),
    "capture_point_in_support_polygon": RewardTermCfg(
      func=wbc_rewards.simplified_capture_point_in_support_polygon,
      weight=0.3,
      params={"feet_sensor_cfg": SceneEntityCfg("feet_ground_contact")},
    ),
    "ankle_hip_step": RewardTermCfg(
      func=wbc_rewards.AnkleHipStepReward,
      weight=0.2,
      params={"feet_sensor_cfg": SceneEntityCfg("feet_ground_contact")},
    ),
    "linear_momentum_change": RewardTermCfg(
      func=wbc_rewards.LinearMomentumChangePenalty,
      weight=1e-6,
      params={},
    ),
    "angular_momentum_change": RewardTermCfg(
      func=wbc_rewards.AngularMomentumChangePenalty,
      weight=1e-5,
      params={},
    ),
  }


def _apply_stability_rewards(cfg: ManagerBasedRlEnvCfg) -> None:
  """Opt-in: add whole-body stability reward terms to ``cfg.rewards``.

  Used by the ``*_stable`` env-cfg variants to keep stability rewards
  off the default training tasks. Existing keys with the same name
  are overwritten.
  """
  for _name, _term in _stability_reward_cfg().items():
    cfg.rewards[_name] = _term


def _handoff_regularization_reward_cfg() -> dict[str, RewardTermCfg]:
  all_joints = SceneEntityCfg("robot", joint_names=(".*",))
  return {
    "alive": RewardTermCfg(func=env_mdp.is_alive, weight=0.5),
    "feet_slip": RewardTermCfg(
      func=wbc_rewards.feet_slip,
      weight=-0.1,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "feet_contact_forces": RewardTermCfg(
      func=wbc_rewards.feet_contact_forces,
      weight=-5e-4,
      params={"sensor_name": "feet_ground_contact", "max_contact_force": 500.0},
    ),
    "feet_stumble": RewardTermCfg(
      func=wbc_rewards.feet_stumble,
      weight=-1.25,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "dof_pos_limits": RewardTermCfg(
      func=env_mdp.joint_pos_limits,
      weight=-5.0,
      params={"asset_cfg": all_joints},
    ),
    "dof_torque_limits": RewardTermCfg(
      func=wbc_rewards.dof_torque_limits,
      weight=-1.0,
      params={"asset_cfg": SceneEntityCfg("robot", actuator_names=[".*"]), "soft_torque_limit": 0.95},
    ),
    "dof_vel": RewardTermCfg(
      func=env_mdp.joint_vel_l2,
      weight=-1e-4,
      params={"asset_cfg": all_joints},
    ),
    "dof_acc": RewardTermCfg(
      func=env_mdp.joint_acc_l2,
      weight=-5e-8,
      params={"asset_cfg": all_joints},
    ),
    "action_rate_l2": RewardTermCfg(func=env_mdp.action_rate_l2, weight=-0.1),
    "joint_limit": RewardTermCfg(
      func=env_mdp.joint_pos_limits,
      weight=-10.0,
      params={"asset_cfg": all_joints},
    ),
    "self_collisions": RewardTermCfg(
      func=tracking_mdp.self_collision_cost,
      weight=-10.0,
      params={"sensor_name": "self_collision", "force_threshold": 10.0},
    ),
    "feet_air_time": RewardTermCfg(
      func=wbc_rewards.feet_air_time,
      weight=5.0,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "motion",
        "feet_air_time_target": 0.5,
      },
    ),
    "ang_vel_xy": RewardTermCfg(
      func=wbc_rewards.ang_vel_xy,
      weight=-0.01,
    ),
    "ankle_dof_acc": RewardTermCfg(
      func=wbc_rewards.ankle_dof_acc,
      weight=-1e-7,
      params={"asset_cfg": all_joints},
    ),
    "ankle_dof_vel": RewardTermCfg(
      func=wbc_rewards.ankle_dof_vel,
      weight=-2e-4,
      params={"asset_cfg": all_joints},
    ),
  }


def _wbc_teacher_task_cfg() -> dict[str, RewardTermCfg]:
  """Stable reward mix: HANDOFF task rewards plus the selected regularization set."""
  rewards = _handoff_tracking_reward_cfg()
  rewards.update(_handoff_regularization_reward_cfg())
  return rewards


def _handoff_termination_cfg(play: bool = False) -> dict[str, TerminationTermCfg]:
  terminations = {
    "time_out": TerminationTermCfg(func=env_mdp.time_out, time_out=True),
    "motion_end": TerminationTermCfg(
      func=wbc_terms.handoff_motion_end,
      params={"command_name": "motion"},
      time_out=True,
    ),
    "root_height_diff": TerminationTermCfg(
      func=wbc_terms.handoff_root_height_diff,
      params={"command_name": "motion", "threshold": 0.3},
    ),
    "roll_limit": TerminationTermCfg(
      func=wbc_terms.handoff_roll_limit,
      params={"threshold": 4.0},
    ),
    "pitch_limit": TerminationTermCfg(
      func=wbc_terms.handoff_pitch_limit,
      params={"threshold": 4.0},
    ),
    "velocity_too_large": TerminationTermCfg(
      func=wbc_terms.handoff_velocity_too_large,
      params={"threshold": 5.0},
    ),
    # Defensive: catch non-finite robot state per-env so MuJoCo physics
    # divergence (e.g. stair-fall edge cases on a cold-start policy)
    # resets cleanly instead of crashing the trainer's NaN guard. All
    # other handoff_* terminations use ``abs(x) > thr`` checks which
    # silently miss NaN. Pair with the ``torch.nan_to_num`` defense in
    # the runner (rewards computed pre-reset still see the NaN qpos and
    # produce NaN reward terms).
    "nan_state": TerminationTermCfg(func=wbc_terms.nan_state),
    "pose_fail": TerminationTermCfg(
      func=wbc_terms.handoff_pose_fail,
      params={
        "command_name": "motion",
        "threshold": 0.7,
        "track_root": False,
        "root_tracking_threshold": 2.0,
      },
    ),
  }
  if play:
    # Keep only time_out during play so a bad student doesn't get reset
    # mid-evaluation and the user can see sustained failure modes.
    terminations = {"time_out": terminations["time_out"]}
  return terminations


def _apply_handoff_domain_rand(
  cfg: ManagerBasedRlEnvCfg, *, play: bool = False
) -> None:
  """Approximate the active HANDOFF DR terms using mjlab-native mechanisms."""
  if play:
    return

  robot_cfg = cfg.scene.entities["robot"]
  assert isinstance(robot_cfg, EntityCfg)
  robot_cfg = copy.deepcopy(robot_cfg)
  assert robot_cfg.articulation is not None

  delayed_actuators = []
  for actuator_cfg in robot_cfg.articulation.actuators:
    if isinstance(actuator_cfg, BuiltinPositionActuatorCfg):
      delayed_actuators.append(
        replace(
          actuator_cfg,
          delay_min_lag=0,
          delay_max_lag=cfg.decimation,
          delay_hold_prob=0.0,
          delay_update_period=cfg.decimation,
          delay_per_env_phase=True,
        )
      )
    else:
      delayed_actuators.append(copy.deepcopy(actuator_cfg))

  robot_cfg.articulation.actuators = tuple(delayed_actuators)
  cfg.scene.entities["robot"] = robot_cfg

  cfg.events["handoff_base_mass"] = EventTermCfg(
    mode="startup",
    func=dr.body_mass,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
      "operation": "add",
      "ranges": _HANDOFF_BASE_MASS_RANGE,
    },
  )
  cfg.events["handoff_motor_strength"] = EventTermCfg(
    mode="startup",
    func=dr.pd_gains,
    params={
      "asset_cfg": SceneEntityCfg("robot", actuator_names=(".*",)),
      "kp_range": _HANDOFF_MOTOR_STRENGTH_RANGE,
      "kd_range": _HANDOFF_MOTOR_STRENGTH_RANGE,
      "operation": "scale",
    },
  )


def _disable_play_randomization(cfg: ManagerBasedRlEnvCfg) -> None:
  """Force deterministic playback without modifying mjlab base task code."""
  for event_name in ("base_com", "encoder_bias", "foot_friction", "push_robot"):
    cfg.events.pop(event_name, None)

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.joint_position_range = (0.0, 0.0)


def _set_wbc_default_num_envs(cfg: ManagerBasedRlEnvCfg, *, play: bool) -> None:
  """Use a larger training default for WBC tasks without affecting play configs."""
  if not play:
    cfg.scene.num_envs = _WBC_DEFAULT_NUM_ENVS


def _make_loco_teacher_obs_terms(
  *,
  enable_noise: bool,
  include_base_lin_vel: bool = True,
  include_height_scan: bool = False,
) -> dict[str, ObservationTermCfg]:
  terms: dict[str, ObservationTermCfg] = {}
  if include_base_lin_vel:
    terms["base_lin_vel"] = ObservationTermCfg(
      func=env_mdp.base_lin_vel,
      scale=wbc_obs.LOCO_BASE_LIN_VEL_SCALE,
      noise=Unoise(n_min=-0.1, n_max=0.1) if enable_noise else None,
    )
  terms.update({
    "base_ang_vel": ObservationTermCfg(
      func=env_mdp.base_ang_vel,
      scale=wbc_obs.LOCO_BASE_ANG_VEL_SCALE,
      noise=Unoise(n_min=-0.2, n_max=0.2) if enable_noise else None,
    ),
    "projected_gravity": ObservationTermCfg(
      func=env_mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05) if enable_noise else None,
    ),
    "command": ObservationTermCfg(
      func=env_mdp.generated_commands,
      params={"command_name": "twist"},
    ),
    "joint_pos": ObservationTermCfg(
      func=env_mdp.joint_pos_rel,
      params={"biased": True},
      scale=wbc_obs.LOCO_JOINT_POS_SCALE,
      noise=Unoise(n_min=-0.01, n_max=0.01) if enable_noise else None,
    ),
    "joint_vel": ObservationTermCfg(
      func=env_mdp.joint_vel_rel,
      scale=wbc_obs.LOCO_JOINT_VEL_SCALE,
      noise=Unoise(n_min=-1.5, n_max=1.5) if enable_noise else None,
    ),
    "actions": ObservationTermCfg(func=env_mdp.last_action),
    "phase": ObservationTermCfg(
      func=wbc_obs.loco_phase_features,
      params={
        "command_name": "twist",
        "gait_period": wbc_obs.LOCO_GAIT_PERIOD,
        "gait_offset": wbc_obs.LOCO_GAIT_OFFSET,
      },
    ),
  })
  if include_height_scan:
    terms["height_scan"] = ObservationTermCfg(
      func=env_mdp.height_scan,
      params={"sensor_name": "terrain_scan"},
      scale=0.2,  # 1 / terrain_scan.max_distance (5.0)
      noise=Unoise(n_min=-0.1, n_max=0.1) if enable_noise else None,
    )
  return terms


def _make_wbc_teacher_actor_terms(*, enable_noise: bool) -> dict[str, ObservationTermCfg]:
  return {
    "motion_root_vel_xy_b": ObservationTermCfg(
      func=wbc_obs.motion_root_vel_xy_b,
      params={"command_name": "motion"},
    ),
    "motion_root_z": ObservationTermCfg(
      func=wbc_obs.motion_root_z,
      params={"command_name": "motion"},
    ),
    "motion_root_roll_pitch": ObservationTermCfg(
      func=wbc_obs.motion_root_roll_pitch,
      params={"command_name": "motion"},
    ),
    "motion_root_yaw_ang_vel_b": ObservationTermCfg(
      func=wbc_obs.motion_root_yaw_ang_vel_b,
      params={"command_name": "motion"},
    ),
    "motion_joint_pos": ObservationTermCfg(
      func=wbc_obs.motion_joint_pos,
      params={"command_name": "motion"},
    ),
    "base_ang_vel": ObservationTermCfg(
      func=env_mdp.base_ang_vel,
      scale=_BASE_ANG_VEL_SCALE,
      noise=Unoise(n_min=-0.1, n_max=0.1) if enable_noise else None,
    ),
    "imu_roll_pitch": ObservationTermCfg(
      func=wbc_obs.imu_roll_pitch,
      noise=Unoise(n_min=-0.1, n_max=0.1) if enable_noise else None,
    ),
    "joint_pos": ObservationTermCfg(
      func=env_mdp.joint_pos_rel,
      params={"biased": True},
      scale=_JOINT_POS_SCALE,
      noise=Unoise(n_min=-0.01, n_max=0.01) if enable_noise else None,
    ),
    "joint_vel": ObservationTermCfg(
      func=env_mdp.joint_vel_rel,
      scale=_JOINT_VEL_SCALE_WITH_ANKLE_MASK,
      noise=Unoise(n_min=-0.1, n_max=0.1) if enable_noise else None,
    ),
    "actions": ObservationTermCfg(func=env_mdp.last_action),
  }


def _make_hand_student_actor_terms(*, enable_noise: bool) -> dict[str, ObservationTermCfg]:
  return {
    "hand_mimic": ObservationTermCfg(
      func=wbc_obs.hand_student_mimic_observation,
      params={"command_name": "motion"},
    ),
    "base_ang_vel": ObservationTermCfg(
      func=env_mdp.base_ang_vel,
      scale=_BASE_ANG_VEL_SCALE,
      noise=Unoise(n_min=-0.1, n_max=0.1) if enable_noise else None,
    ),
    "imu_roll_pitch": ObservationTermCfg(
      func=wbc_obs.imu_roll_pitch,
      noise=Unoise(n_min=-0.1, n_max=0.1) if enable_noise else None,
    ),
    "joint_pos": ObservationTermCfg(
      func=env_mdp.joint_pos_rel,
      params={"biased": True},
      scale=_JOINT_POS_SCALE,
      noise=Unoise(n_min=-0.01, n_max=0.01) if enable_noise else None,
    ),
    "joint_vel": ObservationTermCfg(
      func=env_mdp.joint_vel_rel,
      scale=_JOINT_VEL_SCALE_WITH_ANKLE_MASK,
      noise=Unoise(n_min=-0.1, n_max=0.1) if enable_noise else None,
    ),
    "actions": ObservationTermCfg(func=env_mdp.last_action),
  }


def _make_wbc_critic_current_terms() -> dict[str, ObservationTermCfg]:
  return {
    "base_ang_vel": ObservationTermCfg(
      func=env_mdp.base_ang_vel,
      scale=_BASE_ANG_VEL_SCALE,
    ),
    "imu_roll_pitch": ObservationTermCfg(func=wbc_obs.imu_roll_pitch),
    "joint_pos": ObservationTermCfg(
      func=env_mdp.joint_pos_rel,
      scale=_JOINT_POS_SCALE,
    ),
    "joint_vel": ObservationTermCfg(
      func=env_mdp.joint_vel_rel,
      scale=_JOINT_VEL_SCALE_WITH_ANKLE_MASK,
    ),
    "actions": ObservationTermCfg(func=env_mdp.last_action),
  }


def _priv_future_obs_group() -> ObservationGroupCfg:
  """Privileged future sequence observation group for the critic."""
  return ObservationGroupCfg(
    terms={
      "privileged_future": ObservationTermCfg(
        func=wbc_obs.privileged_future_sequence,
        params={
          "command_name": "motion",
          "step_offsets": wbc_obs.PRIV_FUTURE_STEP_OFFSETS,
        },
      )
    },
    concatenate_terms=True,
    enable_corruption=False,
  )


def _make_wbc_critic_extras_terms() -> dict[str, ObservationTermCfg]:
  return {
    "base_lin_vel": ObservationTermCfg(func=env_mdp.base_lin_vel),
    "root_pos_w": ObservationTermCfg(
      func=wbc_obs.critic_root_pos_w,
      params={"command_name": "motion"},
    ),
    "root_quat_w": ObservationTermCfg(
      func=wbc_obs.critic_root_quat_w,
      params={"command_name": "motion"},
    ),
    "key_body_pos_b": ObservationTermCfg(
      func=wbc_obs.critic_key_body_pos_b,
      params={"command_name": "motion"},
    ),
    "foot_contact": ObservationTermCfg(
      func=wbc_obs.critic_foot_contact,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "base_com_offset": ObservationTermCfg(
      func=wbc_obs.critic_base_com_offset,
      params={"command_name": "motion"},
    ),
    "foot_friction": ObservationTermCfg(
      func=wbc_obs.critic_foot_friction,
      params={"command_name": "motion"},
    ),
    "added_mass": ObservationTermCfg(
      func=wbc_obs.critic_added_mass,
      params={"command_name": "motion"},
    ),
    "motor_scales": ObservationTermCfg(
      func=wbc_obs.critic_motor_scales,
      params={"command_name": "motion"},
    ),
    "encoder_bias": ObservationTermCfg(
      func=wbc_obs.critic_encoder_bias,
      params={"command_name": "motion"},
    ),
  }


def _make_hand_critic_extras_terms(
  *, include_force_randomization: bool = False
) -> dict[str, ObservationTermCfg]:
  terms = _make_wbc_critic_extras_terms()
  if include_force_randomization:
    terms["hand_force"] = ObservationTermCfg(
      func=wbc_obs.critic_hand_force,
      params={"event_name": wbc_events.HAND_FORCE_EVENT_NAME},
    )
  return terms


def _hand_force_randomization_cfg() -> dict[str, object]:
  return {
    "force_apply_links": wbc_events.HAND_FORCE_APPLY_LINKS,
    "force_scale_curriculum": True,
    "force_scale_initial_scale": 0.1,
    "force_scale_up_threshold": 210,
    "force_scale_down_threshold": 200,
    "force_scale_up": 0.02,
    "force_scale_down": 0.02,
    "force_scale_max": 1.0,
    "force_scale_min": 0.0,
    "apply_force_x_range": (-30.0, 30.0),
    "apply_force_y_range": (-30.0, 30.0),
    "apply_force_z_range": (-40.0, 5.0),
    "zero_force_prob": (0.25, 0.25, 0.25),
    "randomize_force_duration_steps": (10, 50),
    "use_lpf": False,
    "force_filter_alpha": 0.05,
  }


def _hand_dual_teacher_task_cfg(
  *,
  enable_foot_clearance: bool,
  foot_clearance_weight: float,
  foot_clearance_target_height: float,
  enable_motion_gait_phase_contact: bool,
) -> dict[str, RewardTermCfg]:
  rewards = {
    "tracking_hand_pos": RewardTermCfg(
      func=wbc_rewards.tracking_hand_pos,
      weight=6.0,
      params={"command_name": "motion"},
    ),
    "tracking_root_translation_z": RewardTermCfg(
      func=wbc_rewards.tracking_root_translation_z,
      weight=1.0,
      params={"command_name": "motion"},
    ),
    "tracking_root_rotation": RewardTermCfg(
      func=wbc_rewards.tracking_root_rotation,
      weight=1.0,
      params={"command_name": "motion"},
    ),
    "tracking_root_linear_vel": RewardTermCfg(
      func=wbc_rewards.tracking_root_linear_vel,
      weight=1.0,
      params={"command_name": "motion"},
    ),
    "tracking_root_angular_vel": RewardTermCfg(
      func=wbc_rewards.tracking_root_angular_vel,
      weight=1.0,
      params={"command_name": "motion"},
    ),
    # "standing_tracking_keybody_pos": RewardTermCfg(
    #   func=wbc_rewards.standing_tracking_keybody_pos,
    #   weight=1.5,
    #   params={"command_name": "motion", "command_threshold": 0.1},
    # ),
    # "standing_tracking_joint_dof": RewardTermCfg(
    #   func=wbc_rewards.standing_tracking_joint_dof,
    #   weight=0.5,
    #   params={"command_name": "motion", "command_threshold": 0.1},
    # ),
    # "standing_tracking_torso_pitch": RewardTermCfg(
    #   func=wbc_rewards.standing_tracking_torso_pitch,
    #   weight=2.0,
    #   params={"command_name": "motion", "command_threshold": 0.1},
    # ),
    # "tracking_torso_pitch": RewardTermCfg(
    #   func=wbc_rewards.tracking_torso_pitch,
    #   weight=2.0,
    #   params={"command_name": "motion"},
    # ),
    # "tracking_waist_pitch_dof": RewardTermCfg(
    #   func=wbc_rewards.tracking_waist_pitch_dof,
    #   weight=2.0,
    #   params={"command_name": "motion"},
    # ),
  }
  if enable_foot_clearance:
    rewards["foot_clearance"] = RewardTermCfg(
      func=velocity_mdp.feet_clearance,
      weight=foot_clearance_weight,
      params={
        "target_height": foot_clearance_target_height,
        "height_sensor_name": "foot_height_scan",
        "command_name": "motion",
        "command_threshold": 0.1,
        "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
      },
    )
  if enable_motion_gait_phase_contact:
    rewards["gait_phase_contact"] = RewardTermCfg(
      func=wbc_rewards.motion_gait_phase_contact,
      weight=0.5,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "motion",
        "gait_period": wbc_obs.LOCO_GAIT_PERIOD,
        "gait_offset": wbc_obs.LOCO_GAIT_OFFSET,
        "stance_ratio": wbc_obs.LOCO_STANCE_RATIO,
        "stand_vel_threshold": 0.1,
        "cmd_limits": (1.0, 1.0, 1.0),
      },
    )
  rewards.update(_handoff_regularization_reward_cfg())
  rewards.pop("joint_limit", None)
  rewards.pop("self_collisions", None)
  return rewards


def _hand_dual_teacher_task_cfg_nonbaseline() -> dict[str, RewardTermCfg]:
  return _hand_dual_teacher_task_cfg(
    enable_foot_clearance=False,
    foot_clearance_weight=-6.0,
    foot_clearance_target_height=0.05,
    enable_motion_gait_phase_contact=True,
  )


def unitree_g1_pkl_tracking_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create G1 tracking config that uses PKL motion files."""
  cfg = unitree_g1_flat_tracking_env_cfg(has_state_estimation=False, play=play)

  # Replace the NPZ MotionCommandCfg with our PKL version
  old_cmd = cfg.commands["motion"]
  assert isinstance(old_cmd, MotionCommandCfg)

  cfg.commands["motion"] = PklMotionCommandCfg(
    entity_name=old_cmd.entity_name,
    resampling_time_range=old_cmd.resampling_time_range,
    debug_vis=old_cmd.debug_vis,
    pose_range=old_cmd.pose_range,
    velocity_range=old_cmd.velocity_range,
    joint_position_range=old_cmd.joint_position_range,
    motion_file=old_cmd.motion_file,
    anchor_body_name=old_cmd.anchor_body_name,
    body_names=old_cmd.body_names,
    adaptive_kernel_size=old_cmd.adaptive_kernel_size,
    adaptive_lambda=old_cmd.adaptive_lambda,
    adaptive_uniform_ratio=old_cmd.adaptive_uniform_ratio,
    adaptive_alpha=old_cmd.adaptive_alpha,
    sampling_mode=old_cmd.sampling_mode,
  )

  if play:
    _disable_play_randomization(cfg)

  _set_wbc_default_num_envs(cfg, play=play)
  return cfg


def unitree_g1_pkl_tracking_custom_ppo_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the custom PPO tracking config with HANDOFF-style observation groups."""
  cfg = unitree_g1_pkl_tracking_env_cfg(play=play)

  feet_ground_cfg = _feet_ground_sensor_cfg()
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet_ground_cfg,)

  cfg.observations = {
    "actor_current": ObservationGroupCfg(
      terms=_make_wbc_teacher_actor_terms(enable_noise=not play),
      concatenate_terms=True,
      enable_corruption=not play,
    ),
    "actor_history": ObservationGroupCfg(
      terms=_make_wbc_teacher_actor_terms(enable_noise=not play),
      concatenate_terms=True,
      enable_corruption=not play,
      history_length=11,
      flatten_history_dim=False,
    ),
    "critic_priv_future_sequence": _priv_future_obs_group(),
    "critic_current": ObservationGroupCfg(
      terms=_make_wbc_critic_current_terms(),
      concatenate_terms=True,
      enable_corruption=False,
    ),
    "critic_extras": ObservationGroupCfg(
      terms=_make_wbc_critic_extras_terms(),
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  _set_wbc_default_num_envs(cfg, play=play)
  return cfg


def unitree_g1_wbc_teacher_flat_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the teacher-flat task with the explicit reward mix used for the stable port."""
  cfg = unitree_g1_pkl_tracking_custom_ppo_env_cfg(play=play)

  sensors = []
  for sensor_cfg in cfg.scene.sensors or ():
    if sensor_cfg.name == "feet_ground_contact":
      sensors.append(_feet_ground_sensor_cfg(track_air_time=True))
    else:
      sensors.append(sensor_cfg)
  cfg.scene.sensors = tuple(sensors)
  _apply_handoff_domain_rand(cfg, play=play)
  cfg.rewards = _wbc_teacher_task_cfg()
  cfg.terminations = _handoff_termination_cfg(play=play)
  _set_wbc_default_num_envs(cfg, play=play)
  return cfg


def unitree_g1_wbc_teacher_flat_stable_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """WBC teacher variant with whole-body stability rewards added."""
  cfg = unitree_g1_wbc_teacher_flat_env_cfg(play=play)
  _apply_stability_rewards(cfg)
  return cfg


def unitree_g1_hand_dual_teacher_flat_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the non-baseline-loco dual-teacher hand-student task."""
  cfg = unitree_g1_wbc_teacher_flat_env_cfg(play=play)
  cfg.enable_falcon_force_randomization = False
  cfg.falcon_force_randomization = _hand_force_randomization_cfg()
  cfg.observations = {
    "actor_current": ObservationGroupCfg(
      terms=_make_hand_student_actor_terms(enable_noise=not play),
      concatenate_terms=True,
      enable_corruption=not play,
    ),
    "actor_history": ObservationGroupCfg(
      terms=_make_hand_student_actor_terms(enable_noise=not play),
      concatenate_terms=True,
      enable_corruption=not play,
      history_length=wbc_obs.ACTOR_HISTORY_LENGTH,
      flatten_history_dim=False,
    ),
    "wbc_teacher_actor_current": ObservationGroupCfg(
      terms=_make_wbc_teacher_actor_terms(enable_noise=not play),
      concatenate_terms=True,
      enable_corruption=not play,
    ),
    "wbc_teacher_actor_history": ObservationGroupCfg(
      terms=_make_wbc_teacher_actor_terms(enable_noise=not play),
      concatenate_terms=True,
      enable_corruption=not play,
      history_length=wbc_obs.ACTOR_HISTORY_LENGTH,
      flatten_history_dim=False,
    ),
    "critic_priv_future_sequence": _priv_future_obs_group(),
    "critic_current": ObservationGroupCfg(
      terms=_make_wbc_critic_current_terms(),
      concatenate_terms=True,
      enable_corruption=False,
    ),
    "critic_extras": ObservationGroupCfg(
      terms=_make_hand_critic_extras_terms(include_force_randomization=False),
      concatenate_terms=True,
      enable_corruption=False,
    ),
    "loco_teacher_actor": ObservationGroupCfg(
      terms={
        "loco_teacher_actor": ObservationTermCfg(
          func=wbc_obs.motion_loco_teacher_observation,
          params={
            "command_name": "motion",
            "gait_period": wbc_obs.LOCO_GAIT_PERIOD,
            "gait_offset": wbc_obs.LOCO_GAIT_OFFSET,
            "stand_vel_threshold": 0.1,
            "cmd_limits": (1.0, 1.0, 1.0),
          },
        )
      },
      concatenate_terms=True,
      enable_corruption=False,
    ),
    "loco_blend": ObservationGroupCfg(
      terms={
        "loco_blend": ObservationTermCfg(
          func=wbc_obs.motion_loco_blend,
          params={
            "command_name": "motion",
            "threshold": 0.1,
            "width": 0.02,
            "cmd_limits": (1.0, 1.0, 1.0),
          },
        )
      },
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }
  cfg.rewards = _hand_dual_teacher_task_cfg_nonbaseline()
  _set_wbc_default_num_envs(cfg, play=play)
  return cfg


_LOCO_UNIFORM_CMD_NAME = "loco_uniform_cmd"


def unitree_g1_hand_dual_teacher_flat_unicmd_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Dual-teacher MoE variant with uniform-random velocity command source.

  Decouples the velocity command the student / loco-expert sees (and the
  velocity-tracking rewards track against) from the motion dataset's velocity
  distribution, which is heavily low-speed biased on vx/vy and causes tracking
  degradation at high commanded velocities.

  Diff vs ``unitree_g1_hand_dual_teacher_flat_env_cfg``:
    1. Adds a ``loco_uniform_cmd`` command (UniformVelocityCommand) sampling
       (vx, vy, wz) i.i.d. uniformly in [-1, 1] on resampling intervals.
    2. Sets ``uniform_cmd_name="loco_uniform_cmd"`` on the student ``hand_mimic``
       observation, on ``loco_teacher_actor``, and on ``loco_blend`` — so the
       student, loco-expert and blend all see the uniform command.
    3. Swaps the velocity-tracking rewards to target the uniform cmd:
       - ``tracking_root_linear_vel`` targets uniform vx/vy.
       - ``tracking_root_angular_vel`` targets uniform wz.
       - ``tracking_root_rotation`` drops yaw (roll/pitch only from motion).
       - ``gait_phase_contact`` and ``feet_air_time`` gate on uniform cmd.

  WBC tracking of hand pose, root z, joint dof, and all regularization rewards
  are unchanged: hand targets and base height still come from motion, and the
  WBC teacher branch still receives motion-derived observations.
  """
  cfg = unitree_g1_hand_dual_teacher_flat_env_cfg(play=play)

  cfg.commands[_LOCO_UNIFORM_CMD_NAME] = UniformVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(3.0, 8.0),
    heading_command=False,
    rel_standing_envs=0.1,
    rel_heading_envs=0.0,
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=_LOCO_FIXED_LIN_VEL_RANGE,
      lin_vel_y=_LOCO_FIXED_LIN_VEL_RANGE,
      ang_vel_z=_LOCO_FIXED_ANG_VEL_RANGE,
    ),
  )

  # (2) Swap velocity-cmd observation sources to uniform.
  for group_name in ("actor_current", "actor_history"):
    hand_mimic = cfg.observations[group_name].terms["hand_mimic"]
    hand_mimic.params = {
      **(hand_mimic.params or {}),
      "uniform_cmd_name": _LOCO_UNIFORM_CMD_NAME,
    }

  loco_actor = cfg.observations["loco_teacher_actor"].terms["loco_teacher_actor"]
  loco_actor.params = {
    **(loco_actor.params or {}),
    "uniform_cmd_name": _LOCO_UNIFORM_CMD_NAME,
  }

  loco_blend = cfg.observations["loco_blend"].terms["loco_blend"]
  loco_blend.params = {
    **(loco_blend.params or {}),
    "uniform_cmd_name": _LOCO_UNIFORM_CMD_NAME,
  }

  # (3) Swap velocity-tracking / gait-gating rewards to uniform-cmd.
  # Combined xy error (matches loco teacher formulation), std=0.5 (linear)
  # / sqrt(0.5) (angular). Weight bumped to 4.0 (up from the old 2.0) so
  # velocity tracking has more gradient budget against the dominant
  # tracking_hand_pos reward (weight 6.0).
  rew = cfg.rewards
  rew["tracking_root_linear_vel"].weight = 4.0
  rew["tracking_root_linear_vel"].params = {
    **(rew["tracking_root_linear_vel"].params or {}),
    "uniform_cmd_name": _LOCO_UNIFORM_CMD_NAME,
    "std": 0.5,
  }
  rew["tracking_root_angular_vel"].weight = 4.0
  rew["tracking_root_angular_vel"].params = {
    **(rew["tracking_root_angular_vel"].params or {}),
    "uniform_cmd_name": _LOCO_UNIFORM_CMD_NAME,
    "std": math.sqrt(0.5),
  }
  rew["tracking_root_rotation"].params = {
    **(rew["tracking_root_rotation"].params or {}),
    "include_yaw": False,
  }
  if "gait_phase_contact" in rew:
    rew["gait_phase_contact"].params = {
      **(rew["gait_phase_contact"].params or {}),
      "uniform_cmd_name": _LOCO_UNIFORM_CMD_NAME,
    }
  if "feet_air_time" in rew:
    rew["feet_air_time"].params = {
      **(rew["feet_air_time"].params or {}),
      "uniform_cmd_name": _LOCO_UNIFORM_CMD_NAME,
    }

  _set_wbc_default_num_envs(cfg, play=play)
  return cfg


def unitree_g1_hand_dual_teacher_flat_unicmd_nobv_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """UniCmd student variant using the non-privileged (no base_lin_vel) loco teacher.

  Diff vs ``unitree_g1_hand_dual_teacher_flat_unicmd_env_cfg``:
    1. ``loco_teacher_actor`` observation drops ``base_lin_vel`` (86 dims)
       to match the nobv teacher's training obs.
    2. Adds a velocity curriculum on ``loco_uniform_cmd`` mirroring the
       loco teacher's warmup (``[-0.5, 0.5]`` until step 5000*24, then
       ramps to ``[-1.0, 1.0]``).

  Inherits reward weights from the base unicmd env cfg:
  ``tracking_hand_pos`` (weight=6.0), ``tracking_root_linear_vel``
  (weight=4.0, std=0.5), ``tracking_root_angular_vel`` (weight=4.0,
  std=sqrt(0.5)).
  """
  cfg = unitree_g1_hand_dual_teacher_flat_unicmd_env_cfg(play=play)

  loco_actor = cfg.observations["loco_teacher_actor"].terms["loco_teacher_actor"]
  loco_actor.params = {
    **(loco_actor.params or {}),
    "include_base_lin_vel": False,
  }

  # Mirror the loco teacher's velocity warmup curriculum on loco_uniform_cmd.
  cfg.curriculum["unicmd_vel"] = CurriculumTermCfg(
    func=velocity_mdp.commands_vel,
    params={
      "command_name": _LOCO_UNIFORM_CMD_NAME,
      "velocity_stages": [
        {
          "step": 0,
          "lin_vel_x": _scale_range(_LOCO_FIXED_LIN_VEL_RANGE, _LOCO_WARMUP_SCALE),
          "lin_vel_y": _scale_range(_LOCO_FIXED_LIN_VEL_RANGE, _LOCO_WARMUP_SCALE),
          "ang_vel_z": _scale_range(_LOCO_FIXED_ANG_VEL_RANGE, _LOCO_WARMUP_SCALE),
        },
        {
          "step": 5000 * 24,
          "lin_vel_x": _LOCO_FIXED_LIN_VEL_RANGE,
          "lin_vel_y": _LOCO_FIXED_LIN_VEL_RANGE,
          "ang_vel_z": _LOCO_FIXED_ANG_VEL_RANGE,
        },
      ],
    },
  )

  _set_wbc_default_num_envs(cfg, play=play)
  return cfg


def unitree_g1_hand_dual_teacher_flat_unicmd_nobv_amp_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """3-teacher MoE student variant: WBC + NoBV loco + AMP recovery teacher.

  Diff vs ``unitree_g1_hand_dual_teacher_flat_unicmd_nobv_env_cfg``:
    1. Installs ``DelayedTerminationManager`` + ``MotionResetManager`` from
       the AMP teacher env: ~20% of envs are tagged as recovery envs and
       on termination stay in the fallen pose for up to ``max_delay_steps``
       before resetting to a Recovery/ motion clip.
    2. Adds an ``amp_teacher_actor`` obs group (proprio + uniform-cmd, 4-frame
       flat history) shaped exactly like the AMP teacher's training obs.
    3. Adds a ``recovery_active`` obs group (single [B,1] flag) sourced from
       the delay manager's active mask — DaggerPPO reads this each step to
       route per-env between the WBC/loco split-KL (recovery off) and the
       AMP-teacher full-body KL (recovery on).
    4. Wraps velocity / hand-position / motion-tracked-root rewards with
       ``amp_mdp.wbc_reward_with_delay_mask`` so they are zeroed on
       in-recovery envs (mirrors AMP teacher's training reward gating).
    5. Adds ``track_root_height_recovery`` (std=0.3, scale=3.5) — the
       AMP teacher's get-up reward, active only on recovery envs.
  """
  from wbc_mjlab import amp_mdp
  from wbc_mjlab.amp_config import _RECOVERY_DIR, _WALK_RUN_DIR
  from wbc_mjlab.commands import (
    DelayRecoveryPklMotionCommandCfg,
    PklMotionCommandCfg,
  )

  cfg = unitree_g1_hand_dual_teacher_flat_unicmd_nobv_env_cfg(play=play)

  # --- (0) Recovery-aware motion command. --------------------------------
  # mjlab runs ``event_manager.apply(reset)`` BEFORE ``command_manager.reset``,
  # so the fallen pose written by ``reset_delay_envs_to_recovery`` is
  # immediately overwritten when the motion command resamples and writes
  # its own reference frame back to sim. Swap the existing PklMotionCommand
  # for the recovery-aware subclass so the fallen pose is re-written for
  # delay envs AFTER the command's standard write — making the delay-env
  # reset chain idempotent end-to-end.
  old_motion = cfg.commands["motion"]
  assert isinstance(old_motion, PklMotionCommandCfg)
  cfg.commands["motion"] = DelayRecoveryPklMotionCommandCfg(
    entity_name=old_motion.entity_name,
    resampling_time_range=old_motion.resampling_time_range,
    debug_vis=old_motion.debug_vis,
    pose_range=old_motion.pose_range,
    velocity_range=old_motion.velocity_range,
    joint_position_range=old_motion.joint_position_range,
    motion_file=old_motion.motion_file,
    anchor_body_name=old_motion.anchor_body_name,
    body_names=old_motion.body_names,
    adaptive_kernel_size=old_motion.adaptive_kernel_size,
    adaptive_lambda=old_motion.adaptive_lambda,
    adaptive_uniform_ratio=old_motion.adaptive_uniform_ratio,
    adaptive_alpha=old_motion.adaptive_alpha,
    sampling_mode=old_motion.sampling_mode,
  )

  # --- (1) Install delay-reset machinery. --------------------------------
  # ``init_motion_loader`` patches ``env.termination_manager`` with a
  # DelayedTerminationManager and loads WalkandRun + Recovery clips into the
  # process-global MotionResetManager singleton.
  cfg.events["init_amp_motion_loader"] = EventTermCfg(
    func=amp_mdp.init_motion_loader,
    mode="startup",
    params={
      "motion_dir": _WALK_RUN_DIR,
      "recovery_dir": _RECOVERY_DIR,
      "delay_reset_env_ratio": 0.0 if play else 0.20,
      "max_delay_steps": 250,
      # Filter the Recovery/ frame buffer at load to keep only genuinely
      # fallen frames (low pelvis + tilted). Without this, ~70% of post-
      # reset spawns land in standing/crouched frames that don't trigger
      # ``recovery_active`` — those delay envs walk normally for the rest
      # of the episode and contribute zero recovery training samples.
      # Threshold defaults match the pose detector via the shared
      # ``POSE_RECOVERY_*`` constants in amp_mdp.py.
      "recovery_fallen_only": True,
    },
  )
  # ``reset_delay_envs_to_recovery`` overrides ONLY the delay-tagged envs
  # with a fallen Recovery/ frame on each reset. Non-delay envs keep the
  # PKL motion-driven reset that already runs upstream.
  cfg.events["reset_delay_envs_to_recovery"] = EventTermCfg(
    func=amp_mdp.reset_delay_envs_to_recovery,
    mode="reset",
    params={
      "motion_dir": _WALK_RUN_DIR,
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
    },
  )

  # --- (2) AMP-teacher actor obs group. ----------------------------------
  # Mirrors AMP_mjlab/amp_config.py:_build_amp_observations actor terms,
  # but the command source is ``loco_uniform_cmd`` (the unicmd env's twist
  # command) rather than the velocity task's ``twist`` term.
  noise = not play
  amp_actor_terms: dict[str, ObservationTermCfg] = {
    "base_ang_vel": ObservationTermCfg(
      func=env_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2) if noise else None,
    ),
    "projected_gravity": ObservationTermCfg(
      func=env_mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05) if noise else None,
    ),
    "command": ObservationTermCfg(
      func=env_mdp.generated_commands,
      params={"command_name": _LOCO_UNIFORM_CMD_NAME},
    ),
    "joint_pos": ObservationTermCfg(
      func=env_mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01) if noise else None,
    ),
    "joint_vel": ObservationTermCfg(
      func=env_mdp.joint_vel_rel,
      noise=Unoise(n_min=-0.5, n_max=0.5) if noise else None,
    ),
    "actions": ObservationTermCfg(func=env_mdp.last_action),
  }
  cfg.observations["amp_teacher_actor"] = ObservationGroupCfg(
    terms=amp_actor_terms,
    concatenate_terms=True,
    enable_corruption=noise,
    history_length=4,
    # Default flatten_history_dim=True so the AMP teacher actor receives a
    # flat 4×96 = 384-D tensor (matches its training obs shape).
  )
  cfg.observations["recovery_active"] = ObservationGroupCfg(
    terms={
      # Stateful pose-based detector: enters recovery on (low pelvis +
      # tilted), holds until full upright (high pelvis + low tilt). See
      # ``RecoveryActiveTerm`` docstring for hysteresis details. Fixes
      # the spurious deactivation when the env's motion command happens
      # to expect a low-pelvis reference (sitting / crouch / kneel /
      # crawl clips — ~10% of seed_cbf_standing).
      "recovery_active": ObservationTermCfg(
        func=wbc_obs.RecoveryActiveTerm,
        params={
          "fall_z_threshold": 0.45,
          "fall_tilt_threshold_rad": 1.0,
          "recover_z_threshold": 0.70,
          "recover_tilt_threshold_rad": 0.3,
          "asset_name": "robot",
        },
      ),
    },
    concatenate_terms=True,
    enable_corruption=False,
  )

  # --- (3) Recovery-aware reward gating. ---------------------------------
  # Wrap velocity / hand-tracking / motion-root rewards so they are × 0
  # on delay envs in the down state. Add the AMP-style track_root_height
  # reward (only active on those envs, ×3.5).
  rew = cfg.rewards
  _RECOVERY_GATED_REWARDS: tuple[tuple[str, str], ...] = (
    ("tracking_root_linear_vel", "wbc_mjlab.rewards:tracking_root_linear_vel"),
    ("tracking_root_angular_vel", "wbc_mjlab.rewards:tracking_root_angular_vel"),
    ("tracking_hand_pos", "wbc_mjlab.rewards:tracking_hand_pos"),
    ("tracking_root_translation_z", "wbc_mjlab.rewards:tracking_root_translation_z"),
    ("tracking_root_rotation", "wbc_mjlab.rewards:tracking_root_rotation"),
  )
  for term_name, reward_func_path in _RECOVERY_GATED_REWARDS:
    term = rew.get(term_name)
    if term is None:
      continue
    term.params = {
      **(term.params or {}),
      "reward_func_path": reward_func_path,
      "delay_env_rew_ratio": 0.0,
    }
    # Pose-based gating shared with ``RecoveryActiveTerm`` — keeps reward
    # gating consistent with AMP-KL gating in DaggerPPO.
    term.func = amp_mdp.wbc_reward_with_pose_recovery_mask

  # AMP-style get-up reward: gated to recovery envs only, ×3.5. Uses
  # pose-based gating shared with the AMP-KL routing.
  rew["track_root_height_recovery"] = RewardTermCfg(
    func=amp_mdp.track_root_height_pose,
    weight=1.0,
    params={"std": 0.3, "mask_delay": True, "delay_env_rew_ratio": 3.5},
  )

  # --- (4) Re-fall termination — keeps delay envs cycling through recovery. ---
  # When a delay env stands up (pose-based recovery_active goes True→False),
  # this termination fires → standard reset machinery routes it back to a
  # Recovery/ frame (fallen pose) → AMP-KL fires again. Without this, a
  # delay env spends ~95% of its episode walking normally after a single
  # recovery, contributing zero recovery training samples for the rest of
  # the episode. With it, delay envs are dedicated to recovery training:
  #   spawn fallen → recover → fire → reset → spawn fallen → ...
  # NOT a fall termination — DelayedTerminationManager passes it through.
  cfg.terminations["delay_env_recovered"] = TerminationTermCfg(
    func=amp_mdp.delay_env_recovered_termination,
    time_out=False,
  )

  # --- (5) Metrics. ------------------------------------------------------
  from mjlab.managers.metrics_manager import MetricsTermCfg

  cfg.metrics = {**(cfg.metrics or {})}
  cfg.metrics["mean_delay_steps"] = MetricsTermCfg(func=amp_mdp.mean_delay_steps)

  _set_wbc_default_num_envs(cfg, play=play)
  return cfg


def unitree_g1_hand_dual_teacher_flat_unicmd_nobv_amp_stable_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """MoE UniCmd NoBV AMP student variant with whole-body stability rewards added."""
  cfg = unitree_g1_hand_dual_teacher_flat_unicmd_nobv_amp_env_cfg(play=play)
  _apply_stability_rewards(cfg)
  return cfg


def unitree_g1_loco_teacher_flat_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the native 15-DoF locomotion teacher on top of the flat velocity task."""
  cfg = _configure_loco_teacher_15dof_arm(
    cfg=unitree_g1_flat_env_cfg(play=play),
    play=play,
  )
  if not play:
    cfg.events["loco_teacher_base_mass"] = EventTermCfg(
      mode="startup",
      func=dr.body_mass,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
        "operation": "add",
        "ranges": _HANDOFF_BASE_MASS_RANGE,
      },
    )

  _apply_loco_teacher_reward_shaping(cfg)

  _set_wbc_default_num_envs(cfg, play=play)
  return cfg


def unitree_g1_loco_teacher_flat_nobv_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Non-privileged locomotion teacher: actor observation drops ``base_lin_vel``.

  Identical to ``unitree_g1_loco_teacher_flat_env_cfg`` except the actor
  observation excludes base linear velocity so the teacher's policy is
  representable by a deployment-aware student that also cannot observe
  base linear velocity.  Critic keeps the full (privileged) observation.
  """
  cfg = unitree_g1_loco_teacher_flat_env_cfg(play=play)
  cfg.observations["actor"] = ObservationGroupCfg(
    terms=_make_loco_teacher_obs_terms(
      enable_noise=not play, include_base_lin_vel=False
    ),
    concatenate_terms=True,
    enable_corruption=not play,
  )
  # Critic keeps the full privileged observation (base_lin_vel included).
  return cfg


def unitree_g1_loco_teacher_flat_nobv_stable_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """NoBV loco teacher variant with whole-body stability rewards added."""
  cfg = unitree_g1_loco_teacher_flat_nobv_env_cfg(play=play)
  _apply_stability_rewards(cfg)
  return cfg


def _apply_loco_teacher_reward_shaping(cfg: ManagerBasedRlEnvCfg) -> None:
  """Apply shared 15-DoF loco-teacher reward + termination shaping.

  Rescales pose/foot-height penalties on top of the base velocity task and
  adds gait, stance-width, and foot-overlap terms. Callers that start from
  either ``unitree_g1_flat_env_cfg`` or ``unitree_g1_rough_env_cfg`` get the
  same downstream reward stack; task-specific overrides (e.g. the stair
  variant's ``feet_clearance_stair``) must run AFTER this helper so
  they can consume the updated weights.
  """
  cfg.rewards["pose"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=wbc_obs.BODY_JOINT_NAMES
  )
  cfg.rewards["pose"].params["std_walking"] = dict(_LOCO_BODY_STD_WALKING)
  cfg.rewards["pose"].params["std_running"] = dict(_LOCO_BODY_STD_RUNNING)
  cfg.rewards["foot_clearance"].weight = -6.0
  cfg.rewards["foot_clearance"].params["target_height"] = 0.05
  cfg.rewards["foot_swing_height"].weight = -0.75
  cfg.rewards["foot_swing_height"].params["target_height"] = 0.08
  cfg.rewards["stand_pose"] = RewardTermCfg(
    func=wbc_rewards.stand_pose,
    weight=-5.0,
    params={
      "command_name": "twist",
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
    },
  )
  cfg.rewards["flat_foot"] = RewardTermCfg(
    func=wbc_rewards.flat_foot,
    weight=-0.5,
    params={
      "sensor_name": "feet_ground_contact",
      "asset_cfg": SceneEntityCfg("robot", body_names=wbc_obs.FEET_BODY_NAMES),
    },
  )
  cfg.rewards["gait_phase_contact"] = RewardTermCfg(
    func=wbc_rewards.gait_phase_contact,
    weight=0.5,
    params={
      "sensor_name": "feet_ground_contact",
      "command_name": "twist",
      "gait_period": wbc_obs.LOCO_GAIT_PERIOD,
      "gait_offset": wbc_obs.LOCO_GAIT_OFFSET,
      "stance_ratio": wbc_obs.LOCO_STANCE_RATIO,
    },
  )
  cfg.rewards["feet_distance_lateral"] = RewardTermCfg(
    func=wbc_rewards.feet_distance_lateral,
    weight=0.5,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=wbc_obs.FEET_BODY_NAMES),
      "min_distance": 0.2,
      "max_distance": 0.35,
    },
  )
  cfg.rewards["knee_distance_lateral"] = RewardTermCfg(
    func=wbc_rewards.knee_distance_lateral,
    weight=1.0,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=wbc_obs.KNEE_BODY_NAMES),
      "min_distance": 0.2,
      "max_distance": 0.35,
    },
  )
  cfg.terminations["foot_overlap"] = TerminationTermCfg(
    func=wbc_terms.foot_overlap,
    params={
      "sensor_name": "feet_ground_contact",
      "threshold": 0.05,
      "asset_cfg": SceneEntityCfg("robot", body_names=wbc_obs.FEET_BODY_NAMES),
    },
  )


def _configure_loco_teacher_15dof_arm(
  cfg: ManagerBasedRlEnvCfg,
  *,
  play: bool,
  command_ranges: dict[str, tuple[float, float]] | None = None,
  include_height_scan: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Apply shared locomotion-teacher 15-DoF + motion-arm wiring to a velocity cfg."""
  if command_ranges is None:
    command_ranges = {
      "lin_vel_x": (-1.0, 1.0),
      "lin_vel_y": (-1.0, 1.0),
      "ang_vel_z": (-1.0, 1.0),
    }
    warmup_ranges = {
      "lin_vel_x": (-0.5, 0.5),
      "lin_vel_y": (-0.5, 0.5),
      "ang_vel_z": (-0.5, 0.5),
    }
  else:
    warmup_ranges = {
      "lin_vel_x": _scale_range(command_ranges["lin_vel_x"], _LOCO_WARMUP_SCALE),
      "lin_vel_y": _scale_range(command_ranges["lin_vel_y"], _LOCO_WARMUP_SCALE),
      "ang_vel_z": _scale_range(command_ranges["ang_vel_z"], _LOCO_WARMUP_SCALE),
    }

  twist_cmd = cast(UniformVelocityCommandCfg, cfg.commands["twist"])
  twist_cmd.ranges.lin_vel_x = command_ranges["lin_vel_x"]
  twist_cmd.ranges.lin_vel_y = command_ranges["lin_vel_y"]
  twist_cmd.ranges.ang_vel_z = command_ranges["ang_vel_z"]
  cfg.curriculum["command_vel"] = CurriculumTermCfg(
    func=velocity_mdp.commands_vel,
    params={
      "command_name": "twist",
      "velocity_stages": [
        {
          "step": 0,
          "lin_vel_x": warmup_ranges["lin_vel_x"],
          "lin_vel_y": warmup_ranges["lin_vel_y"],
          "ang_vel_z": warmup_ranges["ang_vel_z"],
        },
        {
          "step": 5000 * 24,
          "lin_vel_x": command_ranges["lin_vel_x"],
          "lin_vel_y": command_ranges["lin_vel_y"],
          "ang_vel_z": command_ranges["ang_vel_z"],
        },
      ],
    },
  )

  cfg.commands["motion"] = LocoArmMotionCommandCfg(
    motion_file=_HANDOFF_LOCO_MOTION_FILE,
    body_names=("torso_link",),
  )

  import os
  try:
    arm_blend = float(os.environ.get("ARM_BLEND", "0.0"))
  except ValueError:
    print(f"[WARN] Invalid ARM_BLEND value: {os.environ.get('ARM_BLEND')}. Defaulting to 0.0")
    arm_blend = 0.0


  joint_pos_action = G1LocoTeacherActionCfg(
    entity_name="robot",
    body_joint_names=wbc_obs.BODY_JOINT_NAMES,
    arm_joint_names=wbc_obs.ARM_JOINT_NAMES,
    motion_command_name="motion",
    scale=DEFAULT_G1_LOCO_TEACHER_ACTION_SCALE,
    use_default_offset=True,
    init_blend=arm_blend,
    curriculum_start_step=12500 * 24,
    curriculum_step=0.002,
    curriculum_threshold_lin=0.7,
    curriculum_threshold_ang=0.45,
  )
  cfg.actions["joint_pos"] = joint_pos_action
  cfg.curriculum["loco_arm_blend"] = CurriculumTermCfg(
    func=wbc_curriculums.loco_arm_blend,
    params={
      "action_term_name": "joint_pos",
      "command_name": joint_pos_action.standing_command_name,
      "tracking_lin_reward_name": joint_pos_action.tracking_lin_reward_name,
      "tracking_ang_reward_name": joint_pos_action.tracking_ang_reward_name,
    },
  )

  cfg.observations = {
    "actor": ObservationGroupCfg(
      terms=_make_loco_teacher_obs_terms(
        enable_noise=not play,
        include_height_scan=include_height_scan,
      ),
      concatenate_terms=True,
      enable_corruption=not play,
    ),
    "critic": ObservationGroupCfg(
      terms=_make_loco_teacher_obs_terms(
        enable_noise=False,
        include_height_scan=include_height_scan,
      ),
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  _set_wbc_default_num_envs(cfg, play=play)
  return cfg


