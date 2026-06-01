"""Env + runner configs for AMP-teacher tasks.

The env cfg builds on mjlab's stock G1 velocity task (same starting point as
``unitree_g1_loco_teacher_flat_env_cfg``) and overrides observations, rewards,
events, and terminations to match AMP_mjlab's setup.

Actor / critic use a single ``actor`` / ``critic`` obs group with 4-frame
history flattened into the input (``history_length=4``,
``flatten_history_dim=True``, default ``history_ordering="term"``) — same
ordering convention as wbc_mjlab's WBC and loco teachers, which lets us avoid
both ``history_ordering="time"`` and the AMP_mjlab ``mjlab_patch``. The actor
is a plain MLP (the input dim just grows 4×); we don't need the WBC teacher's
split ``actor_current`` + ``actor_history`` groups + temporal encoder. AMP
teacher is intended to eventually replace the loco teacher slot in the
dual-teacher hand-student stack, so the ONNX export contract (single tensor
in, action mean out) is compatible with
``DaggerOnPolicyRunner._load_optional_teacher``.
"""

from __future__ import annotations

import math
import os
from copy import deepcopy

import numpy as np

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg
from mjlab.tasks.velocity import mdp as vel_mdp
from mjlab.tasks.velocity.config.g1.env_cfgs import (
  unitree_g1_flat_env_cfg,
  unitree_g1_rough_env_cfg,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from wbc_mjlab import amp_mdp
from wbc_mjlab.amp_ppo_config import AmpAlgorithmCfg, AmpRunnerCfg
from wbc_mjlab.g1_constants_custom import _WBC_SAVE_INTERVAL


# =============================================================================
# Constants — body / anchor names + motion data path.

_AMP_BODY_NAMES: tuple[str, ...] = (
  "pelvis",
  "left_hip_roll_link",
  "left_knee_link",
  "left_ankle_roll_link",
  "right_hip_roll_link",
  "right_knee_link",
  "right_ankle_roll_link",
  "left_shoulder_roll_link",
  "left_elbow_link",
  "left_wrist_yaw_link",
  "right_shoulder_roll_link",
  "right_elbow_link",
  "right_wrist_yaw_link",
)
_AMP_ANCHOR_NAME = "torso_link"
_AMP_ROOT_NAME = "pelvis"

_AMP_DATA_DIR = os.path.abspath(
  os.path.join(os.path.dirname(__file__), "data", "motions", "g1", "amp")
)
_WALK_RUN_DIR = os.path.join(_AMP_DATA_DIR, "WalkandRun")
_RECOVERY_DIR = os.path.join(_AMP_DATA_DIR, "Recovery")


# =============================================================================
# Env cfg builders.

def _build_amp_observations(play: bool) -> dict[str, ObservationGroupCfg]:
  """3 obs groups: actor (proprio), critic (privileged), amp (discriminator)."""
  noise = not play
  actor_terms: dict[str, ObservationTermCfg] = {
    "base_ang_vel": ObservationTermCfg(
      func=envs_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2) if noise else None,
    ),
    "projected_gravity": ObservationTermCfg(
      func=envs_mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05) if noise else None,
    ),
    "command": ObservationTermCfg(
      func=envs_mdp.generated_commands, params={"command_name": "twist"}
    ),
    "joint_pos": ObservationTermCfg(
      func=envs_mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01) if noise else None,
    ),
    "joint_vel": ObservationTermCfg(
      func=envs_mdp.joint_vel_rel,
      noise=Unoise(n_min=-0.5, n_max=0.5) if noise else None,
    ),
    "actions": ObservationTermCfg(func=envs_mdp.last_action),
  }

  critic_terms: dict[str, ObservationTermCfg] = {
    **actor_terms,
    "base_lin_vel": ObservationTermCfg(
      func=envs_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
    ),
    "body_pos_b": ObservationTermCfg(
      func=amp_mdp.robot_body_pos_b,
      params={
        "anchor_cfg": SceneEntityCfg("robot", body_names=(_AMP_ANCHOR_NAME,)),
        "body_cfg": SceneEntityCfg("robot", body_names=_AMP_BODY_NAMES),
      },
    ),
    "body_ori_b": ObservationTermCfg(
      func=amp_mdp.robot_body_ori_b,
      params={
        "anchor_cfg": SceneEntityCfg("robot", body_names=(_AMP_ANCHOR_NAME,)),
        "body_cfg": SceneEntityCfg("robot", body_names=_AMP_BODY_NAMES),
      },
    ),
  }

  amp_terms: dict[str, ObservationTermCfg] = {
    "body_pos_b": ObservationTermCfg(
      func=amp_mdp.robot_body_pos_b,
      params={
        "anchor_cfg": SceneEntityCfg("robot", body_names=(_AMP_ANCHOR_NAME,)),
        "body_cfg": SceneEntityCfg("robot", body_names=_AMP_BODY_NAMES),
      },
    ),
    "body_ori_b": ObservationTermCfg(
      func=amp_mdp.robot_body_ori_b,
      params={
        "anchor_cfg": SceneEntityCfg("robot", body_names=(_AMP_ANCHOR_NAME,)),
        "body_cfg": SceneEntityCfg("robot", body_names=_AMP_BODY_NAMES),
      },
    ),
    "body_lin_vel_b": ObservationTermCfg(
      func=amp_mdp.robot_body_lin_vel_b,
      params={
        "anchor_cfg": SceneEntityCfg("robot", body_names=(_AMP_ANCHOR_NAME,)),
        "body_cfg": SceneEntityCfg("robot", body_names=_AMP_BODY_NAMES),
      },
    ),
    "body_ang_vel_b": ObservationTermCfg(
      func=amp_mdp.robot_body_ang_vel_b,
      params={
        "anchor_cfg": SceneEntityCfg("robot", body_names=(_AMP_ANCHOR_NAME,)),
        "body_cfg": SceneEntityCfg("robot", body_names=_AMP_BODY_NAMES),
      },
    ),
  }
  # 4-frame actor / critic history matches AMP_mjlab; we use mjlab's default
  # ``history_ordering="term"`` (same convention as wbc_mjlab's WBC + loco
  # teachers, neither of which sets ``"time"``) so we don't need the
  # ``mjlab_patch`` AMP_mjlab carries. ``flatten_history_dim=True`` (default)
  # means the actor still sees a flat tensor — no temporal encoder needed,
  # the MLP input dim just grows by 4×. The discriminator's ``amp`` group
  # stays single-frame because it scores (s, s') pairs.
  return {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=noise,
      history_length=4,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
      history_length=4,
    ),
    "amp": ObservationGroupCfg(
      terms=amp_terms, concatenate_terms=True, enable_corruption=False,
    ),
  }


def _replace_with_amp_rewards(cfg: ManagerBasedRlEnvCfg) -> None:
  """Drop the velocity-task reward set and install AMP's reward shaping."""
  # Preserve mjlab's foot_slip term — its asset_cfg.site_names are already
  # wired by unitree_g1_{flat,rough}_env_cfg. Override the weight to match
  # AMP_mjlab (-0.25 vs mjlab's stock -0.1).
  foot_slip_term = cfg.rewards.get("foot_slip")
  if foot_slip_term is None:
    raise ValueError(
      "Expected mjlab's velocity env cfg to provide a 'foot_slip' reward "
      "term; AMP teacher relies on it. Did mjlab's stock cfg change?"
    )
  foot_slip_term.weight = -0.25

  cfg.rewards = {
    "track_anchor_linear_velocity": RewardTermCfg(
      func=amp_mdp.track_anchor_linear_velocity,
      weight=1.0,
      params={
        "command_name": "twist",
        "std": 1.0,
        "mask_delay": True,
        "delay_env_rew_ratio": 0.0,
        "anchor_cfg": SceneEntityCfg("robot", body_names=(_AMP_ANCHOR_NAME,)),
      },
    ),
    "track_anchor_angular_velocity": RewardTermCfg(
      func=amp_mdp.track_anchor_angular_velocity,
      weight=1.0,
      params={
        "command_name": "twist",
        "std": 3.14,
        "mask_delay": True,
        "delay_env_rew_ratio": 0.0,
        "anchor_cfg": SceneEntityCfg("robot", body_names=(_AMP_ANCHOR_NAME,)),
      },
    ),
    "track_root_height": RewardTermCfg(
      func=amp_mdp.track_root_height,
      weight=1.0,
      params={"std": 0.3, "mask_delay": True, "delay_env_rew_ratio": 3.5},
    ),
    "body_ang_vel_xy_l2": RewardTermCfg(
      func=amp_mdp.body_ang_vel_xy_l2,
      weight=0.5,
      params={
        "std": 3.14,
        "mask_delay": True,
        "delay_env_rew_ratio": 0.0,
        "body_cfg": SceneEntityCfg("robot", body_names=(_AMP_ROOT_NAME,)),
      },
    ),
    "is_terminated": RewardTermCfg(func=envs_mdp.is_terminated, weight=-200.0),
    "joint_acc_l2": RewardTermCfg(func=envs_mdp.joint_acc_l2, weight=-2.5e-7),
    "joint_pos_limits": RewardTermCfg(
      func=envs_mdp.joint_pos_limits, weight=-10.0
    ),
    "action_rate_l2": RewardTermCfg(
      func=envs_mdp.action_rate_l2, weight=-0.01
    ),
    "foot_slip": foot_slip_term,
    "self_collisions": RewardTermCfg(
      func=amp_mdp.self_collision_cost,
      weight=-0.1,
      params={"sensor_name": "self_collision", "force_threshold": 10.0},
    ),
  }


def _replace_with_amp_terminations(cfg: ManagerBasedRlEnvCfg) -> None:
  """AMP uses a simpler termination set than the velocity task."""
  cfg.terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "bad_orientation": TerminationTermCfg(
      func=envs_mdp.bad_orientation,
      params={"limit_angle": math.radians(70.0)},
    ),
    "bad_base_height": TerminationTermCfg(
      func=envs_mdp.root_height_below_minimum,
      params={"minimum_height": 0.5},
    ),
  }


def _add_amp_motion_events(
  cfg: ManagerBasedRlEnvCfg,
  *,
  motion_dir: str,
  recovery_dir: str | None,
  delay_reset_env_ratio: float,
  max_delay_steps: int,
) -> None:
  """Inject AMP's motion-loader startup + per-reset events."""
  cfg.events["init_motion_loader"] = EventTermCfg(
    func=amp_mdp.init_motion_loader,
    mode="startup",
    params={
      "motion_dir": motion_dir,
      "recovery_dir": recovery_dir,
      "delay_reset_env_ratio": delay_reset_env_ratio,
      "max_delay_steps": max_delay_steps,
    },
  )
  cfg.events["reset_from_motion"] = EventTermCfg(
    func=amp_mdp.reset_from_motion_data,
    mode="reset",
    params={
      "motion_dir": motion_dir,
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
    },
  )
  # Drop velocity-task's "reset_robot_*" events that AMP replaces.
  for k in ("reset_base", "reset_joints", "reset_robot_joints"):
    cfg.events.pop(k, None)


def _add_amp_metrics(cfg: ManagerBasedRlEnvCfg) -> None:
  cfg.metrics = {**(cfg.metrics or {})}
  cfg.metrics["mean_delay_steps"] = MetricsTermCfg(func=amp_mdp.mean_delay_steps)




def unitree_g1_amp_teacher_flat_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """AMP teacher on flat terrain. Mirrors AMP_mjlab/Unitree-G1-AMP-Flat."""
  cfg = unitree_g1_flat_env_cfg(play=play)

  cfg.observations = _build_amp_observations(play=play)
  _replace_with_amp_rewards(cfg)
  _replace_with_amp_terminations(cfg)
  _add_amp_motion_events(
    cfg,
    motion_dir=_WALK_RUN_DIR,
    recovery_dir=_RECOVERY_DIR,
    delay_reset_env_ratio=1.0 if play else 0.4,
    max_delay_steps=250,
  )
  _add_amp_metrics(cfg)

  cfg.episode_length_s = int(1e9) if play else 20.0
  return cfg


# =============================================================================
# Runner cfg builders.

def _amp_runner_cfg(experiment_name: str) -> AmpRunnerCfg:
  return AmpRunnerCfg(
    seed=42,
    num_steps_per_env=24,
    max_iterations=10_001,
    obs_groups={
      "actor": ("actor",),
      "critic": ("critic",),
    },
    save_interval=_WBC_SAVE_INTERVAL,
    experiment_name=experiment_name,
    run_name=experiment_name,
    logger="wandb",
    wandb_project="wbc_mjlab",
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=AmpAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
      amp_loss_coef=1.0,
      amp_grad_pen_lambda=10.0,
    ),
    amp_motion_files=_AMP_DATA_DIR,
    amp_body_names=_AMP_BODY_NAMES,
    amp_anchor_name=_AMP_ANCHOR_NAME,
    amp_reward_coef=0.1,
    amp_task_reward_lerp=0.75,
    amp_discr_hidden_dims=(1024, 512, 256),
    amp_replay_buffer_size=100_000,
  )


def unitree_g1_amp_teacher_flat_runner_cfg() -> AmpRunnerCfg:
  return _amp_runner_cfg(experiment_name="g1_amp_teacher_flat")


