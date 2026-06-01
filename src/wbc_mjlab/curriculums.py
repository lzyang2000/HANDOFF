"""Custom curriculum terms for wbc_mjlab tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

from wbc_mjlab.events import HAND_FORCE_EVENT_NAME, get_event_term

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")




def loco_arm_blend(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice,
  action_term_name: str = "joint_pos",
  command_name: str = "twist",
  tracking_lin_reward_name: str = "track_linear_velocity",
  tracking_ang_reward_name: str = "track_angular_velocity",
) -> torch.Tensor:
  """Advance the locomotion teacher's arm-motion blend using episodic rewards."""
  action_term = env.action_manager.get_term(action_term_name)
  env_ids_tensor = _slice_to_env_ids(env, env_ids)
  blend = float(action_term.arm_blend_factor)
  if env_ids_tensor.numel() == 0:
    return torch.tensor(blend, device=env.device)
  if env.common_step_counter < action_term.cfg.curriculum_start_step:
    return torch.tensor(blend, device=env.device)

  twist_cmd = env.command_manager.get_term(command_name)
  walking_ids = env_ids_tensor[~twist_cmd.is_standing_env[env_ids_tensor]]
  if walking_ids.numel() == 0:
    return torch.tensor(blend, device=env.device)

  reward_manager = env.reward_manager
  episode_sums = reward_manager._episode_sums
  if (
    tracking_lin_reward_name not in episode_sums
    or tracking_ang_reward_name not in episode_sums
  ):
    return torch.tensor(blend, device=env.device)

  mean_lin = (
    torch.mean(episode_sums[tracking_lin_reward_name][walking_ids])
    / env.max_episode_length_s
  )
  mean_ang = (
    torch.mean(episode_sums[tracking_ang_reward_name][walking_ids])
    / env.max_episode_length_s
  )
  lin_weight = reward_manager.get_term_cfg(tracking_lin_reward_name).weight
  ang_weight = reward_manager.get_term_cfg(tracking_ang_reward_name).weight
  if (
    mean_lin > action_term.cfg.curriculum_threshold_lin * lin_weight
    and mean_ang > action_term.cfg.curriculum_threshold_ang * ang_weight
  ):
    blend = min(blend + action_term.cfg.curriculum_step, 1.0)
    action_term.set_arm_blend_factor(blend)
  return torch.tensor(blend, device=env.device)


def _slice_to_env_ids(
  env: ManagerBasedRlEnv, env_ids: torch.Tensor | slice
) -> torch.Tensor:
  if isinstance(env_ids, slice):
    return torch.arange(env.num_envs, device=env.device)[env_ids]
  return env_ids


