"""Stateful event terms for `wbc_mjlab` tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.event_manager import EventTermCfg


HAND_FORCE_EVENT_NAME = "falcon_hand_force_randomization"
HAND_FORCE_APPLY_LINKS: tuple[str, str] = (
  "left_wrist_yaw_link",
  "right_wrist_yaw_link",
)
HAND_FORCE_OBS_DIM = 6


def get_event_term(env: "ManagerBasedRlEnv", event_name: str):
  for mode in env.event_manager._mode_term_names:
    names = env.event_manager._mode_term_names[mode]
    if event_name not in names:
      continue
    index = names.index(event_name)
    return env.event_manager._mode_term_cfgs[mode][index].func
  raise ValueError(f"Event term '{event_name}' not found in active event terms.")


