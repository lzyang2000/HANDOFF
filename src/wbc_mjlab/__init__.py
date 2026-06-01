"""wbc_mjlab — fully-stable 3-teacher MoE student pipeline for the Unitree G1.

This is a stripped build that registers only the four tasks needed to produce
the fully-stable (seed + AMP + stability-reward) dual-teacher MoE student:

  1. Wbc-Teacher-Flat-Unitree-G1-Stable               (stable WBC teacher)
  2. Loco-Teacher-Flat-Unitree-G1-NoBV-Stable         (stable NoBV loco teacher)
  3. Amp-Teacher-Flat-Unitree-G1                       (AMP recovery teacher)
  4. Wbc-Hand-Dual-Teacher-Flat-Unitree-G1-MoE-UniCmd-NoBV-AMP-Stable (student)

See README.md for the training order and the wrapper scripts.
"""

from mjlab.tasks.registry import register_mjlab_task

from wbc_mjlab.config import unitree_g1_wbc_teacher_flat_stable_env_cfg
from wbc_mjlab.config import unitree_g1_loco_teacher_flat_nobv_stable_env_cfg
from wbc_mjlab.config import (
  unitree_g1_hand_dual_teacher_flat_unicmd_nobv_amp_stable_env_cfg,
)
from wbc_mjlab.amp_config import (
  unitree_g1_amp_teacher_flat_env_cfg,
  unitree_g1_amp_teacher_flat_runner_cfg,
)
from wbc_mjlab.rl import AmpOnPolicyRunner
from wbc_mjlab.rl import DaggerOnPolicyRunner
from wbc_mjlab.rl_cfg import unitree_g1_pkl_tracking_custom_ppo_runner_cfg
from wbc_mjlab.rl_cfg import unitree_g1_loco_teacher_flat_nobv_runner_cfg
from wbc_mjlab.rl_cfg import unitree_g1_hand_moe_flat_unicmd_nobv_amp_runner_cfg
from wbc_mjlab.g1_constants_custom import attach_payloads_to_scene_robot


def _build_env_cfg_with_payloads(env_fn, *, play: bool):
  """Build env_cfg via ``env_fn(play=play)`` and wrap the robot's spec_fn
  with the sim-to-real hardware payload (Jetson + Dex1-1). Idempotent."""
  cfg = env_fn(play=play)
  attach_payloads_to_scene_robot(cfg)
  return cfg


# DAgger-runner tasks: the two stable teachers + the fully-stable MoE student.
_DAGGER_TASKS = [
    ("Wbc-Teacher-Flat-Unitree-G1-Stable",
     unitree_g1_wbc_teacher_flat_stable_env_cfg,
     unitree_g1_pkl_tracking_custom_ppo_runner_cfg),
    ("Loco-Teacher-Flat-Unitree-G1-NoBV-Stable",
     unitree_g1_loco_teacher_flat_nobv_stable_env_cfg,
     unitree_g1_loco_teacher_flat_nobv_runner_cfg),
    ("Wbc-Hand-Dual-Teacher-Flat-Unitree-G1-MoE-UniCmd-NoBV-AMP-Stable",
     unitree_g1_hand_dual_teacher_flat_unicmd_nobv_amp_stable_env_cfg,
     unitree_g1_hand_moe_flat_unicmd_nobv_amp_runner_cfg),
]

for _task_id, _env_fn, _rl_fn in _DAGGER_TASKS:
  register_mjlab_task(
    task_id=_task_id,
    env_cfg=_build_env_cfg_with_payloads(_env_fn, play=False),
    play_env_cfg=_build_env_cfg_with_payloads(_env_fn, play=True),
    rl_cfg=_rl_fn(),
    runner_cls=DaggerOnPolicyRunner,
  )


# AMP recovery teacher uses AmpOnPolicyRunner + AmpPPO (separate runner).
register_mjlab_task(
  task_id="Amp-Teacher-Flat-Unitree-G1",
  env_cfg=_build_env_cfg_with_payloads(
    unitree_g1_amp_teacher_flat_env_cfg, play=False
  ),
  play_env_cfg=_build_env_cfg_with_payloads(
    unitree_g1_amp_teacher_flat_env_cfg, play=True
  ),
  rl_cfg=unitree_g1_amp_teacher_flat_runner_cfg(),
  runner_cls=AmpOnPolicyRunner,
)
