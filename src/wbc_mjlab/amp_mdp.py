"""AMP-teacher MDP terms: observations, events, terminations, rewards, metrics.

Ported from AMP_mjlab/src/tasks/amp_loco/{ampmotion_loader,mdp/*}.py and bundled
into a single module so the env-side AMP code lives next to ``pkl_motion_lib.py``
without sprouting a sub-package.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationManager
from mjlab.utils.lab_api.math import (
  euler_xyz_from_quat,
  matrix_from_quat,
  quat_apply,
  quat_apply_inverse,
  subtract_frame_transforms,
  yaw_quat,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


# =============================================================================
# Pose-based recovery state tracker (shared by obs term + reward wrappers).
#
# The dual-teacher recovery student uses pose, not termination, to flag the
# "fallen / mid-recovery" state. A single source of truth shared between the
# ``RecoveryActiveTerm`` observation term and the reward wrappers below
# eliminates the failure mode where rewards and KL gating disagree on which
# envs are recovering — that disagreement is what caused the iter-4k
# stuck-mid-recovery training symptom (KL pulls toward standing while task
# reward pulls toward whatever motion clip is sampled).
#
# Thresholds are module-level so the obs term and reward wrappers can't
# silently diverge. Override per env-cfg if needed.

POSE_RECOVERY_FALL_Z: float = 0.45              # m — pelvis below this AND tilted → enter recovery
POSE_RECOVERY_FALL_TILT_RAD: float = 1.0        # ≈57° — squat keeps torso < 0.3 rad
POSE_RECOVERY_RECOVER_Z: float = 0.70           # m — pelvis above this AND upright → exit
POSE_RECOVERY_RECOVER_TILT_RAD: float = 0.3     # ≈17° — well within commanded squat envelope


class _PoseRecoveryStateTracker:
  """Per-env sticky pose-based recovery flag, shared via singleton.

  Updated lazily by whichever caller hits first in a given env step
  (memoized by ``env.common_step_counter``). Both obs term and reward
  wrappers query the same instance, guaranteeing they see the same
  recovery mask within a step.

  State semantics: ``active[i] = True`` iff env ``i`` is currently in
  the fallen-or-mid-recovery window. Sticky: enters on (low pelvis +
  tilted), holds until (high pelvis + upright) — see module-level
  ``POSE_RECOVERY_*`` thresholds.
  """

  _instance: "_PoseRecoveryStateTracker | None" = None

  def __init__(self) -> None:
    self.active: torch.Tensor | None = None
    # Snapshot of ``active`` from the previous step. Used to detect
    # state transitions (e.g., "this env just transitioned from
    # active=True to active=False" → robot just stood up). Consumed by
    # ``delay_env_recovered_termination`` to immediately re-spawn delay
    # envs into a fallen pose, maximizing recovery-training density.
    self.prev_active: torch.Tensor | None = None
    self._last_step: int = -1

  @classmethod
  def get(cls) -> "_PoseRecoveryStateTracker":
    if cls._instance is None:
      cls._instance = cls()
    return cls._instance

  def reset(self, env_ids) -> None:
    if self.active is None:
      return
    if env_ids is None:
      self.active[:] = False
      if self.prev_active is not None:
        self.prev_active[:] = False
    else:
      self.active[env_ids] = False
      if self.prev_active is not None:
        self.prev_active[env_ids] = False
    # Clear memo so next ``ensure_updated`` call definitely runs the state
    # transition for the affected envs.
    self._last_step = -1

  def _ensure_buffer(self, env: "ManagerBasedRlEnv") -> None:
    if self.active is None or self.active.shape[0] != env.num_envs:
      self.active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
      self.prev_active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  def ensure_updated(
    self,
    env: "ManagerBasedRlEnv",
    fall_z: float = POSE_RECOVERY_FALL_Z,
    fall_tilt: float = POSE_RECOVERY_FALL_TILT_RAD,
    recover_z: float = POSE_RECOVERY_RECOVER_Z,
    recover_tilt: float = POSE_RECOVERY_RECOVER_TILT_RAD,
    asset_name: str = "robot",
  ) -> torch.Tensor:
    """Update the sticky state from the robot's current pose. Idempotent
    within a step (memoized by ``env.common_step_counter``).

    Snapshots the previous ``active`` into ``prev_active`` first so
    ``just_recovered_mask`` can detect True→False transitions this step.
    """
    self._ensure_buffer(env)
    step = int(getattr(env, "common_step_counter", -1))
    if step != self._last_step:
      # Snapshot prev state BEFORE the update — used by transition
      # detectors (e.g., ``just_recovered_mask``) within this step.
      assert self.prev_active is not None and self.active is not None
      self.prev_active.copy_(self.active)
      robot = env.scene[asset_name]
      root_z = robot.data.root_link_pos_w[:, 2]
      roll, pitch, _ = euler_xyz_from_quat(robot.data.root_link_quat_w)
      tilted = (roll.abs() > fall_tilt) | (pitch.abs() > fall_tilt)
      upright = (roll.abs() < recover_tilt) & (pitch.abs() < recover_tilt)
      enter_fall = (root_z < fall_z) & tilted
      exit_recover = (root_z > recover_z) & upright
      self.active = (self.active | enter_fall) & ~exit_recover
      self._last_step = step
    return self.active

  def just_recovered_mask(self, env: "ManagerBasedRlEnv") -> torch.Tensor | None:
    """Per-env bool tensor: True for envs that transitioned from
    active=True last step to active=False this step (i.e., just stood up).
    AND-ed with ``delay_env_mask`` so only delay-tagged envs are flagged.

    Caller must have invoked :meth:`ensure_updated` this step (otherwise
    ``prev_active`` and ``active`` may be stale).
    """
    if self.prev_active is None or self.active is None:
      return None
    transitioned = self.prev_active & ~self.active  # True last step, False now
    tm = getattr(env, "termination_manager", None)
    delay_env_mask = getattr(tm, "_delay_env_mask", None) if tm is not None else None
    if delay_env_mask is None:
      return transitioned
    return transitioned & delay_env_mask

  def get_mask(self, env: "ManagerBasedRlEnv") -> torch.Tensor | None:
    """Read the current state AND-ed with the static delay-env subset.

    Returns None when no per-env state has been initialized yet (e.g.,
    called before any obs/reward computation). When the env has no
    ``DelayedTerminationManager`` installed, returns the bare state
    (no delay-subset filtering).
    """
    if self.active is None:
      return None
    tm = getattr(env, "termination_manager", None)
    delay_env_mask = getattr(tm, "_delay_env_mask", None) if tm is not None else None
    if delay_env_mask is None:
      return self.active
    return self.active & delay_env_mask


def pose_recovery_mask(env: "ManagerBasedRlEnv") -> torch.Tensor | None:
  """Public read-only accessor — returns the current pose-based recovery
  mask AND-ed with the delay-env subset. Used by reward wrappers."""
  tracker = _PoseRecoveryStateTracker.get()
  if tracker.active is None:
    # Lazily initialize so first reward call before first obs call still
    # has a well-defined mask (all-False on first step).
    tracker.ensure_updated(env)
  return tracker.get_mask(env)


# =============================================================================
# Env-side motion loader (used by reset events to sample reference poses).

def _filter_recovery_frames_to_fallen(
  motion: dict,
  z_max: float = POSE_RECOVERY_FALL_Z,
  tilt_min: float = POSE_RECOVERY_FALL_TILT_RAD,
) -> dict | None:
  """Drop non-fallen frames from a single recovery motion clip.

  Keeps frames where ``root_z < z_max`` AND
  ``(|roll| > tilt_min OR |pitch| > tilt_min)`` — matching the pose-based
  detector used by ``RecoveryActiveTerm``. Returns a new motion dict
  with all per-frame tensors sliced to the kept indices, or None if no
  frames pass.
  """
  body_pos = motion["body_pos_w"]      # [T, num_bodies, 3]
  body_quat = motion["body_quat_w"]    # [T, num_bodies, 4]
  root_z = body_pos[:, 0, 2]
  root_quat = body_quat[:, 0]
  roll, pitch, _ = euler_xyz_from_quat(root_quat)
  tilted = (roll.abs() > tilt_min) | (pitch.abs() > tilt_min)
  is_fallen = (root_z < z_max) & tilted
  if not is_fallen.any():
    return None
  idx = is_fallen.nonzero(as_tuple=False).squeeze(-1)
  return {
    "motion_name": motion["motion_name"] + "_fallen_only",
    "fps": motion["fps"],
    "dof_pos": motion["dof_pos"][idx],
    "dof_vel": motion["dof_vel"][idx],
    "body_pos_w": motion["body_pos_w"][idx],
    "body_quat_w": motion["body_quat_w"][idx],
    "body_lin_vel_w": motion["body_lin_vel_w"][idx],
    "body_ang_vel_w": motion["body_ang_vel_w"][idx],
  }


class MotionLoader:
  """Loads NPZ motion clips into per-clip torch tensors on a target device.

  Supports a separate ``recovery_dir`` for fall-recovery reference motions,
  used when the ``DelayedTerminationManager`` keeps an env in a 'down' state.

  ``recovery_fallen_only=True`` filters the recovery clips at load time to
  keep only frames that pass the pose-based fall criterion (low pelvis
  AND tilted). Used by the dual-teacher recovery student so 100% of
  delay-env resets spawn into a genuinely fallen pose. Default ``False``
  preserves AMP teacher's training behavior, which depends on the full
  fall-and-recover cycle in the Recovery/ clip for its discriminator.
  """

  def __init__(
    self,
    motion_dir: str,
    device: str | torch.device = "cpu",
    recovery_dir: str | None = None,
    recovery_fallen_only: bool = False,
    recovery_fallen_z_max: float = POSE_RECOVERY_FALL_Z,
    recovery_fallen_tilt_min_rad: float = POSE_RECOVERY_FALL_TILT_RAD,
  ) -> None:
    self.motion_data: list[dict] = self._load_dir(motion_dir, device)
    assert self.motion_data, f"No npz files found in: {motion_dir}"

    self.motion_data_recovery: list[dict] = []
    if recovery_dir is not None and os.path.isdir(recovery_dir):
      raw_recovery = self._load_dir(recovery_dir, device)
      if recovery_fallen_only:
        filtered: list[dict] = []
        total_in, total_out = 0, 0
        for motion in raw_recovery:
          total_in += motion["dof_pos"].shape[0]
          m_filt = _filter_recovery_frames_to_fallen(
            motion,
            z_max=recovery_fallen_z_max,
            tilt_min=recovery_fallen_tilt_min_rad,
          )
          if m_filt is not None:
            filtered.append(m_filt)
            total_out += m_filt["dof_pos"].shape[0]
        if total_in > 0:
          print(
            f"[MotionLoader] Recovery fallen-only filter: kept "
            f"{total_out}/{total_in} frames "
            f"({100 * total_out / total_in:.1f}%) — "
            f"z<{recovery_fallen_z_max:.2f}m AND |tilt|>"
            f"{recovery_fallen_tilt_min_rad:.2f}rad"
          )
        self.motion_data_recovery = filtered
      else:
        self.motion_data_recovery = raw_recovery

    self.motion_names = [
      m["motion_name"] for m in self.motion_data + self.motion_data_recovery
    ]

  @staticmethod
  def _load_dir(dir_path: str, device: str | torch.device) -> list[dict]:
    assert os.path.isdir(dir_path), f"Not a directory: {dir_path}"
    out: list[dict] = []
    for filename in sorted(os.listdir(dir_path)):
      if not filename.endswith(".npz"):
        continue
      data = np.load(os.path.join(dir_path, filename))
      out.append(
        {
          "motion_name": os.path.splitext(filename)[0],
          "fps": float(np.asarray(data["fps"]).item()),
          "dof_pos": torch.tensor(
            data["joint_pos"], dtype=torch.float32, device=device
          ),
          "dof_vel": torch.tensor(
            data["joint_vel"], dtype=torch.float32, device=device
          ),
          "body_pos_w": torch.tensor(
            data["body_pos_w"], dtype=torch.float32, device=device
          ),
          "body_quat_w": torch.tensor(
            data["body_quat_w"], dtype=torch.float32, device=device
          ),
          "body_lin_vel_w": torch.tensor(
            data["body_lin_vel_w"], dtype=torch.float32, device=device
          ),
          "body_ang_vel_w": torch.tensor(
            data["body_ang_vel_w"], dtype=torch.float32, device=device
          ),
        }
      )
    return out


# =============================================================================
# Termination manager subclass — delays the reset signal for a subset of envs.

_DEFAULT_FALL_TERMINATION_NAMES: tuple[str, ...] = (
  # Dual-teacher / wbc env true-fall signals.
  "root_height_diff",
  "roll_limit",
  "pitch_limit",
  # AMP-teacher native terminations (kept for parity if installed elsewhere).
  "bad_orientation",
  "bad_base_height",
)


class DelayedTerminationManager(TerminationManager):
  """Lets a fraction of envs stay 'down' for a few extra steps after a fall.

  Used so a recovery curriculum (sampled from the ``recovery`` motion dir)
  can start from a fallen pose. Wraps a base ``TerminationManager`` and
  intercepts the per-env done signal — but only for **fall** terminations,
  named via ``fall_termination_names`` (default: a mixture of the dual-teacher
  env's true-fall names and the AMP teacher's). Names that aren't present
  in the wrapped manager's term list are silently ignored.

  Behavior per delay env per step:
    - No termination fired             → counter resets to 0, no-op.
    - Only a fall-named term fired     → suppress the done, counter += 1
                                          (until counter ≥ max_delay_steps).
    - A non-fall term fired (alone or
      alongside a fall)                → do not suppress, counter resets to 0,
                                          env terminates / truncates normally.

  This makes "fallen" mean "the robot triggered one of the fall terminations
  AND nothing else". Things like ``motion_end`` / ``pose_fail`` /
  ``time_out`` reset the env normally even on a delay-tagged env.
  """

  def __init__(
    self,
    base: TerminationManager,
    delay_env_mask: torch.Tensor,
    max_delay_steps: int,
    fall_termination_names: tuple[str, ...] = _DEFAULT_FALL_TERMINATION_NAMES,
  ) -> None:
    # Inherit all internal state from the base manager (avoid re-init).
    self.__dict__.update(base.__dict__)
    self._delay_env_mask = delay_env_mask
    self._delay_counters = torch.zeros_like(delay_env_mask, dtype=torch.long)
    self._max_delay_steps = max_delay_steps
    # Keep only names that the base manager actually registered, so a typo
    # or a name from a different env cfg doesn't silently match nothing.
    self._fall_termination_names: tuple[str, ...] = tuple(
      name for name in fall_termination_names if name in self._term_dones
    )

  def compute(self) -> torch.Tensor:
    dones = super().compute()

    if self._max_delay_steps <= 0:
      return dones

    if not self._fall_termination_names:
      # Nothing to gate on — preserve the original any-done behavior so this
      # class is still a drop-in for the AMP teacher's intent.
      fall_dones = dones
    else:
      fall_dones = torch.zeros_like(dones)
      for name in self._fall_termination_names:
        fall_dones = fall_dones | self._term_dones[name]

    # An env is suppressible iff: it's a delay env AND a fall fired AND
    # nothing else (non-fall termination or truncation) is asking to end the
    # episode this step.
    non_fall_terminated = self._terminated_buf & ~fall_dones
    suppressible = (
      self._delay_env_mask
      & fall_dones
      & ~non_fall_terminated
      & ~self._truncated_buf
    )

    self._delay_counters[suppressible] += 1

    not_ready = suppressible & (self._delay_counters < self._max_delay_steps)
    self._terminated_buf[not_ready] = False

    ready = suppressible & (self._delay_counters >= self._max_delay_steps)
    self._delay_counters[ready] = 0

    # Any delay env that we're NOT currently suppressing this step has its
    # counter reset (whether it stood up, didn't fall, or fell-plus-something).
    self._delay_counters[self._delay_env_mask & ~suppressible] = 0

    return self._truncated_buf | self._terminated_buf


# =============================================================================
# Singleton manager threaded through the events.
#
# AMP_mjlab uses a process-global singleton because mjlab's event system has no
# place to stash per-env-cfg state. We keep that pattern here.

class MotionResetManager:
  _instance: "MotionResetManager | None" = None

  def __init__(self) -> None:
    self.walk_run_frames: dict[str, dict[str, torch.Tensor]] = {}
    self.recovery_frames: dict[str, dict[str, torch.Tensor]] = {}

  @classmethod
  def get(cls) -> "MotionResetManager":
    if cls._instance is None:
      cls._instance = cls()
    return cls._instance

  def init(
    self,
    env: "ManagerBasedRlEnv",
    motion_dir: str,
    recovery_dir: str | None = None,
    recovery_fallen_only: bool = False,
    recovery_fallen_z_max: float = POSE_RECOVERY_FALL_Z,
    recovery_fallen_tilt_min_rad: float = POSE_RECOVERY_FALL_TILT_RAD,
  ) -> None:
    if motion_dir in self.walk_run_frames:
      return

    loader = MotionLoader(
      motion_dir=motion_dir,
      device=str(env.device),
      recovery_dir=recovery_dir,
      recovery_fallen_only=recovery_fallen_only,
      recovery_fallen_z_max=recovery_fallen_z_max,
      recovery_fallen_tilt_min_rad=recovery_fallen_tilt_min_rad,
    )
    self.walk_run_frames[motion_dir] = self._concat_frames(loader.motion_data)
    print(
      f"[MotionResetManager] Loaded {len(loader.motion_data)} clips, "
      f"{self.walk_run_frames[motion_dir]['root_pos'].shape[0]} frames "
      f"from {motion_dir}"
    )
    if loader.motion_data_recovery:
      self.recovery_frames[motion_dir] = self._concat_frames(
        loader.motion_data_recovery
      )
      print(
        f"[MotionResetManager] Loaded {len(loader.motion_data_recovery)} "
        f"recovery clips, "
        f"{self.recovery_frames[motion_dir]['root_pos'].shape[0]} frames "
        f"from {recovery_dir}"
      )

  def reset(
    self,
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    motion_dir: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> None:
    if env_ids is None:
      env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    if env_ids.numel() == 0:
      return

    delay_mask = self._get_delay_env_mask(env)
    if delay_mask is not None:
      is_delay = delay_mask[env_ids]
      delay_ids = env_ids[is_delay]
      normal_ids = env_ids[~is_delay]
    else:
      delay_ids = env_ids[:0]
      normal_ids = env_ids

    if normal_ids.numel():
      self._write_reset_state(
        env, normal_ids, self.walk_run_frames[motion_dir], asset_cfg
      )
    if delay_ids.numel():
      recovery = self.recovery_frames.get(motion_dir)
      frames = recovery if recovery is not None else self.walk_run_frames[motion_dir]
      self._write_reset_state(env, delay_ids, frames, asset_cfg)

  @staticmethod
  def _get_delay_env_mask(env: "ManagerBasedRlEnv") -> torch.Tensor | None:
    tm = env.termination_manager
    if isinstance(tm, DelayedTerminationManager):
      return tm._delay_env_mask
    return None

  @staticmethod
  def _write_reset_state(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor,
    frames: dict[str, torch.Tensor],
    asset_cfg: SceneEntityCfg,
  ) -> None:
    total_frames = frames["root_pos"].shape[0]
    num_reset = env_ids.numel()
    idx = torch.randint(0, total_frames, (num_reset,), device=env.device)

    asset: Entity = env.scene[asset_cfg.name]

    root_pos = frames["root_pos"][idx]
    root_quat = frames["root_quat"][idx]
    positions = env.scene.env_origins[env_ids].clone()
    positions[:, 2] = root_pos[:, 2]
    root_pose = torch.cat([positions, root_quat], dim=-1)
    asset.write_root_link_pose_to_sim(root_pose, env_ids=env_ids)

    root_vel = torch.cat(
      [frames["root_lin_vel"][idx], frames["root_ang_vel"][idx]], dim=-1
    )
    asset.write_root_link_velocity_to_sim(root_vel, env_ids=env_ids)

    joint_pos = frames["joint_pos"][idx]
    joint_vel = frames["joint_vel"][idx]
    soft_joint_pos_limits = asset.data.soft_joint_pos_limits
    assert soft_joint_pos_limits is not None
    joint_pos_limits = soft_joint_pos_limits[env_ids][:, asset_cfg.joint_ids]
    joint_pos_clamped = joint_pos[:, asset_cfg.joint_ids].clamp_(
      joint_pos_limits[..., 0], joint_pos_limits[..., 1]
    )

    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, list):
      joint_ids = torch.tensor(joint_ids, device=env.device)

    asset.write_joint_state_to_sim(
      joint_pos_clamped,
      joint_vel[:, asset_cfg.joint_ids],
      env_ids=env_ids,
      joint_ids=joint_ids,
    )

  @staticmethod
  def _concat_frames(motions: list[dict]) -> dict[str, torch.Tensor]:
    keys = ("root_pos", "root_quat", "root_lin_vel", "root_ang_vel", "joint_pos", "joint_vel")
    bufs: dict[str, list[torch.Tensor]] = {k: [] for k in keys}
    for motion in motions:
      bufs["root_pos"].append(motion["body_pos_w"][:, 0, :])
      bufs["root_quat"].append(motion["body_quat_w"][:, 0, :])
      bufs["root_lin_vel"].append(motion["body_lin_vel_w"][:, 0, :])
      bufs["root_ang_vel"].append(motion["body_ang_vel_w"][:, 0, :])
      bufs["joint_pos"].append(motion["dof_pos"])
      bufs["joint_vel"].append(motion["dof_vel"])
    return {k: torch.cat(v, dim=0) for k, v in bufs.items()}


# =============================================================================
# Event hooks consumed by the env via EventTermCfg.

def _flat_terrain_candidate_envs(
  env: "ManagerBasedRlEnv",
  flat_substrs: tuple[str, ...],
) -> torch.Tensor | None:
  """Return env IDs whose initial terrain column matches any flat sub-terrain.

  Returns ``None`` if the env has no terrain or no terrain_generator
  configured, or if no sub-terrain name matches any substring in
  ``flat_substrs`` (caller should fall back to unfiltered sampling).
  """
  terrain = getattr(env.scene, "terrain", None)
  if terrain is None:
    return None
  gen = terrain.cfg.terrain_generator if terrain.cfg is not None else None
  types = getattr(terrain, "terrain_types", None)
  if gen is None or gen.sub_terrains is None or types is None:
    return None
  sub_names = list(gen.sub_terrains.keys())
  flat_cols = [
    i for i, n in enumerate(sub_names) if any(s in n for s in flat_substrs)
  ]
  if not flat_cols:
    return None
  flat_col_t = torch.tensor(flat_cols, dtype=torch.long, device=env.device)
  is_flat = torch.zeros_like(types, dtype=torch.bool)
  for c in flat_col_t.tolist():
    is_flat = is_flat | (types == c)
  return torch.nonzero(is_flat, as_tuple=False).flatten()


def init_motion_loader(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor | None,
  motion_dir: str,
  recovery_dir: str | None = None,
  delay_reset_env_ratio: float = 0.0,
  max_delay_steps: int = 0,
  fall_termination_names: tuple[str, ...] | None = None,
  recovery_fallen_only: bool = False,
  recovery_fallen_z_max: float = POSE_RECOVERY_FALL_Z,
  recovery_fallen_tilt_min_rad: float = POSE_RECOVERY_FALL_TILT_RAD,
  flat_only_delay_envs: bool = False,
  flat_sub_terrain_substrs: tuple[str, ...] = ("flat",),
) -> None:
  """Startup event: load motion clips and (optionally) install delayed termination.

  ``fall_termination_names`` selects which termination terms count as a
  "fall" for the down-state suppression. ``None`` falls back to
  ``_DEFAULT_FALL_TERMINATION_NAMES`` (a superset covering both the AMP
  teacher's and the dual-teacher env's fall conditions).

  ``recovery_fallen_only`` (default False): when True, recovery clip
  frames are filtered at load to keep only frames that pass the
  pose-based fall detector (``root_z < recovery_fallen_z_max`` AND
  ``|tilt| > recovery_fallen_tilt_min_rad``). The dual-teacher recovery
  student should set this to True so 100% of delay-env resets spawn
  fallen. The AMP teacher should leave it at False — its discriminator
  was trained on the unfiltered cyclic clip.

  ``flat_only_delay_envs`` (default False): when True, the random
  ``delay_idx`` is post-filtered to env IDs whose initial terrain column
  matches a sub-terrain whose name contains any of
  ``flat_sub_terrain_substrs`` (default ``("flat",)``). This enforces
  the 4-teacher MoE invariant "recovery only on flat" by ensuring that
  delay envs spawn on flat tiles (where the recovery clip's pose
  trajectory is well-defined). Sample without replacement; if too few
  flat candidates exist, log a warning and use all available.
  """
  MotionResetManager.get().init(
    env=env,
    motion_dir=motion_dir,
    recovery_dir=recovery_dir,
    recovery_fallen_only=recovery_fallen_only,
    recovery_fallen_z_max=recovery_fallen_z_max,
    recovery_fallen_tilt_min_rad=recovery_fallen_tilt_min_rad,
  )

  num_delay = int(env.num_envs * delay_reset_env_ratio)
  if num_delay > 0 and max_delay_steps > 0:
    delay_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if flat_only_delay_envs:
      # Constrain delay envs to those on flat sub-terrain columns. With a
      # 70% flat / 30% stair partition, the expected retained delay count
      # is ~ delay_reset_env_ratio * num_envs (since num_envs *
      # delay_reset_env_ratio < num_envs * 0.70).
      candidate = _flat_terrain_candidate_envs(env, flat_sub_terrain_substrs)
      if candidate is None:
        # No terrain or no flat columns found — fall through to standard
        # behavior with a warning.
        print(
          "[init_motion_loader][WARN] flat_only_delay_envs=True but no "
          "terrain / flat sub-terrains found; falling back to unfiltered "
          "delay-env sampling."
        )
        delay_idx = torch.randperm(env.num_envs, device=env.device)[:num_delay]
      else:
        n_candidates = candidate.numel()
        if n_candidates < num_delay:
          print(
            f"[init_motion_loader][WARN] flat_only_delay_envs=True requested "
            f"{num_delay} delay envs but only {n_candidates} flat envs exist; "
            f"using all flat candidates."
          )
          delay_idx = candidate
        else:
          perm = torch.randperm(n_candidates, device=env.device)
          delay_idx = candidate[perm[:num_delay]]
        num_delay = int(delay_idx.numel())  # reflect actual count
    else:
      delay_idx = torch.randperm(env.num_envs, device=env.device)[:num_delay]
    delay_mask[delay_idx] = True
    fall_names = (
      _DEFAULT_FALL_TERMINATION_NAMES
      if fall_termination_names is None
      else tuple(fall_termination_names)
    )
    env.termination_manager = DelayedTerminationManager(
      base=env.termination_manager,
      delay_env_mask=delay_mask,
      max_delay_steps=max_delay_steps,
      fall_termination_names=fall_names,
    )
    matched = env.termination_manager._fall_termination_names
    print(
      f"[init_motion_loader] DelayedTerminationManager: {num_delay}/{env.num_envs} "
      f"envs, max_delay_steps={max_delay_steps}, fall_terms={list(matched)}"
    )


def reset_from_motion_data(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor | None,
  motion_dir: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Reset event: pick a random motion frame for each env (recovery for delay envs)."""
  MotionResetManager.get().reset(
    env=env, env_ids=env_ids, motion_dir=motion_dir, asset_cfg=asset_cfg
  )


def reset_delay_envs_to_recovery(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor | None,
  motion_dir: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Reset event variant for envs that already have their own normal-env reset
  (e.g. PKL motion tracking). Only overwrites delay envs with a recovery
  frame, leaving non-delay envs untouched. Order this event AFTER the
  baseline reset in ``EventTermCfg`` insertion order.
  """
  mgr = MotionResetManager.get()
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  if env_ids.numel() == 0:
    return
  delay_mask = mgr._get_delay_env_mask(env)
  if delay_mask is None:
    return
  is_delay = delay_mask[env_ids]
  delay_ids = env_ids[is_delay]
  if delay_ids.numel() == 0:
    return
  recovery = mgr.recovery_frames.get(motion_dir)
  frames = recovery if recovery is not None else mgr.walk_run_frames.get(motion_dir)
  if frames is None:
    return
  mgr._write_reset_state(env, delay_ids, frames, asset_cfg)


# =============================================================================
# Motion command subclass that re-writes fallen pose for delay envs.
#
# Required because mjlab's ``ManagerBasedRlEnv._reset_idx`` runs
# ``event_manager.apply(mode="reset")`` BEFORE ``command_manager.reset``.
# Without this subclass, the fallen Recovery/ pose written by the
# ``reset_delay_envs_to_recovery`` event is silently overwritten when the
# command manager subsequently calls ``_resample_command`` →
# ``_write_reference_state_to_sim`` (which writes a motion-clip frame for
# ALL env_ids, including delay envs). The override is invisible — the
# pose-based detector reads the post-command pose (a walking/standing
# motion frame) and never enters recovery state, so AMP-KL never fires
# from spawn-fallen events. The symptom is ``recov_active_frac``
# collapsing to ~0.005 once the policy stops falling organically.

def _override_delay_envs_with_recovery_pose(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor | None:
  """Re-write fallen Recovery/ pose for delay-tagged envs in ``env_ids``.

  Returns the per-call ``delay_ids`` tensor (or None if nothing was
  written). Caller is expected to refresh the entity's derived sim
  state for those ids afterward.
  """
  mgr = MotionResetManager.get()
  if not mgr.recovery_frames:
    return None
  delay_mask = mgr._get_delay_env_mask(env)
  if delay_mask is None:
    return None
  is_delay = delay_mask[env_ids]
  delay_ids = env_ids[is_delay]
  if delay_ids.numel() == 0:
    return None
  frames = next(iter(mgr.recovery_frames.values()))
  mgr._write_reset_state(env, delay_ids, frames, asset_cfg)
  return delay_ids


# =============================================================================
# Observations: body pos/ori/lin_vel/ang_vel in the anchor body's frame.

def robot_body_pos_b(
  env: "ManagerBasedRlEnv",
  anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
  body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
  asset: Entity = env.scene[anchor_cfg.name]
  anchor_pos_w = asset.data.body_link_pos_w[:, anchor_cfg.body_ids[0]]
  anchor_quat_w = asset.data.body_link_quat_w[:, anchor_cfg.body_ids[0]]
  body_pos_w = asset.data.body_link_pos_w[:, body_cfg.body_ids]
  body_quat_w = asset.data.body_link_quat_w[:, body_cfg.body_ids]
  num_bodies = body_pos_w.shape[1]
  pos_b, _ = subtract_frame_transforms(
    anchor_pos_w[:, None, :].expand(-1, num_bodies, -1),
    anchor_quat_w[:, None, :].expand(-1, num_bodies, -1),
    body_pos_w,
    body_quat_w,
  )
  return pos_b.reshape(env.num_envs, -1)


def robot_body_ori_b(
  env: "ManagerBasedRlEnv",
  anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
  body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
  asset: Entity = env.scene[anchor_cfg.name]
  anchor_pos_w = asset.data.body_link_pos_w[:, anchor_cfg.body_ids[0]]
  anchor_quat_w = asset.data.body_link_quat_w[:, anchor_cfg.body_ids[0]]
  body_pos_w = asset.data.body_link_pos_w[:, body_cfg.body_ids]
  body_quat_w = asset.data.body_link_quat_w[:, body_cfg.body_ids]
  num_bodies = body_pos_w.shape[1]
  _, ori_b = subtract_frame_transforms(
    anchor_pos_w[:, None, :].expand(-1, num_bodies, -1),
    anchor_quat_w[:, None, :].expand(-1, num_bodies, -1),
    body_pos_w,
    body_quat_w,
  )
  mat = matrix_from_quat(ori_b)
  return mat[..., :2].reshape(mat.shape[0], -1)


def robot_body_lin_vel_b(
  env: "ManagerBasedRlEnv",
  anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
  body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
  asset: Entity = env.scene[body_cfg.name]
  body_lin_vel_w = asset.data.body_link_lin_vel_w[:, body_cfg.body_ids]
  body_quat_w = asset.data.body_link_quat_w[:, body_cfg.body_ids]
  num_bodies = body_lin_vel_w.shape[1]
  body_lin_vel_b = quat_apply_inverse(
    body_quat_w.reshape(-1, 4), body_lin_vel_w.reshape(-1, 3)
  ).reshape(env.num_envs, num_bodies, 3)
  return body_lin_vel_b.reshape(env.num_envs, -1)


def robot_body_ang_vel_b(
  env: "ManagerBasedRlEnv",
  anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
  body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
  asset: Entity = env.scene[body_cfg.name]
  body_ang_vel_w = asset.data.body_link_ang_vel_w[:, body_cfg.body_ids]
  body_quat_w = asset.data.body_link_quat_w[:, body_cfg.body_ids]
  num_bodies = body_ang_vel_w.shape[1]
  body_ang_vel_b = quat_apply_inverse(
    body_quat_w.reshape(-1, 4), body_ang_vel_w.reshape(-1, 3)
  ).reshape(env.num_envs, num_bodies, 3)
  return body_ang_vel_b.reshape(env.num_envs, -1)


# =============================================================================
# Rewards (delay-env aware).

def _get_delay_active_mask(env: "ManagerBasedRlEnv") -> torch.Tensor | None:
  tm = getattr(env, "termination_manager", None)
  if tm is None:
    return None
  delay_env_mask = getattr(tm, "_delay_env_mask", None)
  delay_counters = getattr(tm, "_delay_counters", None)
  if isinstance(delay_env_mask, torch.Tensor) and isinstance(delay_counters, torch.Tensor):
    return delay_env_mask & (delay_counters > 0)
  return None


def _scale_for_delay(
  env: "ManagerBasedRlEnv",
  reward: torch.Tensor,
  mask_delay: bool,
  delay_env_rew_ratio: float,
) -> torch.Tensor:
  if not mask_delay:
    return reward
  mask = _get_delay_active_mask(env)
  if mask is None:
    return reward
  return torch.where(mask, reward * delay_env_rew_ratio, reward)


def _mask_only_for_delay(
  env: "ManagerBasedRlEnv",
  reward: torch.Tensor,
  mask_delay: bool,
  delay_env_rew_ratio: float,
) -> torch.Tensor:
  if not mask_delay:
    return torch.zeros_like(reward)
  mask = _get_delay_active_mask(env)
  if mask is None:
    return torch.zeros_like(reward)
  return torch.where(mask, reward * delay_env_rew_ratio, torch.zeros_like(reward))


# ---- Pose-based variants (shared with RecoveryActiveTerm via the singleton) ----
#
# Same semantics as ``_scale_for_delay`` / ``_mask_only_for_delay`` but the
# active mask comes from the pose-based ``_PoseRecoveryStateTracker`` instead
# of the termination manager's ``_delay_env_mask & (counters > 0)``. Used by
# the dual-teacher recovery student so reward gating matches AMP-KL gating
# on the same per-env recovery flag (no termination/motion-command aliasing).

def _scale_for_pose_recovery(
  env: "ManagerBasedRlEnv",
  reward: torch.Tensor,
  mask_delay: bool,
  delay_env_rew_ratio: float,
) -> torch.Tensor:
  if not mask_delay:
    return reward
  mask = pose_recovery_mask(env)
  if mask is None:
    return reward
  return torch.where(mask, reward * delay_env_rew_ratio, reward)


def _mask_only_for_pose_recovery(
  env: "ManagerBasedRlEnv",
  reward: torch.Tensor,
  mask_delay: bool,
  delay_env_rew_ratio: float,
) -> torch.Tensor:
  if not mask_delay:
    return torch.zeros_like(reward)
  mask = pose_recovery_mask(env)
  if mask is None:
    return torch.zeros_like(reward)
  return torch.where(mask, reward * delay_env_rew_ratio, torch.zeros_like(reward))


def track_anchor_linear_velocity(
  env: "ManagerBasedRlEnv",
  std: float,
  command_name: str,
  mask_delay: bool = False,
  delay_env_rew_ratio: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  cmd_xyz_b = torch.cat((command[:, :2], torch.zeros_like(command[:, :1])), dim=-1)
  cmd_xyz_w = quat_apply(
    yaw_quat(asset.data.body_link_quat_w[:, anchor_cfg.body_ids[0]]),
    cmd_xyz_b,
  )
  err = torch.sum(
    torch.square(
      cmd_xyz_w[:, :3]
      - asset.data.body_link_lin_vel_w[:, anchor_cfg.body_ids[0], :3]
    ),
    dim=1,
  )
  reward = torch.exp(-err / std**2)
  return _scale_for_delay(env, reward, mask_delay, delay_env_rew_ratio)


def track_anchor_angular_velocity(
  env: "ManagerBasedRlEnv",
  std: float,
  command_name: str,
  mask_delay: bool = False,
  delay_env_rew_ratio: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  ang_vel_w = asset.data.body_link_ang_vel_w[:, anchor_cfg.body_ids[0]]
  err_z = torch.square(command[:, 2] - ang_vel_w[:, 2])
  ang_vel_b = quat_apply_inverse(
    asset.data.body_link_quat_w[:, anchor_cfg.body_ids[0]], ang_vel_w
  )
  err_xy = torch.sum(torch.square(ang_vel_b[:, :2]), dim=-1)
  reward = torch.exp(-(err_z + err_xy) / std**2)
  return _scale_for_delay(env, reward, mask_delay, delay_env_rew_ratio)


def body_ang_vel_xy_l2(
  env: "ManagerBasedRlEnv",
  std: float,
  mask_delay: bool = False,
  delay_env_rew_ratio: float = 1.0,
  body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
  asset: Entity = env.scene[body_cfg.name]
  ang_vel_w = asset.data.body_link_ang_vel_w[:, body_cfg.body_ids[0]]
  ang_vel_b = quat_apply_inverse(
    asset.data.body_link_quat_w[:, body_cfg.body_ids[0]], ang_vel_w
  )
  err = torch.sum(torch.square(ang_vel_b[:, :2]), dim=-1)
  reward = torch.exp(-err / std**2)
  return _scale_for_delay(env, reward, mask_delay, delay_env_rew_ratio)


def track_root_height(
  env: "ManagerBasedRlEnv",
  std: float,
  mask_delay: bool = False,
  delay_env_rew_ratio: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking the default root height. Active only on delay envs.

  Termination-based gating — used by the AMP teacher's own training. The
  dual-teacher recovery student should use :func:`track_root_height_pose`
  instead so the gating matches the AMP-KL gating in DaggerPPO.
  """
  asset: Entity = env.scene[asset_cfg.name]
  desired_h = asset.data.default_root_state[:, 2]
  cur_h = asset.data.body_link_pos_w[:, 0, 2]
  err = torch.square(desired_h - cur_h)
  reward = torch.exp(-err / std**2)
  return _mask_only_for_delay(env, reward, mask_delay, delay_env_rew_ratio)


def track_root_height_pose(
  env: "ManagerBasedRlEnv",
  std: float,
  mask_delay: bool = False,
  delay_env_rew_ratio: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Same as :func:`track_root_height` but gated by the pose-based
  recovery state shared with ``RecoveryActiveTerm``. Use this in the
  dual-teacher recovery student so reward-side and KL-side gating
  read the same per-env mask."""
  asset: Entity = env.scene[asset_cfg.name]
  desired_h = asset.data.default_root_state[:, 2]
  cur_h = asset.data.body_link_pos_w[:, 0, 2]
  err = torch.square(desired_h - cur_h)
  reward = torch.exp(-err / std**2)
  return _mask_only_for_pose_recovery(env, reward, mask_delay, delay_env_rew_ratio)


_REWARD_FN_CACHE: dict = {}




def wbc_reward_with_pose_recovery_mask(
  env: "ManagerBasedRlEnv",
  *,
  reward_func_path: str,
  delay_env_rew_ratio: float = 0.0,
  **reward_kwargs,
) -> torch.Tensor:
  """Same as :func:`wbc_reward_with_delay_mask` but uses the pose-based
  recovery mask (shared with ``RecoveryActiveTerm``).

  Used by the dual-teacher recovery student so the velocity / hand
  tracking rewards are zeroed on the same per-env subset that AMP-KL
  is supervising. Keeps reward-side and KL-side gating consistent —
  prevents the mid-recovery oscillation symptom where AMP-KL pulls
  toward standing while task reward pulls toward whatever motion clip
  the env happened to sample.
  """
  fn = _REWARD_FN_CACHE.get(reward_func_path)
  if fn is None:
    from rsl_rl.utils import resolve_callable

    fn = resolve_callable(reward_func_path)
    _REWARD_FN_CACHE[reward_func_path] = fn
  reward = fn(env, **reward_kwargs)
  return _scale_for_pose_recovery(env, reward, mask_delay=True, delay_env_rew_ratio=delay_env_rew_ratio)


def self_collision_cost(
  env: "ManagerBasedRlEnv",
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  from mjlab.sensor import ContactSensor

  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    force_mag = torch.norm(data.force_history, dim=-1)
    hit = (force_mag > force_threshold).any(dim=1)
    return hit.sum(dim=-1).float()
  assert data.found is not None
  return data.found.squeeze(-1)


# =============================================================================
# Terminations (recovery-cycle termination, used by the student).

def delay_env_recovered_termination(
  env: "ManagerBasedRlEnv",
  fall_z_threshold: float = POSE_RECOVERY_FALL_Z,
  fall_tilt_threshold_rad: float = POSE_RECOVERY_FALL_TILT_RAD,
  recover_z_threshold: float = POSE_RECOVERY_RECOVER_Z,
  recover_tilt_threshold_rad: float = POSE_RECOVERY_RECOVER_TILT_RAD,
  asset_name: str = "robot",
) -> torch.Tensor:
  """Termination: fires when a delay-tagged env just successfully stood up.

  Used by the dual-teacher recovery student to *immediately recycle* delay
  envs back into a fallen pose after every successful recovery. Without
  this termination, a delay env that recovers spends the rest of its
  episode walking normally — contributing zero recovery samples — until
  ``time_out`` finally fires (~1000 steps later). With this termination:

    fallen → recover → done → reset to fallen → recover → done → reset → ...

  The reset routes through the standard machinery, including
  :func:`reset_delay_envs_to_recovery` which writes a Recovery/ frame
  back to the delay env. Net effect: every delay env is essentially
  always either falling or in mid-recovery, maximizing the
  ``recov_active_frac`` training signal.

  Detection re-uses the shared ``_PoseRecoveryStateTracker`` —
  ``ensure_updated`` is idempotent within a step, so the termination
  computing this state is shared with obs and reward gating (single
  source of truth for "is the robot mid-recovery").

  This is NOT a fall termination, so the ``DelayedTerminationManager``
  passes it through normally (does not suppress).

  Should only be added to env_cfgs that install the delay-recovery
  machinery — non-delay envs naturally never trigger this since the mask
  is AND-ed with ``delay_env_mask`` inside the tracker.
  """
  tracker = _PoseRecoveryStateTracker.get()
  tracker.ensure_updated(
    env,
    fall_z=fall_z_threshold,
    fall_tilt=fall_tilt_threshold_rad,
    recover_z=recover_z_threshold,
    recover_tilt=recover_tilt_threshold_rad,
    asset_name=asset_name,
  )
  mask = tracker.just_recovered_mask(env)
  if mask is None:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  return mask


# =============================================================================
# Metrics.

def mean_delay_steps(env: "ManagerBasedRlEnv") -> torch.Tensor:
  tm = env.termination_manager
  counters = getattr(tm, "_delay_counters", None)
  mask = getattr(tm, "_delay_env_mask", None)
  if isinstance(mask, torch.Tensor) and isinstance(counters, torch.Tensor):
    total = torch.sum(counters.float())
    n = torch.sum(mask.float())
    mean = total / (n + 1e-8)
    return mean.expand(env.num_envs)
  return torch.zeros(env.num_envs, device=env.device)
