"""PKL motion commands for tracking and locomotion-teacher tasks."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch

from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.tasks.tracking.mdp.commands import MotionCommandCfg
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_error_magnitude,
  quat_from_euler_xyz,
  quat_inv,
  quat_mul,
  sample_uniform,
  wrap_to_pi,
  yaw_quat,
)
from mjlab.viewer.debug_visualizer import DebugVisualizer

from wbc_mjlab.pkl_motion_lib import PklMotionLib

if TYPE_CHECKING:
  from collections.abc import Callable
  from typing import Any

  import viser

  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_DESIRED_FRAME_COLORS = ((1.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 1.0))


class PklMotionCommand(CommandTerm):
  """Motion command that loads multi-motion PKL datasets via PklMotionLib."""

  cfg: PklMotionCommandCfg
  _env: ManagerBasedRlEnv

  def __init__(self, cfg: PklMotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]
    self.robot_anchor_body_index = self.robot.body_names.index(
      self.cfg.anchor_body_name
    )
    self.motion_anchor_body_index = self.cfg.body_names.index(
      self.cfg.anchor_body_name
    )
    self.body_indexes = torch.tensor(
      self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )

    # Load multi-motion library
    self.motion_lib = PklMotionLib(
      self.cfg.motion_file, self.cfg.body_names, device=self.device
    )

    # Per-env state
    self.motion_ids = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    self.motion_times = torch.zeros(self.num_envs, device=self.device)

    # Cached frame data (updated every step)
    n_joints = self.robot.data.joint_pos.shape[-1]
    n_bodies = len(cfg.body_names)
    self._joint_pos = torch.zeros(self.num_envs, n_joints, device=self.device)
    self._joint_vel = torch.zeros(self.num_envs, n_joints, device=self.device)
    self._body_pos_w = torch.zeros(self.num_envs, n_bodies, 3, device=self.device)
    self._body_quat_w = torch.zeros(self.num_envs, n_bodies, 4, device=self.device)
    self._body_quat_w[:, :, 0] = 1.0  # identity
    self._body_lin_vel_w = torch.zeros(
      self.num_envs, n_bodies, 3, device=self.device
    )
    self._body_ang_vel_w = torch.zeros(
      self.num_envs, n_bodies, 3, device=self.device
    )

    # Relative poses
    self.body_pos_relative_w = torch.zeros(
      self.num_envs, n_bodies, 3, device=self.device
    )
    self.body_quat_relative_w = torch.zeros(
      self.num_envs, n_bodies, 4, device=self.device
    )
    self.body_quat_relative_w[:, :, 0] = 1.0

    # Adaptive sampling state
    max_motion_length = self.motion_lib._motion_lengths.max().item()
    self.bin_count = max(int(max_motion_length / self._env.step_dt) + 1, 1)
    self.bin_failed_count = torch.zeros(
      self.bin_count, dtype=torch.float, device=self.device
    )
    self._current_bin_failed = torch.zeros(
      self.bin_count, dtype=torch.float, device=self.device
    )
    self.kernel = torch.tensor(
      [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)],
      device=self.device,
    )
    self.kernel = self.kernel / self.kernel.sum()

    # Metrics
    self.metrics["error_anchor_pos"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_anchor_rot"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_anchor_lin_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_anchor_ang_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_body_pos"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_body_rot"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_joint_pos"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_joint_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["sampling_entropy"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["sampling_top1_prob"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["sampling_top1_bin"] = torch.zeros(
      self.num_envs, device=self.device
    )

    # Ghost model for visualization
    self._ghost_model: mujoco.MjModel | None = None
    self._ghost_color = np.array(cfg.viz.ghost_color, dtype=np.float32)

    # Teacher ghost visualization buffers (populated externally by runner)
    self._wbc_teacher_joint_targets: torch.Tensor | None = None
    self._loco_teacher_joint_targets: torch.Tensor | None = None
    self._wbc_ghost_model: mujoco.MjModel | None = None
    self._loco_ghost_model: mujoco.MjModel | None = None
    self._wbc_ghost_color = np.array([0.3, 0.5, 1.0, 0.4], dtype=np.float32)
    self._loco_ghost_color = np.array([1.0, 0.6, 0.2, 0.4], dtype=np.float32)

  # ==================== Properties ====================

  @property
  def command(self) -> torch.Tensor:
    return torch.cat([self._joint_pos, self._joint_vel], dim=1)

  @property
  def joint_pos(self) -> torch.Tensor:
    return self._joint_pos

  @property
  def joint_vel(self) -> torch.Tensor:
    return self._joint_vel

  @property
  def body_pos_w(self) -> torch.Tensor:
    return self._body_pos_w + self._env.scene.env_origins[:, None, :]

  @property
  def body_quat_w(self) -> torch.Tensor:
    return self._body_quat_w

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    return self._body_lin_vel_w

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    return self._body_ang_vel_w

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    return (
      self._body_pos_w[:, self.motion_anchor_body_index]
      + self._env.scene.env_origins
    )

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    return self._body_quat_w[:, self.motion_anchor_body_index]

  @property
  def anchor_lin_vel_w(self) -> torch.Tensor:
    return self._body_lin_vel_w[:, self.motion_anchor_body_index]

  @property
  def anchor_ang_vel_w(self) -> torch.Tensor:
    return self._body_ang_vel_w[:, self.motion_anchor_body_index]

  @property
  def robot_joint_pos(self) -> torch.Tensor:
    return self.robot.data.joint_pos

  @property
  def robot_joint_vel(self) -> torch.Tensor:
    return self.robot.data.joint_vel

  @property
  def robot_body_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.body_indexes]

  @property
  def robot_body_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.body_indexes]

  @property
  def robot_body_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.body_indexes]

  @property
  def robot_body_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.body_indexes]

  @property
  def robot_anchor_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.robot_anchor_body_index]

  # ==================== Command lifecycle ====================

  def _update_metrics(self) -> None:
    self.metrics["error_anchor_pos"] = torch.norm(
      self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1
    )
    self.metrics["error_anchor_rot"] = quat_error_magnitude(
      self.anchor_quat_w, self.robot_anchor_quat_w
    )
    self.metrics["error_anchor_lin_vel"] = torch.norm(
      self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1
    )
    self.metrics["error_anchor_ang_vel"] = torch.norm(
      self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1
    )
    self.metrics["error_body_pos"] = torch.norm(
      self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_rot"] = quat_error_magnitude(
      self.body_quat_relative_w, self.robot_body_quat_w
    ).mean(dim=-1)
    self.metrics["error_body_lin_vel"] = torch.norm(
      self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_ang_vel"] = torch.norm(
      self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_joint_pos"] = torch.norm(
      self._joint_pos - self.robot_joint_pos, dim=-1
    )
    self.metrics["error_joint_vel"] = torch.norm(
      self._joint_vel - self.robot_joint_vel, dim=-1
    )

  def _write_reference_state_to_sim(
    self,
    env_ids: torch.Tensor,
    root_pos: torch.Tensor,
    root_ori: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
  ) -> None:
    """Clip joint positions and write root + joint state to sim."""
    soft_limits = self.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos = torch.clip(joint_pos, soft_limits[:, :, 0], soft_limits[:, :, 1])
    vel_factor = 0.8
    joint_vel = joint_vel * vel_factor
    self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    root_lin_vel = root_lin_vel * vel_factor
    root_ang_vel = root_ang_vel * vel_factor
    root_state = torch.cat(
      [root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1
    )
    self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    self.robot.reset(env_ids=env_ids)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)

    # Sample motions and phases
    self.motion_ids[env_ids] = self.motion_lib.sample_motions(n)

    if self.cfg.sampling_mode == "start":
      self.motion_times[env_ids] = 0.0
    elif self.cfg.sampling_mode == "uniform":
      self.motion_times[env_ids] = self.motion_lib.sample_time(
        self.motion_ids[env_ids]
      )
    else:
      assert self.cfg.sampling_mode == "adaptive"
      self._adaptive_sampling(env_ids)

    # Fetch frame at sampled time
    frame = self.motion_lib.get_frame(
      self.motion_ids[env_ids], self.motion_times[env_ids]
    )

    # Root state from anchor body (body index 0 in the tracked list is pelvis)
    root_pos = frame.body_pos_w[:, 0].clone() + self._env.scene.env_origins[env_ids]
    root_ori = frame.body_quat_w[:, 0].clone()
    root_lin_vel = frame.body_lin_vel_w[:, 0].clone()
    root_ang_vel = frame.body_ang_vel_w[:, 0].clone()

    # Apply random perturbations
    range_list = [
      self.cfg.pose_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (n, 6), device=self.device
    )
    root_pos += rand_samples[:, 0:3]
    orientations_delta = quat_from_euler_xyz(
      rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
    )
    root_ori = quat_mul(orientations_delta, root_ori)

    range_list = [
      self.cfg.velocity_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (n, 6), device=self.device
    )
    root_lin_vel += rand_samples[:, :3]
    root_ang_vel += rand_samples[:, 3:]

    joint_pos = frame.joint_pos.clone()
    joint_vel = frame.joint_vel

    joint_pos += sample_uniform(
      lower=self.cfg.joint_position_range[0],
      upper=self.cfg.joint_position_range[1],
      size=joint_pos.shape,
      device=joint_pos.device,
    )

    self._write_reference_state_to_sim(
      env_ids, root_pos, root_ori, root_lin_vel, root_ang_vel, joint_pos, joint_vel
    )

  def _adaptive_sampling(self, env_ids: torch.Tensor) -> None:
    """Bin-based failure-aware sampling (same algorithm as mjlab MotionCommand)."""
    episode_failed = self._env.termination_manager.terminated[env_ids]
    if torch.any(episode_failed):
      motion_lengths = self.motion_lib.get_motion_length(self.motion_ids[env_ids])
      phase = (self.motion_times[env_ids] / motion_lengths.clamp(min=1e-6)).clamp(
        0, 1
      )
      current_bin_index = (phase * (self.bin_count - 1)).long().clamp(
        0, self.bin_count - 1
      )
      fail_bins = current_bin_index[episode_failed]
      self._current_bin_failed[:] = torch.bincount(
        fail_bins, minlength=self.bin_count
      ).float()

    # Sample
    sampling_probabilities = (
      self.bin_failed_count
      + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
    )
    sampling_probabilities = torch.nn.functional.pad(
      sampling_probabilities.unsqueeze(0).unsqueeze(0),
      (0, self.cfg.adaptive_kernel_size - 1),
      mode="replicate",
    )
    sampling_probabilities = torch.nn.functional.conv1d(
      sampling_probabilities, self.kernel.view(1, 1, -1)
    ).view(-1)
    sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

    sampled_bins = torch.multinomial(
      sampling_probabilities, len(env_ids), replacement=True
    )
    phase = (
      sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device)
    ) / self.bin_count

    # Convert phase to motion time — sample random motion first
    self.motion_ids[env_ids] = self.motion_lib.sample_motions(len(env_ids))
    motion_lengths = self.motion_lib.get_motion_length(self.motion_ids[env_ids])
    self.motion_times[env_ids] = phase * motion_lengths

    # Update metrics
    H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
    H_norm = H / math.log(self.bin_count) if self.bin_count > 1 else 1.0
    pmax, imax = sampling_probabilities.max(dim=0)
    self.metrics["sampling_entropy"][:] = H_norm
    self.metrics["sampling_top1_prob"][:] = pmax
    self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

  def update_relative_body_poses(self) -> None:
    """Recompute body_pos_relative_w and body_quat_relative_w."""
    n_bodies = len(self.cfg.body_names)
    anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, n_bodies, 1)
    anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, n_bodies, 1)
    robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(
      1, n_bodies, 1
    )
    robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(
      1, n_bodies, 1
    )

    delta_pos_w = robot_anchor_pos_w_repeat.clone()
    delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
    delta_ori_w = yaw_quat(
      quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat))
    )

    self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
    self.body_pos_relative_w = delta_pos_w + quat_apply(
      delta_ori_w, self.body_pos_w - anchor_pos_w_repeat
    )

  def _update_command(self) -> None:
    # Advance time
    self.motion_times += self._env.step_dt

    # Resample envs that exceeded motion length
    motion_lengths = self.motion_lib.get_motion_length(self.motion_ids)
    env_ids = torch.where(self.motion_times >= motion_lengths)[0]
    if env_ids.numel() > 0:
      self._resample_command(env_ids)

    # Fetch frame for all envs
    frame = self.motion_lib.get_frame(self.motion_ids, self.motion_times)
    self._joint_pos = frame.joint_pos
    self._joint_vel = frame.joint_vel
    self._body_pos_w = frame.body_pos_w
    self._body_quat_w = frame.body_quat_w
    self._body_lin_vel_w = frame.body_lin_vel_w
    self._body_ang_vel_w = frame.body_ang_vel_w

    self.update_relative_body_poses()

    # Update adaptive sampling EMA
    if self.cfg.sampling_mode == "adaptive":
      self.bin_failed_count = (
        self.cfg.adaptive_alpha * self._current_bin_failed
        + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
      )
      self._current_bin_failed.zero_()

  # ==================== Visualization ====================

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    if self.cfg.viz.mode == "ghost":
      if self._ghost_model is None:
        self._ghost_model = copy.deepcopy(self._env.sim.mj_model)
        self._ghost_model.geom_rgba[:] = self._ghost_color

      entity: Entity = self._env.scene[self.cfg.entity_name]
      indexing = entity.indexing
      free_joint_q_adr = indexing.free_joint_q_adr.cpu().numpy()
      joint_q_adr = indexing.joint_q_adr.cpu().numpy()

      for batch in env_indices:
        qpos = np.zeros(self._env.sim.mj_model.nq)
        qpos[free_joint_q_adr[0:3]] = self.body_pos_w[batch, 0].cpu().numpy()
        qpos[free_joint_q_adr[3:7]] = self.body_quat_w[batch, 0].cpu().numpy()
        qpos[joint_q_adr] = self._joint_pos[batch].cpu().numpy()
        visualizer.add_ghost_mesh(
          qpos, model=self._ghost_model, label=f"ghost_{batch}"
        )

    elif self.cfg.viz.mode == "frames":
      for batch in env_indices:
        desired_body_pos = self.body_pos_w[batch].cpu().numpy()
        desired_body_quat = self.body_quat_w[batch]
        desired_body_rotm = matrix_from_quat(desired_body_quat).cpu().numpy()

        current_body_pos = self.robot_body_pos_w[batch].cpu().numpy()
        current_body_quat = self.robot_body_quat_w[batch]
        current_body_rotm = matrix_from_quat(current_body_quat).cpu().numpy()

        for i, body_name in enumerate(self.cfg.body_names):
          visualizer.add_frame(
            position=desired_body_pos[i],
            rotation_matrix=desired_body_rotm[i],
            scale=0.08,
            label=f"desired_{body_name}_{batch}",
            axis_colors=_DESIRED_FRAME_COLORS,
          )
          visualizer.add_frame(
            position=current_body_pos[i],
            rotation_matrix=current_body_rotm[i],
            scale=0.12,
            label=f"current_{body_name}_{batch}",
          )

    # Teacher ghost overlays (WBC + Loco, both grounded to the live robot root)
    # are commented out — they clutter the dual-student play view. Motion
    # reference ghost above stays on for teacher debugging.
    # self._render_teacher_ghosts(visualizer, env_indices)

  def _render_teacher_ghosts(
    self, visualizer: DebugVisualizer, env_indices: list[int] | range
  ) -> None:
    """Render ghost robots for WBC and Loco teacher joint-position predictions."""
    entity: Entity = self._env.scene[self.cfg.entity_name]
    indexing = entity.indexing
    free_joint_q_adr = indexing.free_joint_q_adr.cpu().numpy()
    joint_q_adr = indexing.joint_q_adr.cpu().numpy()

    # Current robot root state (teachers predict joints, not root pose)
    root_pos = entity.data.root_link_pos_w  # (N, 3)
    root_quat = entity.data.root_link_quat_w  # (N, 4)

    for targets, ghost_attr, color_attr, label_prefix in (
      (self._wbc_teacher_joint_targets, "_wbc_ghost_model", "_wbc_ghost_color", "wbc_teacher"),
      (self._loco_teacher_joint_targets, "_loco_ghost_model", "_loco_ghost_color", "loco_teacher"),
    ):
      if targets is None:
        continue
      ghost_model = getattr(self, ghost_attr)
      if ghost_model is None:
        ghost_model = copy.deepcopy(self._env.sim.mj_model)
        ghost_model.geom_rgba[:] = getattr(self, color_attr)
        setattr(self, ghost_attr, ghost_model)

      for batch in env_indices:
        qpos = np.zeros(self._env.sim.mj_model.nq)
        qpos[free_joint_q_adr[0:3]] = root_pos[batch].cpu().numpy()
        qpos[free_joint_q_adr[3:7]] = root_quat[batch].cpu().numpy()
        qpos[joint_q_adr] = targets[batch].cpu().numpy()
        visualizer.add_ghost_mesh(
          qpos, model=ghost_model, label=f"{label_prefix}_{batch}"
        )

  # ==================== GUI ====================

  def create_gui(
    self,
    name: str,
    server: viser.ViserServer,
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    max_frame = int(
      self.motion_lib._motion_num_frames.max().item()
    ) - 1

    with server.gui.add_folder(name.capitalize()):
      scrubber = server.gui.add_slider(
        "Frame", min=0, max=max_frame, step=1, initial_value=0
      )

      @scrubber.on_update
      def _(_) -> None:
        idx = get_env_idx()
        motion_length = self.motion_lib.get_motion_length(
          self.motion_ids[idx : idx + 1]
        )
        num_frames = self.motion_lib._motion_num_frames[self.motion_ids[idx]]
        frame_time = float(scrubber.value) / max(num_frames.item() - 1, 1) * motion_length.item()
        self.motion_times[idx] = frame_time
        if on_change is not None:
          on_change()

      all_envs_cb = server.gui.add_checkbox("All envs", initial_value=True)
      start_btn = server.gui.add_button("Start Here")

      @start_btn.on_click
      def _(_) -> None:
        if request_action is not None:
          request_action(
            "CUSTOM", {"type": "gui_reset", "all_envs": all_envs_cb.value}
          )

    self._scrubber_handles = (scrubber, all_envs_cb, start_btn)
    self._set_scrubber_disabled(True)

  def _set_scrubber_disabled(self, disabled: bool) -> None:
    for handle in self._scrubber_handles:
      handle.disabled = disabled

  def on_viewer_pause(self, paused: bool) -> None:
    if hasattr(self, "_scrubber_handles"):
      self._set_scrubber_disabled(not paused)

  def apply_gui_reset(self, env_ids: torch.Tensor) -> bool:
    if not hasattr(self, "_scrubber_handles"):
      return False
    # Reset to current motion time
    frame = self.motion_lib.get_frame(
      self.motion_ids[env_ids], self.motion_times[env_ids]
    )
    root_pos = frame.body_pos_w[:, 0] + self._env.scene.env_origins[env_ids]
    self._write_reference_state_to_sim(
      env_ids,
      root_pos,
      frame.body_quat_w[:, 0],
      frame.body_lin_vel_w[:, 0],
      frame.body_ang_vel_w[:, 0],
      frame.joint_pos,
      frame.joint_vel,
    )
    self.update_relative_body_poses()
    return True


@dataclass(kw_only=True)
class PklMotionCommandCfg(MotionCommandCfg):
  """Configuration for PKL motion command. Inherits all fields from MotionCommandCfg."""

  def build(self, env: ManagerBasedRlEnv) -> PklMotionCommand:
    return PklMotionCommand(self, env)


class DelayRecoveryPklMotionCommand(PklMotionCommand):
  """PKL motion command variant that re-writes a fallen Recovery/ pose for
  delay-tagged envs after its standard reference-state write.

  Required because mjlab's ``_reset_idx`` runs ``event_manager.apply(reset)``
  BEFORE ``command_manager.reset``. Without this subclass, the fallen pose
  written by the ``reset_delay_envs_to_recovery`` event is silently
  overwritten by ``_write_reference_state_to_sim`` when the command manager
  resamples — leaving ``recovery_active`` False on all spawn-fallen envs
  and starving the AMP-recovery-teacher KL of any active samples.
  """

  def _write_reference_state_to_sim(
    self,
    env_ids: torch.Tensor,
    root_pos: torch.Tensor,
    root_ori: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
  ) -> None:
    super()._write_reference_state_to_sim(
      env_ids, root_pos, root_ori, root_lin_vel, root_ang_vel, joint_pos, joint_vel
    )
    from wbc_mjlab.amp_mdp import _override_delay_envs_with_recovery_pose

    delay_ids = _override_delay_envs_with_recovery_pose(self._env, env_ids)
    if delay_ids is not None:
      # super() already called self.robot.reset for the now-stale
      # motion-frame state on these env_ids; refresh again for the
      # fallen-frame state we just wrote on top.
      self.robot.reset(env_ids=delay_ids)


@dataclass(kw_only=True)
class DelayRecoveryPklMotionCommandCfg(PklMotionCommandCfg):
  """Configuration for the recovery-aware PKL motion command. Same fields
  as :class:`PklMotionCommandCfg`; the only difference is which class the
  ``build`` hook constructs."""

  def build(self, env: ManagerBasedRlEnv) -> DelayRecoveryPklMotionCommand:
    return DelayRecoveryPklMotionCommand(self, env)


class LocoArmMotionCommand(CommandTerm):
  """Lightweight motion dataset holder for locomotion-teacher arm trajectories."""

  cfg: "LocoArmMotionCommandCfg"

  def __init__(self, cfg: "LocoArmMotionCommandCfg", env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.motion_lib = PklMotionLib(
      cfg.motion_file,
      cfg.body_names,
      device=self.device,
    )
    self._command = torch.zeros(self.num_envs, 0, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self._command

  def _update_metrics(self) -> None:
    return None

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    del env_ids

  def _update_command(self) -> None:
    return None










@dataclass(kw_only=True)
class LocoArmMotionCommandCfg(CommandTermCfg):
  """Config for the locomotion-teacher arm-motion dataset."""

  motion_file: str
  body_names: tuple[str, ...] = ("torso_link",)
  resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
  debug_vis: bool = False

  def build(self, env: ManagerBasedRlEnv) -> LocoArmMotionCommand:
    return LocoArmMotionCommand(self, env)
