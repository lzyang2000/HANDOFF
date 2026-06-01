"""Multi-motion library for enriched PKL files.

Loads enriched PKL files (with body_pos_w, body_quat_w) from a YAML dataset
or single PKL. Supports multi-motion sampling and frame interpolation.
"""


import os
import pickle
from dataclasses import dataclass

import torch
import yaml
from tqdm import tqdm

import numpy as np


class _NumpyCompatUnpickler(pickle.Unpickler):
  """Load NumPy-2-era pickles under NumPy 1.x by remapping module names."""

  _MODULE_MAP = {
    "numpy._core.multiarray": "numpy.core.multiarray",
    "numpy._core.numeric": "numpy.core.numeric",
  }

  def find_class(self, module, name):
    module = self._MODULE_MAP.get(module, module)
    return super().find_class(module, name)


def _load_pickle_compat(file_obj):
  return _NumpyCompatUnpickler(file_obj).load()


@dataclass
class FrameData:
  """Interpolated frame data for a batch of environments."""

  joint_pos: torch.Tensor  # [B, N_joints]
  joint_vel: torch.Tensor  # [B, N_joints]
  body_pos_w: torch.Tensor  # [B, N_tracked, 3]
  body_quat_w: torch.Tensor  # [B, N_tracked, 4] in [w,x,y,z]
  body_lin_vel_w: torch.Tensor  # [B, N_tracked, 3]
  body_ang_vel_w: torch.Tensor  # [B, N_tracked, 3]


def _batched_slerp(
  q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor
) -> torch.Tensor:
  """Batched SLERP between quaternions in [w,x,y,z] format.

  Args:
    q0: Start quaternions, shape [..., 4].
    q1: End quaternions, shape [..., 4].
    t: Interpolation factors, shape [...] (broadcastable).

  Returns:
    Interpolated quaternions, shape [..., 4].
  """
  # Broadcast t to match q0 shape: [..., 4]
  while t.dim() < q0.dim():
    t = t.unsqueeze(-1)

  # Ensure shortest path
  dot = (q0 * q1).sum(dim=-1, keepdim=True)
  q1 = torch.where(dot < 0, -q1, q1)
  dot = dot.abs().clamp(max=1.0 - 1e-6)

  omega = torch.acos(dot)
  sin_omega = torch.sin(omega)

  # For very small angles, fall back to linear interpolation
  small = sin_omega.abs() < 1e-6
  s0 = torch.where(small, 1.0 - t, torch.sin((1.0 - t) * omega) / sin_omega)
  s1 = torch.where(small, t, torch.sin(t * omega) / sin_omega)

  return s0 * q0 + s1 * q1


def _compute_ang_vel_from_quat(
  quats: torch.Tensor, dt: float
) -> torch.Tensor:
  """Compute angular velocities from a quaternion sequence using central differences.

  Args:
    quats: Quaternions in [w,x,y,z], shape [T, ..., 4].
    dt: Time step between frames.

  Returns:
    Angular velocities, shape [T, ..., 3].
  """
  from mjlab.utils.lab_api.math import quat_box_minus

  T = quats.shape[0]
  orig_shape = quats.shape[1:-1]  # middle dims (e.g., N_tracked)
  flat_q = quats.reshape(T, -1, 4)  # [T, B, 4]
  B = flat_q.shape[1]

  omega = torch.zeros(T, B, 3, device=quats.device, dtype=quats.dtype)

  if T < 2:
    return omega.reshape(T, *orig_shape, 3)

  if T == 2:
    # Forward difference only
    omega[0] = quat_box_minus(flat_q[1], flat_q[0]) / dt
    omega[1] = omega[0]
    return omega.reshape(T, *orig_shape, 3)

  # Central differences for interior points
  omega[1:-1] = (
    quat_box_minus(
      flat_q[2:].reshape(-1, 4), flat_q[:-2].reshape(-1, 4)
    ).reshape(T - 2, B, 3)
    / (2.0 * dt)
  )

  # Boundary: forward/backward differences
  omega[0] = quat_box_minus(flat_q[1], flat_q[0]) / dt
  omega[-1] = quat_box_minus(flat_q[-1], flat_q[-2]) / dt

  return omega.reshape(T, *orig_shape, 3)


class PklMotionLib:
  """Multi-motion library that loads enriched PKL files."""

  def __init__(
    self,
    motion_file: str,
    body_names: tuple[str, ...],
    device: str = "cpu",
  ) -> None:
    self._device = device
    self._body_names = body_names

    # Accumulators during loading
    self._acc_joint_pos: list[torch.Tensor] = []
    self._acc_joint_vel: list[torch.Tensor] = []
    self._acc_body_pos_w: list[torch.Tensor] = []
    self._acc_body_quat_w: list[torch.Tensor] = []
    self._acc_body_lin_vel_w: list[torch.Tensor] = []
    self._acc_body_ang_vel_w: list[torch.Tensor] = []
    self._acc_num_frames: list[int] = []
    self._acc_lengths: list[float] = []
    self._acc_weights: list[float] = []

    self._load_motions(motion_file)

  def _load_motions(self, motion_file: str) -> None:
    files, weights = self._parse_motion_file(motion_file)

    for i in tqdm(range(len(files)), desc="[PklMotionLib] Loading motions"):
      path = files[i]
      if not os.path.exists(path):
        print(f"[PklMotionLib] Skipping missing file: {path}")
        continue

      try:
        with open(path, "rb") as f:
          data = _load_pickle_compat(f)
      except Exception as e:
        print(f"[PklMotionLib] Error loading {path}: {e}")
        continue

      self._add_motion(data, weights[i])

    self._finalize()
    print(
      f"[PklMotionLib] Loaded {self._num_motions} motions, "
      f"{self._total_frames} total frames."
    )

  def _parse_motion_file(
    self, motion_file: str
  ) -> tuple[list[str], list[float]]:
    if motion_file.endswith(".yaml") or motion_file.endswith(".yml"):
      with open(motion_file) as f:
        config = yaml.safe_load(f)
      root_path = config["root_path"]
      files = [os.path.join(root_path, m["file"]) for m in config["motions"]]
      weights = [m.get("weight", 1.0) for m in config["motions"]]
      return files, weights
    else:
      return [motion_file], [1.0]

  def _add_motion(self, data: dict, weight: float) -> None:
    fps = data["fps"]
    dt = 1.0 / fps

    # Joint data — PKL stores as dof_pos
    joint_pos = torch.tensor(
      np.asarray(data["dof_pos"]), dtype=torch.float32, device=self._device
    )
    T = joint_pos.shape[0]
    if T < 2:
      return

    joint_vel = torch.gradient(joint_pos, spacing=(dt,), dim=0)[0]

    # Body data — from enriched PKL
    if "body_pos_w" not in data or "body_quat_w" not in data:
      raise ValueError(
        f"PKL missing 'body_pos_w' or 'body_quat_w'. "
        f"Run enrich_pkl.py first to add world-frame body data."
      )

    link_body_list = data["link_body_list"]
    tracked_indices = self._get_tracked_indices(link_body_list)

    all_body_pos_w = torch.tensor(
      np.asarray(data["body_pos_w"]), dtype=torch.float32, device=self._device
    )
    all_body_quat_w = torch.tensor(
      np.asarray(data["body_quat_w"]), dtype=torch.float32, device=self._device
    )

    body_pos_w = all_body_pos_w[:, tracked_indices]  # [T, N_tracked, 3]
    body_quat_w = all_body_quat_w[:, tracked_indices]  # [T, N_tracked, 4]

    body_lin_vel_w = torch.gradient(body_pos_w, spacing=(dt,), dim=0)[0]
    body_ang_vel_w = _compute_ang_vel_from_quat(body_quat_w, dt)

    motion_length = dt * (T - 1)

    self._acc_joint_pos.append(joint_pos)
    self._acc_joint_vel.append(joint_vel)
    self._acc_body_pos_w.append(body_pos_w)
    self._acc_body_quat_w.append(body_quat_w)
    self._acc_body_lin_vel_w.append(body_lin_vel_w)
    self._acc_body_ang_vel_w.append(body_ang_vel_w)
    self._acc_num_frames.append(T)
    self._acc_lengths.append(motion_length)
    self._acc_weights.append(weight)

  def _get_tracked_indices(self, link_body_list: list[str]) -> list[int]:
    indices = []
    for name in self._body_names:
      if name not in link_body_list:
        raise ValueError(
          f"Tracked body '{name}' not found in PKL link_body_list: "
          f"{link_body_list}"
        )
      indices.append(link_body_list.index(name))
    return indices

  def _finalize(self) -> None:
    """Concatenate accumulators into flat GPU tensors."""
    if not self._acc_weights:
      raise RuntimeError("[PklMotionLib] No motions loaded!")

    self._num_motions = len(self._acc_weights)
    self._total_frames = sum(self._acc_num_frames)

    self._all_joint_pos = torch.cat(self._acc_joint_pos, dim=0)
    self._all_joint_vel = torch.cat(self._acc_joint_vel, dim=0)
    self._all_body_pos_w = torch.cat(self._acc_body_pos_w, dim=0)
    self._all_body_quat_w = torch.cat(self._acc_body_quat_w, dim=0)
    self._all_body_lin_vel_w = torch.cat(self._acc_body_lin_vel_w, dim=0)
    self._all_body_ang_vel_w = torch.cat(self._acc_body_ang_vel_w, dim=0)

    self._motion_num_frames = torch.tensor(
      self._acc_num_frames, dtype=torch.long, device=self._device
    )
    self._motion_lengths = torch.tensor(
      self._acc_lengths, dtype=torch.float32, device=self._device
    )

    weights = torch.tensor(
      self._acc_weights, dtype=torch.float32, device=self._device
    )
    self._motion_weights = weights / weights.sum()

    # Build cumulative start index
    shifted = self._motion_num_frames.roll(1)
    shifted[0] = 0
    self._motion_start_idx = shifted.cumsum(0)

    # Clear accumulators
    del self._acc_joint_pos, self._acc_joint_vel
    del self._acc_body_pos_w, self._acc_body_quat_w
    del self._acc_body_lin_vel_w, self._acc_body_ang_vel_w
    del self._acc_num_frames, self._acc_lengths, self._acc_weights

  # ==================== Public API ====================

  def num_motions(self) -> int:
    return self._num_motions

  def get_motion_length(self, motion_ids: torch.Tensor) -> torch.Tensor:
    return self._motion_lengths[motion_ids]

  def sample_motions(self, n: int) -> torch.Tensor:
    """Sample n motion IDs via weighted multinomial."""
    return torch.multinomial(
      self._motion_weights, num_samples=n, replacement=True
    )

  def sample_time(self, motion_ids: torch.Tensor) -> torch.Tensor:
    """Sample random phase within each motion."""
    phase = torch.rand(motion_ids.shape, device=self._device)
    lengths = self._motion_lengths[motion_ids]
    return lengths * phase

  def get_frame(
    self, motion_ids: torch.Tensor, motion_times: torch.Tensor
  ) -> FrameData:
    """Get interpolated frame data for given motion IDs and times."""
    lengths = self._motion_lengths[motion_ids]
    num_frames = self._motion_num_frames[motion_ids]
    start_idx = self._motion_start_idx[motion_ids]

    # Handle looping
    motion_times = motion_times % lengths.clamp(min=1e-6)

    # Compute blend factor
    phase = (motion_times / lengths).clamp(0.0, 1.0)
    frame_idx_f = phase * (num_frames - 1).float()
    frame_idx0 = frame_idx_f.long()
    frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
    blend = frame_idx_f - frame_idx0.float()

    # Offset by motion start
    idx0 = frame_idx0 + start_idx
    idx1 = frame_idx1 + start_idx

    # Fetch and interpolate
    b = blend.unsqueeze(-1)  # [B, 1]

    joint_pos = (1.0 - b) * self._all_joint_pos[idx0] + b * self._all_joint_pos[idx1]
    joint_vel = (1.0 - b) * self._all_joint_vel[idx0] + b * self._all_joint_vel[idx1]

    b2 = b.unsqueeze(-1)  # [B, 1, 1] for body tensors [B, N, D]
    body_pos_w = (
      (1.0 - b2) * self._all_body_pos_w[idx0] + b2 * self._all_body_pos_w[idx1]
    )
    body_quat_w = _batched_slerp(
      self._all_body_quat_w[idx0],
      self._all_body_quat_w[idx1],
      blend,
    )
    body_lin_vel_w = (
      (1.0 - b2) * self._all_body_lin_vel_w[idx0]
      + b2 * self._all_body_lin_vel_w[idx1]
    )
    body_ang_vel_w = (
      (1.0 - b2) * self._all_body_ang_vel_w[idx0]
      + b2 * self._all_body_ang_vel_w[idx1]
    )

    return FrameData(
      joint_pos=joint_pos,
      joint_vel=joint_vel,
      body_pos_w=body_pos_w,
      body_quat_w=body_quat_w,
      body_lin_vel_w=body_lin_vel_w,
      body_ang_vel_w=body_ang_vel_w,
    )
