"""Hand-command PD: close a feedback loop around the Molmo hand target.

The hand-student policy tracks a pelvis-frame wrist target. When Molmo
also provides the raw object point, this module measures the physical
gripper midpoint via FK and adds an XY-only correction so the midpoint,
not just the wrist, aligns with the raw point:

    shaped = staged_wrist_target + PD(raw_gripper_point - actual_gripper_midpoint)

When no raw point is available, the controller falls back to the
historical wrist-position shaping behavior.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import pinocchio as pin


class HandCommandPD:
  """Closed-loop shaping of per-hand pelvis-frame position targets.

  Reuses the pinocchio model/data already parsed by ``WristLeveller`` —
  see :meth:`from_leveller`. Self-contained FK call so ordering against
  other leveler/FK consumers doesn't matter: each caller rebuilds ``q``
  from scratch.
  """

  def __init__(
    self,
    model: pin.Model,
    data: pin.Data,
    q_index_map: np.ndarray,
    left_wrist_frame_id: int,
    right_wrist_frame_id: int,
    pelvis_frame_id: int,
    wrist_policy_indices: Sequence[int],
    *,
    left_tool_point_wrist: np.ndarray,
    right_tool_point_wrist: np.ndarray,
    kp: float,
    kd: float,
    kp_xy: Sequence[float],
    kd_xy: Sequence[float],
    deadband_m: float,
    max_shape_m: float,
  ):
    self._model = model
    self._data = data
    self._q_idx = np.asarray(q_index_map, dtype=np.int64)
    self._left_fid = int(left_wrist_frame_id)
    self._right_fid = int(right_wrist_frame_id)
    self._pelvis_fid = int(pelvis_frame_id)
    # Wrist joints are zeroed in the FK query to match WristLeveller's
    # convention: FK the arm pose the policy nominally commanded, not
    # what the leveler overrode on top of the policy's wrist action.
    self._wrist_q_zero = tuple(int(i) for i in wrist_policy_indices)
    self._left_tool_point_wrist = np.asarray(
      left_tool_point_wrist, dtype=np.float64,
    ).reshape(3).copy()
    self._right_tool_point_wrist = np.asarray(
      right_tool_point_wrist, dtype=np.float64,
    ).reshape(3).copy()
    self._kp = float(kp)
    self._kd = float(kd)
    self._kp_xy = np.asarray(kp_xy, dtype=np.float64).reshape(2).copy()
    self._kd_xy = np.asarray(kd_xy, dtype=np.float64).reshape(2).copy()
    self._deadband = float(deadband_m)
    self._max_shape = float(max_shape_m)
    self._prev_err_left: Optional[np.ndarray] = None
    self._prev_err_right: Optional[np.ndarray] = None

  @classmethod
  def from_leveller(
    cls,
    lev,
    *,
    left_tool_point_wrist: Optional[np.ndarray] = None,
    right_tool_point_wrist: Optional[np.ndarray] = None,
    kp: float,
    kd: float,
    kp_xy: Sequence[float],
    kd_xy: Sequence[float],
    deadband_m: float,
    max_shape_m: float,
  ) -> "HandCommandPD":
    return cls(
      lev.pin_model,
      lev.pin_data,
      lev.q_index_map,
      lev.left_wrist_frame_id,
      lev.right_wrist_frame_id,
      lev.pelvis_frame_id,
      tuple(lev.left_policy_indices) + tuple(lev.right_policy_indices),
      left_tool_point_wrist=(
        lev.left_tool_point_wrist
        if left_tool_point_wrist is None else left_tool_point_wrist
      ),
      right_tool_point_wrist=(
        lev.right_tool_point_wrist
        if right_tool_point_wrist is None else right_tool_point_wrist
      ),
      kp=kp,
      kd=kd,
      kp_xy=kp_xy,
      kd_xy=kd_xy,
      deadband_m=deadband_m,
      max_shape_m=max_shape_m,
    )

  def reset(self) -> None:
    self._prev_err_left = None
    self._prev_err_right = None

  def shape(
    self,
    joint_pos: np.ndarray,
    root_pos: np.ndarray,
    root_quat_wxyz: np.ndarray,
    target_b_left: np.ndarray,
    target_b_right: np.ndarray,
    dt: float,
    active_left: bool,
    active_right: bool,
    raw_target_b_left: Optional[np.ndarray] = None,
    raw_target_b_right: Optional[np.ndarray] = None,
  ) -> Tuple[np.ndarray, np.ndarray]:
    """Return shaped (left, right) pelvis-frame position targets.

    Inactive side: passes through unchanged and drops that side's D-term
    memory so the next activation starts fresh.
    """
    left = np.asarray(target_b_left, dtype=np.float64).reshape(3).copy()
    right = np.asarray(target_b_right, dtype=np.float64).reshape(3).copy()
    raw_left = (
      None if raw_target_b_left is None
      else np.asarray(raw_target_b_left, dtype=np.float64).reshape(3).copy()
    )
    raw_right = (
      None if raw_target_b_right is None
      else np.asarray(raw_target_b_right, dtype=np.float64).reshape(3).copy()
    )

    if not active_left and not active_right:
      self._prev_err_left = None
      self._prev_err_right = None
      return left.astype(np.float32), right.astype(np.float32)

    need_measured_wrist = (
      (active_left and raw_left is not None)
      or (active_right and raw_right is not None)
    )
    need_zero_wrist = (
      (active_left and raw_left is None)
      or (active_right and raw_right is None)
    )
    left_actual = left.copy()
    right_actual = right.copy()
    if need_measured_wrist:
      pelvis_tf_measured = self._forward_kinematics(
        joint_pos, root_pos, root_quat_wxyz, zero_wrist=False,
      )
      if active_left and raw_left is not None:
        left_actual = self._tool_point_body(
          self._left_fid, self._left_tool_point_wrist, pelvis_tf_measured,
        )
      if active_right and raw_right is not None:
        right_actual = self._tool_point_body(
          self._right_fid, self._right_tool_point_wrist, pelvis_tf_measured,
        )
    if need_zero_wrist:
      pelvis_tf_zero = self._forward_kinematics(
        joint_pos, root_pos, root_quat_wxyz, zero_wrist=True,
      )
      if active_left and raw_left is None:
        left_actual = self._frame_body(self._left_fid, pelvis_tf_zero)
      if active_right and raw_right is None:
        right_actual = self._frame_body(self._right_fid, pelvis_tf_zero)

    left = self._shape_side(
      "left",
      left,
      raw_left if raw_left is not None else left,
      left_actual,
      dt,
      active_left,
      xy_only=raw_left is not None,
    )
    right = self._shape_side(
      "right",
      right,
      raw_right if raw_right is not None else right,
      right_actual,
      dt,
      active_right,
      xy_only=raw_right is not None,
    )
    return left.astype(np.float32), right.astype(np.float32)

  def _shape_side(
    self,
    side: str,
    base_target: np.ndarray,
    ref_target: np.ndarray,
    actual: np.ndarray,
    dt: float,
    active: bool,
    *,
    xy_only: bool,
  ) -> np.ndarray:
    attr = "_prev_err_left" if side == "left" else "_prev_err_right"
    if not active:
      setattr(self, attr, None)
      return base_target
    err = ref_target - actual
    # Per-axis deadband: ignore errors small enough to be noise,
    # otherwise small sensor jitter would translate into nonzero shape.
    err = np.where(np.abs(err) < self._deadband, 0.0, err)
    if xy_only:
      err[2] = 0.0
    prev = getattr(self, attr)
    if prev is None:
      # First tick since (re-)activation — no D term yet, so the shape
      # ramps smoothly from 0 on this tick's error rather than jumping
      # on a stale derivative across the gate edge.
      err_dot = np.zeros_like(err)
    else:
      err_dot = (err - prev) / max(float(dt), 1e-6)
    setattr(self, attr, err.copy())
    if xy_only:
      shape = np.zeros(3, dtype=np.float64)
      shape[:2] = self._kp_xy * err[:2] + self._kd_xy * err_dot[:2]
      norm = float(np.linalg.norm(shape[:2]))
      if norm > self._max_shape and norm > 0.0:
        shape[:2] = shape[:2] * (self._max_shape / norm)
    else:
      shape = self._kp * err + self._kd * err_dot
      norm = float(np.linalg.norm(shape))
      if norm > self._max_shape and norm > 0.0:
        shape = shape * (self._max_shape / norm)
    return base_target + shape

  def _forward_kinematics(
    self,
    joint_pos: np.ndarray,
    root_pos: np.ndarray,
    root_quat_wxyz: np.ndarray,
    *,
    zero_wrist: bool,
  ):
    q = pin.neutral(self._model)
    q[0:3] = np.asarray(root_pos, dtype=np.float64).reshape(3)
    quat = np.asarray(root_quat_wxyz, dtype=np.float64).reshape(4)
    q[3:7] = np.array([quat[1], quat[2], quat[3], quat[0]], dtype=np.float64)
    jp = np.asarray(joint_pos, dtype=np.float64).reshape(-1)
    q[self._q_idx] = jp
    if zero_wrist:
      for idx in self._wrist_q_zero:
        q[self._q_idx[idx]] = 0.0
    pin.forwardKinematics(self._model, self._data, q)
    pin.updateFramePlacements(self._model, self._data)
    return self._data.oMf[self._pelvis_fid]

  def _frame_body(self, fid: int, pelvis_tf) -> np.ndarray:
    R_pw = np.asarray(pelvis_tf.rotation, dtype=np.float64)
    t_pw = np.asarray(pelvis_tf.translation, dtype=np.float64)
    frame_w = np.asarray(self._data.oMf[fid].translation, dtype=np.float64)
    return R_pw.T @ (frame_w - t_pw)

  def _tool_point_body(
    self,
    fid: int,
    tool_point_wrist: np.ndarray,
    pelvis_tf,
  ) -> np.ndarray:
    wrist_tf = self._data.oMf[fid]
    tool_world = (
      np.asarray(wrist_tf.translation, dtype=np.float64)
      + np.asarray(wrist_tf.rotation, dtype=np.float64)
      @ np.asarray(tool_point_wrist, dtype=np.float64).reshape(3)
    )
    R_pw = np.asarray(pelvis_tf.rotation, dtype=np.float64)
    t_pw = np.asarray(pelvis_tf.translation, dtype=np.float64)
    return R_pw.T @ (tool_world - t_pw)
