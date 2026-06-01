"""Wrist-leveller: keep the gripper's 'up' axis aligned with world +z.

The control policy is trained with wrist joints at zero. We override the
policy's wrist commands post-hoc so the gripper physically stays level
(pinch plane horizontal, fingers horizontal forward — side-grasp pose)
regardless of the arm pose. Observations sent back to the policy keep
the wrist slots at zero so training/deployment distribution stays
consistent.

Toggled via two independent bool constants in
wbc_mjlab.g1_constants_custom:
  MOLMO_WRIST_LEVEL_OVERRIDE — apply the action-side override.
  MOLMO_WRIST_ZERO_OBS        — zero the observation-side wrist slots.
Flip the constants and restart the policy node.

Math: the wrist chain applies R_x(roll) · R_y(pitch) · R_z(yaw) after
the forearm's (elbow link's) world rotation. The z-column of that
composite rotation — which represents the gripper's "up" vector in
world — is independent of yaw (yaw rotates around the local z axis,
which leaves the z-column unchanged). We pin that z-column to world +z
via a closed-form decomposition:

    v     = R_forearm^T · ẑ_world
    pitch = arcsin(v[0])
    roll  = atan2(-v[1], v[2])     (or 0 at gimbal lock)
    yaw   = 0                      (free DOF)

Cost: one pinocchio FK + trivial arithmetic per call. Sub-ms.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Literal, Optional, Sequence, Tuple

import numpy as np
import pinocchio as pin


# Per-side wrist joint names (order: roll, pitch, yaw) + forearm/wrist frame names.
_LEFT_WRIST_JOINTS = ("left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint")
_RIGHT_WRIST_JOINTS = ("right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint")
_LEFT_FOREARM_FRAME = "left_elbow_link"
_RIGHT_FOREARM_FRAME = "right_elbow_link"
_LEFT_WRIST_FRAME = "left_wrist_yaw_link"
_RIGHT_WRIST_FRAME = "right_wrist_yaw_link"
_PELVIS_FRAME = "pelvis"
_DEFAULT_LEFT_TOOL_POINT_WRIST = np.array([0.1830, 0.0030, 0.0], dtype=np.float64)
_DEFAULT_RIGHT_TOOL_POINT_WRIST = np.array([0.1830, -0.0030, 0.0], dtype=np.float64)
WristOrientationMode = Literal["level", "vertical"]


class WristLeveller:
    """Computes level-keeping wrist angles (roll, pitch, yaw) per side.

    Usage:
        lev = WristLeveller(urdf_path, policy_joint_names)
        # per tick (dt seconds since last call):
        left_rpy, right_rpy = lev.compute(
            joint_pos,      # current (policy_joint_names ordering)
            root_pos,       # world pelvis xyz
            root_quat_wxyz, # world pelvis wxyz
            dt=0.02,
        )
        # Either may be None → caller should pass through the policy's
        # own wrist commands for that side (e.g. near the gimbal-lock
        # region where roll can flip by π across a small arm move).

    ``policy_joint_names`` must include the 6 wrist joints; their
    indices are cached so callers can pick the overrides out of the
    returned arrays.

    Two stability knobs:
      * max_rate_rad_per_sec — slew-limits the commanded wrist angles
        so a sudden activation doesn't send a ~75° PD step into the
        forearm in one tick.
      * gimbal_v0_threshold — if the world-up expressed in forearm
        frame lands close to the roll-axis singularity, atan2 becomes
        ill-conditioned (small arm moves flip the roll command by π).
        Above the threshold we return None for that side so the caller
        keeps the policy's wrist command instead.
    """

    # Defaults chosen so a wrist-at-zero → fully-leveled transition takes
    # ~0.5 s at max rate (plenty of time for the PD controller to follow
    # without torque spikes).
    _DEFAULT_MAX_RATE_RAD_PER_SEC: float = 2.5
    _DEFAULT_GIMBAL_V0_THRESH: float = 0.98
    _DEFAULT_YAW_EMA_ALPHA: float = 0.2

    def __init__(
        self,
        urdf_path: str | Path,
        policy_joint_names: Sequence[str],
        *,
        max_rate_rad_per_sec: float = _DEFAULT_MAX_RATE_RAD_PER_SEC,
        gimbal_v0_threshold: float = _DEFAULT_GIMBAL_V0_THRESH,
        yaw_ema_alpha: float = _DEFAULT_YAW_EMA_ALPHA,
        left_tool_point_wrist: Optional[np.ndarray] = None,
        right_tool_point_wrist: Optional[np.ndarray] = None,
        vertical_roll_rad: float = math.pi / 2.0,
    ):
        path = Path(urdf_path)
        if not path.exists():
            raise FileNotFoundError(f"URDF not found for WristLeveller: {path}")
        self._model = pin.buildModelFromUrdf(str(path), pin.JointModelFreeFlyer())
        self._data = self._model.createData()

        # q-index map for the policy joints into the pinocchio q vector.
        self._q_idx = self._build_q_index_map(policy_joint_names)

        # Forearm frame ids + wrist + pelvis ids for target-aware yaw solve.
        self._left_forearm_fid = self._model.getFrameId(_LEFT_FOREARM_FRAME)
        self._right_forearm_fid = self._model.getFrameId(_RIGHT_FOREARM_FRAME)
        self._left_wrist_fid = self._model.getFrameId(_LEFT_WRIST_FRAME)
        self._right_wrist_fid = self._model.getFrameId(_RIGHT_WRIST_FRAME)
        self._pelvis_fid = self._model.getFrameId(_PELVIS_FRAME)
        for fid, name in (
            (self._left_forearm_fid, _LEFT_FOREARM_FRAME),
            (self._right_forearm_fid, _RIGHT_FOREARM_FRAME),
            (self._left_wrist_fid, _LEFT_WRIST_FRAME),
            (self._right_wrist_fid, _RIGHT_WRIST_FRAME),
            (self._pelvis_fid, _PELVIS_FRAME),
        ):
            if fid >= self._model.nframes:
                raise ValueError(f"URDF missing frame: {name}")

        self._left_policy_idx = tuple(policy_joint_names.index(n) for n in _LEFT_WRIST_JOINTS)
        self._right_policy_idx = tuple(policy_joint_names.index(n) for n in _RIGHT_WRIST_JOINTS)

        # Joint limits (from the URDF) for clamping.
        self._left_limits = self._joint_limits(_LEFT_WRIST_JOINTS)
        self._right_limits = self._joint_limits(_RIGHT_WRIST_JOINTS)

        self._policy_joint_names = tuple(policy_joint_names)
        self._left_tool_point_wrist = (
            np.asarray(_DEFAULT_LEFT_TOOL_POINT_WRIST, dtype=np.float64).copy()
            if left_tool_point_wrist is None
            else np.asarray(left_tool_point_wrist, dtype=np.float64).reshape(3).copy()
        )
        self._right_tool_point_wrist = (
            np.asarray(_DEFAULT_RIGHT_TOOL_POINT_WRIST, dtype=np.float64).copy()
            if right_tool_point_wrist is None
            else np.asarray(right_tool_point_wrist, dtype=np.float64).reshape(3).copy()
        )

        # Slew + gimbal guards + yaw EMA
        self._max_rate = float(max_rate_rad_per_sec)
        self._gimbal_v0_thresh = float(gimbal_v0_threshold)
        self._yaw_ema_alpha = float(yaw_ema_alpha)
        self._vertical_roll_rad = float(vertical_roll_rad)
        self._prev_left_rpy: Optional[np.ndarray] = None
        self._prev_right_rpy: Optional[np.ndarray] = None
        self._prev_yaw_ema_left: Optional[float] = None
        self._prev_yaw_ema_right: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def left_policy_indices(self) -> Tuple[int, int, int]:
        """(roll, pitch, yaw) indices into the policy joint array."""
        return self._left_policy_idx

    @property
    def right_policy_indices(self) -> Tuple[int, int, int]:
        return self._right_policy_idx

    @property
    def left_limits(self) -> np.ndarray:
        """(3, 2) (lower, upper) URDF joint limits for (roll, pitch, yaw)."""
        return self._left_limits

    @property
    def right_limits(self) -> np.ndarray:
        return self._right_limits

    # Expose pin internals so sibling modules (e.g. HandCommandPD) can
    # reuse the parsed URDF instead of loading a second copy.
    @property
    def pin_model(self) -> pin.Model:
        return self._model

    @property
    def pin_data(self) -> pin.Data:
        return self._data

    @property
    def q_index_map(self) -> np.ndarray:
        return self._q_idx

    @property
    def left_wrist_frame_id(self) -> int:
        return self._left_wrist_fid

    @property
    def right_wrist_frame_id(self) -> int:
        return self._right_wrist_fid

    @property
    def pelvis_frame_id(self) -> int:
        return self._pelvis_fid

    @property
    def left_tool_point_wrist(self) -> np.ndarray:
        return self._left_tool_point_wrist.copy()

    @property
    def right_tool_point_wrist(self) -> np.ndarray:
        return self._right_tool_point_wrist.copy()

    def reset(self) -> None:
        self._prev_left_rpy = None
        self._prev_right_rpy = None
        self._prev_yaw_ema_left = None
        self._prev_yaw_ema_right = None

    def gripper_midpoint_body(
        self,
        joint_pos: np.ndarray,
        root_pos: np.ndarray,
        root_quat_wxyz: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return the pelvis-frame midpoint between the finger bars.

        This uses the measured wrist joints (no zeroing) because the
        correction path needs the physical gripper midpoint, not the
        wrist-neutral pose used by the leveller solve itself.
        """
        pelvis_tf = self._forward_kinematics(
            joint_pos, root_pos, root_quat_wxyz, zero_wrist=False,
        )
        left = self._tool_point_body(
            self._left_wrist_fid, self._left_tool_point_wrist, pelvis_tf,
        )
        right = self._tool_point_body(
            self._right_wrist_fid, self._right_tool_point_wrist, pelvis_tf,
        )
        return left.astype(np.float32), right.astype(np.float32)

    def target_in_horizontal_grasp_frame(
        self,
        hand: str,
        target_body: np.ndarray,
        joint_pos: np.ndarray,
        root_pos: np.ndarray,
        root_quat_wxyz: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Return ``(depth, lateral, vertical)`` error from the gripper
        midpoint to ``target_body``, expressed in a frame whose horizontal
        axes follow the gripper's world-horizontal finger heading and
        whose vertical axis is gravity (world +z).

        Decouples the depth/lateral/vertical decomposition from any wrist
        pitch the policy may apply during the approach: a leveled gripper
        and a pitched gripper produce the same result so long as the
        finger axis projects onto the same horizontal heading.

        ``target_body`` is assumed to be in the *current* pelvis frame —
        callers handing in a stale snapshot accept the small body-pose
        drift between query and now (pick chain is roughly stationary).
        Returns ``None`` if the gripper heading is degenerate (finger
        axis nearly vertical in world).
        """
        if hand not in ("l", "r"):
            raise ValueError(f"hand must be 'l' or 'r', got {hand!r}")
        pelvis_tf = self._forward_kinematics(
            joint_pos, root_pos, root_quat_wxyz, zero_wrist=False,
        )
        wrist_fid = self._left_wrist_fid if hand == "l" else self._right_wrist_fid
        wrist_tf = self._data.oMf[wrist_fid]
        tool_point = (
            self._left_tool_point_wrist if hand == "l"
            else self._right_tool_point_wrist
        )
        R_world_wrist = np.asarray(wrist_tf.rotation, dtype=np.float64)
        t_world_wrist = np.asarray(wrist_tf.translation, dtype=np.float64)
        midpoint_world = (
            t_world_wrist + R_world_wrist @ np.asarray(tool_point, dtype=np.float64).reshape(3)
        )

        finger_world = R_world_wrist[:, 0]  # wrist +x in world (finger axis)
        heading_xy_world = np.array([finger_world[0], finger_world[1], 0.0], dtype=np.float64)
        n = float(np.linalg.norm(heading_xy_world))
        if n < 1e-6:
            return None
        heading_xy_world /= n
        # Lateral axis: world up × heading (CCW perpendicular in horizontal plane).
        perp_xy_world = np.array(
            [-heading_xy_world[1], heading_xy_world[0], 0.0], dtype=np.float64,
        )

        R_world_pelvis = np.asarray(pelvis_tf.rotation, dtype=np.float64)
        t_world_pelvis = np.asarray(pelvis_tf.translation, dtype=np.float64)
        target_world = (
            t_world_pelvis
            + R_world_pelvis @ np.asarray(target_body, dtype=np.float64).reshape(3)
        )
        err_world = target_world - midpoint_world
        depth = float(err_world @ heading_xy_world)
        lateral = float(err_world @ perp_xy_world)
        vertical = float(err_world[2])
        return np.array([depth, lateral, vertical], dtype=np.float32)

    def horizontal_grasp_frame_world(
        self,
        hand: str,
        joint_pos: np.ndarray,
        root_pos: np.ndarray,
        root_quat_wxyz: np.ndarray,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return ``(midpoint_world, R_world_grasp)`` for the same frame
        ``target_in_horizontal_grasp_frame`` projects into. Useful for
        rendering: a box with this rotation drawn at this position is
        axis-aligned with depth (heading_xy), lateral (perp), vertical
        (world +z). Returns None if the heading is degenerate.
        """
        if hand not in ("l", "r"):
            raise ValueError(f"hand must be 'l' or 'r', got {hand!r}")
        pelvis_tf = self._forward_kinematics(
            joint_pos, root_pos, root_quat_wxyz, zero_wrist=False,
        )
        wrist_fid = self._left_wrist_fid if hand == "l" else self._right_wrist_fid
        wrist_tf = self._data.oMf[wrist_fid]
        tool_point = (
            self._left_tool_point_wrist if hand == "l"
            else self._right_tool_point_wrist
        )
        R_world_wrist = np.asarray(wrist_tf.rotation, dtype=np.float64)
        t_world_wrist = np.asarray(wrist_tf.translation, dtype=np.float64)
        midpoint_world = (
            t_world_wrist + R_world_wrist @ np.asarray(tool_point, dtype=np.float64).reshape(3)
        )
        finger_world = R_world_wrist[:, 0]
        heading_xy = np.array([finger_world[0], finger_world[1], 0.0], dtype=np.float64)
        n = float(np.linalg.norm(heading_xy))
        if n < 1e-6:
            return None
        heading_xy /= n
        perp_xy = np.array([-heading_xy[1], heading_xy[0], 0.0], dtype=np.float64)
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        R = np.column_stack((heading_xy, perp_xy, up))
        return midpoint_world.astype(np.float32), R.astype(np.float32)

    def target_in_wrist_frame(
        self,
        hand: str,
        target_body: np.ndarray,
        joint_pos: np.ndarray,
        root_pos: np.ndarray,
        root_quat_wxyz: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Return ``target_body`` re-expressed in the wrist's local frame.

        Useful for "is the target on the grasp plane?" checks: in the
        wrist's local coordinates, +x is the finger axis (forward), +y is
        the closing/spread direction, and the grasp plane is y=0. So
        |result[1]| is the perpendicular distance from the target to the
        grasp plane, and result[0] is the depth along the finger axis.
        Uses measured wrist joints (no zeroing).
        """
        if hand == "l":
            wrist_fid = self._left_wrist_fid
        elif hand == "r":
            wrist_fid = self._right_wrist_fid
        else:
            raise ValueError(f"hand must be 'l' or 'r', got {hand!r}")
        pelvis_tf = self._forward_kinematics(
            joint_pos, root_pos, root_quat_wxyz, zero_wrist=False,
        )
        wrist_tf = self._data.oMf[wrist_fid]
        R_world_wrist = np.asarray(wrist_tf.rotation, dtype=np.float64)
        t_world_wrist = np.asarray(wrist_tf.translation, dtype=np.float64)
        R_world_pelvis = np.asarray(pelvis_tf.rotation, dtype=np.float64)
        t_world_pelvis = np.asarray(pelvis_tf.translation, dtype=np.float64)
        target_world = (
            t_world_pelvis
            + R_world_pelvis @ np.asarray(target_body, dtype=np.float64).reshape(3)
        )
        target_wrist = R_world_wrist.T @ (target_world - t_world_wrist)
        return target_wrist.astype(np.float32)

    def gripper_pose_body(
        self,
        hand: str,
        joint_pos: np.ndarray,
        root_pos: np.ndarray,
        root_quat_wxyz: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(midpoint_b, R_wrist_b)`` for ``hand`` ('l' or 'r').

        ``midpoint_b`` is the pelvis-frame finger-bar midpoint (3,).
        ``R_wrist_b`` is the pelvis-frame rotation of ``*_wrist_yaw_link``
        such that ``pelvis_vec = R_wrist_b @ wrist_vec``. Useful when a
        caller needs to express a body-frame error in the wrist's own
        coordinates (e.g. asymmetric per-axis tolerances along the
        finger axis).
        """
        if hand == "l":
            wrist_fid = self._left_wrist_fid
            tool_point = self._left_tool_point_wrist
        elif hand == "r":
            wrist_fid = self._right_wrist_fid
            tool_point = self._right_tool_point_wrist
        else:
            raise ValueError(f"hand must be 'l' or 'r', got {hand!r}")
        pelvis_tf = self._forward_kinematics(
            joint_pos, root_pos, root_quat_wxyz, zero_wrist=False,
        )
        midpoint_b = self._tool_point_body(wrist_fid, tool_point, pelvis_tf)
        wrist_tf = self._data.oMf[wrist_fid]
        R_pw = np.asarray(pelvis_tf.rotation, dtype=np.float64)
        R_ww = np.asarray(wrist_tf.rotation, dtype=np.float64)
        R_wrist_b = R_pw.T @ R_ww
        return midpoint_b.astype(np.float32), R_wrist_b.astype(np.float32)

    def gripper_heading_xy(
        self,
        hand: str,
        joint_pos: np.ndarray,
        root_pos: np.ndarray,
        root_quat_wxyz: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Return the gripper's +x (finger) axis projected to the **world**
        horizontal plane and re-expressed in pelvis frame as a 3-vector;
        ``hand`` is "l" or "r". None if the world-horizontal projection
        is degenerate (finger axis nearly vertical in world).

        The leveller keeps the gripper level w.r.t. world (gravity), so
        the finger axis is approximately horizontal in world. Projecting
        in world (not pelvis) and rotating back into pelvis frame yields
        a body-frame direction whose z component compensates for any
        pelvis pitch — using only the xy components would silently drag
        a body-frame retreat partly downward when the pelvis is squatted
        forward. The returned vector is unit-length in world (its xy in
        world sum to one); its norm in pelvis frame is also one because
        pure rotation preserves length.

        Uses measured wrist joints (no zeroing) so the heading reflects
        the physical gripper, matching ``gripper_midpoint_body``.
        """
        pelvis_tf = self._forward_kinematics(
            joint_pos, root_pos, root_quat_wxyz, zero_wrist=False,
        )
        fid = self._left_wrist_fid if hand == "l" else self._right_wrist_fid
        R_world_wrist = np.asarray(self._data.oMf[fid].rotation, dtype=np.float64)
        R_world_pelvis = np.asarray(pelvis_tf.rotation, dtype=np.float64)
        finger_axis_world = R_world_wrist[:, 0]
        n = float(np.linalg.norm(finger_axis_world[:2]))
        if n < 1e-6:
            return None
        finger_world_horiz = np.array(
            [finger_axis_world[0] / n, finger_axis_world[1] / n, 0.0],
            dtype=np.float64,
        )
        return (R_world_pelvis.T @ finger_world_horiz).astype(np.float32)

    def compute(
        self,
        joint_pos: np.ndarray,
        root_pos: np.ndarray,
        root_quat_wxyz: np.ndarray,
        left_target_body: Optional[np.ndarray] = None,
        right_target_body: Optional[np.ndarray] = None,
        yaw_body_direction: Optional[np.ndarray] = None,
        left_yaw_body_direction: Optional[np.ndarray] = None,
        right_yaw_body_direction: Optional[np.ndarray] = None,
        left_orientation_mode: WristOrientationMode = "level",
        right_orientation_mode: WristOrientationMode = "level",
        dt: float = 0.02,
        active_left: Optional[bool] = None,
        active_right: Optional[bool] = None,
        freeze_yaw_left: bool = False,
        freeze_yaw_right: bool = False,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return (left_rpy, right_rpy) — each either a (3,) np.float32
        array, or None if the side is gated off or the gimbal guard tripped
        (caller should keep the policy's wrist command for that side).

        If a per-side target is supplied (pelvis-frame xyz of the hand
        command), the third wrist DOF (yaw) is solved so the gripper's
        finger axis points horizontally at that target. Otherwise, if
        ``yaw_body_direction`` is given (a pelvis-frame direction shared
        by both sides), yaw is solved so the finger axis aligns with
        that direction — useful for holding a fixed gripper heading
        relative to the torso. If neither is set, yaw stays at 0.

        ``active_left`` / ``active_right`` force per-side activeness,
        overriding the default (target-or-yaw-direction) derivation. Pass
        False to gate a side off explicitly — slew state is cleared and
        the next re-enable re-inits from the physical wrist. Per-side yaw
        directions let callers fall back independently when only one side
        has a raw target."""
        pelvis_tf = self._forward_kinematics(
            joint_pos, root_pos, root_quat_wxyz, zero_wrist=True,
        )
        jp = np.asarray(joint_pos, dtype=np.float64).reshape(-1)

        # Per-side active: explicit caller override (active_left/active_right)
        # wins; else derive from whether the side has a target or a shared
        # yaw direction.
        if active_left is not None:
            left_active = bool(active_left)
        else:
            left_active = (
                left_target_body is not None
                or left_yaw_body_direction is not None
                or yaw_body_direction is not None
            )
        if active_right is not None:
            right_active = bool(active_right)
        else:
            right_active = (
                right_target_body is not None
                or right_yaw_body_direction is not None
                or yaw_body_direction is not None
            )

        # Initialize per-side slew state from the *current* wrist angles
        # on the first tick after a (re-)enable, so the leveler's output
        # ramps gradually from wherever the physical wrist is to the
        # leveled target. Without this the first tick would snap to the
        # full target RPY, sending a large PD step into the forearm.
        if left_active and self._prev_left_rpy is None:
            self._prev_left_rpy = np.asarray(jp[list(self._left_policy_idx)], dtype=np.float64).copy()
        if right_active and self._prev_right_rpy is None:
            self._prev_right_rpy = np.asarray(jp[list(self._right_policy_idx)], dtype=np.float64).copy()
        # Fully-gated-off side: clear slew memory so the next re-enable
        # re-inits from the physical wrist via the block above. Don't run
        # the solve+slew path for inactive sides — _slew with prev=None
        # snaps and re-sets prev, defeating the reset.
        if not left_active:
            self._prev_left_rpy = None
            self._prev_yaw_ema_left = None
        if not right_active:
            self._prev_right_rpy = None
            self._prev_yaw_ema_right = None

        # Per-side solve + yaw override + clamp + slew. Inactive sides
        # short-circuit to None without touching slew state (preserved as
        # None by the reset above).
        left: Optional[np.ndarray] = None
        right: Optional[np.ndarray] = None

        if left_active:
            left = self._solve_with_guard(self._data.oMf[self._left_forearm_fid].rotation)
            if left is not None:
                left = self._apply_orientation_mode(left, left_orientation_mode)
                # Freeze takes precedence over any yaw target — pin yaw to
                # the previously emitted EMA value and keep slew memory in
                # sync, so the next non-frozen tick resumes from the same
                # yaw the robot has been holding. First-tick fallback:
                # _prev_yaw_ema_left isn't seeded yet, so use _prev_left_rpy
                # (which the just-ran reset block above seeded from the
                # measured wrist yaw) and lazy-init the EMA state too.
                hold_yaw_left = freeze_yaw_left
                if hold_yaw_left:
                    if self._prev_yaw_ema_left is None and self._prev_left_rpy is not None:
                        self._prev_yaw_ema_left = float(self._prev_left_rpy[2])
                    if self._prev_yaw_ema_left is None:
                        hold_yaw_left = False
                if hold_yaw_left:
                    left[2] = float(self._prev_yaw_ema_left)
                elif left_target_body is not None:
                    left = self._apply_yaw_toward_target(
                        left,
                        self._data.oMf[self._left_forearm_fid].rotation,
                        self._data.oMf[self._left_wrist_fid].translation,
                        pelvis_tf.translation,
                        pelvis_tf.rotation,
                        left_target_body,
                    )
                elif left_yaw_body_direction is not None:
                    left = self._apply_yaw_toward_body_direction(
                        left,
                        self._data.oMf[self._left_forearm_fid].rotation,
                        pelvis_tf.rotation,
                        left_yaw_body_direction,
                    )
                elif yaw_body_direction is not None:
                    left = self._apply_yaw_toward_body_direction(
                        left,
                        self._data.oMf[self._left_forearm_fid].rotation,
                        pelvis_tf.rotation,
                        yaw_body_direction,
                    )
                left = self._clamp(left, self._left_limits)
                left = self._slew("_prev_left_rpy", left, dt)
                if hold_yaw_left:
                    # Slew may have moved yaw if _prev_left_rpy[2] had drifted
                    # from _prev_yaw_ema_left. Pin both back to the held value
                    # so neither output nor next-tick slew reference moves.
                    left[2] = float(self._prev_yaw_ema_left)
                    if self._prev_left_rpy is not None:
                        self._prev_left_rpy[2] = float(self._prev_yaw_ema_left)
                else:
                    left[2] = self._ema_yaw(left[2], "_prev_yaw_ema_left")
            elif self._prev_left_rpy is not None:
                # Gimbal guard: hold the last-valid leveled command rather
                # than returning None (which makes the caller revert to the
                # policy's wrist and visibly step the commanded angle in
                # one tick). Slew state is preserved, so when the solve
                # recovers the ramp continues from here. Only if this is
                # the very first call and guard trips immediately do we
                # fall through to left=None.
                left = self._prev_left_rpy.copy()

        if right_active:
            right = self._solve_with_guard(self._data.oMf[self._right_forearm_fid].rotation)
            if right is not None:
                right = self._apply_orientation_mode(right, right_orientation_mode)
                hold_yaw_right = freeze_yaw_right
                if hold_yaw_right:
                    if self._prev_yaw_ema_right is None and self._prev_right_rpy is not None:
                        self._prev_yaw_ema_right = float(self._prev_right_rpy[2])
                    if self._prev_yaw_ema_right is None:
                        hold_yaw_right = False
                if hold_yaw_right:
                    right[2] = float(self._prev_yaw_ema_right)
                elif right_target_body is not None:
                    right = self._apply_yaw_toward_target(
                        right,
                        self._data.oMf[self._right_forearm_fid].rotation,
                        self._data.oMf[self._right_wrist_fid].translation,
                        pelvis_tf.translation,
                        pelvis_tf.rotation,
                        right_target_body,
                    )
                elif right_yaw_body_direction is not None:
                    right = self._apply_yaw_toward_body_direction(
                        right,
                        self._data.oMf[self._right_forearm_fid].rotation,
                        pelvis_tf.rotation,
                        right_yaw_body_direction,
                    )
                elif yaw_body_direction is not None:
                    right = self._apply_yaw_toward_body_direction(
                        right,
                        self._data.oMf[self._right_forearm_fid].rotation,
                        pelvis_tf.rotation,
                        yaw_body_direction,
                    )
                right = self._clamp(right, self._right_limits)
                right = self._slew("_prev_right_rpy", right, dt)
                if hold_yaw_right:
                    right[2] = float(self._prev_yaw_ema_right)
                    if self._prev_right_rpy is not None:
                        self._prev_right_rpy[2] = float(self._prev_yaw_ema_right)
                else:
                    right[2] = self._ema_yaw(right[2], "_prev_yaw_ema_right")
            elif self._prev_right_rpy is not None:
                right = self._prev_right_rpy.copy()

        return (
            left.astype(np.float32) if left is not None else None,
            right.astype(np.float32) if right is not None else None,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

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
            for idx in self._left_policy_idx + self._right_policy_idx:
                q[self._q_idx[idx]] = 0.0
        pin.forwardKinematics(self._model, self._data, q)
        pin.updateFramePlacements(self._model, self._data)
        return self._data.oMf[self._pelvis_fid]

    def _tool_point_body(
        self,
        wrist_frame_id: int,
        tool_point_wrist: np.ndarray,
        pelvis_tf,
    ) -> np.ndarray:
        wrist_tf = self._data.oMf[wrist_frame_id]
        tool_world = (
            np.asarray(wrist_tf.translation, dtype=np.float64)
            + np.asarray(wrist_tf.rotation, dtype=np.float64)
            @ np.asarray(tool_point_wrist, dtype=np.float64).reshape(3)
        )
        R_pw = np.asarray(pelvis_tf.rotation, dtype=np.float64)
        t_pw = np.asarray(pelvis_tf.translation, dtype=np.float64)
        return R_pw.T @ (tool_world - t_pw)

    def _build_q_index_map(self, policy_joint_names: Sequence[str]) -> np.ndarray:
        idx = []
        missing = []
        for name in policy_joint_names:
            if not self._model.existJointName(name):
                missing.append(name)
                continue
            joint = self._model.joints[self._model.getJointId(name)]
            if joint.nq <= 0:
                missing.append(name)
                continue
            idx.append(int(joint.idx_q))
        if missing:
            raise ValueError(f"URDF missing joints for WristLeveller: {missing}")
        return np.asarray(idx, dtype=np.int64)

    def _joint_limits(self, names: Sequence[str]) -> np.ndarray:
        """Return (3, 2) array of (lower, upper) limits for the wrist joints."""
        lo_up = np.zeros((len(names), 2), dtype=np.float64)
        for i, name in enumerate(names):
            jid = self._model.getJointId(name)
            joint = self._model.joints[jid]
            lo = float(self._model.lowerPositionLimit[joint.idx_q])
            up = float(self._model.upperPositionLimit[joint.idx_q])
            lo_up[i, 0] = lo
            lo_up[i, 1] = up
        return lo_up

    @staticmethod
    def _solve(R_forearm: np.ndarray) -> np.ndarray:
        """Analytic wrist roll/pitch that aligns gripper +z with world +z.

        Returns (roll, pitch, yaw); yaw is 0 (free DOF).
        """
        R = np.asarray(R_forearm, dtype=np.float64).reshape(3, 3)
        v = R.T @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        pitch = float(np.arcsin(np.clip(v[0], -1.0, 1.0)))
        cp = float(np.cos(pitch))
        if abs(cp) < 1e-6:
            roll = 0.0
        else:
            roll = float(np.arctan2(-v[1], v[2]))
        return np.array([roll, pitch, 0.0], dtype=np.float64)

    def _apply_orientation_mode(
        self,
        rpy: np.ndarray,
        orientation_mode: WristOrientationMode,
    ) -> np.ndarray:
        out = np.asarray(rpy, dtype=np.float64).copy()
        if orientation_mode == "level":
            return out
        if orientation_mode == "vertical":
            out[0] += self._vertical_roll_rad
            return out
        raise ValueError(f"Unsupported wrist orientation mode: {orientation_mode}")

    @staticmethod
    def _apply_yaw_toward_body_direction(
        rpy: np.ndarray,
        R_forearm: np.ndarray,
        R_pelvis_world: np.ndarray,
        body_dir: np.ndarray,
    ) -> np.ndarray:
        """Return rpy with yaw replaced by the angle that aligns the
        finger axis (wrist_yaw_link +x) with ``body_dir`` expressed in
        pelvis frame, projected to the horizontal plane.

        Unlike _apply_yaw_toward_target, the desired direction is fixed
        in the pelvis frame — it doesn't depend on wrist position — so
        the gripper holds its heading relative to the torso regardless
        of where the arm is.
        """
        roll, pitch = float(rpy[0]), float(rpy[1])
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        Rx = np.array([[1.0, 0.0, 0.0],
                       [0.0, cr, -sr],
                       [0.0, sr,  cr]])
        Ry = np.array([[ cp, 0.0, sp],
                       [0.0, 1.0, 0.0],
                       [-sp, 0.0, cp]])
        R_wrist_rp = np.asarray(R_forearm, dtype=np.float64) @ Rx @ Ry
        cur = R_wrist_rp[:, 0]
        cur_xy = np.array([cur[0], cur[1]], dtype=np.float64)
        cur_norm = float(np.linalg.norm(cur_xy))
        if cur_norm < 1e-6:
            return rpy
        cur_xy /= cur_norm

        des_world = (
            np.asarray(R_pelvis_world, dtype=np.float64).reshape(3, 3)
            @ np.asarray(body_dir, dtype=np.float64).reshape(3)
        )
        des_xy = np.array([des_world[0], des_world[1]], dtype=np.float64)
        des_norm = float(np.linalg.norm(des_xy))
        if des_norm < 1e-6:
            return rpy
        des_xy /= des_norm

        cross_z = cur_xy[0] * des_xy[1] - cur_xy[1] * des_xy[0]
        dot = cur_xy[0] * des_xy[0] + cur_xy[1] * des_xy[1]
        yaw = math.atan2(float(cross_z), float(dot))
        return np.array([roll, pitch, yaw], dtype=np.float64)

    @staticmethod
    def _apply_yaw_toward_target(
        rpy: np.ndarray,
        R_forearm: np.ndarray,
        wrist_world_pos: np.ndarray,
        pelvis_world_pos: np.ndarray,
        R_pelvis_world: np.ndarray,
        target_body: np.ndarray,
    ) -> np.ndarray:
        """Return rpy with the yaw component replaced by the angle that
        points wrist_yaw_link's +x axis (finger axis) at target_body
        projected to the horizontal plane.

        After the roll+pitch solve has made gripper_up coincide with
        world +z, the wrist_yaw joint rotates around world +z, i.e. it
        turns the finger axis in the horizontal plane. The needed angle
        is the signed horizontal-plane angle between the current finger
        axis and the direction from wrist to target."""
        roll, pitch = float(rpy[0]), float(rpy[1])
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        Rx = np.array([[1.0, 0.0, 0.0],
                       [0.0, cr, -sr],
                       [0.0, sr,  cr]])
        Ry = np.array([[ cp, 0.0, sp],
                       [0.0, 1.0, 0.0],
                       [-sp, 0.0, cp]])
        R_wrist_rp = np.asarray(R_forearm, dtype=np.float64) @ Rx @ Ry
        # Finger axis at yaw=0 (first column = R @ [1, 0, 0]).
        cur = R_wrist_rp[:, 0]
        cur_xy = np.array([cur[0], cur[1]], dtype=np.float64)
        cur_norm = float(np.linalg.norm(cur_xy))
        if cur_norm < 1e-6:
            return rpy
        cur_xy /= cur_norm

        # Target in world.
        tgt_body = np.asarray(target_body, dtype=np.float64).reshape(3)
        tgt_world = np.asarray(pelvis_world_pos, dtype=np.float64).reshape(3) + \
            np.asarray(R_pelvis_world, dtype=np.float64).reshape(3, 3) @ tgt_body
        delta = tgt_world - np.asarray(wrist_world_pos, dtype=np.float64).reshape(3)
        des_xy = np.array([delta[0], delta[1]], dtype=np.float64)
        des_norm = float(np.linalg.norm(des_xy))
        if des_norm < 1e-6:
            return rpy
        des_xy /= des_norm

        # Signed angle from cur_xy to des_xy, around world +z.
        cross_z = cur_xy[0] * des_xy[1] - cur_xy[1] * des_xy[0]
        dot = cur_xy[0] * des_xy[0] + cur_xy[1] * des_xy[1]
        yaw = math.atan2(float(cross_z), float(dot))
        return np.array([roll, pitch, yaw], dtype=np.float64)

    def _solve_with_guard(self, R_forearm: np.ndarray) -> Optional[np.ndarray]:
        """Like _solve, but returns None near the atan2 roll singularity
        (|v[0]| ≈ 1 → pitch near ±π/2, roll ill-defined)."""
        R = np.asarray(R_forearm, dtype=np.float64).reshape(3, 3)
        v = R.T @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(v[0]) > self._gimbal_v0_thresh:
            return None
        pitch = float(np.arcsin(np.clip(v[0], -1.0, 1.0)))
        roll = float(np.arctan2(-v[1], v[2]))
        return np.array([roll, pitch, 0.0], dtype=np.float64)

    def _ema_yaw(self, yaw: float, attr: str) -> float:
        prev = getattr(self, attr)
        if prev is None:
            setattr(self, attr, yaw)
            return yaw
        new = prev + self._yaw_ema_alpha * (yaw - prev)
        setattr(self, attr, new)
        return new

    def _slew(self, attr: str, target: np.ndarray, dt: float) -> np.ndarray:
        prev = getattr(self, attr)
        if prev is None:
            # First valid call since (re-)enable: snap to target and
            # let the PD controller track — the slew-limit applies to
            # *subsequent* changes.
            setattr(self, attr, target.copy())
            return target
        step_max = self._max_rate * max(float(dt), 1e-6)
        delta = np.clip(target - prev, -step_max, step_max)
        new = prev + delta
        setattr(self, attr, new.copy())
        return new

    @staticmethod
    def _clamp(rpy: np.ndarray, limits: np.ndarray) -> np.ndarray:
        return np.clip(rpy, limits[:, 0], limits[:, 1])
