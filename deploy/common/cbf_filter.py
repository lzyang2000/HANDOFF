"""Capture-point safety filter for standing balance.

Two modes:
  - ``"centering"`` (default): proportional controller that nudges the capture
    point toward the support-polygon centroid.  Always-on when standing, gentle.
  - ``"cbf"``: Control Barrier Function that constrains joint velocities to keep
    the capture point inside the support polygon.

Typical usage inside a policy node::

    filt = G1CapturePointCBF(estimator, mode="centering")
    # ... in _process_step, after computing target_pos:
    target_pos, info = filt.filter(
        root_pos, root_quat, root_lin_vel, root_ang_vel,
        joint_pos, joint_vel,
        proposed_target=target_pos,
        last_target=self._last_target,
        velocity_command_norm=vel_norm,
    )
    self._last_target = target_pos.copy()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from deploy.common.capture_point import (
    G1CapturePointEstimator,
    CapturePointDebugData,
    POLICY_JOINT_NAMES,
    signed_distance_gradient,
)
from wbc_mjlab.g1_constants_custom import (
    CBF_ACTIVATION_WIDTH_VEL,
    CBF_ALPHA_DEPLOY,
    CBF_H_TARGET,
    CBF_VEL_MAX_ANKLE_DEPLOY,
    CBF_VEL_MAX_WAIST_DEPLOY,
)

# Adjustable joint indices in the POLICY_JOINT_NAMES order.
#   4  = left_ankle_pitch    5  = left_ankle_roll
#  10  = right_ankle_pitch  11  = right_ankle_roll
#  12  = waist_yaw          13  = waist_roll          14 = waist_pitch
_ANKLE_INDICES = np.array([4, 5, 10, 11], dtype=np.intp)
_WAIST_INDICES = np.array([14], dtype=np.intp)  # waist_pitch only
_ADJUSTABLE_INDICES = np.concatenate([_ANKLE_INDICES, _WAIST_INDICES])

_NUM_POLICY_JOINTS = len(POLICY_JOINT_NAMES)

# Boolean mask over all policy joints, True for non-adjustable joints.
_NON_ADJ_MASK = ~np.isin(np.arange(_NUM_POLICY_JOINTS), _ADJUSTABLE_INDICES)

GRAVITY = 9.81


def _sigmoid(x: float) -> float:
    if x > 20.0:
        return 1.0
    if x < -20.0:
        return 0.0
    return 1.0 / (1.0 + np.exp(-x))


@dataclass(slots=True)
class CBFInfo:
    """Diagnostics returned alongside the filtered target."""

    active: bool = False
    activation: float = 0.0
    h: float = float("nan")
    h_dot_drift: float = 0.0
    constraint_violation: float = 0.0
    delta_target_norm: float = 0.0
    cp_error_xy: np.ndarray | None = None
    capture_point: CapturePointDebugData | None = None


class G1CapturePointCBF:
    """Safety filter with two modes: proportional centering or CBF.

    Adjustable joints: ankle pitch/roll (4 DOF) + waist yaw/roll/pitch (3 DOF).
    Uses the contact-consistent COM Jacobian (feet fixed on ground).
    """

    def __init__(
        self,
        estimator: G1CapturePointEstimator,
        *,
        mode: str = "centering",
        alpha: float = CBF_ALPHA_DEPLOY,
        h_target: float = CBF_H_TARGET,
        kp: float = 10.0,
        activation_vel_threshold: float = 0.1,
        dt: float = 0.02,
        vel_max_ankle: float = CBF_VEL_MAX_ANKLE_DEPLOY,
        vel_max_waist: float = CBF_VEL_MAX_WAIST_DEPLOY,
        activation_width_vel: float = CBF_ACTIVATION_WIDTH_VEL,
    ):
        if mode not in ("centering", "cbf", "none"):
            raise ValueError(f"Unknown mode {mode!r}, expected 'centering', 'cbf', or 'none'")
        self._mode = mode
        self._estimator = estimator
        self._alpha = float(alpha)
        self._h_target = float(h_target)
        self._kp = float(kp)
        self._vel_threshold = float(activation_vel_threshold)
        self._dt = float(dt)
        self._activation_width_vel = float(activation_width_vel)

        # Ordering matches _ADJUSTABLE_INDICES = concat([_ANKLE_INDICES, _WAIST_INDICES]).
        self._vel_limits = np.concatenate([
            np.full(len(_ANKLE_INDICES), vel_max_ankle),
            np.full(len(_WAIST_INDICES), vel_max_waist),
        ])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter(
        self,
        root_pos_w: Sequence[float],
        root_quat_wxyz: Sequence[float],
        root_lin_vel_w: Sequence[float],
        root_ang_vel_w: Sequence[float],
        joint_pos: Sequence[float],
        joint_vel: Sequence[float],
        proposed_target: np.ndarray,
        last_target: np.ndarray,
        velocity_command_norm: float = 0.0,
    ) -> tuple[np.ndarray, CBFInfo]:
        """Return a (possibly modified) target and diagnostic info."""
        info = CBFInfo()
        filtered = np.array(proposed_target, dtype=np.float64)

        if self._mode == "none":
            return filtered, info

        # Override the caller's root_lin_vel_w with a kinematic estimate
        # under the "both feet planted" constraint. The contact-consistent
        # COM Jacobian below already commits to this assumption, so deriving
        # the base velocity the same way makes the filter self-consistent
        # — and lets hardware run without a ZED-derived base velocity
        # (hand_policy passes zeros from hardware_node, which would
        # otherwise leave the capture-point estimate purely static).
        # root_pos_w cancels in all relative quantities so the caller's
        # value (real in sim, zero on hardware) is fine to keep.
        root_lin_vel_w = self._estimator.compute_root_lin_vel_from_contacts(
            root_quat_wxyz, root_ang_vel_w, joint_pos, joint_vel,
            root_pos_w=root_pos_w,
        )

        # 1. Evaluate capture point & support polygon.
        cp_data = self._estimator.evaluate(
            root_pos_w, root_quat_wxyz,
            root_lin_vel_w, root_ang_vel_w,
            joint_pos, joint_vel,
        )
        info.capture_point = cp_data
        h = cp_data.h_now
        info.h = h

        if not np.isfinite(h):
            return filtered, info

        # Standing gate (smooth deactivation when walking).
        activation = _sigmoid(
            (self._vel_threshold - velocity_command_norm) / self._activation_width_vel
        )
        info.activation = activation
        if activation < 1e-4:
            return filtered, info

        # 2. Contact-consistent COM Jacobian → capture-point Jacobian.
        J_cc = self._estimator.compute_contact_consistent_com_jacobian()  # (3, n_joints)

        com_pos = np.asarray(self._estimator.pin_data.com[0], dtype=np.float64)
        com_height = max(float(com_pos[2] - cp_data.ground_z), 0.1)
        omega0 = np.sqrt(GRAVITY / com_height)
        scale = self._dt + 1.0 / omega0

        J_cp_full = J_cc[:2, :] * scale
        J_cp = J_cp_full[:, _ADJUSTABLE_INDICES]

        # 3. Get usable polygon.
        polygon = cp_data.shrunk_polygon_xy
        if polygon.shape[0] < 3:
            polygon = cp_data.raw_polygon_xy
        if polygon.shape[0] < 3:
            return filtered, info

        # 4. Dispatch to mode.
        if self._mode == "centering":
            return self._filter_centering(filtered, J_cp, cp_data, polygon, activation, info)
        else:
            return self._filter_cbf(filtered, J_cp, J_cp_full, cp_data, polygon, activation, last_target, info)

    # ------------------------------------------------------------------
    # Centering mode
    # ------------------------------------------------------------------

    def _filter_centering(
        self,
        filtered: np.ndarray,
        J_cp: np.ndarray,
        cp_data: CapturePointDebugData,
        polygon: np.ndarray,
        activation: float,
        info: CBFInfo,
    ) -> tuple[np.ndarray, CBFInfo]:
        """Proportional controller: push capture point toward polygon centroid."""
        centroid = polygon.mean(axis=0)  # (2,)
        error = cp_data.capture_point_xy - centroid  # (2,)

        v_cp_desired = -self._kp * error  # (2,)
        dq = np.linalg.pinv(J_cp) @ v_cp_desired  # (7,)
        dq = np.clip(dq, -self._vel_limits, self._vel_limits)

        filtered[_ADJUSTABLE_INDICES] += activation * dq * self._dt

        info.active = True
        info.cp_error_xy = error.copy()
        info.constraint_violation = float(np.linalg.norm(error))
        info.delta_target_norm = float(np.linalg.norm(dq * self._dt))
        return filtered, info

    # ------------------------------------------------------------------
    # CBF mode
    # ------------------------------------------------------------------

    def _filter_cbf(
        self,
        filtered: np.ndarray,
        J_cp: np.ndarray,
        J_cp_full: np.ndarray,
        cp_data: CapturePointDebugData,
        polygon: np.ndarray,
        activation: float,
        last_target: np.ndarray,
        info: CBFInfo,
    ) -> tuple[np.ndarray, CBFInfo]:
        """Standalone CBF with shifted barrier.

        Uses ``h_eff = h - h_target`` so the CBF actively pushes the capture
        point inward when it is closer than *h_target* to the polygon edge,
        even when the system is at rest.  This makes the CBF behave like a
        continuous centering controller rather than a boundary-only safety net.
        """
        h = cp_data.h_now
        h_eff = h - self._h_target  # shifted barrier — negative when too close

        last_arr = np.asarray(last_target, dtype=np.float64)
        v_all = (filtered - last_arr) / self._dt
        v_proposed = v_all[_ADJUSTABLE_INDICES]

        nabla_h = signed_distance_gradient(cp_data.capture_point_xy, polygon)
        a = nabla_h @ J_cp
        a_norm_sq = float(a @ a)

        if a_norm_sq < 1e-16:
            return filtered, info

        h_dot_drift = float(nabla_h @ J_cp_full[:, _NON_ADJ_MASK] @ v_all[_NON_ADJ_MASK])
        info.h_dot_drift = h_dot_drift

        # CBF constraint: a @ v_safe + h_dot_drift + alpha * h_eff >= 0
        # When h < h_target, h_eff < 0 and alpha*h_eff is negative,
        # so the constraint demands the adjustable joints push inward.
        margin = float(a @ v_proposed) + h_dot_drift + self._alpha * h_eff
        info.constraint_violation = max(0.0, -margin)

        if margin >= 0.0:
            return filtered, info

        # Project onto safe half-space.
        info.active = True
        lam = -margin / (a_norm_sq + 1e-12)
        dq = v_proposed + lam * a
        dq = np.clip(dq, -self._vel_limits, self._vel_limits)

        margin_after = float(a @ dq) + h_dot_drift + self._alpha * h_eff
        if margin_after < -1e-6:
            dq = self._solve_with_box_constraints(
                a, h_dot_drift + self._alpha * h_eff, v_proposed,
            )

        # Apply: dq is the safe velocity, v_proposed is the original.
        # The delta to add to the target is (dq - v_proposed) * dt.
        delta = (dq - v_proposed) * self._dt
        filtered[_ADJUSTABLE_INDICES] += activation * delta
        info.delta_target_norm = float(np.linalg.norm(delta))
        return filtered, info

    def _solve_with_box_constraints(
        self,
        a: np.ndarray,
        rhs: float,
        v_proposed: np.ndarray,
    ) -> np.ndarray:
        """Greedy projection with box constraints (single linear constraint)."""
        v = v_proposed.copy()
        free = np.ones(len(v), dtype=bool)

        for _ in range(len(v)):
            a_free = a[free]
            a_free_sq = float(a_free @ a_free)
            if a_free_sq < 1e-16:
                break

            margin = float(a @ v) + rhs
            if margin >= -1e-8:
                break

            lam = -margin / (a_free_sq + 1e-12)
            v[free] = v[free] + lam * a[free]

            violations = np.abs(v) > self._vel_limits
            if not np.any(violations & free):
                break
            v = np.clip(v, -self._vel_limits, self._vel_limits)
            free &= ~violations

        return np.clip(v, -self._vel_limits, self._vel_limits)
