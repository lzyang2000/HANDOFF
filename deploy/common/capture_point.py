from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pinocchio as pin

try:
    from std_msgs.msg import Float32MultiArray
except ImportError:
    Float32MultiArray = None  # ROS not available (e.g. offline enrichment)


from wbc_mjlab.g1_constants_custom import CBF_SAFETY_MARGIN

GRAVITY = 9.81
DEFAULT_SAFETY_MARGIN = CBF_SAFETY_MARGIN
DEFAULT_MIN_COM_HEIGHT = 0.1

# The same joint ordering used by the deployed hand policy and sim telemetry.
POLICY_JOINT_NAMES = [
    "left_hip_pitch_joint",  "left_hip_roll_joint",  "left_hip_yaw_joint",
    "left_knee_joint",       "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint",      "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint",       "waist_roll_joint",     "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint",      "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint",     "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# Default G1 foot contact sphere positions relative to the ankle_roll_link frame.
# Kept identical to HANDOFF's support-polygon utilities.
G1_FOOT_CONTACTS_LOCAL = np.array([
    [-0.05,  0.025, -0.03],
    [-0.05, -0.025, -0.03],
    [ 0.12,  0.03,  -0.03],
    [ 0.12, -0.03,  -0.03],
], dtype=np.float64)

_DEFAULT_URDF_PATH = Path(__file__).resolve().parents[1] / "assets" / "g1_29dof_rev_1_0.urdf"


@dataclass(slots=True)
class CapturePointDebugData:
    """Diagnostics that the hand node publishes and the sim node renders."""

    step_id: int
    h_now: float
    back_margin: float
    back_x: float
    capture_point_xy: np.ndarray
    ground_z: float
    raw_polygon_xy: np.ndarray
    shrunk_polygon_xy: np.ndarray


class G1CapturePointEstimator:
    """Pinocchio-backed capture-point and support-polygon estimator for G1."""

    def __init__(
        self,
        urdf_path: str | Path = _DEFAULT_URDF_PATH,
        policy_joint_names: Sequence[str] = POLICY_JOINT_NAMES,
        safety_margin: float = DEFAULT_SAFETY_MARGIN,
        gravity: float = GRAVITY,
        min_com_height: float = DEFAULT_MIN_COM_HEIGHT,
    ):
        self._urdf_path = Path(urdf_path)
        if not self._urdf_path.exists():
            raise FileNotFoundError(f"G1 URDF not found: {self._urdf_path}")

        self._policy_joint_names = tuple(policy_joint_names)
        self._safety_margin = float(safety_margin)
        self._gravity = float(gravity)
        self._min_com_height = float(min_com_height)

        self._pin_model = pin.buildModelFromUrdf(str(self._urdf_path), pin.JointModelFreeFlyer())
        self._pin_data = self._pin_model.createData()
        self._pin_q = pin.neutral(self._pin_model)
        self._pin_v = np.zeros(self._pin_model.nv, dtype=np.float64)

        self._left_foot_frame_id = self._get_required_frame_id("left_ankle_roll_link")
        self._right_foot_frame_id = self._get_required_frame_id("right_ankle_roll_link")

        self._pin_q_indices, self._pin_v_indices = self._build_joint_index_maps(self._policy_joint_names)

    @property
    def urdf_path(self) -> Path:
        return self._urdf_path

    @property
    def pin_model(self):
        return self._pin_model

    @property
    def pin_data(self):
        return self._pin_data

    def evaluate(
        self,
        root_pos_w: Sequence[float],
        root_quat_wxyz: Sequence[float],
        root_lin_vel_w: Sequence[float],
        root_ang_vel_w: Sequence[float],
        joint_pos: Sequence[float],
        joint_vel: Sequence[float],
        step_id: int = 0,
    ) -> CapturePointDebugData:
        """Compute capture point, support polygon, and derived diagnostics."""
        root_pos_w = self._as_vec(root_pos_w, 3, "root_pos_w")
        root_quat_wxyz = self._as_vec(root_quat_wxyz, 4, "root_quat_wxyz")
        root_lin_vel_w = self._as_vec(root_lin_vel_w, 3, "root_lin_vel_w")
        root_ang_vel_w = self._as_vec(root_ang_vel_w, 3, "root_ang_vel_w")
        joint_pos = self._as_vec(joint_pos, len(self._policy_joint_names), "joint_pos")
        joint_vel = self._as_vec(joint_vel, len(self._policy_joint_names), "joint_vel")

        self._pin_q[:] = pin.neutral(self._pin_model)
        self._pin_q[0:3] = root_pos_w
        # Pinocchio free-flyer quaternion layout: [x, y, z, w].
        self._pin_q[3] = root_quat_wxyz[1]
        self._pin_q[4] = root_quat_wxyz[2]
        self._pin_q[5] = root_quat_wxyz[3]
        self._pin_q[6] = root_quat_wxyz[0]
        self._pin_q[self._pin_q_indices] = joint_pos

        self._pin_v[:] = 0.0
        # Keep the same convention used in HANDOFF's reference implementation:
        # translational velocity first, then angular velocity.
        self._pin_v[0:3] = root_lin_vel_w
        self._pin_v[3:6] = root_ang_vel_w
        self._pin_v[self._pin_v_indices] = joint_vel

        pin.forwardKinematics(self._pin_model, self._pin_data, self._pin_q, self._pin_v)
        pin.updateFramePlacements(self._pin_model, self._pin_data)
        pin.centerOfMass(self._pin_model, self._pin_data, self._pin_q, self._pin_v)

        com_pos = np.asarray(self._pin_data.com[0], dtype=np.float64)
        com_vel = np.asarray(self._pin_data.vcom[0], dtype=np.float64)

        foot_L = self._pin_data.oMf[self._left_foot_frame_id]
        foot_R = self._pin_data.oMf[self._right_foot_frame_id]

        contacts_L = foot_contacts_to_world(
            foot_L.translation,
            foot_L.rotation,
            G1_FOOT_CONTACTS_LOCAL,
        )
        contacts_R = foot_contacts_to_world(
            foot_R.translation,
            foot_R.rotation,
            G1_FOOT_CONTACTS_LOCAL,
        )
        all_contacts = np.vstack([contacts_L, contacts_R])
        ground_z = float(np.min(all_contacts[:, 2]))

        raw_polygon_xy = compute_support_polygon(
            foot_L.translation,
            foot_L.rotation,
            foot_R.translation,
            foot_R.rotation,
        )
        shrunk_polygon_xy = shrink_polygon(raw_polygon_xy, self._safety_margin)

        capture_point_xy = self._compute_capture_point_xy(com_pos, com_vel, ground_z)

        if shrunk_polygon_xy.shape[0] >= 3:
            h_now = float(signed_distance_point_to_polygon(capture_point_xy, shrunk_polygon_xy))
            back_x = float(np.min(shrunk_polygon_xy[:, 0]))
            back_margin = float(capture_point_xy[0] - back_x)
        elif raw_polygon_xy.shape[0] >= 3:
            # Fallback for degenerate shrink results.
            h_now = float(signed_distance_point_to_polygon(capture_point_xy, raw_polygon_xy))
            back_x = float(np.min(raw_polygon_xy[:, 0]))
            back_margin = float(capture_point_xy[0] - back_x)
        else:
            h_now = float("-inf")
            back_x = float("nan")
            back_margin = float("nan")

        return CapturePointDebugData(
            step_id=int(step_id),
            h_now=h_now,
            back_margin=back_margin,
            back_x=back_x,
            capture_point_xy=capture_point_xy,
            ground_z=ground_z,
            raw_polygon_xy=raw_polygon_xy,
            shrunk_polygon_xy=shrunk_polygon_xy,
        )

    def compute_contact_consistent_com_jacobian(self) -> np.ndarray:
        """Contact-consistent COM Jacobian for double support (3 x n_joints).

        When both feet are on the ground, the free-floating COM Jacobian
        underestimates ankle authority because it treats the base (pelvis) as
        free.  This method imposes the constraint that both foot velocities are
        zero, solving for the implied base velocity via a stacked pseudo-inverse
        and returning the corrected Jacobian.

        Must be called AFTER :meth:`evaluate` (which populates ``_pin_q`` and
        runs forward kinematics).  Column *i* corresponds to
        ``POLICY_JOINT_NAMES[i]``.
        """
        model, data, q = self._pin_model, self._pin_data, self._pin_q

        pin.computeJointJacobians(model, data, q)
        J_com = pin.jacobianCenterOfMass(model, data, q)  # (3, nv)

        J_L = pin.getFrameJacobian(
            model, data, self._left_foot_frame_id, pin.LOCAL_WORLD_ALIGNED,
        )  # (6, nv)
        J_R = pin.getFrameJacobian(
            model, data, self._right_foot_frame_id, pin.LOCAL_WORLD_ALIGNED,
        )  # (6, nv)

        # Stacked foot constraint: (12, nv)
        J_stack = np.vstack([J_L, J_R])
        J_base_pinv = np.linalg.pinv(J_stack[:, :6])  # (6, 12)

        # J_cc: (3, n_joints) — maps joint velocities to COM velocity
        # with the constraint that both feet remain stationary.
        return J_com[:, 6:] - J_com[:, :6] @ J_base_pinv @ J_stack[:, 6:]

    def compute_root_lin_vel_from_contacts(
        self,
        root_quat_wxyz: Sequence[float],
        root_ang_vel_w: Sequence[float],
        joint_pos: Sequence[float],
        joint_vel: Sequence[float],
        root_pos_w: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        """Solve for the floating-base linear velocity in world under the
        constraint that both ankle_roll frames have zero velocity in world.

        Inputs are the same telemetry the policy already consumes (IMU root
        quaternion + body angular velocity, joint pos/vel). Output is a
        3-vector ``v_root_lin_w`` consistent with double-support kinematics.
        Used by ``G1CapturePointCBF.filter`` so the safety filter no longer
        depends on a base-velocity estimator (ZED on hardware, MuJoCo cvel
        in sim) — instead it shares the "feet planted" assumption already
        baked into ``compute_contact_consistent_com_jacobian``.

        ``root_pos_w`` is only used to populate the free-flyer base for
        FK; it cancels out of the foot-velocity constraint. ``np.zeros(3)``
        is fine.
        """
        if root_pos_w is None:
            root_pos_w = np.zeros(3, dtype=np.float64)
        root_pos_w = self._as_vec(root_pos_w, 3, "root_pos_w")
        root_quat_wxyz = self._as_vec(root_quat_wxyz, 4, "root_quat_wxyz")
        root_ang_vel_w = self._as_vec(root_ang_vel_w, 3, "root_ang_vel_w")
        joint_pos = self._as_vec(joint_pos, len(self._policy_joint_names), "joint_pos")
        joint_vel = self._as_vec(joint_vel, len(self._policy_joint_names), "joint_vel")

        # Build pin_q exactly as evaluate() does so the FK matches.
        self._pin_q[:] = pin.neutral(self._pin_model)
        self._pin_q[0:3] = root_pos_w
        self._pin_q[3] = root_quat_wxyz[1]
        self._pin_q[4] = root_quat_wxyz[2]
        self._pin_q[5] = root_quat_wxyz[3]
        self._pin_q[6] = root_quat_wxyz[0]
        self._pin_q[self._pin_q_indices] = joint_pos

        pin.forwardKinematics(self._pin_model, self._pin_data, self._pin_q)
        pin.computeJointJacobians(self._pin_model, self._pin_data, self._pin_q)
        pin.updateFramePlacements(self._pin_model, self._pin_data)

        J_L = pin.getFrameJacobian(
            self._pin_model, self._pin_data, self._left_foot_frame_id,
            pin.LOCAL_WORLD_ALIGNED,
        )  # (6, nv)
        J_R = pin.getFrameJacobian(
            self._pin_model, self._pin_data, self._right_foot_frame_id,
            pin.LOCAL_WORLD_ALIGNED,
        )  # (6, nv)
        J_stack = np.vstack([J_L, J_R])  # (12, nv)

        # pin_v layout matches evaluate(): [v_lin_w(3), v_ang_w(3), q...].
        # Foot constraint in world: J_stack @ pin_v = 0 →
        #   J_lin @ v_lin + J_ang @ ang_vel + J_q @ qd = 0
        # Solve for v_lin in least-squares sense across both feet (12 eqs,
        # 3 unknowns — well-conditioned whenever the feet are spatially
        # separated, which is any feasible standing pose).
        J_lin = J_stack[:, 0:3]
        J_ang = J_stack[:, 3:6]
        J_q = J_stack[:, self._pin_v_indices]
        rhs = J_ang @ root_ang_vel_w + J_q @ joint_vel
        v_lin_w = -np.linalg.pinv(J_lin) @ rhs
        return np.asarray(v_lin_w, dtype=np.float64).reshape(3)

    def _compute_capture_point_xy(
        self,
        com_pos: np.ndarray,
        com_vel: np.ndarray,
        ground_z: float,
    ) -> np.ndarray:
        com_height = max(float(com_pos[2] - ground_z), self._min_com_height)
        omega0 = np.sqrt(self._gravity / com_height)
        return np.asarray(com_pos[:2], dtype=np.float64) + np.asarray(com_vel[:2], dtype=np.float64) / omega0

    def _build_joint_index_maps(self, policy_joint_names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        q_indices: list[int] = []
        v_indices: list[int] = []
        missing: list[str] = []

        for name in policy_joint_names:
            if not self._pin_model.existJointName(name):
                missing.append(name)
                continue
            joint_id = self._pin_model.getJointId(name)
            joint = self._pin_model.joints[joint_id]
            if joint.nq <= 0 or joint.nv <= 0:
                missing.append(name)
                continue
            q_indices.append(int(joint.idx_q))
            v_indices.append(int(joint.idx_v))

        if missing:
            raise ValueError(
                "Pinocchio URDF is missing required policy joints: "
                + ", ".join(missing)
            )
        if len(q_indices) != len(policy_joint_names) or len(v_indices) != len(policy_joint_names):
            raise ValueError(
                "Failed to build complete Pinocchio joint index maps for the G1 policy joints."
            )

        return np.asarray(q_indices, dtype=np.int32), np.asarray(v_indices, dtype=np.int32)

    def _get_required_frame_id(self, frame_name: str) -> int:
        frame_id = self._pin_model.getFrameId(frame_name)
        if frame_id >= len(self._pin_model.frames):
            raise ValueError(f"Pinocchio frame not found: {frame_name}")
        return int(frame_id)

    @staticmethod
    def _as_vec(values: Sequence[float], expected_len: int, name: str) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if arr.size != expected_len:
            raise ValueError(f"{name} has length {arr.size}, expected {expected_len}")
        return arr



def pack_capture_point_debug(data: CapturePointDebugData) -> Float32MultiArray:
    """Pack a debug snapshot into a step-stamped Float32MultiArray."""
    raw_xy = np.asarray(data.raw_polygon_xy, dtype=np.float32).reshape(-1)
    shrunk_xy = np.asarray(data.shrunk_polygon_xy, dtype=np.float32).reshape(-1)
    header = np.array([
        float(data.step_id),
        float(data.h_now),
        float(data.back_margin),
        float(data.back_x),
        float(data.capture_point_xy[0]),
        float(data.capture_point_xy[1]),
        float(data.ground_z),
        float(len(raw_xy) // 2),
        float(len(shrunk_xy) // 2),
    ], dtype=np.float32)
    payload = np.concatenate([header, raw_xy, shrunk_xy]).astype(np.float32, copy=False)
    msg = Float32MultiArray()
    msg.data = payload.tolist()
    return msg


@dataclass(slots=True)
class CapturePointOverlay:
    """Decoded overlay payload used by the sim viewer."""

    step_id: int
    h_now: float
    back_margin: float
    back_x: float
    capture_point_xy: np.ndarray
    ground_z: float
    raw_polygon_xy: np.ndarray
    shrunk_polygon_xy: np.ndarray



def unpack_capture_point_debug(payload: Sequence[float] | Float32MultiArray) -> CapturePointOverlay:
    """Decode a Float32MultiArray payload into a structured overlay."""
    if isinstance(payload, Float32MultiArray):
        values = np.asarray(payload.data, dtype=np.float64).reshape(-1)
    else:
        values = np.asarray(payload, dtype=np.float64).reshape(-1)

    if values.size < 9:
        raise ValueError(f"Capture-point debug payload is too short: {values.size}")

    step_id = int(round(float(values[0])))
    h_now = float(values[1])
    back_margin = float(values[2])
    back_x = float(values[3])
    capture_point_xy = np.asarray(values[4:6], dtype=np.float64)
    ground_z = float(values[6])
    raw_count = max(0, int(round(float(values[7]))))
    shrunk_count = max(0, int(round(float(values[8]))))

    cursor = 9
    raw_flat_len = 2 * raw_count
    raw_xy = values[cursor:cursor + raw_flat_len]
    if raw_xy.size != raw_flat_len:
        raise ValueError(
            f"Capture-point payload truncated while reading raw polygon: expected {raw_flat_len} values, got {raw_xy.size}"
        )
    cursor += raw_flat_len

    shrunk_flat_len = 2 * shrunk_count
    shrunk_xy = values[cursor:cursor + shrunk_flat_len]
    if shrunk_xy.size != shrunk_flat_len:
        raise ValueError(
            f"Capture-point payload truncated while reading shrunk polygon: expected {shrunk_flat_len} values, got {shrunk_xy.size}"
        )

    raw_polygon_xy = raw_xy.reshape(raw_count, 2).astype(np.float64, copy=False) if raw_count > 0 else np.empty((0, 2), dtype=np.float64)
    shrunk_polygon_xy = shrunk_xy.reshape(shrunk_count, 2).astype(np.float64, copy=False) if shrunk_count > 0 else np.empty((0, 2), dtype=np.float64)

    return CapturePointOverlay(
        step_id=step_id,
        h_now=h_now,
        back_margin=back_margin,
        back_x=back_x,
        capture_point_xy=capture_point_xy,
        ground_z=ground_z,
        raw_polygon_xy=raw_polygon_xy,
        shrunk_polygon_xy=shrunk_polygon_xy,
    )


# ---------------------------------------------------------------------------
# Support polygon computation, signed distance, and shrink utilities
# ---------------------------------------------------------------------------

def foot_contacts_to_world(
    foot_pose_translation: np.ndarray,
    foot_pose_rotation: np.ndarray,
    contacts_local: np.ndarray = G1_FOOT_CONTACTS_LOCAL,
) -> np.ndarray:
    """Transform local foot contact points to world frame."""
    return (foot_pose_rotation @ contacts_local.T).T + foot_pose_translation



def compute_support_polygon(
    foot_L_translation: np.ndarray,
    foot_L_rotation: np.ndarray,
    foot_R_translation: np.ndarray,
    foot_R_rotation: np.ndarray,
    contacts_local: np.ndarray = G1_FOOT_CONTACTS_LOCAL,
) -> np.ndarray:
    """Compute the 2D support polygon from both feet."""
    pts_L = foot_contacts_to_world(foot_L_translation, foot_L_rotation, contacts_local)
    pts_R = foot_contacts_to_world(foot_R_translation, foot_R_rotation, contacts_local)
    all_xy = np.vstack([pts_L[:, :2], pts_R[:, :2]])
    return _convex_hull_2d(all_xy)



def shrink_polygon(vertices: np.ndarray, margin: float) -> np.ndarray:
    """Shrink a convex polygon inward by a uniform margin."""
    if margin <= 0.0:
        return np.asarray(vertices, dtype=vertices.dtype).copy()

    vertices = np.asarray(vertices, dtype=np.float64)
    n = len(vertices)
    if n < 3:
        return np.empty((0, 2), dtype=vertices.dtype)

    offset_edges = []
    for i in range(n):
        p0 = vertices[i]
        p1 = vertices[(i + 1) % n]
        edge = p1 - p0
        normal = np.array([-edge[1], edge[0]], dtype=np.float64)
        length = np.linalg.norm(normal)
        if length < 1e-12:
            continue
        normal = normal / length
        offset_edges.append((p0 + margin * normal, p1 + margin * normal))

    if len(offset_edges) < 3:
        return np.empty((0, 2), dtype=vertices.dtype)

    new_verts = []
    m = len(offset_edges)
    for i in range(m):
        a0, a1 = offset_edges[i]
        b0, b1 = offset_edges[(i + 1) % m]
        pt = _line_line_intersection(a0, a1, b0, b1)
        if pt is not None:
            new_verts.append(pt)

    if len(new_verts) < 3:
        return np.empty((0, 2), dtype=vertices.dtype)

    return np.array(new_verts, dtype=vertices.dtype)



def signed_distance_point_to_polygon(
    point: np.ndarray,
    vertices: np.ndarray,
) -> float:
    """Signed distance from a 2D point to a convex polygon."""
    vertices = np.asarray(vertices, dtype=np.float64)
    n = len(vertices)
    if n < 3:
        return float("-inf")

    point = np.asarray(point, dtype=np.float64)
    min_signed = np.inf
    for i in range(n):
        p0 = vertices[i]
        p1 = vertices[(i + 1) % n]
        edge = p1 - p0
        normal = np.array([-edge[1], edge[0]], dtype=np.float64)
        length = np.linalg.norm(normal)
        if length < 1e-12:
            continue
        normal = normal / length
        d = float(np.dot(normal, point - p0))
        if d < min_signed:
            min_signed = d

    return float(min_signed)



def signed_distance_gradient(
    point: np.ndarray,
    vertices: np.ndarray,
) -> np.ndarray:
    """Gradient of the signed distance w.r.t. the query point."""
    vertices = np.asarray(vertices, dtype=np.float64)
    n = len(vertices)
    if n < 3:
        return np.zeros(2, dtype=np.float64)

    point = np.asarray(point, dtype=np.float64)
    min_signed = np.inf
    best_normal = np.zeros(2, dtype=np.float64)
    for i in range(n):
        p0 = vertices[i]
        p1 = vertices[(i + 1) % n]
        edge = p1 - p0
        normal = np.array([-edge[1], edge[0]], dtype=np.float64)
        length = np.linalg.norm(normal)
        if length < 1e-12:
            continue
        normal = normal / length
        d = float(np.dot(normal, point - p0))
        if d < min_signed:
            min_signed = d
            best_normal = normal

    return best_normal


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """Compute the 2D convex hull using Andrew's monotone chain algorithm."""
    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        return np.empty((0, 2), dtype=np.float64)

    pts = points[np.lexsort((points[:, 1], points[:, 0]))]
    if len(pts) <= 1:
        return pts.copy()

    lower = []
    for p in pts:
        while len(lower) >= 2 and _cross2d(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross2d(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = np.array(lower[:-1] + upper[:-1], dtype=np.float64)
    return hull



def _cross2d(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))



def _line_line_intersection(
    a0: np.ndarray, a1: np.ndarray,
    b0: np.ndarray, b1: np.ndarray,
) -> Optional[np.ndarray]:
    """Intersection of two lines defined by point pairs (a0,a1) and (b0,b1)."""
    da = a1 - a0
    db = b1 - b0
    denom = da[0] * db[1] - da[1] * db[0]
    if abs(denom) < 1e-12:
        return None
    t = ((b0[0] - a0[0]) * db[1] - (b0[1] - a0[1]) * db[0]) / denom
    return a0 + t * da
