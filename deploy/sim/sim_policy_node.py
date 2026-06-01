"""G1 sim+policy combined node — physics and policy in one synchronized process.

Physics runs at 1000 Hz (20× MuJoCo steps per 50 Hz control tick).
The ONNX policy runs in the same thread immediately after each physics step,
eliminating UDP round-trip jitter and the 2-second inter-process startup wait.

ROS 2 is used for controller commands, wrist-leveller gating, and telemetry.
Usage: sim_policy_node.py <path/to/model.onnx>
"""

import argparse
import math
import os
import re
import signal
import threading
import time
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from typing import Optional, Tuple

os.environ.setdefault("WBC_ATTACH_PAYLOADS", "1")
os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")  # suppress ORT warnings (e.g. GPU device discovery on Jetson)

import numpy as np
import mujoco
import mujoco.viewer
import onnxruntime as ort

import rclpy
from rclpy.executors import SingleThreadedExecutor as _SingleThreadedExecutor
from rclpy.node import Node
from rclpy.node import Node as _RclpyNode
from rclpy.qos import (
    DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy,
    DurabilityPolicy as _DurabilityPolicy,
    HistoryPolicy as _HistoryPolicy,
    QoSProfile as _QoSProfile,
    ReliabilityPolicy as _ReliabilityPolicy,
)
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from nav_msgs.msg import Odometry, Path as PathMsg
from sensor_msgs.msg import CameraInfo, Image, JointState, PointCloud2, PointField
from std_msgs.msg import (
    Bool as BoolMsg,
    Float32 as Float32Msg,
    Float32MultiArray,
    Float32MultiArray as _Float32MultiArray,
    Empty,
    String as StringMsg,
    String as _StringMsg,
)
from visualization_msgs.msg import MarkerArray

from deploy.real.g1_robot_constants import (
    HEAD_CAMERA_SIM_FOVY_DEG,
    HEAD_CAMERA_SIM_RENDER_H,
    HEAD_CAMERA_SIM_RENDER_W,
)
from deploy.common.capture_point import (
    CapturePointOverlay,
    G1CapturePointEstimator,
    unpack_capture_point_debug,
)
from deploy.common.cbf_filter import G1CapturePointCBF
from deploy.common.command import (
    CMD_LEFT_GRIPPER, CMD_RIGHT_GRIPPER, CMD_SIZE, COMMAND_TOPIC, make_command,
)
from deploy.common.hand_command_pd import HandCommandPD
from deploy.common.molmo_local_frame import (
    LOCAL_FRAME_ID,
    LocalFrameAnchor,
    local_points_to_world,
    local_xy_to_world_xy,
    make_identity_local_frame_anchor,
    make_local_frame_anchor,
)
from deploy.common.molmo_wrist_phase import (
    MolmoActivePhase as _MolmoActivePhase,
    is_bimanual_active as _is_bimanual_active,
    is_single_pick_leveller_allowed as _is_single_pick_leveller_allowed,
    is_single_pick_retract as _is_single_pick_retract,
    is_single_pick_walkback as _is_single_pick_walkback,
    is_single_pick_yaw_freeze as _is_single_pick_yaw_freeze,
    is_single_place_leveller_active as _is_single_place_leveller_active,
    parse_molmo_active_phase as _parse_molmo_active_phase,
    should_clear_single_pick_walkback_hold as _should_clear_single_pick_walkback_hold,
    update_side_holding as _update_side_holding,
)
from deploy.common.wrist_level import WristLeveller

import mjlab.asset_zoo.robots.unitree_g1.g1_constants as g1_constants
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import KNEES_BENT_KEYFRAME
from mjlab.asset_zoo.robots import get_g1_robot_cfg
from mjlab.entity import Entity

from wbc_mjlab.g1_constants_custom import (
    CAPTURE_POINT_PD_ENABLE,
    CAPTURE_POINT_METHOD,
    CAPTURE_POINT_PD_KP,
    SQUAT_SMOOTHING_HEIGHT,
    FFS_OBB_POSE_TOPIC, FFS_OBB_EXTENT_TOPIC, FFS_MASK_TOPIC,
    MOLMO_VISOR_OVERLAY_TOPIC,
    GRIPPER_OPEN_CMD, GRIPPER_CLOSED_CMD, GRIPPER_PD_KP, GRIPPER_DAMPING_RATIO,
    MOLMO_ACTIVE_PHASE_TOPIC,
    MOLMO_WRIST_VERTICAL_ROLL_RAD,
    MOLMO_DEFAULT_PROMPT, MOLMO_DEFAULT_PROMPT_ENV,
    MOLMO_GRIPPER_MIDPOINT_OFFSET_LEFT,
    MOLMO_GRIPPER_MIDPOINT_OFFSET_RIGHT,
    MOLMO_HAND_CMD_PD_DEADBAND_M,
    MOLMO_HAND_CMD_PD_ENABLE,
    MOLMO_HAND_CMD_PD_KD,
    MOLMO_HAND_CMD_PD_KD_X_MULT,
    MOLMO_HAND_CMD_PD_KD_Y_MULT,
    MOLMO_HAND_CMD_PD_KP,
    MOLMO_HAND_CMD_PD_KP_X_MULT,
    MOLMO_HAND_CMD_PD_KP_Y_MULT,
    MOLMO_HAND_CMD_PD_MAX_SHAPE_M,
    MOLMO_RAW_HAND_TARGET_TOPIC,
    MOLMO_SQUAT_EMA_ALPHA,
    MOLMO_WRIST_LEVEL_OVERRIDE,
    MOLMO_WRIST_LEVEL_ROLL_ONLY,
    MOLMO_WRIST_LEVEL_YAW_EMA_ALPHA,
    MOLMO_WRIST_LEVEL_YAW_TO_TORSO,
    MOLMO_PICK_EARLY_CLOSE_X_BACK_M,
    MOLMO_PICK_EARLY_CLOSE_X_FWD_M,
    MOLMO_PICK_EARLY_CLOSE_Y_M,
    MOLMO_PICK_EARLY_CLOSE_Z_M,
    MOLMO_PICK_EARLY_CLOSE_Z_SQUAT_M,
    MOLMO_WRIST_YAW_FREEZE_RADIUS_M,
    MOLMO_WRIST_YAW_OUTPUT_EMA_ALPHA,
    MOLMO_WRIST_ROLL_AFTER_PICK_ENABLE,
    MOLMO_WRIST_ROLL_AFTER_PICK_RAD,
    MOLMO_WRIST_ZERO_OBS,
    MOLMO_FALL_RECOVERY_PHASE,
    SIM_PERTURB_BODY,
    SIM_PERTURB_T0,
    SIM_PERTURB_DURATION,
    SIM_PERTURB_FORCE,
    SIM_PERTURB_TORQUE,
    wrap_spec_fn_with_payloads,
)

# ---------------------------------------------------------------------------
# Environment variable names (from sim_node.py)
# ---------------------------------------------------------------------------
HAND_SIM_G1_XML_ENV = "WBC_MJLAB_G1_XML"
GRASPNET_OBJECTS_ENV = "GRASPNET_OBJECTS"  # "000,005" | "random:N" | "none"
GRASPNET_SEED_ENV = "GRASPNET_SEED"
MOLMO_CAMERA_NAMES_ENV = "WBC_MJLAB_MOLMO_CAMERA_NAMES"
MOLMO_RENDER_HZ_ENV = "WBC_MJLAB_MOLMO_RENDER_HZ"
MOLMO_DEFAULT_CAMERA_NAMES = ("head_camera",)
MOLMO_DEFAULT_RENDER_HZ = 10.0

VISER_PORT_ENV = "WBC_MJLAB_VISER_PORT"
VISER_URDF_ENV = "WBC_MJLAB_VISER_URDF"
VISER_HZ_ENV = "WBC_MJLAB_VISER_HZ"
VISER_IMAGE_HZ_ENV = "WBC_MJLAB_VISER_IMAGE_HZ"
MUJOCO_VIEWER_ENV = "WBC_MJLAB_MUJOCO_VIEWER"

VISER_DEFAULT_PORT = "8080"
VISER_DEFAULT_HZ = 30.0
VISER_DEFAULT_IMAGE_HZ = 15.0
VISER_DEFAULT_URDF = "g1_29dof_rev_1_0_with_payloads_and_gripper.urdf"
VISER_GRIPPER_JOINT_NAMES = (
    "gripper_prismatic_1_L",
    "gripper_prismatic_2_L",
    "gripper_prismatic_1_R",
    "gripper_prismatic_2_R",
)

# ---------------------------------------------------------------------------
# Sim constants
# ---------------------------------------------------------------------------
POLICY_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

DECIMATION = 20  # 1000 Hz physics / 50 Hz control
NUM_JOINTS = 29

# ---------------------------------------------------------------------------
# Policy / observation constants (from hand_policy.py)
# ---------------------------------------------------------------------------
_ANKLE_DOF_INDICES = {4, 5, 10, 11}
BASE_ANG_VEL_SCALE = 0.25
JOINT_POS_SCALE = 1.0
JOINT_VEL_SCALE = 0.05

HAND_STUDENT_MIMIC_DIM = 14
ACTOR_PROPRIO_DIM = 92
ACTOR_CURRENT_DIM = HAND_STUDENT_MIMIC_DIM + ACTOR_PROPRIO_DIM
ACTOR_HISTORY_LENGTH = 11
ACTOR_OBS_DIM = ACTOR_CURRENT_DIM * (1 + ACTOR_HISTORY_LENGTH)

LOCO_GAIT_OFFSET = 0.5
STAND_VEL_THRESHOLD = 0.1

CMD_VX = 0
CMD_VY = 1
CMD_YAW_RATE = 2
CMD_HEIGHT = 4
CMD_LEFT_HAND = 5
CMD_RIGHT_HAND = 8

_WRIST_LEVEL_Z_THRESHOLD: float = 0.05
_WRIST_LEVEL_GRIPPER_OPEN_MAX: float = 0.5
_WRIST_LEVEL_EMA_ALPHA: float = 0.05
_WRIST_LEVEL_BLEND_EPS: float = 1e-3
_PELVIS_FORWARD = np.array([1.0, 0.0, 0.0], dtype=np.float64)

# ---------------------------------------------------------------------------
# QoS profiles (from sim_node.py)
# ---------------------------------------------------------------------------
# RELIABLE so rviz (which defaults to RELIABLE) can subscribe. BEST_EFFORT
# publishers are silently dropped by RELIABLE subscribers.
VIZ_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST, depth=5,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)

ANCHOR_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST, depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _resolve_keyframe(joint_names, keyframe):
    vals = np.zeros(len(joint_names), dtype=np.float32)
    for i, name in enumerate(joint_names):
        for pattern, value in keyframe.joint_pos.items():
            if re.fullmatch(pattern, name):
                vals[i] = value
                break
    return vals


def _resolve_scales(joint_names, scale_dict):
    scales = np.zeros(len(joint_names), dtype=np.float32)
    for i, name in enumerate(joint_names):
        for pattern, scale in scale_dict.items():
            if re.match(pattern, name):
                scales[i] = scale
                break
    return scales


# 29-element versions (hand_policy.py form)
DEFAULT_POS = _resolve_keyframe(POLICY_JOINT_NAMES, KNEES_BENT_KEYFRAME)
JOINT_SCALES = _resolve_scales(POLICY_JOINT_NAMES, g1_constants.G1_ACTION_SCALE)
JOINT_VEL_SCALES = np.array(
    [0.0 if i in _ANKLE_DOF_INDICES else JOINT_VEL_SCALE for i in range(NUM_JOINTS)],
    dtype=np.float32,
)


def quat_rotate_inverse(q, v):
    w, x, y, z = q
    q_vec = np.array([x, y, z])
    a = v * (2.0 * w * w - 1.0)
    b = np.cross(q_vec, v) * w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c


def quat_rotate(q, v):
    """Rotate vector ``v`` by quaternion ``q`` (wxyz). Inverse of quat_rotate_inverse."""
    w, x, y, z = q
    q_vec = np.array([x, y, z], dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    a = v * (2.0 * w * w - 1.0)
    b = np.cross(q_vec, v) * w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a + b + c


def _mat_to_quat_wxyz(mat: np.ndarray) -> np.ndarray:
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, np.asarray(mat, dtype=np.float64).reshape(9))
    return quat


def _quat_xyzw_to_rotmat(quat_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = [float(v) for v in np.asarray(quat_xyzw, dtype=np.float64).reshape(4)]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def euler_roll_pitch_from_quat(q):
    w, x, y, z = q
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    return np.array([roll, pitch], dtype=np.float32)


def _phase_features(phase_tracker, cmd_3d):
    if np.linalg.norm(cmd_3d) < STAND_VEL_THRESHOLD:
        return np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    left_phase = phase_tracker % 1.0
    right_phase = (phase_tracker + LOCO_GAIT_OFFSET) % 1.0
    two_pi = 2.0 * math.pi
    return np.array([
        math.sin(two_pi * left_phase), math.cos(two_pi * left_phase),
        math.sin(two_pi * right_phase), math.cos(two_pi * right_phase),
    ], dtype=np.float32)


def build_actor_current(joint_pos, joint_vel, root_quat, body_ang_vel,
                        cmd, last_action, phase_tracker):
    vx_ref = cmd[CMD_VX]
    vy_ref = cmd[CMD_VY]
    yaw_ref = cmd[CMD_YAW_RATE]
    z_ref = cmd[CMD_HEIGHT]
    left_hand_b = cmd[CMD_LEFT_HAND:CMD_LEFT_HAND + 3]
    right_hand_b = cmd[CMD_RIGHT_HAND:CMD_RIGHT_HAND + 3]
    cmd_3d = np.array([vx_ref, vy_ref, yaw_ref], dtype=np.float32)
    pf = _phase_features(phase_tracker, cmd_3d)
    mimic = np.concatenate([
        [vx_ref, vy_ref], [z_ref], [yaw_ref],
        left_hand_b, right_hand_b, pf,
    ], dtype=np.float32)

    base_ang_vel = body_ang_vel * BASE_ANG_VEL_SCALE
    imu_rp = euler_roll_pitch_from_quat(root_quat)
    joint_pos_rel = (joint_pos - DEFAULT_POS) * JOINT_POS_SCALE
    joint_vel_rel = joint_vel * JOINT_VEL_SCALES
    proprio = np.concatenate([
        base_ang_vel, imu_rp, joint_pos_rel, joint_vel_rel, last_action,
    ], dtype=np.float32)

    return np.concatenate([mimic, proprio], dtype=np.float32)


# ---------------------------------------------------------------------------
# _MolmoSubscriber (adapted from hand_policy.py)
# ---------------------------------------------------------------------------
class _MolmoSubscriber:
    """Background rclpy node that caches Molmo phase + raw hand targets.

    Adapted from hand_policy.py: node name changed to avoid collision with
    G1SimRosNode, and shutdown() does not call rclpy.shutdown() (main() owns
    the rclpy lifecycle in this combined process).
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._phase: Optional[str] = None
        self._raw_left: Optional[np.ndarray] = None
        self._raw_right: Optional[np.ndarray] = None
        # Per-side world-up extent of the active pick OBB (m), 0 if unknown.
        self._pick_obj_z_left: float = 0.0
        self._pick_obj_z_right: float = 0.0
        # True iff the active step is a squat pick — mirrors the molmo
        # node's branch so the policy-side z fallback matches early-close.
        self._is_squat_pick: bool = False
        self._node = None
        self._thread: Optional[threading.Thread] = None
        try:
            if not rclpy.ok():
                rclpy.init()
            self._node = _RclpyNode("sim_policy_molmo_sub")
            phase_qos = _QoSProfile(
                history=_HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=_ReliabilityPolicy.BEST_EFFORT,
                durability=_DurabilityPolicy.VOLATILE,
            )
            self._node.create_subscription(
                _StringMsg, MOLMO_ACTIVE_PHASE_TOPIC, self._phase_cb, phase_qos,
            )
            self._node.create_subscription(
                _Float32MultiArray, MOLMO_RAW_HAND_TARGET_TOPIC, self._raw_cb, phase_qos,
            )
            self._executor = _SingleThreadedExecutor()
            self._executor.add_node(self._node)
            self._thread = threading.Thread(
                target=self._executor.spin, daemon=True,
            )
            self._thread.start()
            print(
                f"[sim_policy_node] subscribed to {MOLMO_ACTIVE_PHASE_TOPIC} "
                f"and {MOLMO_RAW_HAND_TARGET_TOPIC}"
            )
        except Exception as e:
            print(f"[sim_policy_node] Molmo subscriber disabled: {e}")
            self._node = None

    def _phase_cb(self, msg):
        with self._lock:
            self._phase = str(msg.data) if msg.data is not None else "idle"

    def _raw_cb(self, msg):
        vals = np.asarray(msg.data, dtype=np.float32).reshape(-1)
        with self._lock:
            if vals.size < 8:
                self._raw_left = None
                self._raw_right = None
                self._pick_obj_z_left = 0.0
                self._pick_obj_z_right = 0.0
                self._is_squat_pick = False
                return
            self._raw_left = vals[1:4].copy() if float(vals[0]) > 0.5 else None
            self._raw_right = vals[5:8].copy() if float(vals[4]) > 0.5 else None
            self._pick_obj_z_left = float(vals[8]) if vals.size >= 9 else 0.0
            self._pick_obj_z_right = float(vals[9]) if vals.size >= 10 else 0.0
            self._is_squat_pick = (
                float(vals[10]) > 0.5 if vals.size >= 11 else False
            )

    def leveller_allowed_for(self, side: str) -> bool:
        with self._lock:
            phase = self._phase
        if phase is None:
            return True
        return _is_single_pick_leveller_allowed(
            _parse_molmo_active_phase(phase),
            side,
        )

    def has_phase(self) -> bool:
        with self._lock:
            return self._phase is not None

    def is_fall_recovery(self) -> bool:
        """True iff molmo_node is currently in fall-recovery mode.

        Mirrors hand_policy._MolmoSubscriber.is_fall_recovery — the
        single-token phase string disables the wrist-leveller override
        path automatically (leveller_allowed_for returns False because
        parse_molmo_active_phase returns None), but the obs-zeroing
        block consults this predicate directly.
        """
        with self._lock:
            return self._phase == MOLMO_FALL_RECOVERY_PHASE

    def current_phase(self) -> Optional[_MolmoActivePhase]:
        with self._lock:
            phase = self._phase
        return _parse_molmo_active_phase(phase)

    def raw_target_for(self, side: str) -> Optional[np.ndarray]:
        with self._lock:
            raw = self._raw_left if side == "l" else self._raw_right
            return None if raw is None else raw.copy()

    def pick_object_z_extent_for(self, side: str) -> float:
        with self._lock:
            return self._pick_obj_z_left if side == "l" else self._pick_obj_z_right

    def is_squat_pick_for(self, side: str) -> bool:  # noqa: ARG002
        """Mirrors hand_policy._MolmoSubscriber.is_squat_pick_for. Per-step
        flag (single-hand picks only have one active side at a time)."""
        with self._lock:
            return self._is_squat_pick

    def is_pick_walkback_for(self, side: str) -> bool:
        with self._lock:
            phase = self._phase
        return _is_single_pick_walkback(_parse_molmo_active_phase(phase), side)

    def is_pick_retract_for(self, side: str) -> bool:
        with self._lock:
            phase = self._phase
        return _is_single_pick_retract(_parse_molmo_active_phase(phase), side)

    def is_pick_yaw_freeze_for(self, side: str) -> bool:
        """True during pick:grasp — leveller should hold its last yaw."""
        with self._lock:
            phase = self._phase
        return _is_single_pick_yaw_freeze(_parse_molmo_active_phase(phase), side)

    def is_place_active_for(self, side: str) -> bool:
        with self._lock:
            phase = self._phase
        return _is_single_place_leveller_active(_parse_molmo_active_phase(phase), side)

    def should_clear_walkback_hold_for(self, side: str) -> bool:
        with self._lock:
            phase = self._phase
        if phase is None or phase == "idle":
            return False
        return _should_clear_single_pick_walkback_hold(
            _parse_molmo_active_phase(phase),
            side,
        )

    def shutdown(self):
        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:
                pass
        # Do NOT call rclpy.shutdown() — main() owns the rclpy lifecycle.


# ---------------------------------------------------------------------------
# ROS 2 node (verbatim from sim_node.py)
# ---------------------------------------------------------------------------
class G1SimRosNode(Node):
    def __init__(self):
        super().__init__("g1_sim_node")
        self.command_sub = self.create_subscription(
            Float32MultiArray, COMMAND_TOPIC, self._command_cb, VIZ_QOS
        )
        self.waist_yaw_sub = self.create_subscription(
            Float32Msg, "/molmo/waist_yaw_cmd", self._waist_yaw_cb, VIZ_QOS
        )
        self.capture_point_debug_sub = self.create_subscription(
            Float32MultiArray, "/g1/capture_point_debug", self._capture_point_debug_cb, VIZ_QOS
        )
        self.molmo_marker_sub = self.create_subscription(
            Float32MultiArray, "/molmo/target_markers", self._molmo_marker_cb, VIZ_QOS
        )
        self.create_subscription(
            PoseStamped, FFS_OBB_POSE_TOPIC, self._ffs_obb_pose_cb, VIZ_QOS
        )
        self.create_subscription(
            Vector3Stamped, FFS_OBB_EXTENT_TOPIC, self._ffs_obb_extent_cb, VIZ_QOS
        )
        self.create_subscription(PoseStamped, "/planner/local_anchor", self._local_anchor_cb, ANCHOR_QOS)
        self.create_subscription(PathMsg, "/planner/loco_path", self._loco_path_cb, VIZ_QOS)
        self.create_subscription(Float32MultiArray, "/planner/ee_path", self._ee_path_cb, VIZ_QOS)
        self.create_subscription(MarkerArray, "/planner/esdf_markers", self._esdf_markers_cb, VIZ_QOS)
        self.create_subscription(StringMsg, "/planner/status", self._planner_status_cb, VIZ_QOS)
        self.odom_pub = self.create_publisher(Odometry, "/g1/odom", VIZ_QOS)
        self.joint_state_pub = self.create_publisher(JointState, "/g1/joint_states", VIZ_QOS)
        self.reset_pub = self.create_publisher(Empty, "/g1/reset", VIZ_QOS)

        self._ui_prompt_pub = self.create_publisher(StringMsg, "/molmo/ui/prompt_submit", VIZ_QOS)
        self._ui_voice_toggle_pub = self.create_publisher(Empty, "/molmo/ui/voice_toggle", VIZ_QOS)
        self._ui_reset_pub = self.create_publisher(Empty, "/molmo/ui/reset", VIZ_QOS)

        self.create_subscription(
            Float32MultiArray, "/molmo/ui/overlay_points", self._ui_overlay_points_cb, VIZ_QOS
        )
        self.create_subscription(
            StringMsg, "/molmo/ui/status_text", self._ui_status_cb, VIZ_QOS
        )
        self.create_subscription(
            BoolMsg, "/molmo/ui/voice_state", self._ui_voice_state_cb, VIZ_QOS
        )
        self.create_subscription(
            StringMsg, "/molmo/ui/prompt_text", self._ui_prompt_text_cb, VIZ_QOS
        )
        self.create_subscription(
            StringMsg, "/molmo/ui/plan", self._ui_plan_cb, VIZ_QOS
        )

        self._lock = threading.Lock()
        self._latest_command = make_command()
        self._gripper_cmd = np.array([GRIPPER_OPEN_CMD, GRIPPER_OPEN_CMD], dtype=np.float32)
        self._waist_yaw_override: float = 0.0
        self._capture_point_overlay: CapturePointOverlay | None = None
        self._capture_point_step_id = -1
        self._capture_point_warned = False
        self._planner_local_anchor: LocalFrameAnchor = make_identity_local_frame_anchor()
        self._molmo_markers: list[tuple[int, np.ndarray]] = []
        self._latest_loco_path: np.ndarray | None = None
        self._latest_ee_left: np.ndarray | None = None
        self._latest_ee_right: np.ndarray | None = None
        self._latest_esdf_voxels: np.ndarray | None = None
        self._latest_esdf_voxel_m: float = 0.05
        self._latest_planner_status: str = "idle"
        # FFS OBB halves matched by header.stamp. Pelvis frame; gets cleared
        # once stale so the viser box doesn't stick to the robot forever.
        self._ffs_obb_pending_pose: PoseStamped | None = None
        self._ffs_obb_pending_extent: Vector3Stamped | None = None
        self._latest_ffs_obb: tuple[np.ndarray, np.ndarray, np.ndarray, float] | None = None

        self._ui_overlay_points: list[tuple[int, int]] = []
        self._ui_status_text = "Ready"
        self._ui_voice_recording = False
        self._last_prompt = ""
        self._viser_bridge = None

    def _command_cb(self, msg):
        data = np.array(msg.data, dtype=np.float32)
        if len(data) != CMD_SIZE:
            return
        with self._lock:
            self._latest_command = data.copy()
            self._gripper_cmd = np.array(
                [float(data[CMD_LEFT_GRIPPER]), float(data[CMD_RIGHT_GRIPPER])],
                dtype=np.float32,
            )

    def _waist_yaw_cb(self, msg: Float32Msg):
        with self._lock:
            self._waist_yaw_override = float(msg.data)

    def _capture_point_debug_cb(self, msg):
        try:
            overlay = unpack_capture_point_debug(msg)
        except Exception:
            return
        with self._lock:
            if overlay.step_id < self._capture_point_step_id:
                return
            self._capture_point_overlay = overlay
            self._capture_point_step_id = overlay.step_id

    def _molmo_marker_cb(self, msg):
        data = list(msg.data)
        markers: list[tuple[int, np.ndarray]] = []
        i = 0
        while i + 3 < len(data):
            markers.append((int(data[i]), np.array(data[i+1:i+4], dtype=np.float32)))
            i += 4
        with self._lock:
            self._molmo_markers = markers

    def _ffs_obb_pose_cb(self, msg: PoseStamped) -> None:
        with self._lock:
            self._ffs_obb_pending_pose = msg
            self._match_ffs_obb_locked()

    def _ffs_obb_extent_cb(self, msg: Vector3Stamped) -> None:
        with self._lock:
            self._ffs_obb_pending_extent = msg
            self._match_ffs_obb_locked()

    def _match_ffs_obb_locked(self) -> None:
        pose = self._ffs_obb_pending_pose
        ext = self._ffs_obb_pending_extent
        if pose is None or ext is None:
            return
        ps = float(pose.header.stamp.sec) + float(pose.header.stamp.nanosec) * 1e-9
        es = float(ext.header.stamp.sec) + float(ext.header.stamp.nanosec) * 1e-9
        if abs(ps - es) > 1e-3:
            if ps < es:
                self._ffs_obb_pending_pose = None
            else:
                self._ffs_obb_pending_extent = None
            return
        p = pose.pose
        center = np.array([p.position.x, p.position.y, p.position.z], dtype=np.float32)
        wxyz = np.array(
            [p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z],
            dtype=np.float32,
        )
        extent = np.array([ext.vector.x, ext.vector.y, ext.vector.z], dtype=np.float32)
        self._latest_ffs_obb = (center, wxyz, extent, ps)
        self._ffs_obb_pending_pose = None
        self._ffs_obb_pending_extent = None

    def _local_anchor_cb(self, msg: PoseStamped):
        anchor = make_local_frame_anchor(
            np.array(
                [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
                dtype=np.float64,
            ),
            _quat_xyzw_to_rotmat(
                np.array(
                    [
                        msg.pose.orientation.x,
                        msg.pose.orientation.y,
                        msg.pose.orientation.z,
                        msg.pose.orientation.w,
                    ],
                    dtype=np.float64,
                )
            ),
        )
        with self._lock:
            self._planner_local_anchor = anchor

    def _loco_path_cb(self, msg: PathMsg):
        pts = np.array(
            [[p.pose.position.x, p.pose.position.y] for p in msg.poses],
            dtype=np.float32,
        ).reshape(-1, 2) if msg.poses else np.zeros((0, 2), dtype=np.float32)
        if pts.shape[0] > 0 and str(msg.header.frame_id) == LOCAL_FRAME_ID:
            with self._lock:
                anchor = self._planner_local_anchor
            pts = local_xy_to_world_xy(pts, anchor).astype(np.float32)
        with self._lock:
            self._latest_loco_path = pts if pts.shape[0] > 0 else None

    def _ee_path_cb(self, msg: Float32MultiArray):
        data = np.asarray(msg.data, dtype=np.float32)
        if data.size == 0 or data.size % 4 != 0:
            with self._lock:
                self._latest_ee_left = None
                self._latest_ee_right = None
            return
        rows = data.reshape(-1, 4)
        left = rows[rows[:, 0] < 0.5][:, 1:4]
        right = rows[rows[:, 0] >= 0.5][:, 1:4]
        with self._lock:
            self._latest_ee_left = left.astype(np.float32) if left.size > 0 else None
            self._latest_ee_right = right.astype(np.float32) if right.size > 0 else None

    def _planner_status_cb(self, msg: StringMsg):
        text = str(msg.data or "").strip() or "idle"
        with self._lock:
            self._latest_planner_status = text
        if self._viser_bridge is not None:
            try:
                self._viser_bridge.set_planner_status(text)
            except Exception:
                pass

    def _esdf_markers_cb(self, msg: MarkerArray):
        if not msg.markers:
            with self._lock:
                self._latest_esdf_voxels = None
            return
        marker = msg.markers[0]
        pts = np.array(
            [[pt.x, pt.y, pt.z] for pt in marker.points],
            dtype=np.float32,
        ).reshape(-1, 3) if marker.points else np.zeros((0, 3), dtype=np.float32)
        if pts.shape[0] > 0 and str(marker.header.frame_id) == LOCAL_FRAME_ID:
            with self._lock:
                anchor = self._planner_local_anchor
            pts = local_points_to_world(pts, anchor).astype(np.float32)
        voxel_m = float(marker.scale.x) if marker.scale.x > 0.0 else 0.05
        with self._lock:
            self._latest_esdf_voxels = pts if pts.shape[0] > 0 else None
            self._latest_esdf_voxel_m = voxel_m

    def reset(self) -> None:
        with self._lock:
            self._latest_command = make_command()
            self._gripper_cmd = np.array([GRIPPER_OPEN_CMD, GRIPPER_OPEN_CMD], dtype=np.float32)
            self._waist_yaw_override = 0.0
            self._molmo_markers = []
        self.reset_pub.publish(Empty())

    def bind_viser(self, viser_bridge) -> None:
        self._viser_bridge = viser_bridge

    def publish_prompt(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        self._last_prompt = text
        msg = StringMsg()
        msg.data = text
        self._ui_prompt_pub.publish(msg)

    def publish_voice_toggle(self) -> None:
        self._ui_voice_toggle_pub.publish(Empty())

    def publish_molmo_reset(self) -> None:
        self._ui_reset_pub.publish(Empty())

    @property
    def last_prompt(self) -> str:
        return self._last_prompt

    def _apply_ui_to_viser(self) -> None:
        if self._viser_bridge is None:
            return
        self._viser_bridge.set_overlay(
            self._ui_overlay_points,
            self._ui_status_text,
            self._ui_voice_recording,
        )

    def _ui_overlay_points_cb(self, msg: Float32MultiArray) -> None:
        flat = list(msg.data)
        pts: list[tuple[int, int]] = []
        for i in range(0, len(flat) - 1, 2):
            pts.append((int(flat[i]), int(flat[i + 1])))
        self._ui_overlay_points = pts
        self._apply_ui_to_viser()

    def _ui_status_cb(self, msg: StringMsg) -> None:
        self._ui_status_text = str(msg.data)
        self._apply_ui_to_viser()

    def _ui_voice_state_cb(self, msg: BoolMsg) -> None:
        self._ui_voice_recording = bool(msg.data)
        self._apply_ui_to_viser()

    def _ui_prompt_text_cb(self, msg: StringMsg) -> None:
        if self._viser_bridge is not None:
            self._viser_bridge.set_prompt_text(str(msg.data))

    def _ui_plan_cb(self, msg: StringMsg) -> None:
        if self._viser_bridge is not None:
            try:
                self._viser_bridge.set_plan(str(msg.data))
            except Exception:
                pass

    def _render_viewer_overlays(self, viewer):
        with self._lock:
            overlay = self._capture_point_overlay
        with viewer.lock():
            viewer.user_scn.ngeom = 0
        if overlay is not None:
            overlay.render(viewer)


# ---------------------------------------------------------------------------
# Molmo camera bridge (verbatim from sim_node.py)
# ---------------------------------------------------------------------------
class MolmoCameraBridge:
    def __init__(self, node: Node, model: mujoco.MjModel, viser_bridge=None):
        self._node = node
        self._model = model
        self._viser_bridge = viser_bridge
        camera_names = os.environ.get(MOLMO_CAMERA_NAMES_ENV, "").strip()
        self._camera_names = [n.strip() for n in camera_names.split(",") if n.strip()]
        self._render_hz = float(os.environ.get(MOLMO_RENDER_HZ_ENV, str(MOLMO_DEFAULT_RENDER_HZ)))
        self._render_period = 1.0 / max(self._render_hz, 1e-3)
        self._last_render_t = 0.0
        self._enabled = False
        self._entries = {}
        # Latest SAM2 segmentation mask from ffs_node, keyed only by stamp.
        # We tint the live render rather than the stamp-matched RGB frame
        # — masks lag the matching RGB by ~SAM2 inference latency (~100 ms),
        # but at 10 Hz visor rate the pelvis hasn't moved noticeably and
        # a stamp-mismatched paint looks fine. Cleared after _MASK_STALE_S.
        self._latest_mask: Optional[Tuple[float, np.ndarray]] = None
        self._mask_stale_s = 1.0

        for cam_name in self._camera_names:
            cam_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
            if cam_id == -1:
                continue
            # Override the XML fovy so renders match the real ZED Mini.
            self._model.cam_fovy[cam_id] = HEAD_CAMERA_SIM_FOVY_DEG
            self._entries[cam_name] = {
                "camera_id": int(cam_id),
                "rgb_pub": node.create_publisher(Image, f"/molmo/camera/{cam_name}/rgb", VIZ_QOS),
                "points_pub": node.create_publisher(PointCloud2, f"/molmo/camera/{cam_name}/points", VIZ_QOS),
                "info_pub": node.create_publisher(CameraInfo, f"/molmo/camera/{cam_name}/camera_info", VIZ_QOS),
                "pose_pub": node.create_publisher(PoseStamped, f"/molmo/camera/{cam_name}/pose", VIZ_QOS),
                "overlay_pub": node.create_publisher(Image, MOLMO_VISOR_OVERLAY_TOPIC, VIZ_QOS),
                "rgb_renderer":   mujoco.Renderer(self._model, height=HEAD_CAMERA_SIM_RENDER_H, width=HEAD_CAMERA_SIM_RENDER_W),
                "depth_renderer": mujoco.Renderer(self._model, height=HEAD_CAMERA_SIM_RENDER_H, width=HEAD_CAMERA_SIM_RENDER_W),
            }
            self._entries[cam_name]["depth_renderer"].enable_depth_rendering()
        self._enabled = bool(self._entries)
        if self._enabled:
            node.create_subscription(Image, FFS_MASK_TOPIC, self._mask_cb, VIZ_QOS)

    def _mask_cb(self, msg: Image) -> None:
        """Cache the latest mono8 SAM2 mask + its frame stamp."""
        if msg.encoding != "mono8" or msg.width == 0 or msg.height == 0:
            return
        try:
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
        except Exception:
            return
        stamp_sec = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        self._latest_mask = (stamp_sec, arr.copy())

    def _tinted_overlay(self, rgb: np.ndarray, stamp) -> np.ndarray:
        """Return rgb tinted magenta where the latest SAM2 mask is set.

        Treats a stale (>_mask_stale_s) mask as no-mask so the overlay
        reverts to plain RGB after ffs_node enters IDLE/LOST.
        """
        mask_entry = self._latest_mask
        if mask_entry is None:
            return rgb
        mask_stamp, mask = mask_entry
        now = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        if now - mask_stamp > self._mask_stale_s:
            return rgb
        if mask.shape[:2] != rgb.shape[:2]:
            return rgb
        sel = mask > 0
        if not np.any(sel):
            return rgb
        out = rgb.copy()
        # Magenta (255, 0, 255), alpha 0.45. Distinct from cyan/green
        # used by viser_bridge for capture/target dots.
        tint = np.array([255, 0, 255], dtype=np.float32)
        out[sel] = (0.55 * rgb[sel].astype(np.float32) + 0.45 * tint).astype(np.uint8)
        return out

    @property
    def enabled(self) -> bool:
        return self._enabled

    def publish(self, data: mujoco.MjData, stamp, step_id: int) -> None:
        if not self._enabled:
            return
        now_t = time.perf_counter()
        if (now_t - self._last_render_t) < self._render_period:
            return
        self._last_render_t = now_t

        for cam_name, entry in self._entries.items():
            cam_id = int(entry["camera_id"])
            cam_pos = data.cam_xpos[cam_id].copy()
            cam_mat = data.cam_xmat[cam_id].copy().reshape(3, 3)

            entry["rgb_renderer"].update_scene(data, camera=cam_name)
            rgb = entry["rgb_renderer"].render()
            entry["depth_renderer"].update_scene(data, camera=cam_name)
            depth = entry["depth_renderer"].render().astype(np.float32, copy=True)

            def _img(array, encoding):
                msg = Image()
                msg.header.stamp = stamp
                msg.header.frame_id = cam_name
                msg.height, msg.width = int(array.shape[0]), int(array.shape[1])
                msg.encoding = encoding
                msg.is_bigendian = 0
                msg.data = np.ascontiguousarray(array).tobytes()
                msg.step = int(array.shape[1] * (array.shape[2] if array.ndim == 3 else 1) * array.dtype.itemsize)
                return msg

            fovy = float(self._model.cam_fovy[cam_id])
            fy = 0.5 * float(rgb.shape[0]) / np.tan(np.radians(fovy) / 2.0)
            info = CameraInfo()
            info.header.stamp = stamp
            info.header.frame_id = cam_name
            info.height, info.width = int(rgb.shape[0]), int(rgb.shape[1])
            info.distortion_model = "plumb_bob"
            info.d = [0.0]*5
            info.k = [float(x) for x in [fy, 0, rgb.shape[1]*0.5, 0, fy, rgb.shape[0]*0.5, 0, 0, 1]]
            info.r = [float(x) for x in [1,0,0, 0,1,0, 0,0,1]]
            info.p = [float(x) for x in [fy, 0, rgb.shape[1]*0.5, 0, 0, fy, rgb.shape[0]*0.5, 0, 0, 0, 1, 0]]

            qw = _mat_to_quat_wxyz(cam_mat)
            pose = PoseStamped()
            pose.header.stamp = stamp
            pose.header.frame_id = "world"
            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = map(float, cam_pos)
            pose.pose.orientation.w = float(qw[0])
            pose.pose.orientation.x = float(qw[1])
            pose.pose.orientation.y = float(qw[2])
            pose.pose.orientation.z = float(qw[3])

            # Back-project depth → organized XYZ in MuJoCo camera frame
            # (x=right, y=up, z=backward) to match the convention that
            # cam_mat (cam→world) was built with, so consumers can do
            # p_world = cam_mat @ p_cam + cam_pos without axis swaps.
            H, W = depth.shape
            cx = float(info.k[2])
            cy = float(info.k[5])
            fx_v = float(info.k[0])
            fy_v = float(info.k[4])
            us, vs = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
            z = depth  # positive distance; NaN/0 for invalid
            x_cam = (us - cx) * z / fx_v
            y_cam = -(vs - cy) * z / fy_v
            z_cam = -z
            xyz = np.stack([x_cam, y_cam, z_cam], axis=-1)  # (H, W, 3) float32
            # Mark pixels with no depth as NaN so consumers can filter with np.isfinite
            no_depth = (depth == 0.0) | ~np.isfinite(depth)
            xyz[no_depth] = np.nan

            cloud = PointCloud2()
            cloud.header.stamp = stamp
            cloud.header.frame_id = cam_name
            cloud.height = H
            cloud.width = W
            cloud.fields = [
                PointField(name="x", offset=0,  datatype=PointField.FLOAT32, count=1),
                PointField(name="y", offset=4,  datatype=PointField.FLOAT32, count=1),
                PointField(name="z", offset=8,  datatype=PointField.FLOAT32, count=1),
            ]
            cloud.is_bigendian = False
            cloud.point_step = 12
            cloud.row_step = W * 12
            cloud.is_dense = False
            cloud.data = np.ascontiguousarray(xyz, dtype=np.float32).tobytes()

            entry["rgb_pub"].publish(_img(rgb, "rgb8"))
            entry["points_pub"].publish(cloud)
            entry["info_pub"].publish(info)
            entry["pose_pub"].publish(pose)

            # Debug overlay: tint pixels under the SAM2 mask magenta. Only
            # the overlay topic + viser RGB panel carry the tint — molmo_node
            # grounds on /molmo/camera/<cam>/rgb (untouched above) so the
            # colored region cannot bias the next Hunyuan grounding query.
            overlay = self._tinted_overlay(rgb, stamp)
            entry["overlay_pub"].publish(_img(overlay, "rgb8"))

            if self._viser_bridge is not None and cam_name == self._camera_names[0]:
                self._viser_bridge.push_rgb(overlay)


# ---------------------------------------------------------------------------
# MuJoCo model builder (mirror of sim_node.py)
# ---------------------------------------------------------------------------
def _load_spec_with_graspnet(xml_path: Path) -> mujoco.MjSpec:
    """Load the G1 sim XML and inject graspnet objects per env config."""
    from deploy.sim.graspnet_scene import inject_graspnet_objects

    spec = mujoco.MjSpec.from_file(str(xml_path))
    raw = os.environ.get(GRASPNET_OBJECTS_ENV, "default").strip()
    if raw.lower() == "none":
        return spec
    seed_raw = os.environ.get(GRASPNET_SEED_ENV, "").strip()
    seed = int(seed_raw) if seed_raw else None
    if raw.lower() == "default":
        added = inject_graspnet_objects(spec, object_ids="default", seed=seed)
    elif raw.startswith("random"):
        n = int(raw.split(":", 1)[1]) if ":" in raw else 2
        added = inject_graspnet_objects(
            spec, object_ids="random", num_random=n, seed=seed
        )
    else:
        ids = [s.strip() for s in raw.split(",") if s.strip()]
        added = inject_graspnet_objects(spec, object_ids=ids)
    print(f"Graspnet objects injected: {added}")
    return spec


def build_model():
    xml_override = os.environ.get(HAND_SIM_G1_XML_ENV, "").strip()
    if xml_override:
        xml_path = Path(xml_override).expanduser()
        if not xml_path.exists():
            raise FileNotFoundError(f"{HAND_SIM_G1_XML_ENV} → {xml_path} not found")
        print(f"XML override: {xml_path}")
        robot_cfg = get_g1_robot_cfg()
        robot_cfg.spec_fn = wrap_spec_fn_with_payloads(
            lambda xml_path=xml_path: _load_spec_with_graspnet(xml_path),
            include_jetson=True,
            swap_hands=False,
        )
        robot_cfg.articulation = None
        robot_cfg.collisions = ()
        robot_cfg.init_state.joint_pos = None
        spec = Entity(robot_cfg).spec
        spec.option.timestep = 0.001
        return spec.compile()

    spec = mujoco.MjSpec()
    spec.option.timestep = 0.001
    spec.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
    spec.option.gravity[:] = [0.0, 0.0, -9.81]

    sky = spec.add_texture()
    sky.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
    sky.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
    sky.rgb1[:] = [0.3, 0.5, 0.7]
    sky.rgb2[:] = [0.0, 0.0, 0.0]
    sky.width = sky.height = 512

    tex = spec.add_texture(name="texplane")
    tex.type = mujoco.mjtTexture.mjTEXTURE_2D
    tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    tex.rgb1[:] = [0.2, 0.3, 0.4]
    tex.rgb2[:] = [0.1, 0.15, 0.2]
    tex.width = tex.height = 512
    tex.mark = mujoco.mjtMark.mjMARK_CROSS
    tex.markrgb[:] = [0.8, 0.8, 0.8]

    mat = spec.add_material(name="matplane")
    mat.reflectance = 0.3
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB.value] = tex.name
    mat.texrepeat[:] = [1.0, 1.0]
    mat.texuniform = True

    spec.worldbody.add_light(
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, castshadow=False,
        pos=(0, 0, 5), dir=(0, 0, -1), diffuse=(0.8, 0.8, 0.8), specular=(0.2, 0.2, 0.2),
    )
    floor = spec.worldbody.add_geom(name="floor")
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size[:] = [0, 0, 0.05]
    floor.material = mat.name

    robot_cfg = get_g1_robot_cfg()
    robot_cfg.spec_fn = wrap_spec_fn_with_payloads(robot_cfg.spec_fn)
    robot = Entity(robot_cfg)
    frame = spec.worldbody.add_frame()
    spec.attach(robot.spec, prefix="", frame=frame)
    return spec.compile()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx_path", help="Path to hand_dual_student.onnx")
    args = parser.parse_args()

    # ---- Load ONNX ----
    session = ort.InferenceSession(args.onnx_path, providers=["CPUExecutionProvider"])
    inp_name = session.get_inputs()[0].name
    expected_shape = (1, ACTOR_OBS_DIM)
    actual_shape = tuple(session.get_inputs()[0].shape)
    if actual_shape != expected_shape:
        print(f"WARNING: ONNX input shape {actual_shape} != expected {expected_shape}")

    # ---- Policy state ----
    last_action = np.zeros(NUM_JOINTS, dtype=np.float32)
    phase_tracker = 0.0
    dt = 1.0 / 50.0
    history: deque[np.ndarray] = deque(maxlen=ACTOR_HISTORY_LENGTH)
    history_initialized = False

    wrist_level_latched_left = False
    wrist_level_latched_right = False
    walkback_release_mode = (MOLMO_WRIST_LEVEL_ROLL_ONLY and MOLMO_WRIST_LEVEL_YAW_TO_TORSO)
    wrist_level_hold_left = False
    wrist_level_hold_right = False
    wrist_level_weight_left: float = 0.0
    wrist_level_weight_right: float = 0.0
    wrist_roll_offset_weight_left: float = 0.0
    wrist_roll_offset_weight_right: float = 0.0
    # Per-hand 'carrying an object' latch. See hand_policy.py.
    holding_left = False
    holding_right = False
    # Output-side wrist-yaw EMA. See hand_policy.py.
    prev_left_wrist_yaw_cmd: Optional[float] = None
    prev_right_wrist_yaw_cmd: Optional[float] = None
    # Edge-detect for the yaw-freeze diagnostic prints below.
    prev_left_yaw_freeze: bool = False
    prev_right_yaw_freeze: bool = False
    yaw_trace_print_counter: int = 0
    # EMA-smoothed pelvis height seen by the policy. See hand_policy.py.
    prev_cmd_height: Optional[float] = None
    left_wrist_yaw_idx = POLICY_JOINT_NAMES.index("left_wrist_yaw_joint")
    right_wrist_yaw_idx = POLICY_JOINT_NAMES.index("right_wrist_yaw_joint")

    # ---- Wrist leveller + hand cmd PD ----
    wrist_leveller: WristLeveller | None = None
    wrist_idx_all: np.ndarray | None = None
    if MOLMO_WRIST_LEVEL_OVERRIDE or MOLMO_WRIST_ZERO_OBS or MOLMO_HAND_CMD_PD_ENABLE:
        _urdf = (
            Path(__file__).resolve().parents[1]
            / "assets"
            / "g1_29dof_rev_1_0_with_payloads_and_gripper.urdf"
        )
        wrist_leveller = WristLeveller(
            _urdf,
            POLICY_JOINT_NAMES,
            yaw_ema_alpha=MOLMO_WRIST_LEVEL_YAW_EMA_ALPHA,
            left_tool_point_wrist=np.asarray(
                MOLMO_GRIPPER_MIDPOINT_OFFSET_LEFT, dtype=np.float64,
            ),
            right_tool_point_wrist=np.asarray(
                MOLMO_GRIPPER_MIDPOINT_OFFSET_RIGHT, dtype=np.float64,
            ),
            vertical_roll_rad=MOLMO_WRIST_VERTICAL_ROLL_RAD,
        )
        wrist_idx_all = np.asarray(
            list(wrist_leveller.left_policy_indices)
            + list(wrist_leveller.right_policy_indices),
            dtype=np.int64,
        )
        print(
            f"[sim_policy_node] wrist override={MOLMO_WRIST_LEVEL_OVERRIDE} "
            f"zero_obs={MOLMO_WRIST_ZERO_OBS} (indices {wrist_idx_all.tolist()})"
        )

    hand_cmd_pd: HandCommandPD | None = None
    if MOLMO_HAND_CMD_PD_ENABLE:
        assert wrist_leveller is not None
        kp_xy = (
            MOLMO_HAND_CMD_PD_KP * MOLMO_HAND_CMD_PD_KP_X_MULT,
            MOLMO_HAND_CMD_PD_KP * MOLMO_HAND_CMD_PD_KP_Y_MULT,
        )
        kd_xy = (
            MOLMO_HAND_CMD_PD_KD * MOLMO_HAND_CMD_PD_KD_X_MULT,
            MOLMO_HAND_CMD_PD_KD * MOLMO_HAND_CMD_PD_KD_Y_MULT,
        )
        hand_cmd_pd = HandCommandPD.from_leveller(
            wrist_leveller,
            kp=MOLMO_HAND_CMD_PD_KP,
            kd=MOLMO_HAND_CMD_PD_KD,
            kp_xy=kp_xy,
            kd_xy=kd_xy,
            deadband_m=MOLMO_HAND_CMD_PD_DEADBAND_M,
            max_shape_m=MOLMO_HAND_CMD_PD_MAX_SHAPE_M,
        )
        print(
            f"[sim_policy_node] hand_cmd_pd enabled "
            f"kp={MOLMO_HAND_CMD_PD_KP} kd={MOLMO_HAND_CMD_PD_KD} "
            f"kp_xy={kp_xy} kd_xy={kd_xy} "
            f"deadband={MOLMO_HAND_CMD_PD_DEADBAND_M}m "
            f"max_shape={MOLMO_HAND_CMD_PD_MAX_SHAPE_M}m"
        )

    # ---- Capture-point filter ----
    # Engages only when the commanded pelvis height drops into the squat
    # band — at standing heights the policy already keeps the capture point
    # well centered. CAPTURE_POINT_METHOD picks PD ("centering") or CBF.
    # Mirrors hand_policy.py.
    cp_filter: G1CapturePointCBF | None = None
    if CAPTURE_POINT_PD_ENABLE:
        cp_estimator = G1CapturePointEstimator(policy_joint_names=POLICY_JOINT_NAMES)
        cp_mode = "cbf" if CAPTURE_POINT_METHOD == "cbf" else "centering"
        cp_filter = G1CapturePointCBF(
            cp_estimator,
            mode=cp_mode,
            kp=CAPTURE_POINT_PD_KP,
            dt=dt,
        )
        print(
            f"[sim_policy_node] capture-point filter enabled "
            f"(method={CAPTURE_POINT_METHOD}, kp={CAPTURE_POINT_PD_KP}, "
            f"engages below cmd_height={SQUAT_SMOOTHING_HEIGHT} m)"
        )

    # ---- MuJoCo setup ----
    model = build_model()
    data = mujoco.MjData(model)
    molmo_camera_bridge = None

    qpos_idx, qvel_idx, ctrl_idx = [], [], []
    for name in POLICY_JOINT_NAMES:
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if a == -1:
            raise ValueError(f"Missing actuator for joint {name}")
        qpos_idx.append(model.jnt_qposadr[j])
        qvel_idx.append(model.jnt_dofadr[j])
        ctrl_idx.append(a)
    qpos_idx = np.array(qpos_idx)
    qvel_idx = np.array(qvel_idx)
    ctrl_idx = np.array(ctrl_idx)

    _gripper_names = ["gripper_prismatic_1_L", "gripper_prismatic_1_R"]
    gripper_enabled = True
    gripper_joint_qpos_idx, gripper_joint_qvel_idx, gripper_actuator_idx = [], [], []
    for gn in _gripper_names:
        gj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, gn)
        ga = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, gn)
        if gj < 0 or ga < 0:
            gripper_enabled = False
            print(f"Gripper '{gn}' not found, gripper control disabled.")
            break
        gripper_joint_qpos_idx.append(model.jnt_qposadr[gj])
        gripper_joint_qvel_idx.append(model.jnt_dofadr[gj])
        gripper_actuator_idx.append(ga)
    if gripper_enabled:
        gripper_joint_qpos_idx = np.array(gripper_joint_qpos_idx, dtype=np.int64)
        gripper_joint_qvel_idx = np.array(gripper_joint_qvel_idx, dtype=np.int64)
        gripper_actuator_idx = np.array(gripper_actuator_idx, dtype=np.int64)
        gripper_joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, gn) for gn in _gripper_names]
        gripper_joint_range = model.jnt_range[np.array(gripper_joint_ids)].astype(np.float32)
        gripper_ctrl_range = model.actuator_ctrlrange[gripper_actuator_idx].astype(np.float32)
        nv = model.nv
        mass_matrix = np.zeros((nv, nv), dtype=np.float64)
        print("Gripper PD control enabled.")

    pelvis_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    assert pelvis_body_id >= 0

    # Scripted perturbation: apply the fixed force+torque from
    # g1_constants_custom.py to SIM_PERTURB_BODY for
    # [SIM_PERTURB_T0, SIM_PERTURB_T0 + SIM_PERTURB_DURATION) sim
    # seconds. Used to reproducibly trigger fall-recovery for tuning
    # the gate + snapshot/restore path. Disabled when the body name
    # is empty or absent from the model (lookup returns -1).
    perturb_body_id = (
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, SIM_PERTURB_BODY)
        if SIM_PERTURB_BODY else -1
    )
    perturb_force = np.asarray(SIM_PERTURB_FORCE, dtype=np.float64)
    perturb_torque = np.asarray(SIM_PERTURB_TORQUE, dtype=np.float64)
    perturb_active_prev: bool = False

    viser_gripper_qpos_idx = []
    for gn in VISER_GRIPPER_JOINT_NAMES:
        gj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, gn)
        viser_gripper_qpos_idx.append(model.jnt_qposadr[gj] if gj >= 0 else -1)
    viser_gripper_qpos_idx = np.array(viser_gripper_qpos_idx, dtype=np.int64)

    waist_yaw_actuator_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "waist_yaw_joint")

    _init_yaw = math.radians(float(os.environ.get("WBC_INIT_YAW_DEG", "0")))
    data.qpos[2] = 0.76
    data.qpos[3] = math.cos(_init_yaw / 2)
    data.qpos[6] = math.sin(_init_yaw / 2)
    for i, qi in enumerate(qpos_idx):
        data.qpos[qi] = DEFAULT_POS[i]
    data.ctrl[ctrl_idx] = DEFAULT_POS
    mujoco.mj_forward(model, data)

    # ---- ROS 2 ----
    rclpy.init()
    ros_node = G1SimRosNode()

    reset_event = threading.Event()

    # ---- Viser bridge ----
    viser_bridge = None
    viser_port_str = os.environ.get(VISER_PORT_ENV, VISER_DEFAULT_PORT).strip()
    viser_enabled = bool(viser_port_str) and viser_port_str != "0"
    if viser_enabled:
        from deploy.sim.viser_bridge import ViserBridge
        urdf_path = Path(os.environ.get(
            VISER_URDF_ENV,
            str(Path(__file__).resolve().parents[1] / "assets" / VISER_DEFAULT_URDF),
        )).expanduser()
        viser_bridge = ViserBridge(
            urdf_path=urdf_path,
            joint_names=POLICY_JOINT_NAMES,
            gripper_joint_names=VISER_GRIPPER_JOINT_NAMES,
            port=int(viser_port_str),
            on_submit_prompt=lambda t: ros_node.publish_prompt(t),
            on_resubmit=lambda: ros_node.publish_prompt(ros_node.last_prompt),
            on_toggle_voice=lambda: ros_node.publish_voice_toggle(),
            on_reset_molmo=lambda: ros_node.publish_molmo_reset(),
            on_reset_sim=reset_event.set,
            initial_prompt=os.environ.get(MOLMO_DEFAULT_PROMPT_ENV, MOLMO_DEFAULT_PROMPT),
            robot_hz=float(os.environ.get(VISER_HZ_ENV, str(VISER_DEFAULT_HZ))),
            image_hz=float(os.environ.get(VISER_IMAGE_HZ_ENV, str(VISER_DEFAULT_IMAGE_HZ))),
        )
        ros_node.bind_viser(viser_bridge)

        scene_xml_path = os.environ.get(HAND_SIM_G1_XML_ENV, "").strip()
        if scene_xml_path:
            # Walk the compiled model so we see runtime-injected graspnet objects
            # and render mesh geoms with their textures.
            movable_bodies = viser_bridge.load_mjcf_scene(
                model=model,
                skip_bodies=("pelvis",),
                skip_geoms=("floor",),
            )
        else:
            movable_bodies = []
    else:
        movable_bodies = []

    movable_body_ids: dict[str, int] = {}
    for name in movable_bodies:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            movable_body_ids[name] = bid

    if os.environ.get(MOLMO_CAMERA_NAMES_ENV, "").strip():
        molmo_camera_bridge = MolmoCameraBridge(ros_node, model, viser_bridge=viser_bridge)

    _ros_executor = _SingleThreadedExecutor()
    _ros_executor.add_node(ros_node)
    ros_thread = threading.Thread(target=_ros_executor.spin, daemon=True)
    ros_thread.start()

    # Constructed after rclpy.init() — the rclpy.ok() guard in __init__ will no-op.
    molmo_sub = _MolmoSubscriber()

    # ---- Initial state snapshot for tick-0 obs ----
    root_quat = data.qpos[3:7].astype(np.float32)
    root_pos = data.qpos[0:3].astype(np.float32)
    joint_pos = data.qpos[qpos_idx].astype(np.float32)
    joint_vel = np.zeros(NUM_JOINTS, dtype=np.float32)
    body_ang_vel = np.zeros(3, dtype=np.float32)
    body_lin_vel = np.zeros(3, dtype=np.float32)
    sent_target = DEFAULT_POS.copy()

    control_dt = model.opt.timestep * DECIMATION

    # _policy_reset_pending: set by _do_reset(), cleared by the main loop after
    # wiping all policy carry-over state.
    _policy_reset_pending = [False]

    def _do_reset():
        mujoco.mj_resetData(model, data)
        data.qpos[2] = 0.76
        data.qpos[3] = math.cos(_init_yaw / 2)
        data.qpos[4] = 0.0
        data.qpos[5] = 0.0
        data.qpos[6] = math.sin(_init_yaw / 2)
        for i, qi in enumerate(qpos_idx):
            data.qpos[qi] = DEFAULT_POS[i]
        data.ctrl[ctrl_idx] = DEFAULT_POS
        mujoco.mj_forward(model, data)
        ros_node.reset()
        _policy_reset_pending[0] = True
        print("Simulation reset.")

    # ---- Viewer ----
    mujoco_viewer_enabled = os.environ.get(MUJOCO_VIEWER_ENV, "0").strip().lower() not in (
        "", "0", "false", "no",
    )
    shutdown_event = threading.Event()
    signal.signal(signal.SIGINT,  lambda *_: shutdown_event.set())
    signal.signal(signal.SIGTERM, lambda *_: shutdown_event.set())

    viewer_cm = (
        mujoco.viewer.launch_passive(model, data)
        if mujoco_viewer_enabled else nullcontext(None)
    )
    print(
        f"Launching [sync mode]: mujoco={'on' if mujoco_viewer_enabled else 'off'} "
        f"viser={'http://localhost:' + viser_port_str if viser_enabled else 'off'}"
    )

    step_count = 0

    try:
      with viewer_cm as viewer:
        start_wall = time.perf_counter() - data.time
        prev_sim_time = data.time
        last_status_t = 0.0

        while (
            not shutdown_event.is_set()
            and (viewer is None or viewer.is_running())
            and rclpy.ok()
        ):
            # 1. Real-time pacing
            target_wall = start_wall + data.time + control_dt
            sleep_time = target_wall - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)

            # 2. External reset (viser button)
            if reset_event.is_set():
                reset_event.clear()
                _do_reset()
                step_count = 0
                start_wall = time.perf_counter() - data.time

            # Detect GUI reset via time regression
            if data.time < prev_sim_time - control_dt * 0.5:
                _do_reset()
                step_count = 0
                start_wall = time.perf_counter() - data.time

            # 3. Policy reset — clear all carry-over state after a sim reset
            if _policy_reset_pending[0]:
                _policy_reset_pending[0] = False
                last_action[:] = 0.0
                phase_tracker = 0.0
                history.clear()
                history_initialized = False
                wrist_level_latched_left = False
                wrist_level_latched_right = False
                wrist_level_hold_left = False
                wrist_level_hold_right = False
                wrist_level_weight_left = 0.0
                wrist_level_weight_right = 0.0
                wrist_roll_offset_weight_left = 0.0
                wrist_roll_offset_weight_right = 0.0
                holding_left = False
                holding_right = False
                prev_left_wrist_yaw_cmd = None
                prev_right_wrist_yaw_cmd = None
                prev_cmd_height = None
                if wrist_leveller is not None:
                    wrist_leveller.reset()
                if hand_cmd_pd is not None:
                    hand_cmd_pd.reset()
                # Re-extract state from freshly reset mujoco data for next obs
                root_quat = data.qpos[3:7].astype(np.float32)
                root_pos = data.qpos[0:3].astype(np.float32)
                joint_pos = data.qpos[qpos_idx].astype(np.float32)
                joint_vel = np.zeros(NUM_JOINTS, dtype=np.float32)
                body_ang_vel = np.zeros(3, dtype=np.float32)
                body_lin_vel = np.zeros(3, dtype=np.float32)
                sent_target = DEFAULT_POS.copy()
                data.ctrl[ctrl_idx] = DEFAULT_POS
                print("[sim_policy_node] policy state reset.", flush=True)

            # 4. Read latest command from ROS node
            with ros_node._lock:
                cmd = ros_node._latest_command.copy()

            # EMA-smooth the incoming pelvis height command — same
            # rationale as hand_policy.py: source-side step changes (e.g.
            # molmo flipping locomoting on/off) should not appear in the
            # policy obs as a step.
            raw_cmd_height = float(cmd[CMD_HEIGHT])
            if prev_cmd_height is None:
                prev_cmd_height = raw_cmd_height
            else:
                prev_cmd_height = (
                    (1.0 - MOLMO_SQUAT_EMA_ALPHA) * prev_cmd_height
                    + MOLMO_SQUAT_EMA_ALPHA * raw_cmd_height
                )
            cmd[CMD_HEIGHT] = np.float32(prev_cmd_height)

            # 5. Wrist latch state machine (from hand_policy.py)
            wrist_override_rpy: tuple[np.ndarray | None, np.ndarray | None] | None = None
            left_hand_z = float(cmd[CMD_LEFT_HAND + 2])
            right_hand_z = float(cmd[CMD_RIGHT_HAND + 2])
            left_gripper_open = float(cmd[CMD_LEFT_GRIPPER]) < _WRIST_LEVEL_GRIPPER_OPEN_MAX
            right_gripper_open = float(cmd[CMD_RIGHT_GRIPPER]) < _WRIST_LEVEL_GRIPPER_OPEN_MAX
            use_walkback_release = walkback_release_mode and molmo_sub.has_phase()
            left_allowed = molmo_sub.leveller_allowed_for("l")
            right_allowed = molmo_sub.leveller_allowed_for("r")
            left_raw_target = molmo_sub.raw_target_for("l")
            right_raw_target = molmo_sub.raw_target_for("r")

            # Vertical is the default resting pose. See hand_policy.py
            # for the full semantics — a side stays non-vertical only
            # while carrying an object (pick through place-release).
            phase = molmo_sub.current_phase()
            bimanual_active_now = _is_bimanual_active(phase)
            holding_left = _update_side_holding(holding_left, phase, "l")
            holding_right = _update_side_holding(holding_right, phase, "r")
            left_vertical_wanted = not holding_left
            right_vertical_wanted = not holding_right
            any_vertical_wanted = left_vertical_wanted or right_vertical_wanted

            left_place_active = molmo_sub.is_place_active_for("l")
            right_place_active = molmo_sub.is_place_active_for("r")

            # Latching rule mirrors pick: cross the hand-high threshold while
            # in a "leveler-eligible" sub-phase. For pick that means
            # gripper_open (pre-grasp lift). For single-hand place the
            # gripper stays closed (carrying), so place_active substitutes
            # for gripper_open. Unlatch only when the side is neither
            # carrying-toward-place nor in pre-grasp.
            left_lift_eligible = left_gripper_open or left_place_active
            right_lift_eligible = right_gripper_open or right_place_active

            if left_vertical_wanted:
                wrist_level_latched_left = False
                wrist_level_hold_left = False
            else:
                if left_allowed:
                    if left_hand_z > _WRIST_LEVEL_Z_THRESHOLD and left_lift_eligible:
                        wrist_level_latched_left = True
                        if use_walkback_release:
                            wrist_level_hold_left = True
                    if not left_lift_eligible:
                        wrist_level_latched_left = False
                        if not use_walkback_release:
                            wrist_level_hold_left = False
                    if use_walkback_release and (
                        (not MOLMO_WRIST_ROLL_AFTER_PICK_ENABLE and molmo_sub.is_pick_walkback_for("l"))
                        or molmo_sub.is_pick_retract_for("l")
                    ):
                        wrist_level_hold_left = False
                else:
                    wrist_level_latched_left = False
                    if (
                        not use_walkback_release
                        or molmo_sub.should_clear_walkback_hold_for("l")
                    ):
                        wrist_level_hold_left = False

            if right_vertical_wanted:
                wrist_level_latched_right = False
                wrist_level_hold_right = False
            else:
                if right_allowed:
                    if right_hand_z > _WRIST_LEVEL_Z_THRESHOLD and right_lift_eligible:
                        wrist_level_latched_right = True
                        if use_walkback_release:
                            wrist_level_hold_right = True
                    if not right_lift_eligible:
                        wrist_level_latched_right = False
                        if not use_walkback_release:
                            wrist_level_hold_right = False
                    if use_walkback_release and (
                        (not MOLMO_WRIST_ROLL_AFTER_PICK_ENABLE and molmo_sub.is_pick_walkback_for("r"))
                        or molmo_sub.is_pick_retract_for("r")
                    ):
                        wrist_level_hold_right = False
                else:
                    wrist_level_latched_right = False
                    if (
                        not use_walkback_release
                        or molmo_sub.should_clear_walkback_hold_for("r")
                    ):
                        wrist_level_hold_right = False

            if left_vertical_wanted:
                target_w_left = 1.0
            elif use_walkback_release:
                target_w_left = 1.0 if wrist_level_hold_left else 0.0
            else:
                target_w_left = 1.0 if wrist_level_latched_left else 0.0
            if right_vertical_wanted:
                target_w_right = 1.0
            elif use_walkback_release:
                target_w_right = 1.0 if wrist_level_hold_right else 0.0
            else:
                target_w_right = 1.0 if wrist_level_latched_right else 0.0
            wrist_level_weight_left += _WRIST_LEVEL_EMA_ALPHA * (
                target_w_left - wrist_level_weight_left
            )
            wrist_level_weight_right += _WRIST_LEVEL_EMA_ALPHA * (
                target_w_right - wrist_level_weight_right
            )

            target_off_left = 1.0 if (
                MOLMO_WRIST_ROLL_AFTER_PICK_ENABLE
                and not left_vertical_wanted
                and wrist_level_hold_left
                and not wrist_level_latched_left
            ) else 0.0
            target_off_right = 1.0 if (
                MOLMO_WRIST_ROLL_AFTER_PICK_ENABLE
                and not right_vertical_wanted
                and wrist_level_hold_right
                and not wrist_level_latched_right
            ) else 0.0
            wrist_roll_offset_weight_left += _WRIST_LEVEL_EMA_ALPHA * (
                target_off_left - wrist_roll_offset_weight_left
            )
            wrist_roll_offset_weight_right += _WRIST_LEVEL_EMA_ALPHA * (
                target_off_right - wrist_roll_offset_weight_right
            )

            if any_vertical_wanted or MOLMO_WRIST_LEVEL_ROLL_ONLY:
                left_leveler_active = wrist_level_weight_left > _WRIST_LEVEL_BLEND_EPS
                right_leveler_active = wrist_level_weight_right > _WRIST_LEVEL_BLEND_EPS
            else:
                left_leveler_active = left_allowed and wrist_level_latched_left
                right_leveler_active = right_allowed and wrist_level_latched_right
            left_pd_active = left_allowed and wrist_level_latched_left
            right_pd_active = right_allowed and wrist_level_latched_right
            any_leveler_active = left_leveler_active or right_leveler_active

            # Hard-kill all wrist suppression paths during fall recovery.
            # Mirror of the equivalent block in hand_policy.py — the EMA
            # weights are kept at 1.0 by the vertical_wanted gate even
            # when phase=fall_recovery parses to None, so the leveler
            # would otherwise stay active across the fall. Snap to zero
            # and short-circuit every activity flag this tick.
            if molmo_sub.is_fall_recovery():
                wrist_level_weight_left = 0.0
                wrist_level_weight_right = 0.0
                wrist_roll_offset_weight_left = 0.0
                wrist_roll_offset_weight_right = 0.0
                left_leveler_active = False
                right_leveler_active = False
                any_leveler_active = False
                left_pd_active = False
                right_pd_active = False

            # Wrist yaw freeze: stop chasing perception jitter once gripper
            # is close to target. Mirror of hand_policy.py freeze block.
            left_yaw_freeze = molmo_sub.is_pick_yaw_freeze_for("l")
            right_yaw_freeze = molmo_sub.is_pick_yaw_freeze_for("r")
            left_yaw_freeze_phase = left_yaw_freeze
            right_yaw_freeze_phase = right_yaw_freeze
            left_yaw_freeze_box = False
            right_yaw_freeze_box = False
            left_err_wrist: Optional[np.ndarray] = None
            right_err_wrist: Optional[np.ndarray] = None
            # Box check in the gripper's "horizontal grasp frame": depth +
            # lateral follow the world-horizontal finger heading, vertical
            # is gravity. Z half-extent is per-side from the published OBB
            # extent (falls back to MOLMO_PICK_EARLY_CLOSE_Z_M). See
            # hand_policy for rationale.
            def _err_wrist_for(hand: str, raw_target_b: np.ndarray) -> Optional[np.ndarray]:
                try:
                    return wrist_leveller.target_in_horizontal_grasp_frame(
                        hand, raw_target_b, joint_pos, root_pos, root_quat,
                    )
                except Exception:  # noqa: BLE001
                    return None
            def _z_bound_for(hand: str) -> float:
                obj_z = molmo_sub.pick_object_z_extent_for(hand)
                if obj_z > 0.0:
                    return 0.75 * obj_z
                return float(
                    MOLMO_PICK_EARLY_CLOSE_Z_SQUAT_M
                    if molmo_sub.is_squat_pick_for(hand)
                    else MOLMO_PICK_EARLY_CLOSE_Z_M
                )
            def _in_box(err_w: np.ndarray, z_bound: float) -> bool:
                return bool(
                    -float(MOLMO_PICK_EARLY_CLOSE_X_BACK_M)
                    <= float(err_w[0])
                    <= float(MOLMO_PICK_EARLY_CLOSE_X_FWD_M)
                    and abs(float(err_w[1])) <= float(MOLMO_PICK_EARLY_CLOSE_Y_M)
                    and abs(float(err_w[2])) <= z_bound
                )
            if wrist_leveller is not None and (
                (left_leveler_active and left_raw_target is not None and not left_yaw_freeze)
                or (right_leveler_active and right_raw_target is not None and not right_yaw_freeze)
            ):
                if (
                    not left_yaw_freeze
                    and left_leveler_active
                    and left_raw_target is not None
                ):
                    left_err_wrist = _err_wrist_for("l", left_raw_target)
                    if left_err_wrist is not None and _in_box(left_err_wrist, _z_bound_for("l")):
                        left_yaw_freeze = True
                        left_yaw_freeze_box = True
                if (
                    not right_yaw_freeze
                    and right_leveler_active
                    and right_raw_target is not None
                ):
                    right_err_wrist = _err_wrist_for("r", right_raw_target)
                    if right_err_wrist is not None and _in_box(right_err_wrist, _z_bound_for("r")):
                        right_yaw_freeze = True
                        right_yaw_freeze_box = True

            # Diagnostic: edge-triggered activation/clear, plus a 1Hz trace
            # showing (depth, lateral, vertical) err in the horizontal-grasp
            # frame against the early-close bounds.
            def _box_src(ew: Optional[np.ndarray], side: str) -> str:
                if ew is None:
                    return "box (err=None)"
                z_bnd = _z_bound_for(side)
                obj_z = molmo_sub.pick_object_z_extent_for(side)
                return (
                    f"box err_dlv=({float(ew[0]):+.3f},{float(ew[1]):+.3f},{float(ew[2]):+.3f}) "
                    f"d∈[-{MOLMO_PICK_EARLY_CLOSE_X_BACK_M:.3f},{MOLMO_PICK_EARLY_CLOSE_X_FWD_M:.3f}] "
                    f"|l|<={MOLMO_PICK_EARLY_CLOSE_Y_M:.3f} "
                    f"|v|<={z_bnd:.3f}(obj_z={obj_z:.3f})"
                )
            if left_yaw_freeze and not prev_left_yaw_freeze:
                src = "phase" if left_yaw_freeze_phase else (
                    _box_src(left_err_wrist, "l") if left_yaw_freeze_box else "?"
                )
                # print(f"[yaw_freeze] L active ({src})", flush=True)
            if right_yaw_freeze and not prev_right_yaw_freeze:
                src = "phase" if right_yaw_freeze_phase else (
                    _box_src(right_err_wrist, "r") if right_yaw_freeze_box else "?"
                )
                # print(f"[yaw_freeze] R active ({src})", flush=True)
            # if not left_yaw_freeze and prev_left_yaw_freeze:
                # print("[yaw_freeze] L cleared", flush=True)
            # if not right_yaw_freeze and prev_right_yaw_freeze:
                # print("[yaw_freeze] R cleared", flush=True)
            prev_left_yaw_freeze = left_yaw_freeze
            prev_right_yaw_freeze = right_yaw_freeze
            yaw_trace_print_counter += 1
            if (left_yaw_freeze or right_yaw_freeze) and yaw_trace_print_counter % 25 == 0:
                sides = []
                if left_yaw_freeze:
                    sides.append("L")
                if right_yaw_freeze:
                    sides.append("R")
                # print(f"[yaw_freeze] held: {','.join(sides)}", flush=True)
            if (
                yaw_trace_print_counter % 50 == 0
                and phase is not None
                and phase.action == "pick"
            ):
                def _ew_fmt(ew: Optional[np.ndarray]) -> str:
                    if ew is None:
                        return "n/a"
                    return f"({float(ew[0]):+.3f},{float(ew[1]):+.3f},{float(ew[2]):+.3f})"
                # print(
                #     f"[yaw_freeze.trace] phase={phase.hand}:{phase.motion_phase} "
                #     f"L(active={left_leveler_active}, "
                #     f"raw_target={left_raw_target is not None}, "
                #     f"freeze={left_yaw_freeze}, err_dlv={_ew_fmt(left_err_wrist)}) "
                #     f"R(active={right_leveler_active}, "
                #     f"raw_target={right_raw_target is not None}, "
                #     f"freeze={right_yaw_freeze}, err_dlv={_ew_fmt(right_err_wrist)}) "
                #     f"d∈[-{MOLMO_PICK_EARLY_CLOSE_X_BACK_M:.3f},{MOLMO_PICK_EARLY_CLOSE_X_FWD_M:.3f}] "
                #     f"|l|<={MOLMO_PICK_EARLY_CLOSE_Y_M:.3f} "
                #     f"|v|<={MOLMO_PICK_EARLY_CLOSE_Z_M:.3f}",
                #     flush=True,
                # )

            if MOLMO_WRIST_LEVEL_OVERRIDE and wrist_leveller is not None and any_leveler_active:
                if any_vertical_wanted or MOLMO_WRIST_LEVEL_ROLL_ONLY:
                    left_mode = "vertical" if left_vertical_wanted else "level"
                    right_mode = "vertical" if right_vertical_wanted else "level"
                    left_raw_for_compute = (
                        left_raw_target
                        if left_leveler_active and not left_vertical_wanted
                        else None
                    )
                    right_raw_for_compute = (
                        right_raw_target
                        if right_leveler_active and not right_vertical_wanted
                        else None
                    )
                    left_yaw_dir = (
                        _PELVIS_FORWARD
                        if (
                            MOLMO_WRIST_LEVEL_YAW_TO_TORSO
                            and left_leveler_active
                            and not left_vertical_wanted
                            and left_raw_for_compute is None
                        )
                        else None
                    )
                    right_yaw_dir = (
                        _PELVIS_FORWARD
                        if (
                            MOLMO_WRIST_LEVEL_YAW_TO_TORSO
                            and right_leveler_active
                            and not right_vertical_wanted
                            and right_raw_for_compute is None
                        )
                        else None
                    )
                    if left_yaw_freeze:
                        left_raw_for_compute = None
                        left_yaw_dir = None
                    if right_yaw_freeze:
                        right_raw_for_compute = None
                        right_yaw_dir = None
                    left_rpy, right_rpy = wrist_leveller.compute(
                        joint_pos, root_pos, root_quat,
                        left_target_body=left_raw_for_compute,
                        right_target_body=right_raw_for_compute,
                        left_yaw_body_direction=left_yaw_dir,
                        right_yaw_body_direction=right_yaw_dir,
                        freeze_yaw_left=left_yaw_freeze,
                        freeze_yaw_right=right_yaw_freeze,
                        left_orientation_mode=left_mode,
                        right_orientation_mode=right_mode,
                        dt=dt,
                        active_left=left_leveler_active,
                        active_right=right_leveler_active,
                    )
                    wrist_override_rpy = (left_rpy, right_rpy)
                else:
                    _FWD_TARGET = np.array([5.0, 0.0, 0.0], dtype=np.float64)
                    left_target_body = (
                        left_raw_target
                        if left_leveler_active and left_raw_target is not None
                        else (_FWD_TARGET if left_leveler_active else None)
                    )
                    right_target_body = (
                        right_raw_target
                        if right_leveler_active and right_raw_target is not None
                        else (_FWD_TARGET if right_leveler_active else None)
                    )
                    if left_yaw_freeze:
                        left_target_body = None
                    if right_yaw_freeze:
                        right_target_body = None
                    left_rpy, right_rpy = wrist_leveller.compute(
                        joint_pos, root_pos, root_quat,
                        left_target_body=left_target_body,
                        right_target_body=right_target_body,
                        freeze_yaw_left=left_yaw_freeze,
                        freeze_yaw_right=right_yaw_freeze,
                        dt=dt,
                        active_left=left_leveler_active,
                        active_right=right_leveler_active,
                    )
                    wrist_override_rpy = (left_rpy, right_rpy)
            else:
                if wrist_leveller is not None:
                    wrist_leveller.compute(
                        joint_pos, root_pos, root_quat,
                        left_target_body=None, right_target_body=None, dt=dt,
                        active_left=False, active_right=False,
                    )

            # Hand-position PD shaping — gated off only for bimanual
            # (both wrists vertical). During single-hand non-active
            # vertical, the active side still needs PD and the non-active
            # side's own ``*_pd_active`` is already False.
            if hand_cmd_pd is not None and not bimanual_active_now:
                cmd = cmd.copy()
                shaped_left, shaped_right = hand_cmd_pd.shape(
                    joint_pos, root_pos, root_quat,
                    cmd[CMD_LEFT_HAND:CMD_LEFT_HAND + 3],
                    cmd[CMD_RIGHT_HAND:CMD_RIGHT_HAND + 3],
                    dt=dt,
                    active_left=left_pd_active,
                    active_right=right_pd_active,
                    raw_target_b_left=left_raw_target,
                    raw_target_b_right=right_raw_target,
                )
                cmd[CMD_LEFT_HAND:CMD_LEFT_HAND + 3] = shaped_left
                cmd[CMD_RIGHT_HAND:CMD_RIGHT_HAND + 3] = shaped_right

            # 6. MOLMO_WRIST_ZERO_OBS
            _jp_for_obs = joint_pos
            _jv_for_obs = joint_vel
            if (
                MOLMO_WRIST_ZERO_OBS
                and wrist_idx_all is not None
                and not molmo_sub.is_fall_recovery()
            ):
                _jp_for_obs = joint_pos.copy()
                _jv_for_obs = joint_vel.copy()
                _jp_for_obs[wrist_idx_all] = 0.0
                _jv_for_obs[wrist_idx_all] = 0.0

            # 7. Build observation
            actor_current = build_actor_current(
                _jp_for_obs, _jv_for_obs, root_quat, body_ang_vel,
                cmd, last_action, phase_tracker,
            )
            if not history_initialized:
                for _ in range(ACTOR_HISTORY_LENGTH):
                    history.append(actor_current.copy())
                history_initialized = True
            else:
                history.append(actor_current.copy())
            actor_history = np.concatenate(list(history), dtype=np.float32)
            obs = np.concatenate([actor_current, actor_history], dtype=np.float32).reshape(1, -1)

            # 8. ONNX inference
            policy_action = session.run(None, {inp_name: obs})[0][0]
            phase_tracker = (phase_tracker + dt) % 1.0
            raw_target = DEFAULT_POS + policy_action * JOINT_SCALES
            last_action = ((raw_target - DEFAULT_POS) / JOINT_SCALES).astype(np.float32)

            # 9. Apply wrist override
            sent_target = raw_target.copy()
            if wrist_override_rpy is not None:
                left_rpy, right_rpy = wrist_override_rpy
                roll_only_apply = MOLMO_WRIST_LEVEL_ROLL_ONLY or any_vertical_wanted
                if roll_only_apply:
                    if left_rpy is not None and wrist_level_weight_left > _WRIST_LEVEL_BLEND_EPS:
                        w = float(wrist_level_weight_left)
                        roll_idx = wrist_leveller.left_policy_indices[0]
                        roll_cmd = w * float(left_rpy[0]) + (1.0 - w) * float(sent_target[roll_idx])
                        if (
                            not left_vertical_wanted
                            and wrist_roll_offset_weight_left > _WRIST_LEVEL_BLEND_EPS
                        ):
                            roll_cmd += float(wrist_roll_offset_weight_left) * MOLMO_WRIST_ROLL_AFTER_PICK_RAD
                        lo, hi = wrist_leveller.left_limits[0]
                        sent_target[roll_idx] = float(np.clip(roll_cmd, lo, hi))
                        if MOLMO_WRIST_LEVEL_YAW_TO_TORSO and not left_vertical_wanted:
                            yaw_idx = wrist_leveller.left_policy_indices[2]
                            # Freeze drops the (1-w) policy term so the
                            # leveler's held yaw isn't pulled around by the
                            # policy's atan2 drift near the target.
                            w_yaw = 1.0 if left_yaw_freeze else w
                            sent_target[yaw_idx] = (
                                w_yaw * float(left_rpy[2])
                                + (1.0 - w_yaw) * float(sent_target[yaw_idx])
                            )
                    if right_rpy is not None and wrist_level_weight_right > _WRIST_LEVEL_BLEND_EPS:
                        w = float(wrist_level_weight_right)
                        roll_idx = wrist_leveller.right_policy_indices[0]
                        roll_cmd = w * float(right_rpy[0]) + (1.0 - w) * float(sent_target[roll_idx])
                        if (
                            not right_vertical_wanted
                            and wrist_roll_offset_weight_right > _WRIST_LEVEL_BLEND_EPS
                        ):
                            roll_cmd += float(wrist_roll_offset_weight_right) * MOLMO_WRIST_ROLL_AFTER_PICK_RAD
                        lo, hi = wrist_leveller.right_limits[0]
                        sent_target[roll_idx] = float(np.clip(roll_cmd, lo, hi))
                        if MOLMO_WRIST_LEVEL_YAW_TO_TORSO and not right_vertical_wanted:
                            yaw_idx = wrist_leveller.right_policy_indices[2]
                            w_yaw = 1.0 if right_yaw_freeze else w
                            sent_target[yaw_idx] = (
                                w_yaw * float(right_rpy[2])
                                + (1.0 - w_yaw) * float(sent_target[yaw_idx])
                            )
                else:
                    if left_rpy is not None:
                        sent_target[list(wrist_leveller.left_policy_indices)] = left_rpy
                    if right_rpy is not None:
                        sent_target[list(wrist_leveller.right_policy_indices)] = right_rpy

            # 9b. Output-side wrist-yaw EMA — place phases only. See
            # hand_policy.py for the rationale.
            left_apply_yaw_ema = (
                phase is not None
                and phase.action == "place"
                and (phase.hand_mode == "bimanual" or phase.hand == "l")
            )
            right_apply_yaw_ema = (
                phase is not None
                and phase.action == "place"
                and (phase.hand_mode == "bimanual" or phase.hand == "r")
            )
            if left_apply_yaw_ema:
                left_yaw_cmd = float(sent_target[left_wrist_yaw_idx])
                if prev_left_wrist_yaw_cmd is not None:
                    left_yaw_cmd = prev_left_wrist_yaw_cmd + MOLMO_WRIST_YAW_OUTPUT_EMA_ALPHA * (
                        left_yaw_cmd - prev_left_wrist_yaw_cmd
                    )
                sent_target[left_wrist_yaw_idx] = np.float32(left_yaw_cmd)
                prev_left_wrist_yaw_cmd = left_yaw_cmd
            else:
                prev_left_wrist_yaw_cmd = None

            if right_apply_yaw_ema:
                right_yaw_cmd = float(sent_target[right_wrist_yaw_idx])
                if prev_right_wrist_yaw_cmd is not None:
                    right_yaw_cmd = prev_right_wrist_yaw_cmd + MOLMO_WRIST_YAW_OUTPUT_EMA_ALPHA * (
                        right_yaw_cmd - prev_right_wrist_yaw_cmd
                    )
                sent_target[right_wrist_yaw_idx] = np.float32(right_yaw_cmd)
                prev_right_wrist_yaw_cmd = right_yaw_cmd
            else:
                prev_right_wrist_yaw_cmd = None

            # 9c. Capture-point centering PD — engaged only while squatting.
            # Modifies the ankle pitch/roll + waist pitch slots of sent_target
            # to push the (LIPM) capture point toward the support-polygon
            # centroid. Safety filter only; runs after every other shaping.
            if cp_filter is not None and float(cmd[CMD_HEIGHT]) < SQUAT_SMOOTHING_HEIGHT:
                vel_norm = float(np.linalg.norm(
                    [cmd[CMD_VX], cmd[CMD_VY], cmd[CMD_YAW_RATE]]
                ))
                lin_vel_w = quat_rotate(root_quat, body_lin_vel)
                ang_vel_w = quat_rotate(root_quat, body_ang_vel)
                sent_target, _cp_info = cp_filter.filter(
                    root_pos, root_quat, lin_vel_w, ang_vel_w,
                    joint_pos, joint_vel,
                    proposed_target=sent_target.astype(np.float64),
                    last_target=sent_target.astype(np.float64),
                    velocity_command_norm=vel_norm,
                )
                sent_target = sent_target.astype(np.float32)

            # 10. Apply action immediately — no UDP round-trip
            data.ctrl[ctrl_idx] = sent_target

            # 11. Gripper PD control
            if gripper_enabled:
                with ros_node._lock:
                    gcmd = ros_node._gripper_cmd.copy()
                gcmd = np.clip(gcmd, GRIPPER_OPEN_CMD, GRIPPER_CLOSED_CMD)
                q = data.qpos[gripper_joint_qpos_idx].astype(np.float32)
                qd = data.qvel[gripper_joint_qvel_idx].astype(np.float32)
                q_lo, q_hi = gripper_joint_range[:, 0], gripper_joint_range[:, 1]
                q_target = q_hi - (q_hi - q_lo) * gcmd
                mujoco.mj_fullM(model, mass_matrix, data.qM)
                m_eff = np.maximum(
                    mass_matrix[gripper_joint_qvel_idx, gripper_joint_qvel_idx].astype(np.float32),
                    1e-6,
                )
                kd = GRIPPER_DAMPING_RATIO * 2.0 * np.sqrt(GRIPPER_PD_KP * m_eff)
                tau = np.clip(
                    GRIPPER_PD_KP * (q_target - q) - kd * qd,
                    gripper_ctrl_range[:, 0], gripper_ctrl_range[:, 1],
                )
                data.ctrl[gripper_actuator_idx] = tau

            # 12. Waist yaw override
            if waist_yaw_actuator_idx >= 0:
                with ros_node._lock:
                    wyaw = ros_node._waist_yaw_override
                if abs(wyaw) > 1e-4:
                    data.ctrl[waist_yaw_actuator_idx] = wyaw

            # 12b. Scripted perturbation injection. Writes to
            # data.xfrc_applied[perturb_body_id] only during the active
            # window so that manual viewer drags outside the window are
            # not clobbered. Clears once on the window's falling edge.
            perturb_active = (
                perturb_body_id >= 0
                and SIM_PERTURB_T0 <= data.time < SIM_PERTURB_T0 + SIM_PERTURB_DURATION
            )
            if perturb_active:
                data.xfrc_applied[perturb_body_id, 0:3] = perturb_force
                data.xfrc_applied[perturb_body_id, 3:6] = perturb_torque
            elif perturb_active_prev:
                # Falling edge: clear what we wrote so the body isn't
                # left with a stale perturbation on the next tick.
                data.xfrc_applied[perturb_body_id, 0:6] = 0.0
            perturb_active_prev = perturb_active

            # 13. Physics step
            for _ in range(DECIMATION):
                mujoco.mj_step(model, data)

            # 14. Extract state (stored for next tick's obs)
            root_quat = data.qpos[3:7].astype(np.float32)
            root_pos = data.qpos[0:3].astype(np.float32)
            cvel = data.cvel[pelvis_body_id]
            ang_vel_w = cvel[0:3].astype(np.float32)
            lin_vel_c = cvel[3:6].astype(np.float32)
            pos_f = data.xpos[pelvis_body_id].astype(np.float32)
            stcom = data.subtree_com[pelvis_body_id].astype(np.float32)
            lin_vel_w = lin_vel_c - np.cross(ang_vel_w, stcom - pos_f)
            body_ang_vel = quat_rotate_inverse(root_quat, ang_vel_w)
            body_lin_vel = quat_rotate_inverse(root_quat, lin_vel_w)
            joint_pos = data.qpos[qpos_idx].astype(np.float32)
            joint_vel = data.qvel[qvel_idx].astype(np.float32)

            # 15. Publish
            stamp = ros_node.get_clock().now().to_msg()

            odom = Odometry()
            odom.header.stamp = stamp
            odom.header.frame_id = "world"
            odom.child_frame_id = "pelvis"
            odom.pose.pose.position.x = float(root_pos[0])
            odom.pose.pose.position.y = float(root_pos[1])
            odom.pose.pose.position.z = float(root_pos[2])
            odom.pose.pose.orientation.x = float(root_quat[1])
            odom.pose.pose.orientation.y = float(root_quat[2])
            odom.pose.pose.orientation.z = float(root_quat[3])
            odom.pose.pose.orientation.w = float(root_quat[0])
            odom.twist.twist.linear.x = float(body_lin_vel[0])
            odom.twist.twist.linear.y = float(body_lin_vel[1])
            odom.twist.twist.linear.z = float(body_lin_vel[2])
            odom.twist.twist.angular.x = float(body_ang_vel[0])
            odom.twist.twist.angular.y = float(body_ang_vel[1])
            odom.twist.twist.angular.z = float(body_ang_vel[2])
            ros_node.odom_pub.publish(odom)

            js = JointState()
            js.header.stamp = stamp
            js.name = list(POLICY_JOINT_NAMES) + list(VISER_GRIPPER_JOINT_NAMES)
            body_jp = joint_pos.tolist()
            if (viser_gripper_qpos_idx >= 0).all():
                gripper_jp = data.qpos[viser_gripper_qpos_idx].astype(np.float32).tolist()
            else:
                gripper_jp = [0.0] * len(VISER_GRIPPER_JOINT_NAMES)
            js.position = body_jp + gripper_jp
            ros_node.joint_state_pub.publish(js)

            if molmo_camera_bridge is not None and molmo_camera_bridge.enabled:
                molmo_camera_bridge.publish(data, stamp, step_count)

            if viewer is not None:
                ros_node._render_viewer_overlays(viewer)
                # Draw an arrow showing the perturbation force on
                # whichever body has the largest |F|. Built-in
                # mjVIS_PERTFORCE only renders the viewer's own
                # spring-drag, not direct xfrc_applied writes (which is
                # how the scripted perturbation works), so we draw our
                # own. Visible in the GL viewer for screen-recording.
                _xfrc = data.xfrc_applied
                if _xfrc.size > 0:
                    _fmags = np.linalg.norm(_xfrc[:, :3], axis=1)
                    _bid = int(np.argmax(_fmags))
                    _fmag = float(_fmags[_bid])
                    if _fmag > 1e-3:
                        with viewer.lock():
                            _scn = viewer.user_scn
                            if _scn.ngeom < _scn.maxgeom:
                                # Arrow geom: size = [shaft_radius,
                                # head_radius, length]. Length scales
                                # with force magnitude so a 100N push
                                # ≈ 10cm.
                                _arrow_len = max(0.05, _fmag * 1e-3)
                                _f_world = np.asarray(
                                    _xfrc[_bid, :3], dtype=np.float64
                                )
                                _f_unit = _f_world / max(_fmag, 1e-9)
                                # Rotation mapping arrow's local +z onto
                                # the force direction. mju_quatZ2Vec
                                # gives a stable quat for any direction.
                                _quat = np.zeros(4, dtype=np.float64)
                                mujoco.mju_quatZ2Vec(_quat, _f_unit)
                                _mat = np.zeros(9, dtype=np.float64)
                                mujoco.mju_quat2Mat(_mat, _quat)
                                _g = _scn.geoms[_scn.ngeom]
                                mujoco.mjv_initGeom(
                                    _g,
                                    mujoco.mjtGeom.mjGEOM_ARROW,
                                    [0.01, 0.02, _arrow_len],
                                    np.asarray(
                                        data.xpos[_bid], dtype=np.float64
                                    ),
                                    _mat,
                                    [1.0, 0.2, 0.2, 0.9],  # red
                                )
                                _scn.ngeom += 1

            if viser_bridge is not None:
                with ros_node._lock:
                    markers = list(ros_node._molmo_markers)
                    loco_path = ros_node._latest_loco_path
                    ee_left = ros_node._latest_ee_left
                    ee_right = ros_node._latest_ee_right
                    esdf_voxels = ros_node._latest_esdf_voxels
                    esdf_voxel_m = ros_node._latest_esdf_voxel_m
                    ffs_obb = ros_node._latest_ffs_obb
                # Drop OBB only after a long gap — tracker hiccups (empty
                # mask for a couple of frames, coast, the set_prompt round
                # trip) briefly pause the OBB stream, and a short stale
                # window made the wireframe disappear-and-reappear. 3 s is
                # well beyond the tracker's own EMPTY_MASK_LIMIT (15 frames
                # ≈ 1.5 s at 10 Hz) so we only clear once the tracker has
                # genuinely stopped.
                if ffs_obb is not None:
                    now_sec = time.time()
                    if now_sec - ffs_obb[3] > 3.0:
                        ffs_obb = None
                        with ros_node._lock:
                            ros_node._latest_ffs_obb = None
                if (viser_gripper_qpos_idx >= 0).all():
                    gripper_q = data.qpos[viser_gripper_qpos_idx].astype(np.float32)
                else:
                    gripper_q = None
                viser_bridge.publish_robot(root_pos, root_quat, joint_pos, gripper_pos=gripper_q)
                viser_bridge.publish_markers(markers)
                viser_bridge.publish_loco_path(loco_path)
                viser_bridge.publish_ee_path(ee_left, ee_right)
                viser_bridge.publish_esdf_voxels(esdf_voxels, esdf_voxel_m)
                if ffs_obb is None:
                    viser_bridge.publish_obb_world(None, None, None)
                else:
                    viser_bridge.publish_obb_world(ffs_obb[0], ffs_obb[1], ffs_obb[2])

                # Per-hand grasp-trigger box, drawn in the "horizontal grasp
                # frame": horizontal axes follow the gripper's world-
                # horizontal finger heading, vertical axis is gravity. So
                # the box stays upright relative to the ground even as the
                # wrist pitches/rolls during ee_tracked.
                if wrist_leveller is not None:
                    x_back = float(MOLMO_PICK_EARLY_CLOSE_X_BACK_M)
                    x_fwd = float(MOLMO_PICK_EARLY_CLOSE_X_FWD_M)
                    y_half = float(MOLMO_PICK_EARLY_CLOSE_Y_M)
                    offset_depth = 0.5 * (x_fwd - x_back)
                    # Per-side z half-extent: half the OBB world-up extent
                    # if FFS gave us one for this pick; else the constant
                    # fallback. Same logic _z_bound_for() uses.
                    def _viz_z_bound(hand: str) -> float:
                        obj_z = molmo_sub.pick_object_z_extent_for(hand)
                        if obj_z > 0.0:
                            return 0.75 * obj_z
                        return float(
                            MOLMO_PICK_EARLY_CLOSE_Z_SQUAT_M
                            if molmo_sub.is_squat_pick_for(hand)
                            else MOLMO_PICK_EARLY_CLOSE_Z_M
                        )
                    poses: dict[str, tuple[np.ndarray, np.ndarray] | None] = {
                        "l": None, "r": None,
                    }
                    z_half_per: dict[str, float] = {
                        "l": _viz_z_bound("l"),
                        "r": _viz_z_bound("r"),
                    }
                    mid_world: dict[str, np.ndarray | None] = {"l": None, "r": None}
                    for _hand in ("l", "r"):
                        try:
                            frame = wrist_leveller.horizontal_grasp_frame_world(
                                _hand, joint_pos, root_pos, root_quat,
                            )
                        except Exception:
                            frame = None
                        if frame is None:
                            continue
                        midpoint_w, R_grasp_w = frame
                        midpoint_w = np.asarray(midpoint_w, dtype=np.float64).reshape(3)
                        R_grasp_w = np.asarray(R_grasp_w, dtype=np.float64).reshape(3, 3)
                        centroid_w = midpoint_w + R_grasp_w @ np.array(
                            [offset_depth, 0.0, 0.0], dtype=np.float64,
                        )
                        wxyz = np.zeros(4, dtype=np.float64)
                        mujoco.mju_mat2Quat(wxyz, R_grasp_w.reshape(9))
                        poses[_hand] = (
                            centroid_w.astype(np.float32),
                            wxyz.astype(np.float32),
                        )
                        mid_world[_hand] = midpoint_w.astype(np.float32)
                    # Use the larger of the two side-specific z bounds for
                    # the wireframe (single shared dim — split per-hand
                    # later if both arms ever pick at the same time with
                    # different objects).
                    z_half_box = max(z_half_per["l"], z_half_per["r"])
                    viser_bridge.publish_grasp_boxes(
                        poses["l"], poses["r"],
                        (x_back + x_fwd, 2.0 * y_half, 2.0 * z_half_box),
                    )
                    viser_bridge.publish_grasp_midpoints(
                        mid_world["l"], mid_world["r"],
                    )
                if movable_body_ids:
                    viser_bridge.publish_objects({
                        name: (data.xpos[bid].copy(), data.xquat[bid].copy())
                        for name, bid in movable_body_ids.items()
                    })

                now = time.perf_counter()
                if now - last_status_t > 0.5:
                    last_status_t = now
                    z = float(root_pos[2])
                    viser_bridge.set_sim_status(f"policy OK · pelvis z={z:.2f} m [sync]")

            prev_sim_time = data.time
            step_count += 1
            if viewer is not None:
                viewer.sync()

    finally:
        if viser_bridge is not None:
            viser_bridge.stop()
        molmo_sub.shutdown()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
