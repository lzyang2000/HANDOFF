"""G1 sim2sim node — UDP-synchronous policy exchange.

MuJoCo physics runs locally.  The policy node runs in a separate process
and communicates via UDP (see deploy/common/udp_sync.py).  ROS 2 is used
only for controller commands (keyboard/xbox/molmo) and viewer overlays.
"""

import math
import os
import re
import signal
import socket
import sys
import time
import threading
from contextlib import nullcontext
from pathlib import Path

# Sim2sim runs the payload-enriched MJCF (jetson + dex1_1) so the MuJoCo model
# matches the payload URDF viser loads. `_apply_payloads_to_spec` is gated by
# this env var (read at import time in wbc_mjlab.g1_constants_custom), so it
# must be set *before* that module is imported below. `setdefault` preserves an
# explicit user override.
os.environ.setdefault("WBC_ATTACH_PAYLOADS", "1")

import numpy as np
import mujoco
import mujoco.viewer

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from nav_msgs.msg import Odometry, Path as PathMsg
from sensor_msgs.msg import CameraInfo, Image, JointState, PointCloud2, PointField
from std_msgs.msg import (
    Bool as BoolMsg,
    Float32 as Float32Msg,
    Float32MultiArray,
    Empty,
    String as StringMsg,
)
from visualization_msgs.msg import MarkerArray

from deploy.real.g1_robot_constants import (
    HEAD_CAMERA_SIM_FOVY_DEG,
    HEAD_CAMERA_SIM_RENDER_H,
    HEAD_CAMERA_SIM_RENDER_W,
)
from deploy.common.molmo_local_frame import (
    LOCAL_FRAME_ID,
    LocalFrameAnchor,
    local_points_to_world,
    local_xy_to_world_xy,
    make_identity_local_frame_anchor,
    make_local_frame_anchor,
)
from deploy.common.command import (
    CMD_LEFT_GRIPPER,
    CMD_RIGHT_GRIPPER,
    CMD_SIZE,
    COMMAND_TOPIC,
    make_command,
)
from deploy.common.udp_sync import (
    UDP_HOST, UDP_SIM_PORT, UDP_POLICY_PORT,
    ACTION_BYTES, pack_state, unpack_action, create_udp_socket,
)
from deploy.common.capture_point import CapturePointOverlay, unpack_capture_point_debug
from wbc_mjlab.g1_constants_custom import (
    GRIPPER_OPEN_CMD, GRIPPER_CLOSED_CMD, GRIPPER_PD_KP, GRIPPER_DAMPING_RATIO,
    MOLMO_DEFAULT_PROMPT, MOLMO_DEFAULT_PROMPT_ENV,
    FFS_OBB_POSE_TOPIC, FFS_OBB_EXTENT_TOPIC,
    wrap_spec_fn_with_payloads,
)
from mjlab.asset_zoo.robots import get_g1_robot_cfg, G1_ACTION_SCALE
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import KNEES_BENT_KEYFRAME
from mjlab.entity import Entity

# ---------------------------------------------------------------------------
# Environment variables
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
# Constants
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
NUM_ACTUATED = 15

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


def _resolve_keyframe(joint_names, keyframe):
    vals = np.zeros(len(joint_names), dtype=np.float32)
    for i, name in enumerate(joint_names):
        for pattern, v in keyframe.joint_pos.items():
            if re.fullmatch(pattern, name):
                vals[i] = v
                break
    return vals


def _resolve_scales(joint_names, n, scale_dict):
    scales = np.zeros(n, dtype=np.float32)
    for i in range(n):
        for pattern, scale in scale_dict.items():
            if re.match(pattern, joint_names[i]):
                scales[i] = scale
                break
    return scales


DEFAULT_POS = _resolve_keyframe(POLICY_JOINT_NAMES, KNEES_BENT_KEYFRAME)
JOINT_SCALES = _resolve_scales(POLICY_JOINT_NAMES, NUM_ACTUATED, G1_ACTION_SCALE)


# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------
def quat_rotate_inverse(q, v):
    w, x, y, z = q
    q_vec = np.array([x, y, z])
    a = v * (2.0 * w * w - 1.0)
    b = np.cross(q_vec, v) * w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c


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


# ---------------------------------------------------------------------------
# ROS 2 node — controller commands & viewer overlays only
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

        # Viser UI bridge — sim → molmo (button clicks / prompt submit).
        self._ui_prompt_pub = self.create_publisher(StringMsg, "/molmo/ui/prompt_submit", VIZ_QOS)
        self._ui_voice_toggle_pub = self.create_publisher(Empty, "/molmo/ui/voice_toggle", VIZ_QOS)
        self._ui_reset_pub = self.create_publisher(Empty, "/molmo/ui/reset", VIZ_QOS)

        # Viser UI bridge — molmo → sim (overlay / status / voice state / prompt echo).
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

        # Latest UI state (kept for resubmit + viser bind after the fact).
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

    # ---- viser UI plumbing ----
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
            markers = list(self._molmo_markers)
        with viewer.lock():
            viewer.user_scn.ngeom = 0
        if overlay is not None:
            overlay.render(viewer)
        if markers:
            with viewer.lock():
                scn = viewer.user_scn
                for mtype, pos in markers:
                    if scn.ngeom >= scn.maxgeom:
                        break
                    g = scn.geoms[scn.ngeom]
                    mujoco.mjv_initGeom(
                        g, mujoco.mjtGeom.mjGEOM_SPHERE,
                        [0.02, 0, 0], np.asarray(pos, dtype=np.float64),
                        np.eye(3).flatten(),
                        ([1, 0, 0, 0.8] if mtype == 0 else [0, 1, 0, 0.8]),
                    )
                    scn.ngeom += 1


# ---------------------------------------------------------------------------
# Molmo camera bridge (needs ROS for image topics)
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
                "rgb_renderer":   mujoco.Renderer(self._model, height=HEAD_CAMERA_SIM_RENDER_H, width=HEAD_CAMERA_SIM_RENDER_W),
                "depth_renderer": mujoco.Renderer(self._model, height=HEAD_CAMERA_SIM_RENDER_H, width=HEAD_CAMERA_SIM_RENDER_W),
            }
            self._entries[cam_name]["depth_renderer"].enable_depth_rendering()
        self._enabled = bool(self._entries)

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
            # cam_mat (cam→world) was built with.
            H, W = depth.shape
            cx = float(info.k[2])
            cy = float(info.k[5])
            fx_v = float(info.k[0])
            fy_v = float(info.k[4])
            us, vs = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
            z = depth
            x_cam = (us - cx) * z / fx_v
            y_cam = -(vs - cy) * z / fy_v
            z_cam = -z
            xyz = np.stack([x_cam, y_cam, z_cam], axis=-1)
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

            # Push to viser (first camera only — that's the one the UI shows).
            if self._viser_bridge is not None and cam_name == self._camera_names[0]:
                self._viser_bridge.push_rgb(rgb)


# ---------------------------------------------------------------------------
# MuJoCo model builder
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
            swap_hands=False,  # gripper XML already has its own end-effectors
        )
        robot_cfg.articulation = None
        robot_cfg.collisions = ()
        robot_cfg.init_state.joint_pos = None
        spec = Entity(robot_cfg).spec
        spec.option.timestep = 0.001  # XML has 0.005; match training rate
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
    model = build_model()
    data = mujoco.MjData(model)
    molmo_camera_bridge = None

    # Joint index maps
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

    # Gripper setup
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

    # qpos indices for all 4 viser gripper joints (may include the coupled _2_ joints).
    viser_gripper_qpos_idx = []
    for gn in VISER_GRIPPER_JOINT_NAMES:
        gj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, gn)
        viser_gripper_qpos_idx.append(model.jnt_qposadr[gj] if gj >= 0 else -1)
    viser_gripper_qpos_idx = np.array(viser_gripper_qpos_idx, dtype=np.int64)

    waist_yaw_actuator_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "waist_yaw_joint")

    # Initial pose
    _init_yaw = math.radians(float(os.environ.get("WBC_INIT_YAW_DEG", "0")))
    data.qpos[2] = 0.76
    data.qpos[3] = math.cos(_init_yaw / 2)
    data.qpos[6] = math.sin(_init_yaw / 2)
    for i, qi in enumerate(qpos_idx):
        data.qpos[qi] = DEFAULT_POS[i]
    data.ctrl[ctrl_idx] = DEFAULT_POS
    mujoco.mj_forward(model, data)

    # ---- ROS 2 (background, for controllers only) ----
    rclpy.init()
    ros_node = G1SimRosNode()

    # Signalled by the viser "Reset sim" button; drained by the main loop.
    reset_event = threading.Event()

    # ---- Viser bridge (default on; set WBC_MJLAB_VISER_PORT="" to disable) ----
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

    # Resolve body ids for movable MJCF bodies registered in viser (e.g. cubes).
    movable_body_ids: dict[str, int] = {}
    for name in movable_bodies:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            movable_body_ids[name] = bid

    if os.environ.get(MOLMO_CAMERA_NAMES_ENV, "").strip():
        molmo_camera_bridge = MolmoCameraBridge(ros_node, model, viser_bridge=viser_bridge)
    ros_thread = threading.Thread(
        target=lambda: rclpy.spin(ros_node), daemon=True
    )
    ros_thread.start()

    # ---- UDP socket ----
    udp_sock = create_udp_socket(UDP_HOST, UDP_SIM_PORT)
    udp_sock.setblocking(False)
    policy_addr = (UDP_HOST, UDP_POLICY_PORT)
    print(f"UDP: sim={UDP_HOST}:{UDP_SIM_PORT} → policy={UDP_HOST}:{UDP_POLICY_PORT}")

    step_count = 0
    control_dt = model.opt.timestep * DECIMATION

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
        print("Simulation reset.")

    # ---- Viewer + real-time loop ----
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
        f"Launching: mujoco={'on' if mujoco_viewer_enabled else 'off'} "
        f"viser={'http://localhost:' + viser_port_str if viser_enabled else 'off'}"
    )
    try:
      with viewer_cm as viewer:
        start_wall = time.perf_counter() - data.time
        prev_sim_time = data.time
        policy_seen = False
        no_policy_warned = False
        no_policy_warn_deadline = time.perf_counter() + 1.0
        last_action_t = 0.0
        last_status_t = 0.0

        while (
            not shutdown_event.is_set()
            and (viewer is None or viewer.is_running())
            and rclpy.ok()
        ):
            # Real-time pacing
            target_wall = start_wall + data.time + control_dt
            sleep_time = target_wall - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)

            # External reset (viser "Reset sim" button)
            if reset_event.is_set():
                reset_event.clear()
                _do_reset()
                step_count = 0
                start_wall = time.perf_counter() - data.time

            # Detect GUI reset
            if data.time < prev_sim_time - control_dt * 0.5:
                _do_reset()
                step_count = 0
                start_wall = time.perf_counter() - data.time

            # --- Recv action via UDP (gated: only apply if step matches) ---
            latest_action = None
            try:
                while True:
                    latest_action, _ = udp_sock.recvfrom(ACTION_BYTES + 64)
            except BlockingIOError:
                pass
            if latest_action is not None:
                _, target_pos = unpack_action(latest_action)
                data.ctrl[ctrl_idx] = target_pos
                policy_seen = True
                last_action_t = time.perf_counter()

            if (
                viser_enabled
                and not policy_seen
                and not no_policy_warned
                and time.perf_counter() > no_policy_warn_deadline
            ):
                print(
                    "[sim_node] No policy action received after 1s — the robot "
                    "will fall under gravity. Click 'Reset sim' in the viser UI "
                    "to recover; launch a *_policy.py for closed-loop control.",
                    flush=True,
                )
                no_policy_warned = True

            # --- Gripper PD control ---
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

            # --- Waist yaw override ---
            if waist_yaw_actuator_idx >= 0:
                with ros_node._lock:
                    wyaw = ros_node._waist_yaw_override
                if abs(wyaw) > 1e-4:
                    data.ctrl[waist_yaw_actuator_idx] = wyaw

            # --- Step physics ---
            for _ in range(DECIMATION):
                mujoco.mj_step(model, data)

            # --- Pack & send state via UDP (post-step) ---
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
            with ros_node._lock:
                cmd = ros_node._latest_command.copy()

            udp_sock.sendto(
                pack_state(step_count, root_quat, root_pos,
                           body_lin_vel, body_ang_vel,
                           joint_pos, joint_vel, cmd),
                policy_addr,
            )

            # --- Odom (pelvis pose + body-frame twist) ---
            stamp = ros_node.get_clock().now().to_msg()
            odom = Odometry()
            odom.header.stamp = stamp
            odom.header.frame_id = "world"
            odom.child_frame_id = "pelvis"
            odom.pose.pose.position.x = float(root_pos[0])
            odom.pose.pose.position.y = float(root_pos[1])
            odom.pose.pose.position.z = float(root_pos[2])
            odom.pose.pose.orientation.x = float(root_quat[1])  # wxyz → xyzw
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

            # --- Joint state (29 body joints + 4 gripper joints) ---
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

            # --- Molmo camera (ROS, low-frequency) ---
            if molmo_camera_bridge is not None and molmo_camera_bridge.enabled:
                molmo_camera_bridge.publish(data, stamp, step_count)

            # --- Viewer overlays (MuJoCo GL viewer only) ---
            if viewer is not None:
                ros_node._render_viewer_overlays(viewer)

            # --- Viser publish (robot + marker spheres + sim status) ---
            if viser_bridge is not None:
                with ros_node._lock:
                    markers = list(ros_node._molmo_markers)
                    loco_path = ros_node._latest_loco_path
                    ee_left = ros_node._latest_ee_left
                    ee_right = ros_node._latest_ee_right
                    esdf_voxels = ros_node._latest_esdf_voxels
                    esdf_voxel_m = ros_node._latest_esdf_voxel_m
                    ffs_obb = ros_node._latest_ffs_obb
                # Drop OBB only after a long gap so tracker hiccups (empty
                # mask for a couple of frames, coast, set_prompt round trip)
                # don't make the wireframe disappear-and-reappear. 3 s is
                # well beyond the tracker's own EMPTY_MASK_LIMIT (~1.5 s).
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
                if movable_body_ids:
                    viser_bridge.publish_objects({
                        name: (data.xpos[bid].copy(), data.xquat[bid].copy())
                        for name, bid in movable_body_ids.items()
                    })

                now = time.perf_counter()
                if now - last_status_t > 0.5:
                    last_status_t = now
                    if last_action_t == 0.0:
                        policy_line = "**policy MISSING** — no UDP actions received; robot will fall"
                    elif now - last_action_t > 0.5:
                        policy_line = f"**policy STALE** — last action {now - last_action_t:.1f}s ago"
                    else:
                        policy_line = "policy OK"
                    z = float(root_pos[2])
                    viser_bridge.set_sim_status(f"{policy_line} · pelvis z={z:.2f} m")

            prev_sim_time = data.time
            step_count += 1
            if viewer is not None:
                viewer.sync()

    finally:
        if viser_bridge is not None:
            viser_bridge.stop()
        udp_sock.close()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
