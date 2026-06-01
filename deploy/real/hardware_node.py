"""G1 real-hardware node — drop-in replacement for sim_node.py.

Reads IMU and joint state from the G1 via unitree_interface, sends it to
the policy node over UDP, receives joint-position targets, and applies them
to the motors.  Same UDP protocol and ROS command subscription as sim_node.

Startup sequence (mirrors g1_wrapper.py from HANDOFF):
  1. Press START on the wireless remote  →  interpolate to default pose (2 s)
  2. Press A                             →  enter 50 Hz policy loop
  3. Press B during loop                 →  graceful stop + damp

Usage:
    uv run python deploy/real/hardware_node.py [--net eth0] [--policy-ip 127.0.0.1]
"""

import argparse
import re
import signal
import threading
import time

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty, Float32MultiArray
from std_srvs.srv import Trigger

import unitree_interface
from deploy.common.command import (
    CMD_LEFT_GRIPPER,
    CMD_RIGHT_GRIPPER,
    CMD_SIZE,
    COMMAND_TOPIC,
    make_command,
)
from wbc_mjlab.g1_constants_custom import (
    DEX1_KD,
    DEX1_KP,
    DEX1_Q_CLOSED,
    DEX1_Q_OPEN,
    DEX1_STROKE_M,
    DEX1_TAU_LIMIT,
    GRIPPER_CLOSED_CMD,
    GRIPPER_OPEN_CMD,
)
from deploy.real.g1_robot_constants import (
    HEAD_CAMERA_NAME,
    HEAD_CAMERA_POS_IN_PELVIS,
    HEAD_CAMERA_RPY_IN_PELVIS,
    HEAD_CAMERA_PUBLISH_HZ,
    HEAD_CAMERA_ZED_RESOLUTION,
    HEAD_CAMERA_ZED_FPS,
    HEAD_CAMERA_ZED_DEPTH_MODE,
    HEAD_CAMERA_ZED_MIN_DEPTH_M,
    HEAD_CAMERA_ZED_MAX_DEPTH_M,
    HEAD_CAMERA_ZED_CONFIDENCE_THRESHOLD,
    HEAD_CAMERA_ZED_AEC_AGC_ENABLED,
    HEAD_CAMERA_ZED_AEC_AGC_ROI,
    HEAD_CAMERA_ZED_EXPOSURE,
    HEAD_CAMERA_ZED_GAIN,
)

try:
    from deploy.real.zed_bridge import ZedBridge
    _ZED_AVAILABLE = True
except ImportError as _zed_import_exc:
    print(
        f"[WARN] ZED bridge unavailable ({_zed_import_exc}); "
        "camera topics will be disabled. Install the ZED SDK and the pyzed "
        "Python bindings (https://www.stereolabs.com/docs/app-development/python/install) "
        "to enable the head camera."
    )
    _ZED_AVAILABLE = False

from deploy.real.pelvis_camera_fk import PelvisCameraFK

# Wireless controller button map (mirrors HANDOFF g1_wrapper.py ContollerMapping).
# WirelessController.keys is a raw bitmask; use btn(ctrl, "name") to check.
CONTROLLER_MAPPING = {
    "R1": 0x0001, "L1": 0x0002, "start": 0x0004, "select": 0x0008,
    "R2": 0x0010, "L2": 0x0020, "F1":    0x0040, "F2":     0x0080,
    "A":  0x0100, "B":  0x0200, "X":     0x0400, "Y":      0x0800,
    "up": 0x1000, "right": 0x2000, "down": 0x4000, "left":  0x8000,
}


def btn(ctrl, name: str) -> bool:
    return bool(ctrl.keys & CONTROLLER_MAPPING[name])
from deploy.common.udp_sync import (
    ACTION_BYTES, UDP_HOST, UDP_POLICY_PORT, UDP_SIM_PORT,
    create_udp_socket, pack_state, unpack_action,
)
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
    KNEES_BENT_KEYFRAME,
    STIFFNESS_5020, DAMPING_5020,
    STIFFNESS_7520_14, DAMPING_7520_14,
    STIFFNESS_7520_22, DAMPING_7520_22,
    STIFFNESS_4010, DAMPING_4010,
)

# ---------------------------------------------------------------------------
# Constants (mirrors sim_node.py)
# ---------------------------------------------------------------------------
POLICY_JOINT_NAMES = [
    "left_hip_pitch_joint",    "left_hip_roll_joint",    "left_hip_yaw_joint",
    "left_knee_joint",         "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint",   "right_hip_roll_joint",   "right_hip_yaw_joint",
    "right_knee_joint",        "right_ankle_pitch_joint","right_ankle_roll_joint",
    "waist_yaw_joint",         "waist_roll_joint",       "waist_pitch_joint",
    "left_shoulder_pitch_joint","left_shoulder_roll_joint","left_shoulder_yaw_joint",
    "left_elbow_joint",        "left_wrist_roll_joint",  "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint",
    "right_elbow_joint",       "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
NUM_JOINTS = len(POLICY_JOINT_NAMES)   # 29
CONTROL_DT = 0.02                      # 50 Hz

# RELIABLE so rviz (which defaults to RELIABLE) can subscribe. BEST_EFFORT
# publishers are silently dropped by RELIABLE subscribers.
VIZ_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST, depth=5,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)


def _quat_wxyz_to_mat(wxyz) -> np.ndarray:
    """Unit quaternion (w, x, y, z) → 3×3 rotation matrix (float64)."""
    w, x, y, z = (float(v) for v in wxyz)
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def _yaw_from_quat_wxyz(wxyz) -> float:
    """Extract Z-axis yaw (rad) from a unit quaternion in (w, x, y, z) order."""
    w, x, y, z = (float(v) for v in wxyz)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _apply_yaw_offset_quat_wxyz(q_wxyz, yaw_offset) -> np.ndarray:
    """Return Rz(-yaw_offset) * q_wxyz as (w, x, y, z)."""
    half = -0.5 * float(yaw_offset)
    qw_o, qz_o = float(np.cos(half)), float(np.sin(half))
    qw_r = float(q_wxyz[0]); qx_r = float(q_wxyz[1])
    qy_r = float(q_wxyz[2]); qz_r = float(q_wxyz[3])
    # (qw_o, 0, 0, qz_o) * (qw_r, qx_r, qy_r, qz_r) in wxyz order.
    return np.array([
        qw_o * qw_r - qz_o * qz_r,
        qw_o * qx_r - qz_o * qy_r,
        qw_o * qy_r + qz_o * qx_r,
        qw_o * qz_r + qz_o * qw_r,
    ], dtype=np.float64)


def _apply_yaw_offset_xy(x, y, yaw_offset):
    """Rotate the (x, y) pair by -yaw_offset about world Z."""
    cy, sy = float(np.cos(-yaw_offset)), float(np.sin(-yaw_offset))
    return cy * x - sy * y, sy * x + cy * y


def _rpy_deg_to_mat(rpy_deg) -> np.ndarray:
    """(roll, pitch, yaw) in degrees, ZYX intrinsic → 3×3 rotation matrix."""
    r, p, y = np.radians(rpy_deg)
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx

# ---------------------------------------------------------------------------
# PD gains — mjlab gains per POLICY_JOINT_NAMES order
#
# Motor type per joint index:
#   7520_14 : hip_pitch (0,6), hip_yaw (2,8), waist_yaw (12)
#   7520_22 : hip_roll (1,7), knee (3,9)
#   2×5020  : ankle (4,5,10,11), waist_roll/pitch (13,14)
#   5020    : shoulder (15-17,22-24), elbow (18,25), wrist_roll (19,26)
#   4010    : wrist_pitch (20,27), wrist_yaw (21,28)
# ---------------------------------------------------------------------------
_KP = np.array([
    STIFFNESS_7520_14,   # 0  left_hip_pitch
    STIFFNESS_7520_22,   # 1  left_hip_roll
    STIFFNESS_7520_14,   # 2  left_hip_yaw
    STIFFNESS_7520_22,   # 3  left_knee
    2*STIFFNESS_5020,    # 4  left_ankle_pitch
    2*STIFFNESS_5020,    # 5  left_ankle_roll
    STIFFNESS_7520_14,   # 6  right_hip_pitch
    STIFFNESS_7520_22,   # 7  right_hip_roll
    STIFFNESS_7520_14,   # 8  right_hip_yaw
    STIFFNESS_7520_22,   # 9  right_knee
    2*STIFFNESS_5020,    # 10 right_ankle_pitch
    2*STIFFNESS_5020,    # 11 right_ankle_roll
    STIFFNESS_7520_14,   # 12 waist_yaw
    2*STIFFNESS_5020,    # 13 waist_roll
    2*STIFFNESS_5020,    # 14 waist_pitch
    STIFFNESS_5020,      # 15 left_shoulder_pitch
    STIFFNESS_5020,      # 16 left_shoulder_roll
    STIFFNESS_5020,      # 17 left_shoulder_yaw
    STIFFNESS_5020,      # 18 left_elbow
    STIFFNESS_5020,      # 19 left_wrist_roll
    STIFFNESS_4010,      # 20 left_wrist_pitch
    STIFFNESS_4010,      # 21 left_wrist_yaw
    STIFFNESS_5020,      # 22 right_shoulder_pitch
    STIFFNESS_5020,      # 23 right_shoulder_roll
    STIFFNESS_5020,      # 24 right_shoulder_yaw
    STIFFNESS_5020,      # 25 right_elbow
    STIFFNESS_5020,      # 26 right_wrist_roll
    STIFFNESS_4010,      # 27 right_wrist_pitch
    STIFFNESS_4010,      # 28 right_wrist_yaw
], dtype=np.float64)

_KD = np.array([
    DAMPING_7520_14,     # 0  left_hip_pitch
    DAMPING_7520_22,     # 1  left_hip_roll
    DAMPING_7520_14,     # 2  left_hip_yaw
    DAMPING_7520_22,     # 3  left_knee
    2*DAMPING_5020,      # 4  left_ankle_pitch
    2*DAMPING_5020,      # 5  left_ankle_roll
    DAMPING_7520_14,     # 6  right_hip_pitch
    DAMPING_7520_22,     # 7  right_hip_roll
    DAMPING_7520_14,     # 8  right_hip_yaw
    DAMPING_7520_22,     # 9  right_knee
    2*DAMPING_5020,      # 10 right_ankle_pitch
    2*DAMPING_5020,      # 11 right_ankle_roll
    DAMPING_7520_14,     # 12 waist_yaw
    2*DAMPING_5020,      # 13 waist_roll
    2*DAMPING_5020,      # 14 waist_pitch
    DAMPING_5020,        # 15 left_shoulder_pitch
    DAMPING_5020,        # 16 left_shoulder_roll
    DAMPING_5020,        # 17 left_shoulder_yaw
    DAMPING_5020,        # 18 left_elbow
    DAMPING_5020,        # 19 left_wrist_roll
    DAMPING_4010,        # 20 left_wrist_pitch
    DAMPING_4010,        # 21 left_wrist_yaw
    DAMPING_5020,        # 22 right_shoulder_pitch
    DAMPING_5020,        # 23 right_shoulder_roll
    DAMPING_5020,        # 24 right_shoulder_yaw
    DAMPING_5020,        # 25 right_elbow
    DAMPING_5020,        # 26 right_wrist_roll
    DAMPING_4010,        # 27 right_wrist_pitch
    DAMPING_4010,        # 28 right_wrist_yaw
], dtype=np.float64)

# ---------------------------------------------------------------------------
# Default pose (same source as sim_node.py)
# ---------------------------------------------------------------------------
def _resolve_keyframe(joint_names, keyframe):
    vals = np.zeros(len(joint_names), dtype=np.float32)
    for i, name in enumerate(joint_names):
        for pattern, v in keyframe.joint_pos.items():
            if re.fullmatch(pattern, name):
                vals[i] = v
                break
    return vals


DEFAULT_POS = _resolve_keyframe(POLICY_JOINT_NAMES, KNEES_BENT_KEYFRAME)


# ---------------------------------------------------------------------------
# Dex1-1 gripper helpers
#
# The command-bus scalar at CMD_LEFT_GRIPPER / CMD_RIGHT_GRIPPER is a unitless
# intent in [GRIPPER_OPEN_CMD, GRIPPER_CLOSED_CMD] (= [0, 1]). Both the policy
# (for its wrist-leveler gate) and this node consume the same field, so the
# gate stays in sync with the gripper without a feedback loop. We scale it to
# the calibrated motor-side radian range, then send it to the dex1_1_service
# bridge over rt/dex1/{left,right}/cmd.
# ---------------------------------------------------------------------------
_GRIPPER_CMD_SPAN = max(GRIPPER_CLOSED_CMD - GRIPPER_OPEN_CMD, 1e-6)
_DEX1_Q_SPAN = DEX1_Q_CLOSED - DEX1_Q_OPEN


def _gripper_cmd_to_motor_q(cmd: float) -> float:
    """Map intent scalar in [open_cmd, closed_cmd] to M4010 motor angle (rad)."""
    t = (float(cmd) - GRIPPER_OPEN_CMD) / _GRIPPER_CMD_SPAN
    t = max(0.0, min(1.0, t))
    return DEX1_Q_OPEN + t * _DEX1_Q_SPAN


def _motor_q_to_prismatic_m(q: float) -> float:
    """Map motor angle to one prismatic-jaw position (m). Each side has two
    parallel prismatic joints (URDF gripper_prismatic_{1,2}_{L,R}) that share
    half the total stroke each — symmetric parallel-jaw kinematics. URDF
    convention is prismatic=0 → jaws together (closed), so map motor CLOSED
    to 0 and motor OPEN to half-stroke."""
    if abs(_DEX1_Q_SPAN) < 1e-6:
        return 0.0
    t = (float(q) - DEX1_Q_OPEN) / _DEX1_Q_SPAN
    t = max(0.0, min(1.0, t))
    return 0.5 * DEX1_STROKE_M * (1.0 - t)


def _make_dex1_command(cmd: float) -> "unitree_interface.Dex1Command":
    """Build a Dex1Command in position mode at the scaled motor target."""
    msg = unitree_interface.Dex1Command()
    msg.mode = 1
    msg.q_target = _gripper_cmd_to_motor_q(cmd)
    msg.dq_target = 0.0
    msg.tau_ff = 0.0
    msg.kp = DEX1_KP
    msg.kd = DEX1_KD
    return msg


# ---------------------------------------------------------------------------
# ROS 2 node — command subscription only
# ---------------------------------------------------------------------------
class G1HardwareRosNode(Node):
    def __init__(self):
        super().__init__("g1_hardware_node")
        self.create_subscription(
            Float32MultiArray, COMMAND_TOPIC, self._command_cb, VIZ_QOS
        )
        self.joint_state_pub = self.create_publisher(JointState, "/g1/joint_states", VIZ_QOS)
        self.odom_pub = self.create_publisher(Odometry, "/g1/odom", VIZ_QOS)
        # Cleared on reset so consumers (planner_node) drop ESDF voxels that
        # were integrated in the pre-reset world frame and would otherwise
        # ghost into the new origin.
        self.planner_reset_pub = self.create_publisher(Empty, "/planner/reset", VIZ_QOS)
        self._lock = threading.Lock()
        self._latest_command = make_command()

        # World re-zero: the service fires on every reanchoring Molmo query
        # (prompt / search / approach_requery) and once at policy-mode entry.
        # The handler signals the 50 Hz loop via _reset_pending and waits on
        # _reset_done — the loop owns _origin_pos / _origin_yaw_imu /
        # _origin_yaw_zed to avoid racing the publish path. See the world
        # re-zero plan for the full sequence (zed_bridge.reset_world_tracking
        # → settle → capture origin → /planner/reset publish → success ack).
        self._reset_pending = threading.Event()
        self._reset_done = threading.Event()
        self.create_service(
            Trigger,
            "/g1/reset_world_origin",
            self._reset_world_origin_handler,
            callback_group=ReentrantCallbackGroup(),
        )

        # Virtual-button latches for the viser UI. Each topic is an Empty
        # press that the main thread consumes (edge-triggered) to mirror the
        # physical START / A / SELECT buttons on the wireless remote.
        self._virtual_lock = threading.Lock()
        self._virtual_buttons = {"start": False, "A": False, "select": False}
        self.create_subscription(
            Empty, "/g1/ui/btn_start",
            lambda _msg: self._latch_virtual("start"), VIZ_QOS,
        )
        self.create_subscription(
            Empty, "/g1/ui/btn_a",
            lambda _msg: self._latch_virtual("A"), VIZ_QOS,
        )
        self.create_subscription(
            Empty, "/g1/ui/btn_select",
            lambda _msg: self._latch_virtual("select"), VIZ_QOS,
        )

    def _command_cb(self, msg):
        data = np.array(msg.data, dtype=np.float32)
        if len(data) != CMD_SIZE:
            return
        with self._lock:
            self._latest_command = data.copy()

    @property
    def command(self):
        with self._lock:
            return self._latest_command.copy()

    def _latch_virtual(self, name: str) -> None:
        with self._virtual_lock:
            self._virtual_buttons[name] = True
        print(f"[virtual_button] {name} pressed via UI")

    def consume_virtual_button(self, name: str) -> bool:
        with self._virtual_lock:
            pressed = self._virtual_buttons.get(name, False)
            self._virtual_buttons[name] = False
        return pressed

    def _reset_world_origin_handler(self, request, response):
        """Service callback: kick the loop and wait for it to commit a new origin.

        The 50 Hz loop calls zed_bridge.reset_world_tracking(), waits for the
        ZED to repopulate its cached pose, captures pelvis pos + IMU yaw as
        the new origin, publishes /planner/reset, and signals _reset_done.
        """
        del request  # std_srvs/Trigger has no fields
        self._reset_done.clear()
        self._reset_pending.set()
        if not self._reset_done.wait(timeout=2.0):
            response.success = False
            response.message = "world re-zero timed out (ZED not settling?)"
            return response
        response.success = True
        response.message = "world re-zeroed"
        return response


# ---------------------------------------------------------------------------
# Motor helpers
# ---------------------------------------------------------------------------
def _send_pd(robot, q_target, kp, kd):
    cmd = robot.create_zero_command()
    cmd.q_target = list(q_target)
    cmd.dq_target = [0.0] * NUM_JOINTS
    cmd.kp = list(kp)
    cmd.kd = list(kd)
    cmd.tau_ff = [0.0] * NUM_JOINTS
    robot.write_low_command(cmd)


# ---------------------------------------------------------------------------
# Startup sequence (mirrors g1_wrapper.py remote logic)
# ---------------------------------------------------------------------------
def startup(robot, ros_node):
    # Step 1: damp and wait for START to trigger stand-up
    print("Press START on the wireless remote (or the viser button) "
          "to move to default position ...")
    while True:
        ctrl = robot.read_wireless_controller()
        if btn(ctrl, "start") or ros_node.consume_virtual_button("start"):
            break
        state = robot.read_low_state()
        current = np.array(state.motor.q[:NUM_JOINTS], dtype=np.float32)
        _send_pd(robot, current, np.zeros(NUM_JOINTS), _KD)
        time.sleep(CONTROL_DT)

    # Step 2: 2-second linear interpolation to DEFAULT_POS
    print("Moving to default position ...")
    state = robot.read_low_state()
    q_start = np.array(state.motor.q[:NUM_JOINTS], dtype=np.float32)
    n_steps = int(2.0 / CONTROL_DT)
    for i in range(n_steps):
        alpha = (i + 1) / n_steps
        q_target = (1.0 - alpha) * q_start + alpha * DEFAULT_POS
        _send_pd(robot, q_target, _KP, _KD)
        time.sleep(CONTROL_DT)
    print("Default position reached.")

    # Step 3: hold default, wait for A to start policy
    print("Press A (or the viser button) to start the policy loop ...")
    while True:
        ctrl = robot.read_wireless_controller()
        if btn(ctrl, "A") or ros_node.consume_virtual_button("A"):
            break
        _send_pd(robot, DEFAULT_POS, _KP, _KD)
        time.sleep(CONTROL_DT)
    print("Starting policy loop.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="G1 real-hardware node")
    parser.add_argument("--net", default="eth0",
                        help="Network interface for robot DDS (default: eth0)")
    parser.add_argument("--policy-ip", default=UDP_HOST,
                        help="IP of the policy node (default: 127.0.0.1)")
    parser.add_argument("--dex1-tau-limit", type=float, default=DEX1_TAU_LIMIT,
                        help=f"Gripper force limit N·m — hold position when |tau_est| exceeds this (default: {DEX1_TAU_LIMIT})")
    args = parser.parse_args()

    # Convert SIGTERM (sent by play_real_hand.sh's cleanup pkill) into a
    # KeyboardInterrupt so the finally block runs and parks the grippers
    # closed before we exit. SIGINT already does this by default.
    def _on_sigterm(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _on_sigterm)

    robot = unitree_interface.UnitreeInterface.create_g1(args.net)
    # Release the G1's high-level motion control service (loco/sport) before
    # we start writing LowCmd. Without this, the active high-level service
    # fights our writes and the robot ignores them — same release pattern
    # as g1_ankle_swing_example.cpp in unitree_sdk2.
    if not robot.release_motion_control():
        print("[WARN] Failed to release high-level motion control within timeout; "
              "LowCmd writes may be ignored.")
    robot.set_control_mode(unitree_interface.ControlMode.PR)

    # Dex1-1 grippers. create_g1 already initialized the DDS factory, so we
    # pass re_init=False to avoid double-init. The dex1_1_gripper_server
    # binary must be running separately for these to do anything.
    try:
        dex1_left, dex1_right = unitree_interface.create_dex1_pair(args.net, re_init=False)
        # Park both grippers in the open state immediately. The writer
        # thread inside Dex1Interface will keep republishing this until we
        # update the buffer in the main loop.
        close_cmd = _make_dex1_command(GRIPPER_CLOSED_CMD)
        dex1_left.write_command(close_cmd)
        dex1_right.write_command(close_cmd)
        print("Dex1-1 grippers initialized; sent CLOSE command on both sides.")
    except Exception as exc:
        print(f"[WARN] Dex1-1 init failed ({exc}); gripper control disabled.")
        dex1_left = None
        dex1_right = None

    # ROS 2 spins in background for command subscription
    rclpy.init()
    ros_node = G1HardwareRosNode()
    ros_thread = threading.Thread(target=lambda: rclpy.spin(ros_node), daemon=True)
    ros_thread.start()

    # ZED bridge — optional; node runs normally without pyzed or camera.
    zed_bridge = None
    if _ZED_AVAILABLE:
        try:
            zed_bridge = ZedBridge(
                ros_node,
                HEAD_CAMERA_NAME,
                HEAD_CAMERA_PUBLISH_HZ,
                HEAD_CAMERA_ZED_RESOLUTION,
                HEAD_CAMERA_ZED_FPS,
                HEAD_CAMERA_ZED_DEPTH_MODE,
                HEAD_CAMERA_ZED_MIN_DEPTH_M,
                HEAD_CAMERA_ZED_MAX_DEPTH_M,
                HEAD_CAMERA_ZED_CONFIDENCE_THRESHOLD,
                aec_agc_enabled=HEAD_CAMERA_ZED_AEC_AGC_ENABLED,
                aec_agc_roi=HEAD_CAMERA_ZED_AEC_AGC_ROI,
                exposure=HEAD_CAMERA_ZED_EXPOSURE,
                gain=HEAD_CAMERA_ZED_GAIN,
            )
            print("ZED Mini initialised.")
        except Exception as exc:
            print(f"[WARN] ZED unavailable ({exc}); camera topics disabled.")
            zed_bridge = None

    startup(robot, ros_node)

    # Start ZED grab thread *after* the startup button-gate so the ZED is
    # not holding the USB bus during the interactive remote sequence.
    if zed_bridge is not None:
        zed_bridge.start()
        print("ZED grab thread started.")

    # Camera→pelvis transform. The mount constants are baked at neutral
    # waist, but the real kinematic chain is
    #   pelvis → waist_yaw → waist_roll → torso → (head_camera_mount) → lens
    # so PelvisCameraFK runs the waist-chain FK each tick and composes the
    # truly-static T_torso_camera (derived once from the constants) to get a
    # live camera→pelvis. Neutral fallback kept for when pinocchio/URDF is
    # unavailable.
    _R_pc = _rpy_deg_to_mat(HEAD_CAMERA_RPY_IN_PELVIS)
    _t_pc = np.array(HEAD_CAMERA_POS_IN_PELVIS, dtype=np.float64)
    _t_cp_neutral = -_R_pc.T @ _t_pc
    try:
        pelvis_camera_fk = PelvisCameraFK(
            HEAD_CAMERA_POS_IN_PELVIS,
            HEAD_CAMERA_RPY_IN_PELVIS,
            POLICY_JOINT_NAMES,
        )
        print("PelvisCameraFK initialised (live waist-chain correction).")
    except Exception as exc:
        print(f"[WARN] PelvisCameraFK unavailable ({exc}); using neutral-waist fallback.")
        pelvis_camera_fk = None

    # Velocity EMA state (world positions → body-frame linear velocity).
    _prev_t_wp: np.ndarray | None = None
    _prev_odom_t: float | None = None
    _body_lin_vel = np.zeros(3, dtype=np.float64)
    _VEL_ALPHA = 0.2  # EMA weight; ~5-sample time constant at 50 Hz

    # World re-zero state (loop-owned). Set after a /g1/reset_world_origin
    # service fires. Two distinct yaw offsets are required because /g1/odom
    # mixes sensors:
    #   - Orientation comes from the IMU → subtract IMU yaw at reset.
    #   - Position is built from ZED translation (t_wc + R_wc @ _t_cp_live)
    #     in ZED's world frame → subtract ZED yaw at reset.
    # Using one offset (e.g. IMU yaw for both) leaves a static ZED-vs-IMU
    # yaw mismatch baked into the published pose — the very bug this fix
    # exists to remove. zed_bridge.set_world_origin gets _origin_yaw_zed
    # (its translation and orientation both originate in ZED world).
    _origin_pos: np.ndarray | None = None
    _origin_yaw_imu: float | None = None
    _origin_yaw_zed: float | None = None
    # After reset_world_tracking() the ZED's cached pose is None until the
    # next grab repopulates it; capture origin only once it returns again.
    _capture_origin_next: bool = False

    # Policy-entry self-trigger: enter the loop immediately and fire the
    # first reset after the policy has been running for 5 s. Running the
    # policy first (rather than sleeping at default pose) gives the IMU
    # bias filter a representative motion sample and lets the PD controllers
    # converge under the policy's actual command stream — the world frame
    # then snaps to whatever the IMU+ZED report at that moment.
    _INITIAL_RESET_AFTER_SEC = 5.0
    _initial_reset_fired: bool = False

    udp_sock = create_udp_socket(UDP_HOST, UDP_SIM_PORT)
    udp_sock.setblocking(False)
    policy_addr = (args.policy_ip, UDP_POLICY_PORT)
    print(f"UDP: hardware={UDP_HOST}:{UDP_SIM_PORT}  policy={args.policy_ip}:{UDP_POLICY_PORT}")

    zeros3 = np.zeros(3, dtype=np.float32)
    last_target = DEFAULT_POS.copy()
    step_count = 0
    _loop_start_t = time.perf_counter()

    try:
        while rclpy.ok():
            t0 = time.perf_counter()

            # One-shot policy-entry world re-zero: fire after the policy has
            # been driving the robot for _INITIAL_RESET_AFTER_SEC. Running
            # the policy first lets the IMU bias filter and PD controllers
            # settle under the actual command stream before the world frame
            # is anchored.
            if (
                not _initial_reset_fired
                and (t0 - _loop_start_t) >= _INITIAL_RESET_AFTER_SEC
            ):
                ros_node._reset_pending.set()
                _initial_reset_fired = True
                print(
                    f"[hardware] Policy ran {_INITIAL_RESET_AFTER_SEC:.1f} s; "
                    f"firing initial world re-zero."
                )

            # Read hardware state
            state = robot.read_low_state()
            quat    = np.array(state.imu.quat,       dtype=np.float32)  # wxyz
            ang_vel = np.array(state.imu.omega,       dtype=np.float32)  # body frame
            jpos    = np.array(state.motor.q[:NUM_JOINTS],  dtype=np.float32)
            jvel    = np.array(state.motor.dq[:NUM_JOINTS], dtype=np.float32)

            # Publish joint state for downstream consumers (planner_node
            # self-mask, rviz). Schema matches sim_node: 29 body joints + 4
            # gripper joints. When the Dex1-1 service is up, surface the
            # live motor angles mapped onto the URDF parallel-jaw stroke
            # (each side's two prismatic joints share half-stroke each);
            # otherwise fall back to zero so downstream nodes don't crash.
            stamp = ros_node.get_clock().now().to_msg()
            if dex1_left is not None and dex1_right is not None:
                left_state = dex1_left.read_state()
                right_state = dex1_right.read_state()
                left_jaw = _motor_q_to_prismatic_m(left_state.q)
                right_jaw = _motor_q_to_prismatic_m(right_state.q)
                gripper_pos = [left_jaw, left_jaw, right_jaw, right_jaw]
                gripper_effort = [left_state.tau_est, left_state.tau_est,
                                  right_state.tau_est, right_state.tau_est]
            else:
                gripper_pos = [0.0, 0.0, 0.0, 0.0]
                gripper_effort = [0.0, 0.0, 0.0, 0.0]
            js = JointState()
            js.header.stamp = stamp
            js.name = list(POLICY_JOINT_NAMES) + [
                "gripper_prismatic_1_L", "gripper_prismatic_2_L",
                "gripper_prismatic_1_R", "gripper_prismatic_2_R",
            ]
            js.position = jpos.tolist() + gripper_pos
            js.effort = [0.0] * len(POLICY_JOINT_NAMES) + gripper_effort
            ros_node.joint_state_pub.publish(js)

            # World re-zero state machine. The service handler sets
            # _reset_pending; this branch tells the ZED to drop its world
            # frame, invalidates downstream EMA state, and clears the
            # current origin so we capture a fresh one once the ZED has
            # repopulated its cached pose.
            if ros_node._reset_pending.is_set():
                if zed_bridge is not None:
                    zed_bridge.reset_world_tracking()
                    zed_bridge.set_world_origin(None, None)
                _origin_pos = None
                _origin_yaw_imu = None
                _origin_yaw_zed = None
                _prev_t_wp = None
                _prev_odom_t = None
                _body_lin_vel[:] = 0.0
                _capture_origin_next = True
                ros_node._reset_pending.clear()

            # Publish odometry. Orientation from IMU (wxyz→xyzw). Position and
            # linear velocity from ZED positional tracking if available.
            R_wc, t_wc = zed_bridge.camera_pose_in_world() if zed_bridge else (None, None)

            # Once ZED has resumed publishing post-reset, capture pelvis pos
            # plus *both* yaw references as the new origin and arm zed_bridge
            # with ZED's yaw (translation and orientation both come from ZED).
            if _capture_origin_next and R_wc is not None:
                if pelvis_camera_fk is not None:
                    _, _t_cp_live_init = pelvis_camera_fk.camera_to_pelvis(jpos)
                else:
                    _t_cp_live_init = _t_cp_neutral
                t_wp_raw = t_wc + R_wc @ _t_cp_live_init
                yaw_imu = _yaw_from_quat_wxyz(quat.astype(np.float64))
                # ZED yaw extracted from R_wc — atan2 of the camera's forward
                # axis projected into world XY. Static head-camera mount has
                # zero yaw component (URDF rpy="0 0.83 0"), so this equals the
                # body's yaw in ZED world.
                yaw_zed = float(np.arctan2(R_wc[1, 0], R_wc[0, 0]))
                # Preserve published z by zeroing only x, y of the origin —
                # consumers (planner ESDF, viz TF) get a flat horizontal jump
                # instead of dropping the pelvis to z=0.
                _origin_pos = np.array(
                    [float(t_wp_raw[0]), float(t_wp_raw[1]), 0.0],
                    dtype=np.float64,
                )
                _origin_yaw_imu = yaw_imu
                _origin_yaw_zed = yaw_zed
                if zed_bridge is not None:
                    zed_bridge.set_world_origin(_origin_pos, _origin_yaw_zed)
                # Tell planner to flush ESDF / loco visuals integrated in the
                # pre-reset world frame.
                ros_node.planner_reset_pub.publish(Empty())
                _capture_origin_next = False
                ros_node._reset_done.set()

            odom = Odometry()
            odom.header.stamp = stamp
            odom.header.frame_id = "world"
            odom.child_frame_id = "pelvis"
            # Orientation: subtract IMU yaw at reset (IMU yaw → 0).
            if _origin_yaw_imu is not None:
                quat_pub_wxyz = _apply_yaw_offset_quat_wxyz(
                    quat.astype(np.float64), _origin_yaw_imu
                )
            else:
                quat_pub_wxyz = quat.astype(np.float64)
            odom.pose.pose.orientation.x = float(quat_pub_wxyz[1])
            odom.pose.pose.orientation.y = float(quat_pub_wxyz[2])
            odom.pose.pose.orientation.z = float(quat_pub_wxyz[3])
            odom.pose.pose.orientation.w = float(quat_pub_wxyz[0])
            odom.twist.twist.angular.x = float(ang_vel[0])
            odom.twist.twist.angular.y = float(ang_vel[1])
            odom.twist.twist.angular.z = float(ang_vel[2])
            if R_wc is not None:
                # world→pelvis: apply inverse of the pelvis→camera mount.
                # Use live FK (waist-aware) when available; else neutral.
                if pelvis_camera_fk is not None:
                    _, _t_cp_live = pelvis_camera_fk.camera_to_pelvis(jpos)
                else:
                    _t_cp_live = _t_cp_neutral
                t_wp = t_wc + R_wc @ _t_cp_live
                # Translation comes from ZED world → use ZED yaw at reset so
                # the published pelvis pos shares its frame with /molmo/
                # camera/<name>/pose (also zed_bridge-rotated by ZED yaw).
                if _origin_pos is not None and _origin_yaw_zed is not None:
                    dx = float(t_wp[0] - _origin_pos[0])
                    dy = float(t_wp[1] - _origin_pos[1])
                    rx, ry = _apply_yaw_offset_xy(dx, dy, _origin_yaw_zed)
                    t_wp_pub = np.array(
                        [rx, ry, float(t_wp[2] - _origin_pos[2])],
                        dtype=np.float64,
                    )
                else:
                    t_wp_pub = t_wp
                odom.pose.pose.position.x = float(t_wp_pub[0])
                odom.pose.pose.position.y = float(t_wp_pub[1])
                odom.pose.pose.position.z = float(t_wp_pub[2])
                now_t = time.perf_counter()
                if _prev_t_wp is not None and _prev_odom_t is not None:
                    dt = now_t - _prev_odom_t
                    if dt > 1e-6:
                        # Velocity computed in the published (re-zeroed) frame
                        # so it stays consistent with the published quaternion.
                        vel_world = (t_wp_pub - _prev_t_wp) / dt
                        R_wb = _quat_wxyz_to_mat(quat_pub_wxyz)
                        vel_body = R_wb.T @ vel_world
                        _body_lin_vel[:] = (
                            (1 - _VEL_ALPHA) * _body_lin_vel + _VEL_ALPHA * vel_body
                        )
                _prev_t_wp = t_wp_pub
                _prev_odom_t = now_t
                odom.twist.twist.linear.x = float(_body_lin_vel[0])
                odom.twist.twist.linear.y = float(_body_lin_vel[1])
                odom.twist.twist.linear.z = float(_body_lin_vel[2])
            else:
                _prev_t_wp = None
                _prev_odom_t = None
                _body_lin_vel[:] = 0.0
            ros_node.odom_pub.publish(odom)

            # Send state to policy (body_lin_vel and root_pos → zeros;
            # the student policy does not observe base_lin_vel or root_pos).
            udp_sock.sendto(
                pack_state(step_count, quat, zeros3, zeros3, ang_vel, jpos, jvel,
                           ros_node.command),
                policy_addr,
            )

            # Drain UDP to latest action packet
            latest_raw = None
            try:
                while True:
                    latest_raw, _ = udp_sock.recvfrom(ACTION_BYTES + 64)
            except BlockingIOError:
                pass
            if latest_raw is not None:
                _, last_target = unpack_action(latest_raw)

            _send_pd(robot, last_target, _KP, _KD)

            # Forward the gripper-intent scalar to each Dex1-1 side. The
            # writer thread inside Dex1Interface republishes this buffer
            # at 200 Hz; updating it once per 50 Hz tick is plenty.
            if dex1_left is not None and dex1_right is not None:
                latest_cmd = ros_node.command
                l_state = dex1_left.read_state()
                r_state = dex1_right.read_state()
                left_dex1_cmd = _make_dex1_command(latest_cmd[CMD_LEFT_GRIPPER])
                right_dex1_cmd = _make_dex1_command(latest_cmd[CMD_RIGHT_GRIPPER])
                if abs(l_state.tau_est) > args.dex1_tau_limit:
                    left_dex1_cmd.q_target = l_state.q
                if abs(r_state.tau_est) > args.dex1_tau_limit:
                    right_dex1_cmd.q_target = r_state.q
                dex1_left.write_command(left_dex1_cmd)
                dex1_right.write_command(right_dex1_cmd)

            # select → emergency stop: damp at current position, exit immediately
            # B → graceful stop: interpolate to rest in finally block
            ctrl = robot.read_wireless_controller()
            if btn(ctrl, "select") or ros_node.consume_virtual_button("select"):
                print("SELECT pressed — emergency stop (damp mode).")
                state = robot.read_low_state()
                current = np.array(state.motor.q[:NUM_JOINTS], dtype=np.float32)
                _send_pd(robot, current, np.zeros(NUM_JOINTS), _KD)
                return
            if btn(ctrl, "B"):
                print("B pressed — exiting policy loop.")
                break

            step_count += 1
            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, CONTROL_DT - elapsed))

    finally:
        udp_sock.close()
        if zed_bridge is not None:
            zed_bridge.shutdown()
        # Park grippers closed before tearing down so anything currently
        # held stays held when the body switches to damp.
        if dex1_left is not None and dex1_right is not None:
            close_cmd = _make_dex1_command(GRIPPER_CLOSED_CMD)
            dex1_left.write_command(close_cmd)
            dex1_right.write_command(close_cmd)
        # Return to damped hold at current position
        state = robot.read_low_state()
        current = np.array(state.motor.q[:NUM_JOINTS], dtype=np.float32)
        _send_pd(robot, current, np.zeros(NUM_JOINTS), _KD)
        rclpy.shutdown()


if __name__ == "__main__":
    main()
