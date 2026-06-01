#!/usr/bin/env python3
"""TeleVuer-driven DDS XR controller node for the G1 hand sim.

Publishes:
  /g1/command   (std_msgs/Float32MultiArray) - unified command (19 floats)

The node supports both TeleVuer controller tracking and hand tracking.
Controller tracking uses thumbsticks for base velocity and wrist poses for
hand targets. Hand tracking uses the tracked wrists for hand targets and keeps
base velocity at zero because no controller axes are available.
"""

from __future__ import annotations

import argparse
import math
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Pose
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from deploy.common.command import (
    CMD_HEIGHT,
    CMD_LEFT_GRIPPER,
    CMD_LEFT_HAND,
    CMD_LEFT_WRIST,
    CMD_PITCH,
    CMD_RIGHT_GRIPPER,
    CMD_RIGHT_HAND,
    CMD_RIGHT_WRIST,
    CMD_VX,
    CMD_VY,
    CMD_YAW_RATE,
    COMMAND_TOPIC,
    make_command,
)
from wbc_mjlab.g1_constants_custom import (
    DDS_XR_GRIPPER_MAX_OPEN_M,
    GRIPPER_OPEN_CMD,
    NOMINAL_LEFT_HAND_BODY,
    NOMINAL_RIGHT_HAND_BODY,
    gripper_open_m_to_cmd,
)
from teleop_common import (
    DEADZONE,
    DEFAULT_HAND_X,
    DEFAULT_HAND_Y,
    DEFAULT_HAND_Z,
    DEFAULT_HEIGHT_OFFSET,
    DEFAULT_PITCH,
    VIZ_QOS,
    HAND_ALPHA,
    HAND_NEG_LIMIT_XYZ,
    HAND_POS_LIMIT_XYZ,
    HEAD_HEIGHT_SCALE,
    LEFT_HAND_NEG_LIMIT_XYZ,
    LEFT_HAND_POS_LIMIT_XYZ,
    MAX_VX,
    MAX_VY,
    MAX_YAW,
    NOMINAL_ROOT_Z,
    PITCH_NEG_LIMIT,
    PITCH_POS_LIMIT,
    PUBLISH_RATE_HZ,
    WRIST_RPY_ALPHA,
    WRIST_RPY_MAX_RAD,
    WRIST_RPY_OUTLIER_JUMP_RAD,
)

# DDS XR uses offset-based height limits (relative to NOMINAL_ROOT_Z)
HEIGHT_POS_LIMIT = 0.0
HEIGHT_NEG_LIMIT = -0.5

try:
    from televuer import TeleVuerWrapper
except ImportError:  # pragma: no cover - handled at runtime when dependency is missing
    TeleVuerWrapper = None


def _local_ipv4_candidates():
    ips = set()
    try:
        host_ips = socket.gethostbyname_ex(socket.gethostname())[2]
        ips.update(ip for ip in host_ips if "." in ip and not ip.startswith("127."))
    except Exception:
        pass

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.add(s.getsockname()[0])
        s.close()
    except Exception:
        pass

    return sorted(ips)


def _pose_pos3(pose: Optional[Pose]) -> Optional[np.ndarray]:
    if pose is None:
        return None
    return np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float32)


def _pose_quat_xyzw(pose: Optional[Pose]) -> Optional[np.ndarray]:
    if pose is None:
        return None
    return np.array(
        [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
        dtype=np.float32,
    )


def _roll_pitch_degrees(pose: Optional[Pose]) -> tuple[Optional[float], Optional[float]]:
    if pose is None:
        return None, None
    q = pose.orientation
    sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
    cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (q.w * q.y - q.z * q.x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    return float(np.degrees(roll)), float(np.degrees(pitch))


def _wrap_to_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def _normalize_quat_xyzw(q: np.ndarray):
    quat = np.asarray(q, dtype=np.float32).reshape(4)
    norm = float(np.linalg.norm(quat))
    if (not np.isfinite(norm)) or norm < 1e-6:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), False
    return (quat / norm).astype(np.float32), True


def _quat_conjugate_xyzw(q: np.ndarray):
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float32)


def _quat_mul_xyzw(a: np.ndarray, b: np.ndarray):
    ax, ay, az, aw = float(a[0]), float(a[1]), float(a[2]), float(a[3])
    bx, by, bz, bw = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float32,
    )


def _quat_to_rpy_xyz(q: np.ndarray):
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw], dtype=np.float32)


def _mean_quat_xyzw(samples):
    if len(samples) == 0:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    arr = np.asarray(samples, dtype=np.float32).reshape(-1, 4)
    ref = arr[0].copy()
    for i in range(arr.shape[0]):
        if float(np.dot(arr[i], ref)) < 0.0:
            arr[i] = -arr[i]
    mean = np.mean(arr, axis=0)
    out, _ = _normalize_quat_xyzw(mean)
    return out


@dataclass
class PolicyCommand:
    vx: float = 0.0
    vy: float = 0.0
    yaw_rate: float = 0.0
    pitch_offset: float = 0.0
    height_offset: float = 0.0
    left_hand_offset: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    right_hand_offset: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    left_wrist_rpy: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    right_wrist_rpy: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    # Unitless [GRIPPER_OPEN_CMD..GRIPPER_CLOSED_CMD] bus scalars (0 open, 1 closed).
    left_gripper: float = GRIPPER_OPEN_CMD
    right_gripper: float = GRIPPER_OPEN_CMD


class TeleVuerXrCommandSource:
    """Converts TeleVuer XR data into hand-policy commands."""

    MAX_VX = MAX_VX
    MAX_VY = MAX_VY
    MAX_YAW = MAX_YAW

    RIGHT_HAND_POS_LIMIT_XYZ = HAND_POS_LIMIT_XYZ
    RIGHT_HAND_NEG_LIMIT_XYZ = HAND_NEG_LIMIT_XYZ
    LEFT_HAND_POS_LIMIT_XYZ = LEFT_HAND_POS_LIMIT_XYZ
    LEFT_HAND_NEG_LIMIT_XYZ = LEFT_HAND_NEG_LIMIT_XYZ
    PITCH_POS_LIMIT = PITCH_POS_LIMIT
    PITCH_NEG_LIMIT = PITCH_NEG_LIMIT
    HEIGHT_POS_LIMIT = HEIGHT_POS_LIMIT
    HEIGHT_NEG_LIMIT = HEIGHT_NEG_LIMIT

    def __init__(
        self,
        *,
        use_hand_tracking: bool,
        binocular: bool,
        img_shape: tuple[int, int],
        display_fps: float,
        display_mode: str,
        zmq: bool,
        webrtc: bool,
        webrtc_url: Optional[str],
        cert_file: Optional[str],
        key_file: Optional[str],
        return_hand_rot_data: bool,
        vel_ema_alpha: float = 0.05,
        neutral_samples: int = 30,
    ):
        if TeleVuerWrapper is None:
            raise RuntimeError(
                "televuer is not installed. Add the Git dependency in wbc_mjlab/pyproject.toml and rerun."
            )

        self.use_hand_tracking = bool(use_hand_tracking)
        self._vel_ema_alpha = float(np.clip(vel_ema_alpha, 0.0, 1.0))
        self._neutral_samples = max(1, int(neutral_samples))
        self._wrist_rpy_enabled = False
        self._wrist_rpy_alpha = WRIST_RPY_ALPHA
        self._wrist_rpy_max_rad = WRIST_RPY_MAX_RAD.copy()
        self._wrist_rpy_outlier_jump_rad = WRIST_RPY_OUTLIER_JUMP_RAD.copy()

        self._default_pitch_offset = float(DEFAULT_PITCH)
        self._default_height_offset = float(DEFAULT_HEIGHT_OFFSET)
        self._default_left_hand_offset = np.array(
            [DEFAULT_HAND_X, -DEFAULT_HAND_Y, DEFAULT_HAND_Z], dtype=np.float32
        )
        self._default_right_hand_offset = np.array(
            [DEFAULT_HAND_X, DEFAULT_HAND_Y, DEFAULT_HAND_Z], dtype=np.float32
        )

        self.left_thumb = np.zeros(2, dtype=np.float32)
        self.right_thumb = np.zeros(2, dtype=np.float32)
        # TeleData ctrl trigger values: 10.0 = released, 0.0 = fully pressed.
        # Default released so a missing controller leaves the grippers open.
        self.left_trigger = 10.0
        self.right_trigger = 10.0
        self._gripper_max_open_m = float(DDS_XR_GRIPPER_MAX_OPEN_M)
        self.left_wrist_pos: Optional[np.ndarray] = None
        self.right_wrist_pos: Optional[np.ndarray] = None
        self.head_z: Optional[float] = None
        self.head_pitch_deg: Optional[float] = None
        self.left_wrist_quat: np.ndarray = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        self.right_wrist_quat: np.ndarray = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        self.head_roll_deg = 0.0

        # Button edge tracking mirrors the current HANDOFF DDS controller.
        self.left_a_pressed = False
        self.left_b_pressed = False
        self.right_a_pressed = False
        self.right_b_pressed = False
        self.left_a_reset_hold_enabled = True
        self.hand_motion_enabled = False
        self._stand_reset_requested = True

        self._left_wrist_samples: list[np.ndarray] = []
        self._right_wrist_samples: list[np.ndarray] = []
        self._head_z_samples: list[float] = []
        self._head_pitch_samples: list[float] = []
        self._left_wrist_quat_samples: list[np.ndarray] = []
        self._right_wrist_quat_samples: list[np.ndarray] = []
        self.neutral_ready = False
        self.neutral_left_wrist = np.zeros(3, dtype=np.float32)
        self.neutral_right_wrist = np.zeros(3, dtype=np.float32)
        self.neutral_left_wrist_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        self.neutral_right_wrist_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        self.neutral_head_z = 0.0
        self.neutral_head_pitch_deg = 0.0

        self.cmd = PolicyCommand(
            pitch_offset=self._default_pitch_offset,
            height_offset=self._default_height_offset,
            left_hand_offset=self._default_left_hand_offset.copy(),
            right_hand_offset=self._default_right_hand_offset.copy(),
        )
        self.filtered_left_hand_offset = np.zeros(3, dtype=np.float32)
        self.filtered_right_hand_offset = np.zeros(3, dtype=np.float32)
        self.output_left_hand_offset = self._default_left_hand_offset.copy()
        self.output_right_hand_offset = self._default_right_hand_offset.copy()
        self._held_left_hand_offset = self.output_left_hand_offset.copy()
        self._held_right_hand_offset = self.output_right_hand_offset.copy()
        self.filtered_left_wrist_rpy = np.zeros(3, dtype=np.float32)
        self.filtered_right_wrist_rpy = np.zeros(3, dtype=np.float32)
        self._prev_left_wrist_raw_rpy = np.zeros(3, dtype=np.float32)
        self._prev_right_wrist_raw_rpy = np.zeros(3, dtype=np.float32)
        self._left_wrist_rpy_initialized = False
        self._right_wrist_rpy_initialized = False

        self._output_vx = 0.0
        self._output_vy = 0.0
        self._output_yaw_rate = 0.0

        tv_kwargs: dict[str, Any] = {
            "use_hand_tracking": self.use_hand_tracking,
            "binocular": bool(binocular),
            "img_shape": img_shape,
            "display_fps": float(display_fps),
            "display_mode": display_mode,
            "zmq": bool(zmq),
            "webrtc": bool(webrtc),
            "webrtc_url": webrtc_url,
            "cert_file": cert_file,
            "key_file": key_file,
            "return_hand_rot_data": bool(return_hand_rot_data),
        }
        self._tv_wrapper = TeleVuerWrapper(**tv_kwargs)

    def print_help(self):
        print("TeleVuer DDS XR controls:")
        print("  left/right thumbstick -> vx / vy / yaw (controller tracking only)")
        print("  left/right trigger    -> left/right gripper open length "
              f"(0..{self._gripper_max_open_m * 100.0:.0f} cm; released=open, pressed=closed)")
        print("  head pose -> torso pitch / height")
        print("  tracked wrist poses -> left/right hand references")
        print("  Left-A  : toggle reset-hold mode")
        print("  Left-B  : recalibrate neutral pose")
        print("  Right-A : toggle hand hold/follow")
        print("  Right-B : toggle wrist RPY actuation")
        if self.use_hand_tracking:
            print("  hand tracking mode: base velocity stays at zero (no thumbsticks available)")

    @staticmethod
    def _sample_to_array(value, size: int):
        out = np.zeros(size, dtype=np.float32)
        data = np.asarray(value, dtype=np.float32).reshape(-1)
        if data.size >= size:
            out[:] = data[:size]
        elif data.size > 0:
            out[:data.size] = data
        return out

    def _handle_left_a(self, pressed: bool):
        if pressed and not self.left_a_pressed:
            self.left_a_reset_hold_enabled = not self.left_a_reset_hold_enabled
            if self.left_a_reset_hold_enabled:
                self._stand_reset_requested = True
                self.hand_motion_enabled = False
                mode = "enabled"
            else:
                self.hand_motion_enabled = True
                mode = "disabled"
            print(f"Left-A toggle: reset-hold {mode}")
        self.left_a_pressed = pressed

    def _handle_right_a(self, pressed: bool):
        if pressed and not self.right_a_pressed:
            if self.left_a_reset_hold_enabled:
                print("Right-A ignored: Left-A reset-hold is enabled.")
                self.right_a_pressed = pressed
                return
            self.hand_motion_enabled = not self.hand_motion_enabled
            if self.hand_motion_enabled:
                mode = "follow"
            else:
                self._held_left_hand_offset = self.output_left_hand_offset.copy()
                self._held_right_hand_offset = self.output_right_hand_offset.copy()
                mode = "hold"
            print(f"Right-A toggle: hand mode {mode}")
        self.right_a_pressed = pressed

    def _handle_left_b(self, pressed: bool):
        if pressed and not self.left_b_pressed:
            self._start_neutral_calibration(trigger="Left-B")
        self.left_b_pressed = pressed

    def _handle_right_b(self, pressed: bool):
        if pressed and not self.right_b_pressed:
            self._wrist_rpy_enabled = not self._wrist_rpy_enabled
            if not self._wrist_rpy_enabled:
                self.filtered_left_wrist_rpy[:] = 0.0
                self.filtered_right_wrist_rpy[:] = 0.0
                self._left_wrist_rpy_initialized = False
                self._right_wrist_rpy_initialized = False
            mode = "enabled" if self._wrist_rpy_enabled else "disabled"
            print(f"Right-B toggle: wrist RPY actuation {mode}")
        self.right_b_pressed = pressed

    def _ingest_tele_data(self, tele_data):
        self.left_thumb[:] = self._sample_to_array(getattr(tele_data, "left_ctrl_thumbstickValue", np.zeros(2)), 2)
        self.right_thumb[:] = self._sample_to_array(getattr(tele_data, "right_ctrl_thumbstickValue", np.zeros(2)), 2)
        self.left_trigger = float(getattr(tele_data, "left_ctrl_triggerValue", 10.0))
        self.right_trigger = float(getattr(tele_data, "right_ctrl_triggerValue", 10.0))

        self.left_wrist_pos = _pose_pos3(getattr(tele_data, "left_wrist_pose", None))
        self.right_wrist_pos = _pose_pos3(getattr(tele_data, "right_wrist_pose", None))
        left_wrist_quat = _pose_quat_xyzw(getattr(tele_data, "left_wrist_pose", None))
        right_wrist_quat = _pose_quat_xyzw(getattr(tele_data, "right_wrist_pose", None))
        self.left_wrist_quat = (
            left_wrist_quat if left_wrist_quat is not None else np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        )
        self.right_wrist_quat = (
            right_wrist_quat if right_wrist_quat is not None else np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        )

        self.head_z = None
        self.head_pitch_deg = None
        self.head_roll_deg = 0.0
        head_pose = getattr(tele_data, "head_pose", None)
        head_pos = _pose_pos3(head_pose)
        head_roll_deg, head_pitch_deg = _roll_pitch_degrees(head_pose)
        if head_pos is not None:
            self.head_z = float(head_pos[2])
        if head_pitch_deg is not None:
            self.head_pitch_deg = float(head_pitch_deg)
        if head_roll_deg is not None:
            self.head_roll_deg = float(head_roll_deg)

        self._handle_left_a(bool(getattr(tele_data, "left_ctrl_aButton", False)))
        self._handle_left_b(bool(getattr(tele_data, "left_ctrl_bButton", False)))
        self._handle_right_a(bool(getattr(tele_data, "right_ctrl_aButton", False)))
        self._handle_right_b(bool(getattr(tele_data, "right_ctrl_bButton", False)))

    def _accumulate_neutral_sample(self):
        left_wrist_pos = self.left_wrist_pos
        right_wrist_pos = self.right_wrist_pos
        head_z = self.head_z
        head_pitch_deg = self.head_pitch_deg
        left_wrist_quat = self.left_wrist_quat
        right_wrist_quat = self.right_wrist_quat

        if (
            left_wrist_pos is None
            or right_wrist_pos is None
            or head_z is None
            or head_pitch_deg is None
            or not _normalize_quat_xyzw(left_wrist_quat)[1]
            or not _normalize_quat_xyzw(right_wrist_quat)[1]
        ):
            return
        self._left_wrist_samples.append(left_wrist_pos.copy())
        self._right_wrist_samples.append(right_wrist_pos.copy())
        self._head_z_samples.append(float(head_z))
        self._head_pitch_samples.append(float(head_pitch_deg))

        left_q, _ = _normalize_quat_xyzw(left_wrist_quat)
        right_q, _ = _normalize_quat_xyzw(right_wrist_quat)
        if self._left_wrist_quat_samples and float(np.dot(left_q, self._left_wrist_quat_samples[0])) < 0.0:
            left_q = -left_q
        if self._right_wrist_quat_samples and float(np.dot(right_q, self._right_wrist_quat_samples[0])) < 0.0:
            right_q = -right_q
        self._left_wrist_quat_samples.append(left_q.copy())
        self._right_wrist_quat_samples.append(right_q.copy())

        if len(self._left_wrist_samples) >= self._neutral_samples:
            self.neutral_left_wrist = np.mean(self._left_wrist_samples, axis=0).astype(np.float32)
            self.neutral_right_wrist = np.mean(self._right_wrist_samples, axis=0).astype(np.float32)
            self.neutral_left_wrist_quat = _mean_quat_xyzw(self._left_wrist_quat_samples)
            self.neutral_right_wrist_quat = _mean_quat_xyzw(self._right_wrist_quat_samples)
            self.neutral_head_z = float(np.mean(self._head_z_samples))
            self.neutral_head_pitch_deg = float(np.mean(self._head_pitch_samples))
            self.neutral_ready = True
            print(
                "TeleVuer neutral calibration complete: "
                f"left={self.neutral_left_wrist}, right={self.neutral_right_wrist}, "
                f"head_z={self.neutral_head_z:.3f}, head_pitch_deg={self.neutral_head_pitch_deg:.2f}"
            )

    def _start_neutral_calibration(self, trigger: str = "manual"):
        self._left_wrist_samples.clear()
        self._right_wrist_samples.clear()
        self._head_z_samples.clear()
        self._head_pitch_samples.clear()
        self._left_wrist_quat_samples.clear()
        self._right_wrist_quat_samples.clear()
        self.neutral_ready = False
        self._reset_filtered_offsets()
        print(
            f"{trigger}: collecting {self._neutral_samples} neutral samples "
            "for left/right wrists + head pose."
        )

    def _reset_filtered_offsets(self):
        self.filtered_left_hand_offset[:] = 0.0
        self.filtered_right_hand_offset[:] = 0.0
        self.output_left_hand_offset[:] = self._default_left_hand_offset
        self.output_right_hand_offset[:] = self._default_right_hand_offset
        self._held_left_hand_offset[:] = self._default_left_hand_offset
        self._held_right_hand_offset[:] = self._default_right_hand_offset
        self.cmd.pitch_offset = self._default_pitch_offset
        self.cmd.height_offset = self._default_height_offset
        self.cmd.left_hand_offset = self._default_left_hand_offset.copy()
        self.cmd.right_hand_offset = self._default_right_hand_offset.copy()
        self.cmd.left_wrist_rpy = np.zeros(3, dtype=np.float32)
        self.cmd.right_wrist_rpy = np.zeros(3, dtype=np.float32)
        self.cmd.left_gripper = GRIPPER_OPEN_CMD
        self.cmd.right_gripper = GRIPPER_OPEN_CMD
        self._output_vx = 0.0
        self._output_vy = 0.0
        self._output_yaw_rate = 0.0
        self.filtered_left_wrist_rpy[:] = 0.0
        self.filtered_right_wrist_rpy[:] = 0.0
        self._prev_left_wrist_raw_rpy[:] = 0.0
        self._prev_right_wrist_raw_rpy[:] = 0.0
        self._left_wrist_rpy_initialized = False
        self._right_wrist_rpy_initialized = False

    def _filter_wrist_rpy_from_quat(
        self,
        curr_quat: np.ndarray,
        neutral_quat: np.ndarray,
        prev_raw: np.ndarray,
        filtered: np.ndarray,
        initialized: bool,
    ):
        q_curr, valid_curr = _normalize_quat_xyzw(curr_quat)
        q_neutral, valid_neutral = _normalize_quat_xyzw(neutral_quat)
        if not (valid_curr and valid_neutral):
            return filtered, prev_raw, initialized

        if float(np.dot(q_curr, q_neutral)) < 0.0:
            q_curr = -q_curr

        q_rel = _quat_mul_xyzw(_quat_conjugate_xyzw(q_neutral), q_curr)
        q_rel, valid_rel = _normalize_quat_xyzw(q_rel)
        if not valid_rel:
            return filtered, prev_raw, initialized

        raw_rpy = _quat_to_rpy_xyz(q_rel)
        raw_rpy = _wrap_to_pi(raw_rpy).astype(np.float32)
        raw_rpy = np.clip(raw_rpy, -self._wrist_rpy_max_rad, self._wrist_rpy_max_rad).astype(np.float32)

        if initialized:
            delta = _wrap_to_pi(raw_rpy - prev_raw).astype(np.float32)
            if np.any(np.abs(delta) > self._wrist_rpy_outlier_jump_rad):
                raw_rpy = prev_raw.copy()
            filtered = (1.0 - self._wrist_rpy_alpha) * filtered + self._wrist_rpy_alpha * raw_rpy
        else:
            filtered = raw_rpy.copy()
            initialized = True

        prev_raw = raw_rpy.copy()
        return filtered.astype(np.float32), prev_raw.astype(np.float32), initialized

    def _update_reset_hold_mode(self):
        self.cmd.vx = 0.0
        self.cmd.vy = 0.0
        self.cmd.yaw_rate = 0.0
        self.cmd.pitch_offset = self._default_pitch_offset
        self.cmd.height_offset = self._default_height_offset

        self.filtered_left_wrist_rpy[:] = 0.0
        self.filtered_right_wrist_rpy[:] = 0.0
        self._left_wrist_rpy_initialized = False
        self._right_wrist_rpy_initialized = False
        self.cmd.left_wrist_rpy = np.zeros(3, dtype=np.float32)
        self.cmd.right_wrist_rpy = np.zeros(3, dtype=np.float32)

        self.filtered_left_hand_offset[:] = 0.0
        self.filtered_right_hand_offset[:] = 0.0
        alpha = HAND_ALPHA
        self.output_left_hand_offset = (
            (1.0 - alpha) * self.output_left_hand_offset + alpha * self._default_left_hand_offset
        )
        self.output_right_hand_offset = (
            (1.0 - alpha) * self.output_right_hand_offset + alpha * self._default_right_hand_offset
        )
        self._held_left_hand_offset = self.output_left_hand_offset.copy()
        self._held_right_hand_offset = self.output_right_hand_offset.copy()
        self.cmd.left_hand_offset = self.output_left_hand_offset.astype(np.float32).copy()
        self.cmd.right_hand_offset = self.output_right_hand_offset.astype(np.float32).copy()

    def _update_gripper(self):
        # TeleData triggerValue runs 10.0 (released) -> 0.0 (fully pressed), so
        # pull = 1 - triggerValue/10. A released trigger commands the max jaw
        # opening; a fully pressed trigger commands a closed gripper. The
        # physical opening (m) is then mapped to the unitless bus scalar.
        left_pull = min(max((10.0 - float(self.left_trigger)) / 10.0, 0.0), 1.0)
        right_pull = min(max((10.0 - float(self.right_trigger)) / 10.0, 0.0), 1.0)
        left_open_m = (1.0 - left_pull) * self._gripper_max_open_m
        right_open_m = (1.0 - right_pull) * self._gripper_max_open_m
        self.cmd.left_gripper = gripper_open_m_to_cmd(left_open_m)
        self.cmd.right_gripper = gripper_open_m_to_cmd(right_open_m)

    def _update_from_state(self):
        # Grippers are an independent control axis: they track the triggers
        # regardless of reset-hold / hand-hold state.
        self._update_gripper()

        if self.left_a_reset_hold_enabled:
            self._update_reset_hold_mode()
            return

        if self.use_hand_tracking:
            target_vx = 0.0
            target_vy = 0.0
            target_yaw = 0.0
        else:
            target_vx = float(np.clip(-self.left_thumb[1] * self.MAX_VX, -self.MAX_VX, self.MAX_VX))
            target_vy = float(np.clip(-self.left_thumb[0] * self.MAX_VY, -self.MAX_VY, self.MAX_VY))
            target_yaw = float(np.clip(-self.right_thumb[0] * self.MAX_YAW, -self.MAX_YAW, self.MAX_YAW))
            if abs(self.left_thumb[1]) <= DEADZONE:
                target_vx = 0.0
            if abs(self.left_thumb[0]) <= DEADZONE:
                target_vy = 0.0
            if abs(self.right_thumb[0]) <= DEADZONE:
                target_yaw = 0.0

        self.cmd.vx = target_vx
        self.cmd.vy = target_vy
        self.cmd.yaw_rate = target_yaw

        if self.neutral_ready and self.left_wrist_pos is not None and self.right_wrist_pos is not None:
            left_delta_raw = self.left_wrist_pos - self.neutral_left_wrist
            right_delta_raw = self.right_wrist_pos - self.neutral_right_wrist
            left_delta = np.minimum(
                np.maximum(left_delta_raw, self.LEFT_HAND_NEG_LIMIT_XYZ),
                self.LEFT_HAND_POS_LIMIT_XYZ,
            )
            right_delta = np.minimum(
                np.maximum(right_delta_raw, self.RIGHT_HAND_NEG_LIMIT_XYZ),
                self.RIGHT_HAND_POS_LIMIT_XYZ,
            )
            self.filtered_left_hand_offset = (
                (1.0 - HAND_ALPHA) * self.filtered_left_hand_offset + HAND_ALPHA * left_delta
            )
            self.filtered_right_hand_offset = (
                (1.0 - HAND_ALPHA) * self.filtered_right_hand_offset + HAND_ALPHA * right_delta
            )
        else:
            self.filtered_left_hand_offset[:] = 0.0
            self.filtered_right_hand_offset[:] = 0.0

        if self._wrist_rpy_enabled and self.neutral_ready:
            (
                self.filtered_left_wrist_rpy,
                self._prev_left_wrist_raw_rpy,
                self._left_wrist_rpy_initialized,
            ) = self._filter_wrist_rpy_from_quat(
                curr_quat=self.left_wrist_quat,
                neutral_quat=self.neutral_left_wrist_quat,
                prev_raw=self._prev_left_wrist_raw_rpy,
                filtered=self.filtered_left_wrist_rpy,
                initialized=self._left_wrist_rpy_initialized,
            )
            (
                self.filtered_right_wrist_rpy,
                self._prev_right_wrist_raw_rpy,
                self._right_wrist_rpy_initialized,
            ) = self._filter_wrist_rpy_from_quat(
                curr_quat=self.right_wrist_quat,
                neutral_quat=self.neutral_right_wrist_quat,
                prev_raw=self._prev_right_wrist_raw_rpy,
                filtered=self.filtered_right_wrist_rpy,
                initialized=self._right_wrist_rpy_initialized,
            )
        else:
            self.filtered_left_wrist_rpy[:] = 0.0
            self.filtered_right_wrist_rpy[:] = 0.0
            self._left_wrist_rpy_initialized = False
            self._right_wrist_rpy_initialized = False

        if self.hand_motion_enabled:
            target_left = self._default_left_hand_offset + self.filtered_left_hand_offset
            target_right = self._default_right_hand_offset + self.filtered_right_hand_offset
        else:
            target_left = self._held_left_hand_offset
            target_right = self._held_right_hand_offset

        self.output_left_hand_offset = (
            (1.0 - HAND_ALPHA) * self.output_left_hand_offset + HAND_ALPHA * target_left
        )
        self.output_right_hand_offset = (
            (1.0 - HAND_ALPHA) * self.output_right_hand_offset + HAND_ALPHA * target_right
        )

        if self.neutral_ready and self.head_pitch_deg is not None:
            pitch_offset = self._default_pitch_offset + np.deg2rad(
                self.head_pitch_deg - self.neutral_head_pitch_deg
            ) * 0.35
            self.cmd.pitch_offset = float(np.clip(pitch_offset, self.PITCH_NEG_LIMIT, self.PITCH_POS_LIMIT))
        else:
            self.cmd.pitch_offset = self._default_pitch_offset

        if self.neutral_ready and self.head_z is not None:
            head_dz = float(self.head_z - self.neutral_head_z)
            height_offset = self._default_height_offset + head_dz * HEAD_HEIGHT_SCALE
            self.cmd.height_offset = float(np.clip(height_offset, self.HEIGHT_NEG_LIMIT, self.HEIGHT_POS_LIMIT))
        else:
            self.cmd.height_offset = self._default_height_offset

        self.cmd.left_hand_offset = self.output_left_hand_offset.astype(np.float32).copy()
        self.cmd.right_hand_offset = self.output_right_hand_offset.astype(np.float32).copy()
        self.cmd.left_wrist_rpy = self.filtered_left_wrist_rpy.copy()
        self.cmd.right_wrist_rpy = self.filtered_right_wrist_rpy.copy()

    def update(self, dt: float):
        try:
            tele_data = self._tv_wrapper.get_tele_data()
        except Exception as exc:
            raise RuntimeError(f"TeleVuer data update failed: {exc}") from exc

        if tele_data is not None:
            self._ingest_tele_data(tele_data)
        if not self.neutral_ready:
            self._accumulate_neutral_sample()
        self._update_from_state()

        a = self._vel_ema_alpha
        self._output_vx = (1.0 - a) * self._output_vx + a * float(self.cmd.vx)
        self._output_vy = (1.0 - a) * self._output_vy + a * float(self.cmd.vy)
        self._output_yaw_rate = (1.0 - a) * self._output_yaw_rate + a * float(self.cmd.yaw_rate)

    def get_command(self) -> PolicyCommand:
        return PolicyCommand(
            vx=float(self._output_vx),
            vy=float(self._output_vy),
            yaw_rate=float(self._output_yaw_rate),
            pitch_offset=float(self.cmd.pitch_offset),
            height_offset=float(self.cmd.height_offset),
            left_hand_offset=self.cmd.left_hand_offset.copy(),
            right_hand_offset=self.cmd.right_hand_offset.copy(),
            left_wrist_rpy=self.cmd.left_wrist_rpy.copy(),
            right_wrist_rpy=self.cmd.right_wrist_rpy.copy(),
            left_gripper=float(self.cmd.left_gripper),
            right_gripper=float(self.cmd.right_gripper),
        )

    def consume_stand_reset_request(self) -> bool:
        requested = self._stand_reset_requested
        self._stand_reset_requested = False
        return requested

    def reset_to_neutral(self):
        self._reset_filtered_offsets()
        self._stand_reset_requested = True

    def shutdown(self):
        try:
            self._tv_wrapper.close()
        except Exception:
            pass


class DdsXrNode(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("dds_xr_node")
        self._args = args
        self._lock = threading.Lock()
        self._command_source = TeleVuerXrCommandSource(
            use_hand_tracking=bool(args.use_hand_tracking),
            binocular=bool(args.binocular),
            img_shape=(int(args.img_height), int(args.img_width)),
            display_fps=float(args.display_fps),
            display_mode=str(args.display_mode),
            zmq=bool(args.zmq),
            webrtc=bool(args.webrtc),
            webrtc_url=args.webrtc_url,
            cert_file=args.cert_file,
            key_file=args.key_file,
            return_hand_rot_data=bool(args.publish_hand_rot),
            vel_ema_alpha=float(args.vel_ema_alpha),
            neutral_samples=int(args.neutral_samples),
        )
        self._cmd_pub = self.create_publisher(Float32MultiArray, COMMAND_TOPIC, VIZ_QOS)
        self._timer = self.create_timer(1.0 / max(1e-6, float(args.rate)), self._publish)
        self._measure_fps = bool(args.measure_fps)
        self._fps_t0 = time.time()
        self._fps_count = 0

        mode = "hand tracking" if bool(args.use_hand_tracking) else "controller tracking"
        self.get_logger().info(
            f"TeleVuer DDS XR node active ({mode}); publishing {COMMAND_TOPIC}."
        )
        self._print_vuer_endpoints()
        self._command_source.print_help()

    def destroy_node(self):
        try:
            self._command_source.shutdown()
        except Exception:
            pass
        return super().destroy_node()

    def _resolve_vuer_host_port(self):
        vuer = getattr(getattr(self._command_source, "_tv_wrapper", None), "tvuer", None)
        vuer = getattr(vuer, "vuer", None)

        host = getattr(vuer, "host", None)
        if host is None:
            host = "0.0.0.0"

        port = None
        for attr in ("port", "_port"):
            value = getattr(vuer, attr, None)
            if value is not None:
                try:
                    port = int(value)
                    break
                except Exception:
                    pass
        if port is None:
            port = 8012

        return str(host), int(port)

    def _print_vuer_endpoints(self):
        host, port = self._resolve_vuer_host_port()
        ips = _local_ipv4_candidates()
        if not ips:
            print(f"Vuer bind host={host}, port={port} (no LAN IPv4 detected).")
            return
        print(f"Vuer bind host={host}, port={port}")
        for ip in ips:
            print(f"Vuer URL: https://{ip}:{port}?ws=wss://{ip}:{port}")

    def _publish(self):
        with self._lock:
            self._command_source.update(1.0 / max(1e-6, float(self._args.rate)))
            if not self._command_source.neutral_ready:
                cmd = make_command()
            else:
                src = self._command_source.get_command()
                cmd = make_command()
                cmd[CMD_VX] = float(src.vx)
                cmd[CMD_VY] = float(src.vy)
                cmd[CMD_YAW_RATE] = float(src.yaw_rate)
                cmd[CMD_PITCH] = float(src.pitch_offset)
                cmd[CMD_HEIGHT] = float(NOMINAL_ROOT_Z + src.height_offset)
                cmd[CMD_LEFT_HAND:CMD_LEFT_HAND + 3] = NOMINAL_LEFT_HAND_BODY + src.left_hand_offset
                cmd[CMD_RIGHT_HAND:CMD_RIGHT_HAND + 3] = NOMINAL_RIGHT_HAND_BODY + src.right_hand_offset
                cmd[CMD_LEFT_WRIST:CMD_LEFT_WRIST + 3] = src.left_wrist_rpy
                cmd[CMD_RIGHT_WRIST:CMD_RIGHT_WRIST + 3] = src.right_wrist_rpy
                cmd[CMD_LEFT_GRIPPER] = float(src.left_gripper)
                cmd[CMD_RIGHT_GRIPPER] = float(src.right_gripper)

        msg = Float32MultiArray()
        msg.data = cmd.tolist()
        self._cmd_pub.publish(msg)

        if self._measure_fps:
            self._fps_count += 1
            now = time.time()
            if now - self._fps_t0 >= 1.0:
                self.get_logger().info(f"publisher_fps={self._fps_count / (now - self._fps_t0):.2f}")
                self._fps_t0 = now
                self._fps_count = 0


def parse_args():
    parser = argparse.ArgumentParser(description="Publish TeleVuer XR data to /g1/cmd_vel and /g1/hand_ref")
    parser.add_argument("--rate", type=float, default=PUBLISH_RATE_HZ, help="Publish rate in Hz")
    parser.add_argument("--measure_fps", type=int, default=0, choices=[0, 1], help="Print publisher FPS once per second")
    parser.add_argument("--vel_ema_alpha", type=float, default=0.05, help="EMA smoothing factor for cmd_vel outputs")
    parser.add_argument("--neutral_samples", type=int, default=30, help="Neutral pose samples to collect before tracking")

    parser.add_argument("--use_hand_tracking", type=int, default=0, choices=[0, 1], help="0: controller tracking, 1: hand tracking")
    parser.add_argument("--publish_hand_rot", type=int, default=0, choices=[0, 1], help="Ask TeleVuer to return hand rotation matrices")

    parser.add_argument("--display_mode", type=str, default="pass-through", choices=["immersive", "pass-through", "ego"])
    parser.add_argument("--display_fps", type=float, default=30.0, help="XR display FPS setting for TeleVuer")
    parser.add_argument("--binocular", type=int, default=0, choices=[0, 1], help="Whether XR image mode is binocular")
    parser.add_argument("--img_height", type=int, default=480, help="XR display image height")
    parser.add_argument("--img_width", type=int, default=1280, help="XR display image width")

    parser.add_argument("--zmq", type=int, default=0, choices=[0, 1], help="Enable ZMQ image mode")
    parser.add_argument("--webrtc", type=int, default=0, choices=[0, 1], help="Enable WebRTC image mode")
    parser.add_argument("--webrtc_url", type=str, default=None, help="WebRTC offer URL (required when --webrtc 1)")
    parser.add_argument("--cert_file", type=str, default=None, help="Path to SSL cert.pem")
    parser.add_argument("--key_file", type=str, default=None, help="Path to SSL key.pem")

    args, ros_args = parser.parse_known_args()
    return args, ros_args


def main():
    args, ros_args = parse_args()
    rclpy.init(args=ros_args)
    node = None
    try:
        node = DdsXrNode(args)
        rclpy.spin(node)
    except RuntimeError as exc:
        print(f"dds_xr_node startup failed: {exc}")
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
