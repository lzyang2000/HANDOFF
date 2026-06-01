"""Shared constants and limits for all controller nodes.

Controller-specific input handling (TerminalReader, gamepad, DDS, etc.)
lives in each node.  This module provides the single source of truth for
command limits, step sizes, QoS profiles, and formatting helpers.
"""

import numpy as np
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

# ---------------------------------------------------------------------------
# Publish rate
# ---------------------------------------------------------------------------
PUBLISH_RATE_HZ = 50.0

# ---------------------------------------------------------------------------
# QoS
# ---------------------------------------------------------------------------
# RELIABLE so rviz (which defaults to RELIABLE) can subscribe. BEST_EFFORT
# publishers are silently dropped by RELIABLE subscribers.
VIZ_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)

# Latched profile for /g1/control_source. TRANSIENT_LOCAL replays the last
# value to late-joining subscribers, so launch order between viser_ui_node
# and the controller nodes does not matter.
CONTROL_SOURCE_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)

# ---------------------------------------------------------------------------
# Velocity limits
# ---------------------------------------------------------------------------
MAX_VX = 1.0
MAX_VY = 1.0
MAX_YAW = 1.0
VEL_LIMITS = np.array([MAX_VX, MAX_VY, MAX_YAW], dtype=np.float32)

# ---------------------------------------------------------------------------
# Torso limits
# ---------------------------------------------------------------------------
PITCH_POS_LIMIT = 0.8
PITCH_NEG_LIMIT = -0.2
HEIGHT_MIN = 0.26
HEIGHT_MAX = 0.78

# ---------------------------------------------------------------------------
# Hand position limits (offsets from nominal body-frame positions)
# Single source of truth is reach_clamp.py (which must stay rclpy-free so
# unit tests can import it without a ROS env). Re-exported here for
# backward compatibility with existing imports.
# ---------------------------------------------------------------------------
from reach_clamp import (  # noqa: E402 - re-export after numpy import
    HAND_NEG_LIMIT_XYZ,
    HAND_POS_LIMIT_XYZ,
    LEFT_HAND_NEG_LIMIT_XYZ,
    LEFT_HAND_POS_LIMIT_XYZ,
)

# ---------------------------------------------------------------------------
# Wrist / gripper limits
# ---------------------------------------------------------------------------
WRIST_ROLL_LIMIT = 1.0
GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = 1.0

# ---------------------------------------------------------------------------
# Default offsets (standing pose with arms at rest)
# ---------------------------------------------------------------------------
DEFAULT_PITCH = -0.09
DEFAULT_HEIGHT_OFFSET = -0.0
DEFAULT_HAND_X = 0.12
DEFAULT_HAND_Y = 0.00
DEFAULT_HAND_Z = 0.02
NOMINAL_ROOT_Z = 0.78

# ---------------------------------------------------------------------------
# Input parameters
# ---------------------------------------------------------------------------
DEADZONE = 0.15

# Keyboard step sizes (per keypress)
KEYBOARD_VX_STEP = 0.02
KEYBOARD_VY_STEP = 0.02
KEYBOARD_YAW_STEP = 0.04
KEYBOARD_HAND_STEP = 0.005
KEYBOARD_HEIGHT_STEP = 0.002
KEYBOARD_PITCH_STEP = 0.005
KEYBOARD_WRIST_ROLL_STEP = 0.03
KEYBOARD_GRIPPER_STEP = 0.04

# Xbox / gamepad continuous speeds (per tick at PUBLISH_RATE_HZ)
GAMEPAD_PITCH_SPEED = 0.005
GAMEPAD_HEIGHT_SPEED = 0.002
GAMEPAD_HAND_SPEED = 0.005

# DDS XR smoothing
HAND_ALPHA = 0.8
WRIST_RPY_ALPHA = 0.2
WRIST_RPY_MAX_RAD = np.array([0.8, 0.8, 0.8], dtype=np.float32)
WRIST_RPY_OUTLIER_JUMP_RAD = np.array([0.6, 0.6, 0.6], dtype=np.float32)
HEAD_HEIGHT_SCALE = 0.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clamp_vel(vel: np.ndarray) -> np.ndarray:
    return np.clip(vel, -VEL_LIMITS, VEL_LIMITS)


def format_command(command: np.ndarray) -> str:
    return f"vx={command[0]:+.2f} vy={command[1]:+.2f} yaw={command[2]:+.2f}"
