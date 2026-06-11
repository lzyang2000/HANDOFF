"""Keyboard controller node publishing the unified command message.

Matches the HANDOFF shared_hand_command.py JoystickController keyboard layout:
  W/S:   vx + / -
  A/D:   vy + / -
  Q/E:   yaw + / -
  U/I:   height up / down
  Arrows: hand forward/back/left/right
  . / ,: hand up / down
  Z/X:   left wrist roll + / -
  C/V:   right wrist roll + / -
  G/H:   left gripper close / open
  J/K:   right gripper close / open
  N/M:   both grippers close / open
  T:     toggle right-hand-only mode
  F, R:  reset all to defaults
"""

import os
import select
import sys
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from deploy.common.command import (
    CMD_HEIGHT,
    CMD_LEFT_GRIPPER,
    CMD_LEFT_HAND,
    CMD_LEFT_WRIST,
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
    NOMINAL_COMMAND,
    NOMINAL_LEFT_HAND_BODY,
    NOMINAL_RIGHT_HAND_BODY,
)
from teleop_common import (
    DEFAULT_HAND_X,
    DEFAULT_HAND_Y,
    DEFAULT_HAND_Z,
    VIZ_QOS,
    HAND_NEG_LIMIT_XYZ,
    HAND_POS_LIMIT_XYZ,
    HEIGHT_MAX,
    HEIGHT_MIN,
    KEYBOARD_GRIPPER_STEP,
    KEYBOARD_HAND_STEP,
    KEYBOARD_HEIGHT_STEP,
    KEYBOARD_VX_STEP,
    KEYBOARD_VY_STEP,
    KEYBOARD_WRIST_ROLL_STEP,
    KEYBOARD_YAW_STEP,
    LEFT_HAND_NEG_LIMIT_XYZ,
    LEFT_HAND_POS_LIMIT_XYZ,
    MAX_VX,
    MAX_VY,
    MAX_YAW,
    PUBLISH_RATE_HZ,
    WRIST_ROLL_LIMIT,
)

try:
    import termios
    import tty
except ImportError:
    termios = None
    tty = None

# ---------------------------------------------------------------------------
# Allowed key tokens (superset of all keyboard controls)
# ---------------------------------------------------------------------------
_ALLOWED_CHARS = frozenset((
    "w", "a", "s", "d", "q", "e",       # loco
    "u", "i",                              # height
    ",", ".",                              # hand z
    "z", "x", "c", "v",                   # wrist roll
    "g", "h", "j", "k", "n", "m",         # grippers
    "t",                                   # right-hand-only toggle
    "f", "r",                              # reset
))


class TerminalReader:
    """Reads single keypresses from the terminal in a background thread."""

    def __init__(self, on_key):
        self._on_key = on_key
        self._stop_event = threading.Event()
        self._thread = None
        self._stdin_fd = None
        self._old_tty_settings = None
        self._close_stdin_fd = False

    def start(self):
        if termios is None or tty is None:
            print("Terminal keyboard input disabled (termios/tty not available).")
            return False
        if sys.stdin.isatty():
            self._stdin_fd = sys.stdin.fileno()
            self._close_stdin_fd = False
        else:
            try:
                self._stdin_fd = os.open("/dev/tty", os.O_RDONLY)
                self._close_stdin_fd = True
            except Exception:
                print("Terminal keyboard input disabled (no usable TTY).")
                return False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.2)
        self._restore_terminal_mode()

    def _run(self):
        try:
            fd = self._stdin_fd
            self._old_tty_settings = termios.tcgetattr(fd)
            tty.setcbreak(fd)
            while not self._stop_event.is_set():
                ready, _, _ = select.select([fd], [], [], 0.05)
                if not ready:
                    continue
                token = self._read_key_token(fd)
                if token is not None:
                    self._on_key(token)
        except Exception as exc:
            print(f"Terminal keyboard error: {exc}")
        finally:
            self._restore_terminal_mode()

    def _read_key_token(self, fd):
        ch = os.read(fd, 1)
        if not ch:
            return None
        if ch == b" ":
            return "space"
        if ch == b"\x1b":
            seq = b""
            for _ in range(6):
                ready, _, _ = select.select([fd], [], [], 0.01)
                if not ready:
                    break
                seq += os.read(fd, 1)
            if seq in (b"[A", b"OA"):
                return "UP"
            if seq in (b"[B", b"OB"):
                return "DOWN"
            if seq in (b"[C", b"OC"):
                return "RIGHT"
            if seq in (b"[D", b"OD"):
                return "LEFT"
            return None
        char = ch.decode("utf-8", errors="ignore").lower()
        if char in _ALLOWED_CHARS:
            return char
        return None

    def _restore_terminal_mode(self):
        if self._stdin_fd is not None and self._old_tty_settings is not None:
            try:
                termios.tcsetattr(self._stdin_fd, termios.TCSADRAIN, self._old_tty_settings)
            except Exception:
                pass
        if self._stdin_fd is not None and self._close_stdin_fd:
            try:
                os.close(self._stdin_fd)
            except Exception:
                pass
        self._stdin_fd = None
        self._old_tty_settings = None
        self._close_stdin_fd = False


class KeyboardNode(Node):
    def __init__(self):
        super().__init__('keyboard_node')
        # Loco
        self._vx = 0.0
        self._vy = 0.0
        self._yaw = 0.0
        # Torso
        self._height = float(NOMINAL_COMMAND[CMD_HEIGHT])
        # Hands (offsets from nominal body-frame positions)
        self._right_hand = np.array([DEFAULT_HAND_X, DEFAULT_HAND_Y, DEFAULT_HAND_Z], dtype=np.float32)
        self._left_hand = np.array([DEFAULT_HAND_X, -DEFAULT_HAND_Y, DEFAULT_HAND_Z], dtype=np.float32)
        self._right_hand_only = False
        # Wrist roll
        self._left_wrist_roll = 0.0
        self._right_wrist_roll = 0.0
        # Grippers (0=open, 1=closed)
        self._left_gripper = 0.0
        self._right_gripper = 0.0

        self._lock = threading.Lock()
        self._pub = self.create_publisher(Float32MultiArray, COMMAND_TOPIC, VIZ_QOS)
        self._timer = self.create_timer(1.0 / PUBLISH_RATE_HZ, self._publish_command)
        self._terminal = TerminalReader(self._handle_key)

        self._print_help()
        if self._terminal.start():
            self.get_logger().info("Terminal keyboard input enabled.")
        else:
            self.get_logger().warning("Terminal keyboard input unavailable; publishing nominal command.")

    def destroy_node(self):
        self._terminal.stop()
        return super().destroy_node()

    def _print_help(self):
        print("Keyboard controls:")
        print("  W/S: vx + / -")
        print("  A/D: vy + / -")
        print("  Q/E: yaw + / -")
        print("  U/I: height up / down")
        print("  Arrows: hand forward/back/left/right")
        print("  . / ,: hand up / down")
        print("  Z/X: left wrist roll + / -")
        print("  C/V: right wrist roll + / -")
        print("  G/H: left gripper close / open")
        print("  J/K: right gripper close / open")
        print("  N/M: both grippers close / open")
        print("  T: toggle right-hand-only mode")
        print("  F, R: reset all to defaults")

    def _reset_to_defaults(self):
        self._vx = 0.0
        self._vy = 0.0
        self._yaw = 0.0
        self._height = float(NOMINAL_COMMAND[CMD_HEIGHT])
        self._right_hand[:] = [DEFAULT_HAND_X, DEFAULT_HAND_Y, DEFAULT_HAND_Z]
        self._left_hand[:] = [DEFAULT_HAND_X, -DEFAULT_HAND_Y, DEFAULT_HAND_Z]
        self._left_wrist_roll = 0.0
        self._right_wrist_roll = 0.0
        self._left_gripper = 0.0
        self._right_gripper = 0.0

    def _clamp_state(self):
        self._vx = float(np.clip(self._vx, -MAX_VX, MAX_VX))
        self._vy = float(np.clip(self._vy, -MAX_VY, MAX_VY))
        self._yaw = float(np.clip(self._yaw, -MAX_YAW, MAX_YAW))
        self._height = float(np.clip(self._height, HEIGHT_MIN, HEIGHT_MAX))
        self._right_hand = np.clip(self._right_hand, HAND_NEG_LIMIT_XYZ, HAND_POS_LIMIT_XYZ)
        self._left_hand = np.clip(self._left_hand, LEFT_HAND_NEG_LIMIT_XYZ, LEFT_HAND_POS_LIMIT_XYZ)
        self._left_wrist_roll = float(np.clip(self._left_wrist_roll, -WRIST_ROLL_LIMIT, WRIST_ROLL_LIMIT))
        self._right_wrist_roll = float(np.clip(self._right_wrist_roll, -WRIST_ROLL_LIMIT, WRIST_ROLL_LIMIT))
        self._left_gripper = float(np.clip(self._left_gripper, 0.0, 1.0))
        self._right_gripper = float(np.clip(self._right_gripper, 0.0, 1.0))

    def _handle_key(self, token):
        with self._lock:
            # Loco
            if token == "w":
                self._vx += KEYBOARD_VX_STEP
            elif token == "s":
                self._vx -= KEYBOARD_VX_STEP
            elif token == "a":
                self._vy += KEYBOARD_VY_STEP
            elif token == "d":
                self._vy -= KEYBOARD_VY_STEP
            elif token == "q":
                self._yaw += KEYBOARD_YAW_STEP
            elif token == "e":
                self._yaw -= KEYBOARD_YAW_STEP
            # Height
            elif token == "u":
                self._height += KEYBOARD_HEIGHT_STEP
            elif token == "i":
                self._height -= KEYBOARD_HEIGHT_STEP
            # Hand position
            elif token == "UP":
                self._right_hand[0] += KEYBOARD_HAND_STEP
            elif token == "DOWN":
                self._right_hand[0] -= KEYBOARD_HAND_STEP
            elif token == "LEFT":
                self._right_hand[1] += KEYBOARD_HAND_STEP
            elif token == "RIGHT":
                self._right_hand[1] -= KEYBOARD_HAND_STEP
            elif token == ".":
                self._right_hand[2] += KEYBOARD_HAND_STEP
            elif token == ",":
                self._right_hand[2] -= KEYBOARD_HAND_STEP
            # Wrist roll
            elif token == "z":
                self._left_wrist_roll += KEYBOARD_WRIST_ROLL_STEP
            elif token == "x":
                self._left_wrist_roll -= KEYBOARD_WRIST_ROLL_STEP
            elif token == "c":
                self._right_wrist_roll += KEYBOARD_WRIST_ROLL_STEP
            elif token == "v":
                self._right_wrist_roll -= KEYBOARD_WRIST_ROLL_STEP
            # Grippers
            elif token == "g":
                self._left_gripper += KEYBOARD_GRIPPER_STEP
            elif token == "h":
                self._left_gripper -= KEYBOARD_GRIPPER_STEP
            elif token == "j":
                self._right_gripper += KEYBOARD_GRIPPER_STEP
            elif token == "k":
                self._right_gripper -= KEYBOARD_GRIPPER_STEP
            elif token == "n":
                self._left_gripper += KEYBOARD_GRIPPER_STEP
                self._right_gripper += KEYBOARD_GRIPPER_STEP
            elif token == "m":
                self._left_gripper -= KEYBOARD_GRIPPER_STEP
                self._right_gripper -= KEYBOARD_GRIPPER_STEP
            # Toggle
            elif token == "t":
                self._right_hand_only = not self._right_hand_only
                mode = "ON" if self._right_hand_only else "OFF"
                print(f"Right-hand-only mode: {mode}")
            # Reset
            elif token in ("f", "r"):
                self._reset_to_defaults()
                print("Commands reset to defaults.")

            self._clamp_state()

            # Mirror left hand unless in right-hand-only mode
            if not self._right_hand_only:
                self._left_hand[0] = self._right_hand[0]
                self._left_hand[1] = -self._right_hand[1]
                self._left_hand[2] = self._right_hand[2]

            print(
                f"vx={self._vx:+.2f} vy={self._vy:+.2f} yaw={self._yaw:+.2f}"
                f"  h={self._height:.3f}"
                f"  hand_r=[{self._right_hand[0]:+.2f},{self._right_hand[1]:+.2f},{self._right_hand[2]:+.2f}]"
                f"  grip=[{self._left_gripper:.2f},{self._right_gripper:.2f}]"
            )

    def _publish_command(self):
        with self._lock:
            cmd = make_command()
            cmd[CMD_VX] = self._vx
            cmd[CMD_VY] = self._vy
            cmd[CMD_YAW_RATE] = self._yaw
            cmd[CMD_HEIGHT] = self._height
            cmd[CMD_LEFT_HAND:CMD_LEFT_HAND + 3] = NOMINAL_LEFT_HAND_BODY + self._left_hand
            cmd[CMD_RIGHT_HAND:CMD_RIGHT_HAND + 3] = NOMINAL_RIGHT_HAND_BODY + self._right_hand
            cmd[CMD_LEFT_WRIST] = self._left_wrist_roll
            cmd[CMD_RIGHT_WRIST] = self._right_wrist_roll
            cmd[CMD_LEFT_GRIPPER] = self._left_gripper
            cmd[CMD_RIGHT_GRIPPER] = self._right_gripper

        msg = Float32MultiArray()
        msg.data = cmd.tolist()
        self._pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
