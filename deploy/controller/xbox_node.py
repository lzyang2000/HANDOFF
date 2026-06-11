"""Xbox controller node publishing the unified command message.

Publishes:
  /g1/command  (std_msgs/Float32MultiArray)  – 18-float unified command
               (see deploy/common/command.py for field layout)

Button / axis mapping (Xbox One / 360 layout via linux ``inputs``):
  Left Stick Y   -> vx  (forward / back)
  Left Stick X   -> vy  (left / right)
  Right Stick X  -> yaw rate
  Left Trigger   -> torso height -  (ABS_Z,  0-255)
  Right Trigger  -> torso height +  (ABS_RZ, 0-255)
  D-pad Up/Down  -> right hand forward / back (x offset)
  D-pad Left/Right -> right hand lateral (y offset)
  LB  (BTN_TL)  -> right hand down (z offset)
  RB  (BTN_TR)  -> right hand up   (z offset)
  Start/Back     -> reset all commands to defaults
"""

import argparse
import os
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String

from deploy.common.command import (
    CMD_HEIGHT,
    CMD_LEFT_HAND,
    CMD_RIGHT_HAND,
    CMD_VX,
    CMD_VY,
    CMD_YAW_RATE,
    COMMAND_TOPIC,
    CONTROL_SOURCE_TOPIC,
    make_command,
)
from deploy.common.gamepad_bridge import (
    BridgeConfig,
    GamepadBridgeServer,
    bridge_config_from_env,
)
from teleop_common import (
    CONTROL_SOURCE_QOS,
    DEADZONE,
    VIZ_QOS,
    GAMEPAD_HAND_SPEED,
    GAMEPAD_HEIGHT_SPEED,
    HAND_NEG_LIMIT_XYZ,
    HAND_POS_LIMIT_XYZ,
    HEIGHT_MAX,
    HEIGHT_MIN,
    LEFT_HAND_NEG_LIMIT_XYZ,
    LEFT_HAND_POS_LIMIT_XYZ,
    MAX_VX,
    MAX_VY,
    MAX_YAW,
    PUBLISH_RATE_HZ,
    format_command,
)
from wbc_mjlab.g1_constants_custom import (
    DEFAULT_HAND_X,
    DEFAULT_HAND_Y,
    DEFAULT_HAND_Z,
    DEFAULT_HEIGHT,
    NOMINAL_LEFT_HAND_BODY,
    NOMINAL_RIGHT_HAND_BODY,
)

try:
    import inputs
except ImportError:
    inputs = None


class XboxNode(Node):
    def __init__(self, source: str = "gamepad"):
        super().__init__("xbox_node")
        assert source in ("gamepad", "server"), source
        self._source = source

        # --- Command state (protected by _lock) ---
        self._vx = 0.0
        self._vy = 0.0
        self._yaw = 0.0
        self._height = DEFAULT_HEIGHT

        self._right_hand = np.array([DEFAULT_HAND_X, DEFAULT_HAND_Y, DEFAULT_HAND_Z], dtype=np.float32)
        self._left_hand = np.array([DEFAULT_HAND_X, -DEFAULT_HAND_Y, DEFAULT_HAND_Z], dtype=np.float32)

        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        # --- Publisher ---
        self._pub = self.create_publisher(Float32MultiArray, COMMAND_TOPIC, VIZ_QOS)
        # Default "xbox" so this node publishes when no override exists (sim,
        # standalone xbox controller mode without viser_ui_node).
        self._active_source = "xbox"
        self.create_subscription(
            String, CONTROL_SOURCE_TOPIC, self._control_source_cb, CONTROL_SOURCE_QOS,
        )
        self._timer = self.create_timer(1.0 / PUBLISH_RATE_HZ, self._publish)

        if source == "gamepad":
            self._init_gamepad_source()
        else:
            self._init_server_source()

    # ------------------------------------------------------------------
    # Source: local gamepad (evdev via the ``inputs`` library)
    # ------------------------------------------------------------------
    def _init_gamepad_source(self):
        if inputs is None:
            raise RuntimeError("inputs is not installed.")
        if len(inputs.devices.gamepads) == 0:
            raise RuntimeError("No gamepad found.")

        # Raw axis state (written by input thread).
        self._abs_x  = 0
        self._abs_y  = 0
        self._abs_rx = 0
        self._abs_z  = 0
        self._abs_rz = 0
        self._hat_x  = 0
        self._hat_y  = 0
        self._btn_lb = 0
        self._btn_rb = 0

        self._thread = threading.Thread(target=self._monitor_controller, daemon=True)
        self._thread.start()

        self.get_logger().info(
            f"Using gamepad: {inputs.devices.gamepads[0]}\n"
            "Controls:\n"
            "  Left Stick   -> vx / vy\n"
            "  Right Stick X -> yaw\n"
            "  LT / RT       -> torso height - / +\n"
            "  D-pad         -> right hand forward/back (Y) and lateral (X)\n"
            "  LB / RB       -> right hand down / up (z)\n"
            "  Start / Back  -> reset all to defaults"
        )

    # ------------------------------------------------------------------
    # Source: browser (shared GamepadBridgeServer; W3C Gamepad API)
    # ------------------------------------------------------------------
    def _init_server_source(self):
        cfg = bridge_config_from_env(
            BridgeConfig(
                max_vx=MAX_VX, max_vy=MAX_VY, max_yaw=MAX_YAW,
                height_min=HEIGHT_MIN, height_max=HEIGHT_MAX,
                hand_pos_limit_xyz=tuple(HAND_POS_LIMIT_XYZ.tolist()),
                hand_neg_limit_xyz=tuple(HAND_NEG_LIMIT_XYZ.tolist()),
                left_hand_pos_limit_xyz=tuple(LEFT_HAND_POS_LIMIT_XYZ.tolist()),
                left_hand_neg_limit_xyz=tuple(LEFT_HAND_NEG_LIMIT_XYZ.tolist()),
                default_height=DEFAULT_HEIGHT,
                default_hand_x=DEFAULT_HAND_X,
                default_hand_y=DEFAULT_HAND_Y,
                default_hand_z=DEFAULT_HAND_Z,
                deadzone=DEADZONE,
                tick_hz=PUBLISH_RATE_HZ,
                gamepad_hand_speed=GAMEPAD_HAND_SPEED,
                gamepad_height_speed=GAMEPAD_HEIGHT_SPEED,
                label="deploy xbox_node gamepad bridge",
            )
        )
        self._bridge = GamepadBridgeServer(cfg)
        self._bridge.start()
        self.get_logger().info(
            f"xbox_node running in server mode.\n"
            f"  Open  http://localhost:{cfg.http_port}  in a browser with the "
            "controller connected.\n"
            "  Override ports with XBOX_BRIDGE_HOST / _HTTP_PORT / _WS_PORT."
        )

    def destroy_node(self):
        self._stop_event.set()
        t = getattr(self, "_thread", None)
        if t is not None and t.is_alive():
            t.join(timeout=0.2)
        return super().destroy_node()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_stick(value: int) -> float:
        return value / 32768.0

    def _reset_to_defaults(self):
        self._vx = 0.0
        self._vy = 0.0
        self._yaw = 0.0
        self._height = DEFAULT_HEIGHT
        self._right_hand[:] = [DEFAULT_HAND_X,  DEFAULT_HAND_Y, DEFAULT_HAND_Z]
        self._left_hand[:]  = [DEFAULT_HAND_X, -DEFAULT_HAND_Y, DEFAULT_HAND_Z]

    def _clamp_state(self):
        self._vx    = float(np.clip(self._vx,    -MAX_VX,  MAX_VX))
        self._vy    = float(np.clip(self._vy,    -MAX_VY,  MAX_VY))
        self._yaw   = float(np.clip(self._yaw,   -MAX_YAW, MAX_YAW))
        self._height = float(np.clip(self._height, HEIGHT_MIN, HEIGHT_MAX))
        self._right_hand = np.clip(self._right_hand, HAND_NEG_LIMIT_XYZ, HAND_POS_LIMIT_XYZ)
        self._left_hand = np.clip(self._left_hand, LEFT_HAND_NEG_LIMIT_XYZ, LEFT_HAND_POS_LIMIT_XYZ)

    def _update_from_gamepad(self):
        dz = DEADZONE

        raw_ly = self._normalize_stick(self._abs_y)
        raw_lx = self._normalize_stick(self._abs_x)
        raw_rx = self._normalize_stick(self._abs_rx)

        self._vx  = -raw_ly * MAX_VX  if abs(raw_ly) > dz else 0.0
        self._vy  = -raw_lx * MAX_VY  if abs(raw_lx) > dz else 0.0
        self._yaw = -raw_rx * MAX_YAW if abs(raw_rx) > dz else 0.0

        lt_norm = self._abs_z  / 255.0
        rt_norm = self._abs_rz / 255.0
        if rt_norm > 0.1:
            self._height += GAMEPAD_HEIGHT_SPEED * rt_norm
        if lt_norm > 0.1:
            self._height -= GAMEPAD_HEIGHT_SPEED * lt_norm

        if self._hat_y < 0:
            self._right_hand[0] += GAMEPAD_HAND_SPEED
        elif self._hat_y > 0:
            self._right_hand[0] -= GAMEPAD_HAND_SPEED

        if self._hat_x > 0:
            self._right_hand[1] -= GAMEPAD_HAND_SPEED
        elif self._hat_x < 0:
            self._right_hand[1] += GAMEPAD_HAND_SPEED

        if self._btn_rb:
            self._right_hand[2] += GAMEPAD_HAND_SPEED
        if self._btn_lb:
            self._right_hand[2] -= GAMEPAD_HAND_SPEED

        self._left_hand[0]  = self._right_hand[0]
        self._left_hand[1]  = -self._right_hand[1]
        self._left_hand[2]  = self._right_hand[2]

        self._clamp_state()

    # ------------------------------------------------------------------
    # Input thread
    # ------------------------------------------------------------------

    def _monitor_controller(self):
        last_print = 0.0
        while not self._stop_event.is_set():
            try:
                for event in inputs.get_gamepad():
                    if self._stop_event.is_set():
                        break

                    if event.ev_type == "Absolute":
                        if   event.code == "ABS_X":      self._abs_x  = event.state
                        elif event.code == "ABS_Y":      self._abs_y  = event.state
                        elif event.code == "ABS_RX":     self._abs_rx = event.state
                        elif event.code == "ABS_Z":      self._abs_z  = event.state
                        elif event.code == "ABS_RZ":     self._abs_rz = event.state
                        elif event.code == "ABS_HAT0X":  self._hat_x  = event.state
                        elif event.code == "ABS_HAT0Y":  self._hat_y  = event.state
                        else:
                            continue

                        with self._lock:
                            now = time.monotonic()
                            if now - last_print >= 0.3:
                                self._update_from_gamepad()
                                print(
                                    f"cmd_vel {format_command(np.array([self._vx, self._vy, self._yaw]))}"
                                    f"  hand_r=[{self._right_hand[0]:+.2f},{self._right_hand[1]:+.2f},{self._right_hand[2]:+.2f}]"
                                    f"  h={self._height:.3f}"
                                )
                                last_print = now

                    elif event.ev_type == "Key":
                        if   event.code == "BTN_TL":    self._btn_lb = event.state
                        elif event.code == "BTN_TR":    self._btn_rb = event.state
                        if event.code in ("BTN_START", "BTN_SELECT"):
                            if event.state == 1:
                                with self._lock:
                                    self._reset_to_defaults()
                                    print("Commands reset to defaults.")

            except Exception as exc:
                self.get_logger().warning(f"Gamepad read error: {exc}")
                time.sleep(1.0)

    # ------------------------------------------------------------------
    # Publish
    # ------------------------------------------------------------------

    def _publish(self):
        if self._source == "gamepad":
            with self._lock:
                self._update_from_gamepad()
                vx, vy, yaw = self._vx, self._vy, self._yaw
                height = self._height
                right = self._right_hand.copy()
                left = self._left_hand.copy()
        else:
            # Browser path: the JS integrates triggers/D-pad locally, so the
            # bridge state already holds absolute values. No re-integration.
            s = self._bridge.get_state()
            vx, vy, yaw = s["vx"], s["vy"], s["yaw"]
            height = s["height"]
            right = s["right_hand"]
            left = s["left_hand"]

        cmd = make_command()
        cmd[CMD_VX] = vx
        cmd[CMD_VY] = vy
        cmd[CMD_YAW_RATE] = yaw
        cmd[CMD_HEIGHT] = height
        cmd[CMD_LEFT_HAND:CMD_LEFT_HAND + 3] = NOMINAL_LEFT_HAND_BODY + left
        cmd[CMD_RIGHT_HAND:CMD_RIGHT_HAND + 3] = NOMINAL_RIGHT_HAND_BODY + right

        msg = Float32MultiArray()
        msg.data = cmd.tolist()
        if self._active_source == "xbox":
            self._pub.publish(msg)

    def _control_source_cb(self, msg: String) -> None:
        self._active_source = (msg.data or "xbox").strip() or "xbox"


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=("gamepad", "server"),
        default=os.environ.get("XBOX_SOURCE", "gamepad"),
        help="gamepad=local evdev pad; server=browser bridge on XBOX_BRIDGE_HTTP_PORT.",
    )
    parsed, ros_args = parser.parse_known_args(args)
    rclpy.init(args=ros_args)
    node = None
    try:
        node = XboxNode(source=parsed.source)
        rclpy.spin(node)
    except RuntimeError as exc:
        print(f"xbox_node startup failed: {exc}")
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
