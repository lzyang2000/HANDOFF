"""UDP-based synchronous state↔action exchange for sim↔policy.

Replaces ROS DDS for the tight control loop. ROS is still used for
controller commands, telemetry, and monitoring — just not for the
latency-critical state→action path.

Protocol (all values float32, little-endian):

  State packet (sim → policy):  91 floats = 364 bytes
    [step_id(1), quat_wxyz(4), pos_xyz(3), body_lin_vel(3), body_ang_vel(3),
     joint_pos(29), joint_vel(29), command(19)]

  Action packet (policy → sim):  30 floats = 120 bytes
    [step_id(1), target_pos(29)]
"""

import errno
import os
import signal
import socket
import struct
import subprocess
import time

import numpy as np

# Ports
UDP_SIM_PORT = 9870      # sim listens here for action responses
UDP_POLICY_PORT = 9871   # policy listens here for state packets
UDP_HOST = "127.0.0.1"

# Packet sizes
NUM_JOINTS = 29
CMD_SIZE = 19
STATE_FLOATS = 1 + 4 + 3 + 3 + 3 + NUM_JOINTS + NUM_JOINTS + CMD_SIZE  # 91
ACTION_FLOATS = 1 + NUM_JOINTS  # 30
STATE_BYTES = STATE_FLOATS * 4   # 364
ACTION_BYTES = ACTION_FLOATS * 4  # 120


def _kill_udp_port_holders(port: int) -> None:
    """Find and kill any process bound to the given UDP port."""
    my_pid = os.getpid()
    try:
        result = subprocess.run(
            ["fuser", f"{port}/udp"],
            capture_output=True, text=True, timeout=5,
        )
        pids_str = result.stdout.strip()
        if not pids_str:
            return
        for tok in pids_str.split():
            try:
                pid = int(tok)
            except ValueError:
                continue
            if pid == my_pid:
                continue
            print(f"Killing zombie process {pid} on UDP port {port}")
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        time.sleep(0.2)
    except FileNotFoundError:
        print(f"Warning: 'fuser' not found — cannot auto-kill zombie on port {port}. "
              "Install psmisc or kill the process manually.")
    except subprocess.TimeoutExpired:
        pass


def create_udp_socket(host: str, port: int) -> socket.socket:
    """Create and bind a UDP socket, killing any zombie holder if needed."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((host, port))
    except OSError as e:
        if e.errno != errno.EADDRINUSE:
            raise
        print(f"Port {port} in use — attempting to kill zombie holder...")
        _kill_udp_port_holders(port)
        sock.bind((host, port))
    return sock


def pack_state(step_id: int, root_quat: np.ndarray, root_pos: np.ndarray,
               body_lin_vel: np.ndarray, body_ang_vel: np.ndarray,
               joint_pos: np.ndarray, joint_vel: np.ndarray,
               command: np.ndarray) -> bytes:
    buf = np.empty(STATE_FLOATS, dtype=np.float32)
    buf[0] = float(step_id)
    buf[1:5] = root_quat
    buf[5:8] = root_pos
    buf[8:11] = body_lin_vel
    buf[11:14] = body_ang_vel
    buf[14:43] = joint_pos
    buf[43:72] = joint_vel
    buf[72:72 + CMD_SIZE] = command
    return buf.tobytes()


def unpack_state(data: bytes) -> tuple[int, np.ndarray, np.ndarray,
                                        np.ndarray, np.ndarray,
                                        np.ndarray, np.ndarray,
                                        np.ndarray]:
    buf = np.frombuffer(data, dtype=np.float32)
    step_id = int(buf[0])
    root_quat = buf[1:5].copy()
    root_pos = buf[5:8].copy()
    body_lin_vel = buf[8:11].copy()
    body_ang_vel = buf[11:14].copy()
    joint_pos = buf[14:43].copy()
    joint_vel = buf[43:72].copy()
    command = buf[72:72 + CMD_SIZE].copy()
    return step_id, root_quat, root_pos, body_lin_vel, body_ang_vel, joint_pos, joint_vel, command


def pack_action(step_id: int, target_pos: np.ndarray) -> bytes:
    buf = np.empty(ACTION_FLOATS, dtype=np.float32)
    buf[0] = float(step_id)
    buf[1:30] = target_pos
    return buf.tobytes()


def unpack_action(data: bytes) -> tuple[int, np.ndarray]:
    buf = np.frombuffer(data, dtype=np.float32)
    step_id = int(buf[0])
    target_pos = buf[1:30].copy()
    return step_id, target_pos
