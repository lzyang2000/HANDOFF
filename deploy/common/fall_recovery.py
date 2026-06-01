"""Fall-recovery helpers for molmo_node.

The state-machine itself lives on MolmoNode (it needs access to approach
state and the loco publisher), but the pure-data pieces — the tilt math
and the snapshot dataclass — are factored out here so they can be
exercised in unit tests without bringing up rclpy or the full controller.

`_odom_tilt_rad` consumes a duck-typed object with the same shape as
nav_msgs/Odometry (msg.pose.pose.orientation.{w,x,y,z}) — test fixtures
pass a SimpleNamespace; the live node passes the real ROS message.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import numpy as np

if TYPE_CHECKING:
    # Imported only for type-checking to avoid pulling QueuedMolmoStep
    # (defined in molmo_node) into the runtime import graph.
    from molmo_node import QueuedMolmoStep


def odom_tilt_rad(odom_msg) -> float:
    """max(|roll|, |pitch|) in radians from an Odometry-shaped message.

    Standard ZYX-Euler extraction from the wxyz quaternion. Yaw is
    intentionally not returned — fall detection is heading-independent.
    """
    q = odom_msg.pose.pose.orientation
    w, x, y, z = float(q.w), float(q.x), float(q.y), float(q.z)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_sin = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(pitch_sin)
    return max(abs(roll), abs(pitch))


@dataclass
class FallRecoverySnapshot:
    """Approach + step state captured at fall entry, restored on recovery.

    The local-frame target survives the fall because it is anchored to
    pelvis XY at snapshot time, not to current odom. ``active_step``
    holds a reference to the running step — pausing in place rather than
    clearing keeps the same motion_phase resumable. ``exec_queue``
    mirrors QueuedMolmoStep entries waiting behind the active one.
    ``approach_pending_steps`` holds the steps that were queued behind
    the approach itself (promoted to ``exec_queue`` at arrival) — these
    must survive the fall too, otherwise a mid-approach fall loses the
    pick/place that was about to run.
    """
    approach_active: bool
    approach_target_local: Optional[np.ndarray]
    approach_target_body: Optional[np.ndarray]
    approach_phase: str
    approach_pending_steps: List["QueuedMolmoStep"]
    active_step: Optional["QueuedMolmoStep"]
    exec_queue: List["QueuedMolmoStep"]
    search_active: bool
    search_phase: str


class FallTiltGate:
    """Pure tilt-only entry/exit hysteresis machine.

    Drives the boolean ``in_fall`` state from per-tick tilt samples and
    a pair of thresholds + hold counts. Decoupled from MolmoNode so the
    transitions can be unit-tested without ROS — the live node calls
    ``observe(tilt)`` once per tick, and reads ``triggered_entry`` /
    ``triggered_exit`` to know whether to run the snapshot/restore
    side-effects this tick.
    """

    def __init__(
        self,
        *,
        entry_rad: float,
        exit_rad: float,
        entry_ticks: int,
        exit_ticks: int,
    ) -> None:
        if exit_rad >= entry_rad:
            raise ValueError(
                f"exit_rad ({exit_rad}) must be < entry_rad ({entry_rad}) "
                "for proper hysteresis"
            )
        self._entry_rad = float(entry_rad)
        self._exit_rad = float(exit_rad)
        self._entry_ticks = max(1, int(entry_ticks))
        self._exit_ticks = max(1, int(exit_ticks))
        self.in_fall: bool = False
        self._entry_count: int = 0
        self._exit_count: int = 0
        self.triggered_entry: bool = False
        self.triggered_exit: bool = False

    def observe(self, tilt_rad: float) -> None:
        self.triggered_entry = False
        self.triggered_exit = False
        if not self.in_fall:
            if tilt_rad > self._entry_rad:
                self._entry_count += 1
                if self._entry_count >= self._entry_ticks:
                    self.in_fall = True
                    self._entry_count = 0
                    self._exit_count = 0
                    self.triggered_entry = True
            else:
                self._entry_count = 0
        else:
            if tilt_rad < self._exit_rad:
                self._exit_count += 1
                if self._exit_count >= self._exit_ticks:
                    self.in_fall = False
                    self._entry_count = 0
                    self._exit_count = 0
                    self.triggered_exit = True
            else:
                self._exit_count = 0
