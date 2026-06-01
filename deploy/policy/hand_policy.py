"""Hand dual-student policy — UDP control loop.

Receives state from sim_node via UDP, runs ONNX inference, sends action back.
Control path is pure UDP request-response at 50 Hz; a lightweight rclpy
subscriber runs on a background thread so molmo_node can signal the
current manipulation phase (used to gate the wrist leveller).

See deploy/common/udp_sync.py for the packet protocol.
"""

import argparse
import math
import os
import re
import threading
import time
from pathlib import Path
from typing import Optional

os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")

import numpy as np
import onnxruntime as ort
from collections import deque

import mjlab.asset_zoo.robots.unitree_g1.g1_constants as g1_constants
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import KNEES_BENT_KEYFRAME
from deploy.common.udp_sync import (
    UDP_HOST, UDP_SIM_PORT, UDP_POLICY_PORT,
    STATE_BYTES, unpack_state, pack_action, create_udp_socket,
)
from deploy.common.cbf_filter import G1CapturePointCBF
from deploy.common.capture_point import G1CapturePointEstimator
from deploy.common.hand_command_pd import HandCommandPD
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
from wbc_mjlab.g1_constants_custom import (
    CAPTURE_POINT_PD_ENABLE,
    CAPTURE_POINT_METHOD,
    CAPTURE_POINT_PD_KP,
    SQUAT_SMOOTHING_HEIGHT,
    MOLMO_ACTIVE_PHASE_TOPIC,
    MOLMO_WRIST_VERTICAL_ROLL_RAD,
    MOLMO_HAND_CMD_PD_DEADBAND_M,
    MOLMO_HAND_CMD_PD_ENABLE,
    MOLMO_HAND_CMD_PD_KD,
    MOLMO_HAND_CMD_PD_KD_X_MULT,
    MOLMO_HAND_CMD_PD_KD_Y_MULT,
    MOLMO_HAND_CMD_PD_KP,
    MOLMO_HAND_CMD_PD_KP_X_MULT,
    MOLMO_HAND_CMD_PD_KP_Y_MULT,
    MOLMO_HAND_CMD_PD_MAX_SHAPE_M,
    MOLMO_GRIPPER_MIDPOINT_OFFSET_LEFT,
    MOLMO_GRIPPER_MIDPOINT_OFFSET_RIGHT,
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
)

# When set (by play_*_hand.sh in dds mode), copy cmd[CMD_LEFT_WRIST] /
# cmd[CMD_RIGHT_WRIST] (relative-to-neutral RPY published by dds_xr_node)
# onto the wrist joint targets, bypassing the policy's wrist output.
DDS_XR_WRIST_PASSTHROUGH: bool = os.environ.get("WBC_MJLAB_DDS_XR_WRIST", "0") == "1"

import rclpy
from rclpy.node import Node as _RclpyNode
from rclpy.qos import (
    DurabilityPolicy as _DurabilityPolicy,
    HistoryPolicy as _HistoryPolicy,
    QoSProfile as _QoSProfile,
    ReliabilityPolicy as _ReliabilityPolicy,
)
from std_msgs.msg import Float32MultiArray as _Float32MultiArray, String as _StringMsg


class _MolmoSubscriber:
    """Background rclpy node that caches Molmo phase + raw hand targets.

    The hand policy runs on UDP, but wrist-leveller gating depends on
    which hand is doing a ``pick`` step and whether the user asked for
    single-hand or bimanual manipulation. The phase string format is
    ``"{action}:{motion_phase}:{hand}:{hand_mode}"`` (e.g.,
    ``"pick:ee_tracked:l:single"``) while a step is active, ``"idle"``
    otherwise. ``leveller_allowed_for(side)`` enforces: pick + single
    hand-mode + side matches the active hand, excluding tracked return
    phases like ``ee_tracked_reverse`` where the wrist leveller should
    stay off.

    Non-molmo flows (keyboard, xr) don't publish this topic; in that
    case the gate defaults to True on both sides so the leveller keeps
    its historical behavior until an explicit phase is received.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._phase: Optional[str] = None  # None == never received
        self._raw_left: Optional[np.ndarray] = None
        self._raw_right: Optional[np.ndarray] = None
        # World-up extent of the active pick target's OBB at query time
        # (per side, 0.0 = unknown / not a pick / FFS missed). Consumed
        # by the early-close box check as the per-side vertical bound.
        self._pick_obj_z_left: float = 0.0
        self._pick_obj_z_right: float = 0.0
        # Whether the active step is a squat pick (step.squat_delta > 0 or
        # approach-time squat held). Used to swap MOLMO_PICK_EARLY_CLOSE_Z_M
        # for the squat-specific tolerance in the early-close fallback path.
        self._is_squat_pick: bool = False
        self._node = None
        self._thread: Optional[threading.Thread] = None
        try:
            if not rclpy.ok():
                rclpy.init()
            self._node = _RclpyNode("hand_policy_molmo_sub")
            # Must match molmo_node's VIZ_QOS (reliable, volatile) — a
            # mismatched reliability silently drops every message and the
            # gate defaults permissively open.
            phase_qos = _QoSProfile(
                history=_HistoryPolicy.KEEP_LAST,
                depth=5,
                reliability=_ReliabilityPolicy.RELIABLE,
                durability=_DurabilityPolicy.VOLATILE,
            )
            self._node.create_subscription(
                _StringMsg, MOLMO_ACTIVE_PHASE_TOPIC, self._phase_cb, phase_qos,
            )
            self._node.create_subscription(
                _Float32MultiArray, MOLMO_RAW_HAND_TARGET_TOPIC, self._raw_cb, phase_qos,
            )
            self._thread = threading.Thread(
                target=rclpy.spin, args=(self._node,), daemon=True,
            )
            self._thread.start()
            print(
                f"[hand_policy] subscribed to {MOLMO_ACTIVE_PHASE_TOPIC} "
                f"and {MOLMO_RAW_HAND_TARGET_TOPIC}"
            )
        except Exception as e:
            print(f"[hand_policy] Molmo subscriber disabled: {e}")
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
            # Slots 8/9/10 only present in the newer schemas; fall back to
            # 0/0/False for legacy publishers (= constant z bound, non-squat).
            self._pick_obj_z_left = float(vals[8]) if vals.size >= 9 else 0.0
            self._pick_obj_z_right = float(vals[9]) if vals.size >= 10 else 0.0
            self._is_squat_pick = (
                float(vals[10]) > 0.5 if vals.size >= 11 else False
            )

    def leveller_allowed_for(self, side: str) -> bool:
        """True if the leveller should run on ``side`` ("l" or "r").

        Enforces: active step is a single-hand pick on ``side``, except
        for tracked return phases where the wrist leveller should stay
        off. Returns True if no phase has been received yet (backward
        compat for non-molmo flows).
        """
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

        The wrist override path already drops out for this phase via
        leveller_allowed_for() (the single-token string parses to None
        in molmo_wrist_phase). Obs zeroing is independent of that gate,
        so callers consult this predicate directly.
        """
        with self._lock:
            return self._phase == MOLMO_FALL_RECOVERY_PHASE

    def current_phase(self) -> Optional[_MolmoActivePhase]:
        """Return the parsed active phase, or None for ``idle`` / no publish yet.

        Callers drive the per-side vertical-hold latch off this; the
        latch ignores ``None`` so transient ``idle`` publications
        between queued Molmo sub-steps don't drop the hold.
        """
        with self._lock:
            phase = self._phase
        return _parse_molmo_active_phase(phase)

    def raw_target_for(self, side: str) -> Optional[np.ndarray]:
        with self._lock:
            raw = self._raw_left if side == "l" else self._raw_right
            return None if raw is None else raw.copy()

    def pick_object_z_extent_for(self, side: str) -> float:
        """World-up extent of the active pick target's OBB (m), or 0.0
        if unknown / not a pick / FFS missed at query time."""
        with self._lock:
            return self._pick_obj_z_left if side == "l" else self._pick_obj_z_right

    def is_squat_pick_for(self, side: str) -> bool:  # noqa: ARG002
        """True iff the active step is a squat pick. Mirrors
        molmo_node._raw_target_inside_gripper_volume's branch (
        ``step.squat_delta > 0`` or the approach-time squat is held).
        Per-side signature is for symmetry with the other accessors —
        the flag itself is per-step (single-hand picks have one active
        side at a time)."""
        with self._lock:
            return self._is_squat_pick

    def is_pick_walkback_for(self, side: str) -> bool:
        """True only during the active hand's single-hand pick walkback."""
        with self._lock:
            phase = self._phase
        return _is_single_pick_walkback(_parse_molmo_active_phase(phase), side)

    def is_pick_retract_for(self, side: str) -> bool:
        """True during tracked/staged pick return: arm retracting after grasp."""
        with self._lock:
            phase = self._phase
        return _is_single_pick_retract(_parse_molmo_active_phase(phase), side)

    def is_pick_yaw_freeze_for(self, side: str) -> bool:
        """True during pick:grasp — leveller should hold its last yaw."""
        with self._lock:
            phase = self._phase
        return _is_single_pick_yaw_freeze(_parse_molmo_active_phase(phase), side)

    def is_place_active_for(self, side: str) -> bool:
        """True while ``side`` is performing a single-hand place pre-release."""
        with self._lock:
            phase = self._phase
        return _is_single_place_leveller_active(_parse_molmo_active_phase(phase), side)

    def should_clear_walkback_hold_for(self, side: str) -> bool:
        """True when an explicit non-idle phase conflicts with ``side``'s hold.

        This intentionally ignores transient ``idle`` publications between
        queued Molmo sub-steps so a walkback-held wrist leveler survives
        grasp -> lift/carry handoffs.
        """
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
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

# Pelvis-frame forward direction; the finger axis is yawed to track this
# when MOLMO_WRIST_LEVEL_YAW_TO_TORSO is on.
_PELVIS_FORWARD = np.array([1.0, 0.0, 0.0], dtype=np.float64)

# ---------------------------------------------------------------------------
# Constants (must exactly match training)
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

NUM_JOINTS = 29
_ANKLE_DOF_INDICES = {4, 5, 10, 11}

BASE_ANG_VEL_SCALE = 0.25
JOINT_POS_SCALE = 1.0
JOINT_VEL_SCALE = 0.05

HAND_STUDENT_MIMIC_DIM = 14
ACTOR_PROPRIO_DIM = 92
ACTOR_CURRENT_DIM = HAND_STUDENT_MIMIC_DIM + ACTOR_PROPRIO_DIM
ACTOR_HISTORY_LENGTH = 11
ACTOR_OBS_DIM = ACTOR_CURRENT_DIM * (1 + ACTOR_HISTORY_LENGTH)

LOCO_GAIT_PERIOD = 1.0
LOCO_GAIT_OFFSET = 0.5
STAND_VEL_THRESHOLD = 0.1

CMD_VX = 0
CMD_VY = 1
CMD_YAW_RATE = 2
CMD_HEIGHT = 4
CMD_LEFT_HAND = 5
CMD_RIGHT_HAND = 8
CMD_LEFT_WRIST = 11
CMD_RIGHT_WRIST = 14
CMD_LEFT_GRIPPER = 17
CMD_RIGHT_GRIPPER = 18
# Gate the wrist leveler to the pick-approach rise: commanded hand z
# body-frame is above this threshold, AND the gripper is still open
# (pre-grasp). Covers the "initial lift before lowering/closing" only,
# not the post-grasp lift / carry. No UDP or cmd semantics changed —
# purely a heuristic on fields the policy already reads.
_WRIST_LEVEL_Z_THRESHOLD: float = 0.05    # pelvis-frame hand-z
_WRIST_LEVEL_GRIPPER_OPEN_MAX: float = 0.5  # cmd_gripper < this = open
# Per-tick blend coefficient for the leveler↔policy EMA. At 50 Hz, α=0.05
# gives a time-constant of ~0.4 s (≈20 ticks to settle within 1/e) — long
# enough that latch flips don't step the commanded wrist, short enough
# that the policy regains full control soon after grasp.
_WRIST_LEVEL_EMA_ALPHA: float = 0.05
# Below this weight the blend contribution is negligible — stop running
# the leveler on that side to avoid wasted FK and stale slew state.
_WRIST_LEVEL_BLEND_EPS: float = 1e-3


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


DEFAULT_POS = _resolve_keyframe(POLICY_JOINT_NAMES, KNEES_BENT_KEYFRAME)
JOINT_SCALES = _resolve_scales(POLICY_JOINT_NAMES, g1_constants.G1_ACTION_SCALE)
_LEFT_WRIST_IDX = np.asarray(
    [POLICY_JOINT_NAMES.index(j) for j in
     ("left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint")],
    dtype=np.int64,
)
_RIGHT_WRIST_IDX = np.asarray(
    [POLICY_JOINT_NAMES.index(j) for j in
     ("right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint")],
    dtype=np.int64,
)
JOINT_VEL_SCALES = np.array(
    [0.0 if i in _ANKLE_DOF_INDICES else JOINT_VEL_SCALE for i in range(NUM_JOINTS)],
    dtype=np.float32,
)


# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------
def euler_roll_pitch_from_quat(q):
    w, x, y, z = q
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    return np.array([roll, pitch], dtype=np.float32)


def quat_rotate(q, v):
    """Rotate vector ``v`` by quaternion ``q`` (wxyz). Inverse of quat_rotate_inverse."""
    w, x, y, z = q
    q_vec = np.array([x, y, z], dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    a = v * (2.0 * w * w - 1.0)
    b = np.cross(q_vec, v) * w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a + b + c


# ---------------------------------------------------------------------------
# Observation building
# ---------------------------------------------------------------------------
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
    # hand_mimic (14)
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

    # proprio (92)
    base_ang_vel = body_ang_vel * BASE_ANG_VEL_SCALE
    imu_rp = euler_roll_pitch_from_quat(root_quat)
    joint_pos_rel = (joint_pos - DEFAULT_POS) * JOINT_POS_SCALE
    joint_vel_rel = joint_vel * JOINT_VEL_SCALES
    proprio = np.concatenate([
        base_ang_vel, imu_rp, joint_pos_rel, joint_vel_rel, last_action,
    ], dtype=np.float32)

    return np.concatenate([mimic, proprio], dtype=np.float32)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx_path", help="Path to hand_dual_student.onnx")
    args = parser.parse_args()

    # Load ONNX
    session = ort.InferenceSession(args.onnx_path, providers=["CPUExecutionProvider"])
    inp_name = session.get_inputs()[0].name
    expected_shape = (1, ACTOR_OBS_DIM)
    actual_shape = tuple(session.get_inputs()[0].shape)
    if actual_shape != expected_shape:
        print(f"WARNING: ONNX input {actual_shape} != expected {expected_shape}")

    # Policy state
    last_action = np.zeros(NUM_JOINTS, dtype=np.float32)
    phase_tracker = 0.0
    dt = 1.0 / 50.0
    history: deque[np.ndarray] = deque(maxlen=ACTOR_HISTORY_LENGTH)
    history_initialized = False

    # Per-hand pre-grasp wrist-leveler latches. Edge-triggered by
    # (hand lifted + gripper open) and released when that hand's gripper
    # closes (grasp → carry transition). Ensures brief dips in commanded
    # z or momentary gripper noise during the approach don't flicker the
    # leveler off before grasp.
    wrist_level_latched_left = False
    wrist_level_latched_right = False
    # When both ROLL_ONLY and YAW_TO_TORSO are enabled, keep the wrist
    # leveler's EMA hold armed through grasp/lift/carry and only release
    # it at Molmo pick walkback. Other wrist modes keep the historical
    # gripper-close behavior.
    walkback_release_mode = (
        MOLMO_WRIST_LEVEL_ROLL_ONLY and MOLMO_WRIST_LEVEL_YAW_TO_TORSO
    )
    # Per-hand EMA hold latch. Arms when the pre-grasp latch first engages
    # and, in walkback_release_mode, stays armed until pick:walkback.
    wrist_level_hold_left = False
    wrist_level_hold_right = False
    # Per-hand EMA weight in [0, 1] blending the leveler's wrist command
    # (weight=1) with the policy's own wrist command (weight=0). In
    # walkback_release_mode, target=1 while the hold latch is armed and
    # target=0 starting at walkback; otherwise target follows the legacy
    # pre-grasp latch directly.
    wrist_level_weight_left: float = 0.0
    wrist_level_weight_right: float = 0.0
    # Independent EMA weight for the post-grasp wrist-roll offset. Fades
    # in at grasp close (pre-grasp latch drops, hold stays armed) and
    # fades out at the pick → place transition. Separate from the
    # leveller's own weight so fade-in doesn't step (leveller weight is
    # already ~1 by grasp) and the hold can persist through walkback.
    wrist_roll_offset_weight_left: float = 0.0
    wrist_roll_offset_weight_right: float = 0.0
    # Per-hand 'carrying an object' latch. See update_side_holding for
    # semantics. Drives the inverse of "vertical wanted": a hand that is
    # currently carrying stays non-vertical; a free hand rests vertical.
    holding_left = False
    holding_right = False
    # Output-side EMA state for wrist yaw. Held as Optional[float] so the
    # first tick after (re-)init seeds from the actual command without
    # smoothing — only subsequent ticks are filtered.
    prev_left_wrist_yaw_cmd: Optional[float] = None
    prev_right_wrist_yaw_cmd: Optional[float] = None
    # Edge-detect for the yaw-freeze diagnostic prints below.
    prev_left_yaw_freeze: bool = False
    prev_right_yaw_freeze: bool = False
    yaw_trace_print_counter: int = 0
    # EMA-smoothed pelvis height seen by the policy. Step changes in
    # cmd[CMD_HEIGHT] (e.g. molmo flipping between WALK_HEIGHT and
    # STAND_HEIGHT-squat at approach edges) would otherwise show up as a
    # step in the actor obs and the controller targets.
    prev_cmd_height: Optional[float] = None
    left_wrist_yaw_idx = POLICY_JOINT_NAMES.index("left_wrist_yaw_joint")
    right_wrist_yaw_idx = POLICY_JOINT_NAMES.index("right_wrist_yaw_joint")

    # Background rclpy subscriber to /molmo/active_phase. The gate below
    # restricts the leveller (latch + roll-only continuous override) to
    # pick phases only, so place-retreat doesn't drive a stale leveller.
    molmo_sub = _MolmoSubscriber()

    # Two independent toggles for the wrist behavior (both bool constants
    # in wbc_mjlab.g1_constants_custom, default False):
    #   MOLMO_WRIST_LEVEL_OVERRIDE — overwrite the commanded wrist joint
    #     positions with angles that keep the gripper upright (pinch plane
    #     horizontal, side-grasp pose).
    #   MOLMO_WRIST_ZERO_OBS — zero the wrist joint slots in the
    #     observation fed to the neural net (useful when the policy was
    #     trained with wrist=0).
    # They can run independently: zero-obs only to debug the observation
    # effect, override-only to compare what the policy commands vs what
    # we actually send.
    wrist_leveller: WristLeveller | None = None
    wrist_idx_all: np.ndarray | None = None
    # HandCommandPD shares the WristLeveller's parsed URDF, so enabling
    # either feature loads the pinocchio model exactly once.
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
            f"[hand_policy] wrist override={MOLMO_WRIST_LEVEL_OVERRIDE} "
            f"zero_obs={MOLMO_WRIST_ZERO_OBS} (indices {wrist_idx_all.tolist()})"
        )
    print(f"[hand_policy] dds_xr wrist passthrough={DDS_XR_WRIST_PASSTHROUGH}")

    hand_cmd_pd: HandCommandPD | None = None
    if MOLMO_HAND_CMD_PD_ENABLE:
        assert wrist_leveller is not None  # guarded by the block above
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
            f"[hand_policy] hand_cmd_pd enabled "
            f"kp={MOLMO_HAND_CMD_PD_KP} kd={MOLMO_HAND_CMD_PD_KD} "
            f"kp_xy={kp_xy} kd_xy={kd_xy} "
            f"deadband={MOLMO_HAND_CMD_PD_DEADBAND_M}m "
            f"max_shape={MOLMO_HAND_CMD_PD_MAX_SHAPE_M}m"
        )

    # Capture-point filter. Engages only when the commanded pelvis height
    # drops into the squat band — at standing heights the policy already
    # keeps the capture point well centered. CAPTURE_POINT_METHOD picks PD
    # ("centering") or CBF.
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
            f"[hand_policy] capture-point filter enabled "
            f"(method={CAPTURE_POINT_METHOD}, kp={CAPTURE_POINT_PD_KP}, "
            f"engages below cmd_height={SQUAT_SMOOTHING_HEIGHT} m)"
        )

    # UDP socket
    sock = create_udp_socket(UDP_HOST, UDP_POLICY_PORT)
    sim_addr = (UDP_HOST, UDP_SIM_PORT)
    print(f"Hand policy UDP: listening on {UDP_HOST}:{UDP_POLICY_PORT}")

    next_tick = time.perf_counter()

    # Track sim step_id to detect sim_node resets: step_count is zeroed on
    # viser "Reset sim", so any backwards jump in step_id means we must
    # clear obs history and per-tick latched state.
    prev_step_id = -1

    try:
        while True:
            # 50 Hz real-time pacing
            now = time.perf_counter()
            sleep_time = next_tick - now
            if sleep_time > 0:
                time.sleep(sleep_time)
            next_tick += dt

            # 1. Non-blocking drain to latest state
            latest_data = None
            sock.setblocking(False)
            try:
                while True:
                    latest_data, _ = sock.recvfrom(STATE_BYTES + 64)
            except BlockingIOError:
                pass
            sock.setblocking(True)

            if latest_data is None:
                continue

            step_id, root_quat, root_pos, body_lin_vel, body_ang_vel, \
                joint_pos, joint_vel, cmd = unpack_state(latest_data)

            # Sim reset detected: step_id went backwards (sim_node zeroed
            # step_count). Clear every piece of state that survives a
            # process-local reset so the next tick starts cold.
            if step_id < prev_step_id:
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
                print(
                    "[hand_policy] sim reset detected "
                    f"(step_id {prev_step_id} -> {step_id}); cleared history + state",
                    flush=True,
                )
            prev_step_id = step_id

            # EMA-smooth the incoming pelvis height command. Step changes
            # at the source (e.g. molmo flipping locomoting on/off) would
            # otherwise show up as a step in the policy obs. First tick
            # (or first tick after a sim reset) seeds without smoothing.
            raw_cmd_height = float(cmd[CMD_HEIGHT])
            if prev_cmd_height is None:
                prev_cmd_height = raw_cmd_height
            else:
                prev_cmd_height = (
                    (1.0 - MOLMO_SQUAT_EMA_ALPHA) * prev_cmd_height
                    + MOLMO_SQUAT_EMA_ALPHA * raw_cmd_height
                )
            cmd = cmd.copy()
            cmd[CMD_HEIGHT] = np.float32(prev_cmd_height)

            # Compute leveller output only if we'll use it on the action
            # side. Rotation-only math, so root_pos doesn't matter
            # (hardware_node passes zeros). Either side may be None
            # (gimbal lock) — in that case skip that side's override and
            # let the policy's own wrist command pass through.
            wrist_override_rpy: tuple[np.ndarray | None, np.ndarray | None] | None = None
            # Edge-triggered pre-grasp latch:
            #   on  = hand lifted (z > threshold) AND gripper open.
            #   off = gripper closes (grasp done → carry starting).
            # A separate EMA hold latch keeps the roll-only blend active
            # after grasp and releases it at pick:walkback.
            left_hand_z = float(cmd[CMD_LEFT_HAND + 2])
            right_hand_z = float(cmd[CMD_RIGHT_HAND + 2])
            left_gripper_open = float(cmd[CMD_LEFT_GRIPPER]) < _WRIST_LEVEL_GRIPPER_OPEN_MAX
            right_gripper_open = float(cmd[CMD_RIGHT_GRIPPER]) < _WRIST_LEVEL_GRIPPER_OPEN_MAX
            use_walkback_release = walkback_release_mode and molmo_sub.has_phase()
            left_allowed = molmo_sub.leveller_allowed_for("l")
            right_allowed = molmo_sub.leveller_allowed_for("r")
            left_raw_target = molmo_sub.raw_target_for("l")
            right_raw_target = molmo_sub.raw_target_for("r")

            # Vertical is the default resting pose. A side goes
            # non-vertical only while it is carrying an object — that
            # spans pick (all sub-phases) → idle gap between pick and
            # place → place pre-release → the ``release`` phase itself.
            # Once place advances past release (home/standup/walkback/
            # carry_back/carry_lower/ee_tracked_reverse), the hand is
            # empty again and the latch clears. Bimanual phases reset
            # both latches (bimanual already drives both wrists vertical
            # via the shared pose).
            phase = molmo_sub.current_phase()
            bimanual_active_now = _is_bimanual_active(phase)
            holding_left = _update_side_holding(holding_left, phase, "l")
            holding_right = _update_side_holding(holding_right, phase, "r")
            left_vertical_wanted = not holding_left
            right_vertical_wanted = not holding_right
            any_vertical_wanted = left_vertical_wanted or right_vertical_wanted

            # Single-hand latches run only on sides that are NOT currently
            # vertical_wanted. A vertical_wanted side is driven directly
            # by the vertical-hold latch above, so clear its legacy
            # latches here so they don't fight the vertical blend.
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

            # EMA weight drives the leveler↔policy blend. Vertical_wanted
            # sides pin target=1; other sides keep the legacy single-hand
            # timing.
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

            # Post-grasp roll-offset EMA — skip on vertical_wanted sides
            # (vertical pose already subsumes the "rotate for carry"
            # role). Otherwise: target=1 once the gripper has closed and
            # the hold latch is still armed, decays back to 0 at place
            # start.
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

            # ROLL_ONLY keeps running the leveler while weight > eps so
            # there's a current value to blend against the policy during
            # the fade. Any_vertical_wanted also forces this branch so
            # bimanual / single-non-active vertical mode applies even if
            # ROLL_ONLY were off. Non-ROLL_ONLY uses the latch directly
            # (hard override, no blend — legacy behavior).
            if any_vertical_wanted or MOLMO_WRIST_LEVEL_ROLL_ONLY:
                left_leveler_active = wrist_level_weight_left > _WRIST_LEVEL_BLEND_EPS
                right_leveler_active = wrist_level_weight_right > _WRIST_LEVEL_BLEND_EPS
            else:
                left_leveler_active = left_allowed and wrist_level_latched_left
                right_leveler_active = right_allowed and wrist_level_latched_right
            # Hand-position PD stays on the pre-grasp gate and drops as
            # soon as the gripper closes, even if the wrist leveler keeps
            # blending through carry/walkback.
            left_pd_active = left_allowed and wrist_level_latched_left
            right_pd_active = right_allowed and wrist_level_latched_right
            any_leveler_active = left_leveler_active or right_leveler_active

            # Hard-kill all wrist suppression paths during fall recovery.
            # The EMA + latching above keeps the leveler engaged via the
            # left_vertical_wanted / right_vertical_wanted gate (which is
            # True whenever a side isn't carrying anything — the normal
            # idle-hand state), so the parse-to-None phase trick is NOT
            # enough on its own. Snap weights to zero and force every
            # downstream activity flag off so the override block, hand-PD
            # shaping, and yaw-freeze paths all short-circuit this tick.
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

            # Freeze the leveller's yaw input so the wrist holds its last
            # committed yaw instead of chasing perception jitter while the
            # gripper is closing around the object. Two triggers:
            #   (a) phase == pick:grasp (50 ms close dwell), and
            #   (b) gripper midpoint within MOLMO_WRIST_YAW_FREEZE_RADIUS_M
            #       of the raw target while leveller is active. This kicks
            #       in during the tail of pick_approach, mirroring the
            #       early-close-into-grasp gate in molmo_node, so yaw stops
            #       being adjusted before the gripper begins closing.
            # Roll/pitch leveling stays active in both cases.
            left_yaw_freeze = molmo_sub.is_pick_yaw_freeze_for("l")
            right_yaw_freeze = molmo_sub.is_pick_yaw_freeze_for("r")
            left_yaw_freeze_phase = left_yaw_freeze
            right_yaw_freeze_phase = right_yaw_freeze
            left_yaw_freeze_box = False
            right_yaw_freeze_box = False
            left_err_wrist: Optional[np.ndarray] = None
            right_err_wrist: Optional[np.ndarray] = None
            # Box check expressed in the gripper's "horizontal grasp frame":
            # depth axis = world-horizontal projection of the finger axis,
            # lateral = perpendicular horizontal axis, vertical = gravity.
            # Decouples the test from any pitch the policy applies to the
            # wrist during ee_tracked, so a tilted gripper doesn't shift
            # the box around the world. Z half-extent is per-side: half
            # the OBB's world-up extent published with raw_target (= the
            # vertical reach the fingers need to clear past object center),
            # or MOLMO_PICK_EARLY_CLOSE_Z_M when FFS missed.
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

            # Diagnostic: rising/falling edge of freeze (per side) and a
            # ~1 Hz trace. Source is either the pick:grasp phase or the
            # 3-axis box on err = (depth, lateral, vertical) in the
            # horizontal-grasp frame (heading-aligned XY, gravity-aligned Z).
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
                print(f"[yaw_freeze] L active ({src})", flush=True)
            if right_yaw_freeze and not prev_right_yaw_freeze:
                src = "phase" if right_yaw_freeze_phase else (
                    _box_src(right_err_wrist, "r") if right_yaw_freeze_box else "?"
                )
                print(f"[yaw_freeze] R active ({src})", flush=True)
            if not left_yaw_freeze and prev_left_yaw_freeze:
                print("[yaw_freeze] L cleared", flush=True)
            if not right_yaw_freeze and prev_right_yaw_freeze:
                print("[yaw_freeze] R cleared", flush=True)
            prev_left_yaw_freeze = left_yaw_freeze
            prev_right_yaw_freeze = right_yaw_freeze

            yaw_trace_print_counter += 1
            # While-held heartbeat — every 25 ticks (~0.5s) print that the
            # freeze is still latched, so it's visible at a glance whether
            # we're currently holding yaw or not.
            if (left_yaw_freeze or right_yaw_freeze) and yaw_trace_print_counter % 25 == 0:
                sides = []
                if left_yaw_freeze:
                    sides.append("L")
                if right_yaw_freeze:
                    sides.append("R")
                print(f"[yaw_freeze] held: {','.join(sides)}", flush=True)
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
                    # Per-side orientation mode. A vertical_wanted side
                    # gets roll-only "vertical" pose with policy pitch+yaw
                    # preserved (no raw-target yaw, no torso-forward yaw).
                    # A non-vertical side keeps the legacy roll-only path
                    # with raw-target or torso-forward yaw.
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
                    # Freeze pins yaw to the leveler's last emitted EMA
                    # value (clearing raw_target so the yaw target branches
                    # are skipped — the freeze flag overrides them anyway,
                    # but None'ing the target keeps slew/clamp inputs clean).
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
                    left_target_for_compute = (
                        left_raw_target
                        if left_leveler_active and left_raw_target is not None
                        else (_FWD_TARGET if left_leveler_active else None)
                    )
                    right_target_for_compute = (
                        right_raw_target
                        if right_leveler_active and right_raw_target is not None
                        else (_FWD_TARGET if right_leveler_active else None)
                    )
                    if left_yaw_freeze:
                        left_target_for_compute = None
                    if right_yaw_freeze:
                        right_target_for_compute = None
                    left_rpy, right_rpy = wrist_leveller.compute(
                        joint_pos, root_pos, root_quat,
                        left_target_body=left_target_for_compute,
                        right_target_body=right_target_for_compute,
                        dt=dt,
                        active_left=left_leveler_active,
                        active_right=right_leveler_active,
                        freeze_yaw_left=left_yaw_freeze,
                        freeze_yaw_right=right_yaw_freeze,
                    )
                    wrist_override_rpy = (left_rpy, right_rpy)
            else:
                # Neither side active — poke leveler with both inactive
                # so it clears slew memory. Next activation re-inits from
                # the then-current physical wrist via compute()'s is-None
                # check.
                if wrist_leveller is not None:
                    wrist_leveller.compute(
                        joint_pos, root_pos, root_quat,
                        left_target_body=None, right_target_body=None, dt=dt,
                        active_left=False, active_right=False,
                    )
            # 1b. Command-level hand-position PD: shape cmd[LEFT_HAND] /
            # cmd[RIGHT_HAND] by the measured wrist_yaw_link tracking
            # error so the policy sees a stretched target. Gated off only
            # for bimanual (both wrists vertical) — during single-hand
            # non-active-vertical, the active side still needs PD, and
            # the non-active side's PD is already off via its per-side
            # ``*_pd_active`` flag. Operates on a copy of cmd — upstream
            # bytes are untouched.
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

            # Independently of the action override, optionally zero the
            # wrist slots in the observation so the net sees what it was
            # trained on. Scope the zeroing to obs-only locals so the CP
            # filter below still sees the real wrist state for FK.
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

            # 2. Build observation
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

            # 3. Inference
            policy_action = session.run(None, {inp_name: obs})[0][0]
            phase_tracker = (phase_tracker + dt) % 1.0

            raw_target = DEFAULT_POS + policy_action * JOINT_SCALES
            # last_action feeds the policy's own observation history on
            # the next tick — it must reflect what the *policy* produced,
            # not what the leveller injected. Compute it from raw_target
            # (pre-override) so the policy's recurrent state stays
            # consistent with its training distribution.
            last_action = ((raw_target - DEFAULT_POS) / JOINT_SCALES).astype(np.float32)

            sent_target = raw_target.copy()
            if wrist_override_rpy is not None:
                left_rpy, right_rpy = wrist_override_rpy
                roll_only_apply = MOLMO_WRIST_LEVEL_ROLL_ONLY or any_vertical_wanted
                if roll_only_apply:
                    # Blend leveler and policy wrist by the EMA weight so
                    # roll-only release fades instead of stepping. On a
                    # vertical_wanted side, skip the post-grasp roll
                    # offset and keep policy pitch+yaw (only roll gets
                    # overridden); non-vertical sides keep the legacy
                    # roll-only + torso-yaw behavior.
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
                            # Freeze drops the (1-w) policy term — the leveler
                            # is currently holding yaw and we don't want the
                            # policy's atan2-amplified drift leaking through.
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
                    # Non-ROLL_ONLY: hard override on the active side
                    # (compute returned None for inactive sides).
                    if left_rpy is not None:
                        sent_target[list(wrist_leveller.left_policy_indices)] = left_rpy
                    if right_rpy is not None:
                        sent_target[list(wrist_leveller.right_policy_indices)] = right_rpy

            # 3a-bis. dds_xr VR wrist passthrough. Overwrites the wrist
            # slots of sent_target with default_pose + (RPY relative to
            # Left-B neutral) as published by dds_xr_node on /g1/command.
            # The dds_xr launch path also disables MOLMO_WRIST_LEVEL_OVERRIDE,
            # so the leveller block above is skipped in this mode.
            if DDS_XR_WRIST_PASSTHROUGH:
                sent_target[_LEFT_WRIST_IDX] = (
                    DEFAULT_POS[_LEFT_WRIST_IDX] + cmd[CMD_LEFT_WRIST:CMD_LEFT_WRIST + 3]
                )
                sent_target[_RIGHT_WRIST_IDX] = (
                    DEFAULT_POS[_RIGHT_WRIST_IDX] + cmd[CMD_RIGHT_WRIST:CMD_RIGHT_WRIST + 3]
                )

            # 3b. Output-side wrist-yaw EMA — place phases only. Pick
            # approach needs precise wrist yaw to grasp cleanly; place
            # is where the upstream snaps (vertical_wanted gate flips at
            # release → ee_tracked_reverse, raw_target switches across
            # step boundaries, geometric degeneracies during stack
            # descent) actually need attenuating. Bimanual place engages
            # both hands; single-hand place engages only the active hand.
            # When inactive, prev is reset so the next entry seeds from
            # the current command without backfilling stale state.
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

            # 3c. Capture-point centering PD — engaged only while squatting.
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

            # 4. Send action (with wrist override applied)
            sock.sendto(pack_action(step_id, sent_target.astype(np.float32)), sim_addr)

    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
        molmo_sub.shutdown()
        print("Hand policy stopped.")


if __name__ == "__main__":
    main()
