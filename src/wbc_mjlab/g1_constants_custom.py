"""HANDOFF-matched G1 constants.

All module-level constants for wbc_mjlab live here so config.py, rl_cfg.py,
and any future modules have a single import source.

The robot keeps mjlab's stock G1 actuator model (PD gains, effort limits, and
armatures from the Unitree motor specs). Action scales and clipping below are
taken from the HANDOFF IsaacGym training configs:
  - WBC teacher: HANDOFF/legged_gym/envs/g1/g1_mimic_config.py
  - Loco teacher: HANDOFF/legged_gym/envs/g1/g1_loco_config.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec
from mjlab.entity import EntityCfg

##
# CBF (Control Barrier Function) parameters.
##

CBF_H_TARGET: float = 0.03
CBF_SAFETY_MARGIN: float = 0.03
CBF_ALPHA_TRAINING: float = 0.5
CBF_ALPHA_DEPLOY: float = 0.5
CBF_VEL_MAX_ANKLE_TRAINING: float = 5.0
CBF_VEL_MAX_WAIST_TRAINING: float = 5.0
CBF_VEL_MAX_ANKLE_DEPLOY: float = 5.0
CBF_VEL_MAX_WAIST_DEPLOY: float = 5.0
CBF_ACTIVATION_WIDTH_VEL: float = 0.03

# Squat-only low-pass smoothing on the policy's action targets. Engages
# when the user commands a low pelvis height (squat) — addresses the
# observed high-frequency oscillation at low heights without touching
# standing / walking dynamics. Pure EMA on sent_target:
#     y_n = alpha * x_n + (1 - alpha) * y_{n-1}
# alpha=1 → no smoothing, alpha=0 → frozen. Lower alpha = more damping
# but more lag.
SQUAT_SMOOTHING_ENABLE: bool = False
SQUAT_SMOOTHING_HEIGHT: float = 0.75   # m — engage when cmd height below this
SQUAT_SMOOTHING_ALPHA: float = 0.3     # EMA coefficient on the new action

# Capture-point stabilization on the policy's action targets. Engages only
# when the user commands a low pelvis height (squat) — at low heights the
# capture point drifts toward the back of the support polygon and termination
# rate spikes. CAPTURE_POINT_METHOD selects the algorithm:
#   "pd"  → proportional centering (G1CapturePointCBF mode="centering")
#   "cbf" → control-barrier-function constraint (mode="cbf"), reuses
#           CBF_ALPHA_DEPLOY / CBF_VEL_MAX_*_DEPLOY / CBF_H_TARGET above.
# Reuses SQUAT_SMOOTHING_HEIGHT as the engagement threshold.
CAPTURE_POINT_PD_ENABLE: bool = True
CAPTURE_POINT_METHOD: str = "cbf"      # "pd" or "cbf"
CAPTURE_POINT_PD_KP: float = 5.0      # proportional gain (PD method only)

##
# Sim-to-real hardware payloads: Jetson on back, Dex1-1 hands replacing
# the stock rubber hands. Attached programmatically via MjSpec so we never
# modify the vendored G1 XML.
##

# Rubber-hand mass baked into MJCF wrist_yaw_link.mass (0.254576 kg total).
# The URDF splits this into wrist_yaw_link (0.084576 kg) + left_rubber_hand
# (0.170 kg), so we subtract exactly 0.170 from the MJCF wrist to match the
# URDF's body-wise accounting.
RUBBER_HAND_MASS_ESTIMATE: float = 0.170


@dataclass(frozen=True)
class _NewBodyCfg:
  """A rigid child body welded to a parent link (no joint)."""

  name: str
  parent_body: str
  pos: tuple[float, float, float]
  half_size: tuple[float, float, float]  # box geom half-sizes (x, y, z)
  mass: float
  diaginertia: tuple[float, float, float]
  quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
  rgba: tuple[float, float, float, float] = (0.15, 0.15, 0.15, 1.0)


# NVIDIA Jetson + mount + cameras on the robot's back (−X face of torso_link).
# Dims 243.19 × 112.40 × 56.88 mm; long axis along Z (spine).
# Inertia = m/12 · (sum of squares of the two perpendicular sides).
JETSON_PAYLOAD = _NewBodyCfg(
  name="jetson_payload",
  parent_body="torso_link",
  # torso visual mesh back is at x ≈ -0.067 in torso_link frame; place the
  # box front flush against it: center_x = -0.067 - half_depth (0.02844) ≈ -0.095.
  pos=(-0.09, 0.0, 0.15),
  half_size=(0.02844, 0.05620, 0.12160),
  mass=2.0,
  diaginertia=(0.011962, 0.010396, 0.002645),
  rgba=(0.0, 0.0, 0.0, 1.0),  # pure black (NVIDIA Jetson casing)
)

# Unitree Dex1-1 hand per wrist. Replaces the stock rubber_hand geom.
# Dims 143 × 78 × 67 mm; long axis along X (palm-forward).
_DEX_POS = (0.115, 0.0, 0.0)
_DEX_HALF = (0.0715, 0.039, 0.0335)
_DEX_MASS = 0.55
_DEX_INERTIA = (0.000485, 0.001143, 0.001216)
LEFT_DEX1_1 = _NewBodyCfg(
  name="left_dex1_1",
  parent_body="left_wrist_yaw_link",
  pos=_DEX_POS,
  half_size=_DEX_HALF,
  mass=_DEX_MASS,
  diaginertia=_DEX_INERTIA,
)
RIGHT_DEX1_1 = _NewBodyCfg(
  name="right_dex1_1",
  parent_body="right_wrist_yaw_link",
  pos=_DEX_POS,
  half_size=_DEX_HALF,
  mass=_DEX_MASS,
  diaginertia=_DEX_INERTIA,
)

# (wrist_body, rubber_hand_mesh_name, hand_collision_geom_name)
_HAND_SWAPS: tuple[tuple[str, str, str], ...] = (
  ("left_wrist_yaw_link", "left_rubber_hand", "left_hand_collision"),
  ("right_wrist_yaw_link", "right_rubber_hand", "right_hand_collision"),
)

# Master toggle: sim-to-real hardware payloads (Jetson on back, Dex1-1 hands)
# are only applied when ``WBC_ATTACH_PAYLOADS`` is truthy. Default off — the
# seed-variant train scripts opt in by exporting the env var. Also serves as
# the per-run disable toggle.
ATTACH_PAYLOADS: bool = os.environ.get("WBC_ATTACH_PAYLOADS", "").lower() in (
  "1", "true", "yes", "on",
)


def _add_new_body(spec: mujoco.MjSpec, p: _NewBodyCfg) -> None:
  parent = spec.body(p.parent_body)
  body = parent.add_body(name=p.name, pos=list(p.pos), quat=list(p.quat))
  body.mass = p.mass
  body.inertia = list(p.diaginertia)
  body.ipos = [0.0, 0.0, 0.0]
  body.iquat = [1.0, 0.0, 0.0, 0.0]
  visual_cls = spec.find_default("visual")
  collision_cls = spec.find_default("collision")
  # Attach a matte material per-payload so the configured rgba isn't washed
  # out by headlight specular highlights (default geom specular ≈ 0.5 makes
  # a black geom look grey under our ambient=0.45 headlight).
  mat_name = f"{p.name}_mat"
  spec.add_material(
    name=mat_name,
    rgba=list(p.rgba),
    specular=0.0,
    reflectance=0.0,
    shininess=0.0,
  )
  body.add_geom(
    default=visual_cls,
    name=f"{p.name}_visual",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=list(p.half_size),
    rgba=list(p.rgba),
    material=mat_name,
  )
  body.add_geom(
    default=collision_cls,
    name=f"{p.name}_collision",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=list(p.half_size),
  )


def _delete_geoms(spec: mujoco.MjSpec, body_name: str, *, names=(), meshes=()) -> None:
  body = spec.body(body_name)
  victims = [
    g for g in body.geoms if (g.name in names) or (g.meshname in meshes)
  ]
  for g in victims:
    spec.delete(g)


def _apply_payloads_to_spec(
  spec: mujoco.MjSpec,
  *,
  include_jetson: bool = True,
  swap_hands: bool = True,
) -> mujoco.MjSpec:
  """Mutate an already-loaded MjSpec to attach Jetson + Dex1-1 hardware.

  Flags let callers opt out per piece — e.g., a gripper-XML variant keeps its
  grippers by passing swap_hands=False.
  """
  if not ATTACH_PAYLOADS:
    return spec
  if include_jetson:
    _add_new_body(spec, JETSON_PAYLOAD)
  if swap_hands:
    for wrist_name, rubber_mesh, hand_col in _HAND_SWAPS:
      _delete_geoms(spec, wrist_name, names=(hand_col,), meshes=(rubber_mesh,))
      wrist = spec.body(wrist_name)
      wrist.mass = wrist.mass - RUBBER_HAND_MASS_ESTIMATE
    _add_new_body(spec, LEFT_DEX1_1)
    _add_new_body(spec, RIGHT_DEX1_1)
  return spec


_PAYLOAD_WRAPPED_ATTR = "__wbc_payloads_wrapped__"


def wrap_spec_fn_with_payloads(
  base_spec_fn: "callable[[], mujoco.MjSpec]",
  *,
  include_jetson: bool = True,
  swap_hands: bool = True,
) -> "callable[[], mujoco.MjSpec]":
  """Wrap any EntityCfg.spec_fn so the returned MjSpec carries our hardware.

  The returned wrapper is tagged so repeated wrapping is detectable and can
  be skipped by callers that want idempotent application.
  """
  if getattr(base_spec_fn, _PAYLOAD_WRAPPED_ATTR, False):
    return base_spec_fn

  def _wrapped() -> mujoco.MjSpec:
    return _apply_payloads_to_spec(
      base_spec_fn(),
      include_jetson=include_jetson,
      swap_hands=swap_hands,
    )

  setattr(_wrapped, _PAYLOAD_WRAPPED_ATTR, True)
  return _wrapped


def get_spec_with_payloads() -> mujoco.MjSpec:
  return _apply_payloads_to_spec(get_spec())


# Tag so wrap_spec_fn_with_payloads() treats it as already-wrapped and won't
# double-apply (which would add the Jetson twice and subtract rubber-hand mass
# twice).
setattr(get_spec_with_payloads, _PAYLOAD_WRAPPED_ATTR, True)


def attach_payloads_to_scene_robot(cfg) -> None:
  """Idempotently wrap ``cfg.scene.entities['robot'].spec_fn`` with the
  Jetson+Dex1-1 payload so every task carries the sim-to-real hardware.

  Safe to call multiple times: if the spec_fn is already wrapped, no-op.
  """
  import copy as _copy

  robot_cfg = cfg.scene.entities["robot"]
  if not isinstance(robot_cfg, EntityCfg):
    return
  if getattr(robot_cfg.spec_fn, _PAYLOAD_WRAPPED_ATTR, False):
    return
  robot_cfg = _copy.deepcopy(robot_cfg)
  robot_cfg.spec_fn = wrap_spec_fn_with_payloads(robot_cfg.spec_fn)
  cfg.scene.entities["robot"] = robot_cfg






##
# Constants migrated from config.py.
##

_BASE_ANG_VEL_SCALE: float = 0.25
_JOINT_POS_SCALE: float = 1.0
_JOINT_VEL_SCALE: float = 0.05
_ANKLE_DOF_INDICES: tuple[int, ...] = (4, 5, 10, 11)
_JOINT_VEL_SCALE_WITH_ANKLE_MASK: tuple[float, ...] = tuple(
  0.0 if i in _ANKLE_DOF_INDICES else _JOINT_VEL_SCALE for i in range(29)
)

_HANDOFF_BASE_MASS_RANGE: tuple[float, float] = (-3.0, 3.0)
_HANDOFF_MOTOR_STRENGTH_RANGE: tuple[float, float] = (0.8, 1.2)
_WBC_DEFAULT_NUM_ENVS: int = 4096
_HANDOFF_LOCO_MOTION_FILE: str = "/home/yangl/handoff/wbc_handoff_data/dataset.yaml"
_LOCO_HEIGHT_CAP: float = 0.78
_LOCO_WARMUP_SCALE: float = 0.5
_LOCO_FIXED_LIN_VEL_RANGE: tuple[float, float] = (-1.0, 1.0)
_LOCO_FIXED_ANG_VEL_RANGE: tuple[float, float] = (-1.0, 1.0)
_LOCO_FIXED_BASE_HEIGHT_RANGE: tuple[float, float] = (0.45, 0.78)
_DUAL_BASELINE_MOTION_GATE_THRESHOLD: float = 0.1
_DUAL_BASELINE_ADDED_LOCO_REWARDS: tuple[str, ...] = (
  "foot_clearance",
  "foot_swing_height",
  "air_time",
  "soft_landing",
)
_LOCO_BODY_STD_WALKING: dict[str, float] = {
  r".*hip_pitch.*": 0.3,
  r".*hip_roll.*": 0.15,
  r".*hip_yaw.*": 0.15,
  r".*knee.*": 0.35,
  r".*ankle_pitch.*": 0.25,
  r".*ankle_roll.*": 0.1,
  r".*waist_yaw.*": 0.2,
  r".*waist_roll.*": 0.08,
  r".*waist_pitch.*": 0.1,
}
_LOCO_BODY_STD_RUNNING: dict[str, float] = {
  r".*hip_pitch.*": 0.5,
  r".*hip_roll.*": 0.2,
  r".*hip_yaw.*": 0.2,
  r".*knee.*": 0.6,
  r".*ankle_pitch.*": 0.35,
  r".*ankle_roll.*": 0.15,
  r".*waist_yaw.*": 0.3,
  r".*waist_roll.*": 0.08,
  r".*waist_pitch.*": 0.2,
}

##
# Constant migrated from rl_cfg.py.
##

_WBC_SAVE_INTERVAL: int = 1000

##
# Molmo node config: ROS topic, env var names + defaults, UI window titles,
# and the baseline pose command written by molmo_node._publish_command.
##

MOLMO_STATUS_TOPIC: str = "/molmo/status"
# Current pick/place/idle phase, published by molmo_node once per tick
# and consumed by hand_policy to gate the wrist leveller. String payload:
# ``f"{action}:{motion_phase}:{hand}:{hand_mode}"`` (e.g.,
# "pick:ee_tracked:l:single") while a step is active, "idle" otherwise.
# Keep semantics open (string not bool) so additional phase-dependent
# behavior can key off the same topic later.
MOLMO_ACTIVE_PHASE_TOPIC: str = "/molmo/active_phase"
# Raw Molmo object reference in pelvis frame, published by molmo_node and
# consumed by hand_policy for gripper-midpoint correction. Float payload:
# ``[left_active, lx, ly, lz, right_active, rx, ry, rz]``.
MOLMO_RAW_HAND_TARGET_TOPIC: str = "/molmo/raw_hand_targets"
MOLMO_CAMERA_NAMES_ENV: str = "WBC_MJLAB_MOLMO_CAMERA_NAMES"
MOLMO_SERVER_HOST_ENV: str = "WBC_MJLAB_MOLMO_HOST"
MOLMO_SERVER_PORT_ENV: str = "WBC_MJLAB_MOLMO_PORT"
MOLMO_POINT_COORD_MODE_ENV: str = "WBC_MJLAB_MOLMO_POINT_COORD_MODE"
MOLMO_VISER_FALLBACK_ENV: str = "WBC_MJLAB_MOLMO_VISER_FALLBACK"
MOLMO_DEFAULT_PROMPT_ENV: str = "WBC_MJLAB_MOLMO_DEFAULT_PROMPT"
MOLMO_WRIST_LEVEL_OVERRIDE_ENV: str = "WBC_MJLAB_MOLMO_WRIST_LEVEL_OVERRIDE"
MOLMO_WRIST_ZERO_OBS_ENV: str = "WBC_MJLAB_MOLMO_WRIST_ZERO_OBS"

MOLMO_DEFAULT_CAMERA_NAME: str = "head_camera"
# Default to the Hunyuan backend (HY_HOST / HY_PORT). The legacy Molmo TCP
# server lived on 9876; the hunyuan swap (commit 7e0316d) moved the VLM to
# Hunyuan's OpenAI-compatible endpoint on HY_PORT (8180 on this workstation).
# Keeping the MOLMO_DEFAULT_SERVER_* names so env-var overrides keep working,
# but aliasing the defaults to the HY constants avoids a stale-port connection
# refused at query time.
MOLMO_DEFAULT_SERVER_HOST: str = "127.0.0.1"
MOLMO_DEFAULT_SERVER_PORT: int = 8180
MOLMO_DEFAULT_POINT_COORD_MODE: str = "norm1000"
# MOLMO_DEFAULT_PROMPT: str = "pick up the red box and put it in the green bin, then pick up the blue box and put it in the yellow bin, turn right, then pick up the purple box with both hands then turn right and put it on the table"
MOLMO_DEFAULT_PROMPT: str = "Pick up the blue box with both hands, turn around and hand it over"

DEFAULT_HEIGHT: float = 0.78
DEFAULT_HAND_X: float = 0.162
DEFAULT_HAND_Y: float = 0.00
DEFAULT_HAND_Z: float = 0.07
# X back-off applied during the home_xy phase only (the subsequent home
# phase still uses DEFAULT_HAND_X). Pulls the wrist toward the body
# while it traverses to the home XY at the held arm Z, so the hand
# clears anything the gripper might still be over before the home
# phase descends to DEFAULT_HAND_Z.
MOLMO_HOME_XY_X_BACKOFF_M: float = 0.1

HAND_POS_LIMIT_XYZ: tuple[float, float, float] = (0.6, 0.5, 0.8)
HAND_NEG_LIMIT_XYZ: tuple[float, float, float] = (-0.25, -0.23, -0.1)

MOLMO_UI_WINDOW: str = "Molmo Camera"
MOLMO_DEPTH_WINDOW: str = "Molmo Depth"

# molmo_node._publish_command picks between these based on whether an
# approach / search / walkback is currently driving a nonzero velocity.
# Dropping the pelvis during locomotion buys stability; standing tall
# when parked extends reach.
MOLMO_WALK_HEIGHT: float = 0.78
MOLMO_STAND_HEIGHT: float = 0.78
# Lower bound on commanded pelvis height when auto-squatting to reach a
# low manipulation target. Training distribution covers 0.5–0.78 m, but
# the squat-to-reach path commits to a deep crouch at 0.28 m intentionally
# — picks of low objects benefit more from getting the pelvis fully down
# than from staying inside the trained envelope. Watch for balance
# regressions on the real robot; revert to 0.55 (training-margin floor)
# if the deep squat destabilizes.
MOLMO_SQUAT_MIN_HEIGHT: float = 0.55
# Per-tick EMA coefficient for the height transition. At the 50 Hz
# command publish rate, α=0.02 gives τ ≈ 1 s — slow enough the knee PDs
# don't spike when the squat delta changes between steps.
MOLMO_SQUAT_EMA_ALPHA: float = 0.04

# Place-refine: after the place ee_tracked descent, before opening the
# gripper, re-query Molmo once and move the hand to the new target.
# Hand-only motion — never re-enters approach / walking. The nudge is
# always attempted; MAX_DELTA_M is the reference distance for INTERP_SEC,
# and larger reaches stretch the interp time proportionally to keep the
# hand speed bounded (reach > MAX_DELTA_M → interp = INTERP_SEC * delta /
# MAX_DELTA_M). If the query doesn't come back in TIMEOUT_SEC we just
# proceed to release.
MOLMO_PLACE_REFINE_ENABLE: bool = False
MOLMO_PLACE_REFINE_INTERP_SEC: float = 1.0
MOLMO_PLACE_REFINE_TIMEOUT_SEC: float = 2.0
MOLMO_PLACE_REFINE_MAX_DELTA_M: float = 0.15

# Planner node (deploy/controller/planner_node.py):
#   PLANNER_PLAN_RATE_HZ      — tick/replan frequency for the ESDF build
#                               + A* (loco) + manip planner.
#   PLANNER_STALE_DEPTH_SEC   — max age of the latest depth frame before a
#                               tick publishes "no_depth" and skips the plan.
#   PLANNER_MARKER_PERIOD_SEC — throttle for the ESDF RViz marker publish.
#   PLANNER_LOCO_DEAD_RECKONING — when True, the loco planner skips A* and
#                               emits a canned "strafe y → let tracker yaw →
#                               approach x" path. The forward leg is trimmed
#                               at the first point where the body-height SDF
#                               drops below PLANNER_LOCO_DR_OBSTACLE_STOP_M,
#                               so the robot stops a safe distance before
#                               any box. Useful when A* collapses on
#                               close-approach because ESDF occupancy of the
#                               target's surface dominates the planning
#                               volume — the canned path just drives the
#                               robot toward the goal until it is about to
#                               clip something.
PLANNER_PLAN_RATE_HZ: float = 1.0
PLANNER_STALE_DEPTH_SEC: float = 0.5
PLANNER_MARKER_PERIOD_SEC: float = 0.5
PLANNER_LOCO_DEAD_RECKONING: bool = False
PLANNER_LOCO_DR_OBSTACLE_STOP_M: float = 0.35

##
# Molmo execution state machine (HANDOFF defaults).
##

MOLMO_INTERP_SEC: float = 3.0
MOLMO_GRASP_DWELL_SEC: float = 0.05
MOLMO_SPLIT_MOVE_XY_Z: bool = True
# Wrist leveller (deploy/common/wrist_level.py) exposes two independent
# toggles consumed by deploy/policy/hand_policy.py at startup:
#   MOLMO_WRIST_LEVEL_OVERRIDE — overwrite the commanded wrist joint
#     positions with angles that keep the gripper upright (side-grasp).
#   MOLMO_WRIST_ZERO_OBS — zero the wrist joint slots in the observation
#     fed to the policy's neural net (useful when the policy was trained
#     with wrist=0).
# Either can run on its own. Xbox-driven launchers override these via env
# so teleop keeps the policy's raw wrist behavior.
# Both default to True for normal manipulation. They are suppressed at
# runtime when molmo_node publishes MOLMO_FALL_RECOVERY_PHASE on the
# active-phase topic — see MOLMO_FALL_TILT_ENTRY_RAD below. The
# fall-detect threshold has to fire EARLY (before catastrophic tilt) or
# the override has already driven the wrist to bad joint angles through
# the lead-up to the fall, hurting recovery vs. just keeping the
# override off permanently. The current 0.35 rad entry threshold +
# 0.05 sec hold (3 ticks at 50Hz) is the tuned value for that.
def _env_bool(name: str, default: bool) -> bool:
  raw = os.environ.get(name)
  if raw is None:
    return default
  return raw.strip().lower() in ("1", "true", "yes", "on")


MOLMO_WRIST_LEVEL_OVERRIDE: bool = _env_bool(MOLMO_WRIST_LEVEL_OVERRIDE_ENV, True)
MOLMO_WRIST_ZERO_OBS: bool = _env_bool(MOLMO_WRIST_ZERO_OBS_ENV, True)
# Fall-recovery mode in molmo_node. When the pelvis tilt (max of |roll|,
# |pitch| from /g1/odom) exceeds the entry threshold for the entry-hold
# duration, molmo_node enters fall-recovery: it freezes the active
# manipulation step in place, snapshots approach state, zeros the
# published velocity command, and publishes the fall-recovery phase
# string on MOLMO_ACTIVE_PHASE_TOPIC. hand_policy.py / sim_policy_node.py
# key off that phase to suppress the wrist-leveller override (via the
# existing leveller_allowed_for gate) and the obs-side wrist zeroing.
# No pelvis-height check — hardware /g1/odom z is ZED-derived and
# unreliable during a fall; IMU-driven tilt is the only robust signal.
#
# Entry threshold is intentionally aggressive (0.20 rad ≈ 11°, 0.05 sec
# hold = 3 ticks at 50Hz). The user-facing constraint is that the
# wrist override has to be SUPPRESSED before the robot accumulates
# significant tilt — running the override through the lead-up to a
# fall regresses recovery vs. keeping the override off permanently.
# Trading occasional false-positive entries during aggressive squat
# picks (manipulation pause for ~hysteresis window) for reliable
# pre-damage override suppression. Bumped down from 0.35 → 0.20 to
# catch the disturbance earlier; with 0.20 rad entry the override
# suppression engages roughly when an external push first starts
# tipping the pelvis, not after it has already tilted 20°.
#
# Exit threshold is below entry with a hysteresis gap; if false-
# positive entries during normal manipulation become a problem,
# raise EXIT_HOLD_SEC to require a longer settled period before
# unpausing, or raise EXIT_RAD to require less tilt to "be settled".
MOLMO_FALL_TILT_ENTRY_RAD: float = 0.30
MOLMO_FALL_TILT_EXIT_RAD: float = 0.20
MOLMO_FALL_TILT_ENTRY_HOLD_SEC: float = 0.05
MOLMO_FALL_TILT_EXIT_HOLD_SEC: float = 0.1
MOLMO_FALL_RECOVERY_PHASE: str = "fall_recovery"
# Post-recovery standing phase. After the tilt gate exits fall mode
# (robot is upright), molmo_node keeps publishing the fall_recovery
# phase and zero velocity for this many additional seconds before
# resuming the saved approach. Gives the policy a moment to settle
# into a stable standing pose — walking immediately after upright is
# too aggressive and often re-tips the robot, since the policy's
# proprio + last-action history is still saturated with fall-state
# data and the wrist-leveller EMAs are mid fade-back-in. Set to 0.0
# to disable the stand phase (resume walking immediately on tilt
# exit).
MOLMO_POST_RECOVERY_STAND_SEC: float = 0.0

# Scripted perturbation injection in deploy/sim/sim_policy_node.py.
# When sim time is within [SIM_PERTURB_T0, SIM_PERTURB_T0 +
# SIM_PERTURB_DURATION), the sim writes the listed force + torque
# (world frame) to data.xfrc_applied for the named body. Used to
# reproducibly trigger fall-recovery from a known impulse without
# having to ctrl-drag the GL viewer each run. Set SIM_PERTURB_BODY to
# an empty string (or a body name that doesn't exist in the model) to
# disable the injection.
SIM_PERTURB_BODY: str = "torso_link"
SIM_PERTURB_T0: float = 1000000.0
SIM_PERTURB_DURATION: float = 1.0
SIM_PERTURB_FORCE: tuple[float, float, float] = (-60, 0, -500)
SIM_PERTURB_TORQUE: tuple[float, float, float] = (0.0, 0.0, 0.0)
# When MOLMO_WRIST_LEVEL_OVERRIDE is on, this narrows the override so
# only wrist_roll is overwritten; wrist_pitch and wrist_yaw keep the
# policy-commanded values. Intended for picks where pitch/yaw already
# look good but roll is tilted. No effect when MOLMO_WRIST_LEVEL_OVERRIDE
# is off.
MOLMO_WRIST_LEVEL_ROLL_ONLY: bool = True
# When True (and MOLMO_WRIST_LEVEL_OVERRIDE is on), drive wrist yaw so
# the gripper's finger axis horizontally aligns with pelvis +x — the
# gripper always faces the torso's forward direction regardless of arm
# pose. Independent of ROLL_ONLY and the pick/place latches; active
# continuously whenever the leveller runs.
MOLMO_WRIST_LEVEL_YAW_TO_TORSO: bool = True
# EMA smoothing applied to the leveller's yaw output after the slew limiter.
# Filters arm-oscillation-driven yaw jitter without affecting roll/pitch.
# α=1.0 disables filtering; lower values increase smoothing (τ ≈ dt/α).
MOLMO_WRIST_LEVEL_YAW_EMA_ALPHA: float = 0.2
# Output-side EMA on the *final* commanded wrist yaw, applied after every
# leveler blend in hand_policy.py / sim_policy_node.py. Catches single-tick
# spikes from any upstream cause (gate flips at vertical_wanted boundaries,
# raw_target switches across step boundaries, geometric singularities) by
# bounding how far the wrist yaw can move per tick.
# α=1.0 disables filtering; lower values increase smoothing (τ ≈ dt/α).
MOLMO_WRIST_YAW_OUTPUT_EMA_ALPHA: float = 0.01
# Extra roll added to the active wrist once the gripper closes, held
# through pick:lift/carry, faded out at pick:walkback alongside the
# wrist leveller's EMA. +π/2 rotates the gripper from the leveller's
# horizontal side-grasp pose to a vertical pinch plane so the cube is
# held "on its side" — more stable through the walkback than the
# default pose. Set to 0.0 to disable.
import math as _math
# Master toggle for the post-grasp wrist-roll rotation feature. When
# True, once the gripper closes the active wrist rolls by
# MOLMO_WRIST_ROLL_AFTER_PICK_RAD and holds that orientation through
# pick:walkback until the place sequence begins. When False, the
# feature is fully disabled: no offset is applied and the leveller's
# hold latch releases at pick:walkback as it did before the feature
# was added.
MOLMO_WRIST_ROLL_AFTER_PICK_ENABLE: bool = False
MOLMO_WRIST_ROLL_AFTER_PICK_RAD: float = _math.pi / 2
# Roll offset applied when the leveller is in "level" orientation mode.
# The 90° gripper-mount rotation in the URDF/XML means the leveller's
# no-offset solve already produces a vertical pinch plane, so "vertical"
# mode is a pass-through and "level" mode adds this +π/2 roll to rotate
# the pinch normal up to world +z (horizontal pinch plane / side-grasp
# pose). Used for sides not currently grasping (which want vertical) vs
# sides actively pinching from the side (which want level).
MOLMO_WRIST_VERTICAL_ROLL_RAD: float = _math.pi / 2
# Whole approach trajectory duration for pick/place guided phases.
# Covers the retreat → hover → over-target → descend sequence as a single
# smoothed path; tuned so the tracker traverses the entire ee_tracked
# step in this many seconds regardless of how many waypoints the planner
# emits.
MOLMO_EE_GUIDED_TOTAL_SEC: float = 5.0
# Squat-pick variant: descent leg is partly consumed by the lowered pelvis
# but the wrist works in a cramped envelope near the reach floor; give the
# trajectory more time so the descent stays smooth.
MOLMO_EE_GUIDED_TOTAL_SQUAT_SEC: float = 7.0
# Hand-to-wrist offset in body frame [X, Y, Z].
# Molmo targets the hand, but the policy controls the wrist, so we add this
# to the body-frame target to place the hand at the right spot. X and Z
# apply to both hands unchanged; Y is per-side (positive = outward on the
# left, negative = outward on the right) so asymmetric hand mounting can
# be tuned without losing the sign convention.
MOLMO_HAND_WRIST_OFFSET_X: float = -0.18
MOLMO_HAND_WRIST_OFFSET_Y_LEFT: float = 0.06
MOLMO_HAND_WRIST_OFFSET_Y_RIGHT: float = 0.0
MOLMO_HAND_WRIST_OFFSET_Z: float = 0.05
# Wrist-frame gripper midpoint used by the raw-Molmo correction path.
# Derived from the deploy URDF (Dex1-1 chassis + UMI fingers @ scale 0.65):
#   wrist_yaw_link -> gripper_mount: 0.0415 m in +x
#   gripper_mount  -> rail joint:    0.1000 m in +x, ±0.003 m in y, R_y(90°)
#   rail joint     -> tool point (between fingertip pads): 0.0415 m along rail +z
#                                     (= +x in wrist frame after R_y(90°))
# Total wrist-frame x = 0.0415 + 0.1000 + 0.0415 = 0.1830 m.
MOLMO_GRIPPER_MAX_WIDTH_M: float = 0.120
MOLMO_GRIPPER_MAX_HALF_WIDTH_M: float = 0.5 * MOLMO_GRIPPER_MAX_WIDTH_M
MOLMO_GRIPPER_MIDPOINT_OFFSET_LEFT: tuple[float, float, float] = (
  0.1830, 0.0030, 0.0,
)
MOLMO_GRIPPER_MIDPOINT_OFFSET_RIGHT: tuple[float, float, float] = (
  0.1830, -0.0030, 0.0,
)
# Gripper finger length (along wrist_yaw_link +x). Used by place_return
# to retreat along the actual gripper heading after release so the
# fingertips clear the just-placed object regardless of waist yaw.
# From the deploy URDF: UMI finger mesh extent z=0.123 m at unit scale; with
# scale=0.65 the finger extends ~0..0.080 m from the rail joint.
MOLMO_GRIPPER_FINGER_LENGTH_M: float = 0.08
# Pick early-close: short-circuit ee_tracked(pick_approach) the moment
# the raw Molmo target sits inside the gripper closing volume in
# wrist-yaw-link frame. Tolerances are derived from the URDF gripper
# geometry with ~25% safety margin so the trigger only fires when the
# object is solidly inside the fingers, not at the lip:
#   x in [-0.022, +0.058] (rail to fingertip),    safe → [-0.017, +0.044]
#   y in ±0.060 (12 cm full-open inner gap)
#   z in [-0.015, +0.015] (finger height ≈ 29 mm), safe → ±0.011
# Disabled by default; flip to True after sim tuning.
MOLMO_PICK_EARLY_CLOSE_ENABLE:   bool = True
MOLMO_PICK_EARLY_CLOSE_X_BACK_M: float = 0.05   # toward rail (−x)
MOLMO_PICK_EARLY_CLOSE_X_FWD_M:  float = 0.03   # toward fingertip (+x)
MOLMO_PICK_EARLY_CLOSE_Y_M:      float = MOLMO_GRIPPER_MAX_HALF_WIDTH_M  # ±y across fingers
MOLMO_PICK_EARLY_CLOSE_Z_M:      float = 0.0 # ±z finger height
MOLMO_PICK_EARLY_CLOSE_Z_SQUAT_M: float = 0.0 # ±z finger height (squat pick)
# Block early-close until each finger has physically opened to at least
# this prismatic position (URDF range 0..0.060, larger = more open). Without
# this gate, an early-close that fires on the first ticks of pick_approach
# would issue the OPEN command for only a few ms before the grasp dwell
# ramps it closed again — the actuator never has time to physically open
# and the gripper appears to stay closed throughout the pick.
MOLMO_PICK_EARLY_CLOSE_MIN_OPEN_M: float = 0.05
# Wrist-yaw freeze radius for pick approach. While the gripper midpoint is
# within this body-frame distance of the raw Molmo target, the wrist
# leveller stops applying yaw correction (its internal EMA holds the
# last yaw). Mirrors the spirit of the early-close gate: at close range
# a few cm of perception jitter on the target translates to several
# degrees of yaw demand, and any further yaw rotation while the fingers
# are about to wrap the object knocks them into it. Set slightly larger
# than the early-close box's diagonal (~0.10 m) so the freeze engages
# just before the early-close branch fires. Set to 0.0 to disable.
MOLMO_WRIST_YAW_FREEZE_RADIUS_M: float = 0.20
# Alternative trigger: freeze yaw when the target lies on (or very close to)
# the gripper's grasp plane in the wrist's local frame. The grasp plane is
# y=0 in wrist coords (the symmetry plane between the two finger pads;
# normal = closing direction). |target_wrist_y| < this constant means yaw
# is correctly aimed at the target — independent of how far the gripper
# still has to travel along +x. Captures "yaw is correct" rather than
# "gripper is close", which is more directly what the freeze is meant to
# preserve. Set to 0.0 to fall back to the radius-only trigger.
MOLMO_WRIST_YAW_FREEZE_PLANE_Y_M: float = 0.04
# Companion gate: only allow the plane trigger when the target sits in
# front of the wrist (target_wrist_x > this). Past the wrist, the leveler's
# atan2 swings ±π and any "yaw is correct" reading is meaningless; we want
# to freeze BEFORE that crossover, not after. 0.0 = anything in front of
# wrist origin.
MOLMO_WRIST_YAW_FREEZE_PLANE_X_MIN_M: float = 0.0
# Pick sub-phase geometry
MOLMO_PRE_PICK_RETREAT_X: float = -0.15  # pull hand back before approach #TODO might want to seperate with squat pick (-0.18?)
MOLMO_PRE_PICK_HOVER_Z: float = 0.25     # hover height above target
MOLMO_PRE_PICK_XY_EXTRA_Z: float = 0.05   # extra hover clearance during XY move
# pick_approach lift waypoint sits at target_z + this clearance, capped at
# (MOLMO_PRE_PICK_HOVER_Z + MOLMO_PRE_PICK_XY_EXTRA_Z = 0.30). The clearance
# keeps the back-extended wrist high enough to stay inside the hand-student
# policy's trained envelope; without it (lift_z = target_z) the back-extended
# pose is outside reach and the arm stalls at the start.
MOLMO_PICK_LIFT_CLEARANCE_M: float = 0.15
MOLMO_PICKUP_SCOOP_FORWARD_X: float = 0.0 # forward nudge after lowering
MOLMO_BIMANUAL_CLAMP_DY: float = 0.2       # inward y-nudge per hand during bimanual clamp
MOLMO_BIMANUAL_HOVER_SPREAD_DY: float = 0.3# outward y-spread per hand at hover_z before descent
MOLMO_BIMANUAL_Z_OFFSET: float = -0.05     # extra z offset for bimanual mode targets (e.g. reach lower)
MOLMO_BIMANUAL_PRE_PICK_RETREAT_X: float = -0.25 # retract x for bimanual approach
# Optional dedicated carry_back retract x for bimanual mode.
# If None, carry_back falls back to MOLMO_CARRY_RETRACT_X.
MOLMO_BIMANUAL_CARRY_RETRACT_X: float = 0.15
MOLMO_PICKUP_LIFT_Z: float = 0.2         # lift height after grasp
# Grasp-failure recovery: detect when the gripper closed on empty air and
# retry. Both finger qpos below the threshold (range [0, 0.0425] m) ⇒ empty.
MOLMO_GRASP_FAILURE_THRESH_M: float = 0.02
MOLMO_GRASP_RETRY_LIMIT: int = 0
MOLMO_GRASP_RETRY_LIFT_Z: float = 0.15
MOLMO_GRASP_RETRY_LIFT_SEC: float = 0.8
MOLMO_GRASP_RETRY_APPROACH_SEC: float = 3.0
MOLMO_GRASP_RETRY_REQUERY_TIMEOUT_SEC: float = 10.0
MOLMO_GRASP_RETRY_REQUERY_MAX_ATTEMPTS: int = 3
# Post-pick carry: retract X before swinging arm to carry pose.
MOLMO_CARRY_RETRACT_X: float = -0.25
# Extra Z raise applied during single-hand squat-pick carry_back so the held
# object clears the table edge while the pelvis is still squatted, before
# standup. Stacks on top of the snapshot-based body-frame target.
MOLMO_CARRY_BACK_LIFT_SQUAT_PICK_M: float = 0.3
# Extra Z raise applied during the fused single-hand pick→place tail
# carry_back (no squat). Smaller than the squat-pick lift since there's no
# table-edge clearance need; just enough to clear the just-placed object.
MOLMO_CARRY_BACK_LIFT_M: float = 0.0
# Post-walkback carry lowering (pick flow): optional extra phase that lowers
# the currently carried item by a fixed z distance after walkback.
MOLMO_CARRY_LOWER_AFTER_WALKBACK_ENABLE: bool = True
MOLMO_CARRY_LOWER_AFTER_WALKBACK_Z: float = 0.1
MOLMO_CARRY_LOWER_AFTER_WALKBACK_SEC: float = 0.6
# Post-pick carry pose (body-frame absolute position in pelvis frame).
MOLMO_CARRY_LEFT_OFFSET:  tuple[float, float, float] = (0.05,0,0.1)
MOLMO_CARRY_RIGHT_OFFSET: tuple[float, float, float] = (0.05,0,0.1)

# Hand-over (fixed-target release) waypoints — pelvis-frame absolute
# positions applied identically to MOLMO_CARRY_*_OFFSET. Single-hand
# handover uses the side that is currently carrying; bimanual uses both
# simultaneously. The robot first walks forward via
# MOLMO_HANDOVER_WALK_FORWARD_* below, then reaches these offsets and
# opens the gripper(s) for a human handoff.
MOLMO_HANDOVER_LEFT_OFFSET:  tuple[float, float, float] = (0.40, 0.05, 0.25)
MOLMO_HANDOVER_RIGHT_OFFSET: tuple[float, float, float] = (0.40, -0.05, 0.25)
MOLMO_HANDOVER_INTERP_SEC: float = 1.5
# Pause after the arm(s) reach the handover pose, before the gripper(s)
# start opening. Lets the human position to receive the object and gives
# the arm time to settle from the interp.
MOLMO_HANDOVER_REACH_DWELL_SEC: float = 3.0
# Separate from MOLMO_GRASP_DWELL_SEC so handover can be tuned slower
# (the human needs a beat to grab the object) without lengthening every pick.
MOLMO_HANDOVER_DWELL_SEC: float = 0.5
# Walk forward before the handover reach so the robot meets the human
# partway. Open-loop time × velocity ≈ distance (loco teacher tracks
# commanded vx well on flat ground). Reuses the existing "walkback"
# motion phase with a positive base_vx.
MOLMO_HANDOVER_WALK_FORWARD_VX: float = 0.3        # m/s, +x = forward
MOLMO_HANDOVER_WALK_FORWARD_SEC: float = 2      # ≈ 1.0 m at 0.3 m/s

# Place sub-phase geometry
MOLMO_PLACE_UP_Z: float = 0.12            # lift height after place release (just enough to clear the placed object before carry_back retracts in x)
MOLMO_EE_TRACKED_REVERSE_LIFT_Z: float = 0.25  # higher reverse-track lift before retracting and descending
MOLMO_BIMANUAL_PLACE_PRE_LIFT_Z: float = 0.15  # lift height before starting the bimanual place approach
MOLMO_BIMANUAL_PLACE_DOWN_Z_OFFSET: float = 0.1  # upward z bias applied when lowering for bimanual place
MOLMO_PLACE_RETURN_RETRACT_X: float = -0.2  # absolute pelvis-frame x for single-arm reverse retract before lowering
MOLMO_BIMANUAL_PLACE_RETURN_RETRACT_X: float = -0.2  # absolute pelvis-frame x for reverse retract before lowering
MOLMO_PLACE_XY_DWELL_SEC: float = 0.15    # brief pause after XY approach
# Extra forward x cap for bimanual place approach/reach, applied before the
# generic per-hand reach envelope clamp. Increase toward the envelope max
# (0.6) to relax the cap; lower it to make two-hand placements more
# conservative.
MOLMO_BIMANUAL_PLACE_MAX_X: float = 0.3
# Extra forward x cap for single-hand place targets. Applied to the
# pelvis-frame hand offset after resolving the Molmo point and wrist offset,
# before queueing the place motion. Increase toward HAND_POS_LIMIT_XYZ[0]
# to relax the cap; lower it to keep release targets closer to the body.
MOLMO_PLACE_MAX_X: float = 0.4
# Minimum pelvis-frame hand-z (offset from nominal) during place. The
# hand won't be commanded below this floor, so releases above table-top
# clutter don't drive the arm into the ground. Pick still uses -0.1 so
# the hand can descend below nominal to grasp low objects.
MOLMO_PLACE_MIN_Z: float = 0.2
# Relational placement — pelvis body-frame offsets applied to the place
# target after it's been resolved. Triggered by `classify_place_relation()`
# parsing the atomic place subtask text ("place it on top of X" → on_top_of,
# etc). Only two relations use a fixed offset; left_of / right_of /
# in_front_of ask Hunyuan to ground the placement pixel on the surface
# directly (build_hy_relational_pointing_prompt), so no fixed offset is
# applied for those. Offsets are in metres; signs follow the existing
# `_direction_offset` convention (X forward, Y robot-left, Z up).
MOLMO_STACK_Z_OFFSET: float = 0.20         # on_top_of: last-resort fallback lift when neither FFS nor pointcloud top scan succeed
MOLMO_STACK_CLEARANCE_Z: float = 0.1      # on_top_of: hover this far above the depth-detected target top before release (column-scan path)
# When True, a one-shot FFS (SAM2 + PCA OBB) is run at pick query and at
# stacking-place query to recover (a) the held object's height and (b) the
# target's 3D OBB. Stack hover Z then becomes target_top + held_half +
# MOLMO_STACK_OBB_CLEARANCE_Z. Falls back to the column-scan/_find_target_top_body_z
# path when ffs_node is unavailable or set_prompt times out.
MOLMO_USE_FFS_FOR_SIZE: bool = False
MOLMO_STACK_OBB_CLEARANCE_Z: float = 0.05  # gap above target top once held height is already accounted for
MOLMO_HELD_HEIGHT_FALLBACK_M: float = 0.2 # full body-Z extent fallback when FFS missed at pick query (used as held_full in stack-place geometry)
MOLMO_PICK_GRIP_DEPTH_M: float = 0.02      # fallback: top-grasp assumption used by the stack-place math when the per-pick palm-to-bottom offset wasn't captured (FFS down at pick time). The OBB pick path stashes the actual offset so this fallback rarely fires.
# Tolerance above Molmo's place pixel allowed for the OBB-derived target_top_z.
# Molmo points at the top of the place surface, so the depth-projected pixel Z
# is a near-ground-truth bound on the surface top. When SAM2's mask leaks onto
# background / walls / the held object, the OBB extent inflates and target_top_z
# rises tens of cm above pixel_z, producing the place-too-high symptom. Clamp
# target_top_z to pixel_z + this slack to absorb depth noise + minor pointing
# error without letting OBB inflation propagate.
MOLMO_STACK_PIXEL_Z_SLACK_M: float = 0.05
# When True, the gpt-5.5 grounding path requests a bounding box alongside the
# point and forwards the box to SAM2 (via FFS) as a region prompt. SAM2's
# combined point+box prompt segments more reliably than a point-only seed
# whose pixel landed on the object's edge — the failure mode that produced
# 1 m OBB extents in fused-place runs. Hunyuan path is unaffected.
MOLMO_VLM_USE_BOX: bool = True
# Sanity gate: drop the agent's box if its area covers more than this
# fraction of the camera frame (filters out the "agent boxed the entire
# image" failure mode). Falls back to point-only SAM2.
MOLMO_VLM_BOX_MAX_AREA_FRAC: float = 0.80
# Sanity gate: the agent's box must contain its own point within this
# pixel slack on every side. If not, drop the box and fall back to
# point-only SAM2 (catches obvious agent inconsistencies).
MOLMO_VLM_BOX_CONTAIN_POINT_SLACK_PX: float = 5.0
# When True, drop legs that fail box parsing/sanity instead of falling
# back to point-only. Default False = strictly additive. Reserved for
# a future tighter operating mode once box-quality is understood.
MOLMO_REQUIRE_BOX: bool = False
MOLMO_PLACE_FRONT_OFFSET: float = 0.10     # in_front_of: +X — fixed body-frame bump; ZED is pitched ~32°, image-y is not a clean 3D X proxy
MOLMO_PLACE_BEHIND_OFFSET: float = 0.10    # behind: -X — "behind" is usually occluded, so we keep a fixed offset
# Explicit non-locomotion phase that drives the squat-to-stand
# transition. _start_next_execution_step clears the approach-time squat
# hold ONLY when this phase begins, so this is also the window in which
# the squat EMA (τ ≈ 1 s at MOLMO_SQUAT_EMA_ALPHA=0.02) gets to ramp the
# pelvis from 0.28 m back to 0.78 m. Sized for ~3τ so the residual at
# standup-end is small (~5%) and the existing zero-snap there is a
# clean-up rather than a visible jump. Bumped from the legacy 0.5 s,
# which only worked back when the squat depth was 0.23 m and the hold
# was already cleared upstream during carry_lift.
MOLMO_RETURN_STANDUP_SEC: float = 2.0
# Safety cap on the post-squat-pick stand-before-requery wait. The standup
# phase forcibly zeroes _current_squat_delta but the cascaded output cmd
# height EMA still lags by ~τ; _maybe_start_query holds the next prompt's
# requery until that EMA converges to MOLMO_STAND_HEIGHT. If it has not
# converged within this many seconds (publish stall, controller not
# tracking, etc.), fire the requery anyway so the task does not stall.
MOLMO_POST_SQUAT_STAND_WAIT_MAX_SEC: float = 3.0
# Walkback
MOLMO_PICK_WALKBACK_SEC: float = 2.5 # mod
MOLMO_PLACE_WALKBACK_SEC: float = 0.0
MOLMO_WALKBACK_VX: float = -0.3 # mod
GRIPPER_OPEN_CMD: float = 0.0
GRIPPER_CLOSED_CMD: float = 1.0

##
# Nominal command (default standing pose) — schema indices live in
# deploy/common/command.py, values live here so they can be tuned alongside
# the rest of the project-wide constants.
##
NOMINAL_LEFT_HAND_BODY = np.array([-0.08, 0.23044664, -0.09842005], dtype=np.float32)
NOMINAL_RIGHT_HAND_BODY = np.array([-0.08, -0.23043664, -0.09842005], dtype=np.float32)
DEFAULT_LEFT_HAND_OFFSET = np.array([0.12, 0.0, 0.02], dtype=np.float32)
DEFAULT_RIGHT_HAND_OFFSET = np.array([0.12, 0.0, 0.02], dtype=np.float32)


def _build_nominal_command() -> np.ndarray:
  from deploy.common.command import (
    CMD_HEIGHT,
    CMD_LEFT_HAND,
    CMD_RIGHT_HAND,
    CMD_SIZE,
  )

  cmd = np.zeros(CMD_SIZE, dtype=np.float32)
  # TODO update this to match the walking pose for easier transitions
  cmd[CMD_HEIGHT] = 0.78
  cmd[CMD_LEFT_HAND:CMD_LEFT_HAND + 3] = NOMINAL_LEFT_HAND_BODY + DEFAULT_LEFT_HAND_OFFSET
  cmd[CMD_RIGHT_HAND:CMD_RIGHT_HAND + 3] = NOMINAL_RIGHT_HAND_BODY + DEFAULT_RIGHT_HAND_OFFSET
  return cmd


NOMINAL_COMMAND = _build_nominal_command()
MOLMO_POST_RETURN_ENABLE: bool = False
MOLMO_POST_RELEASE_SEC: float = 0.4
MOLMO_POST_LIFT_Z: float = 0.15
MOLMO_POST_LIFT_SEC: float = 1.0
MOLMO_POST_RETURN_SEC: float = 1.2
GRIPPER_PD_KP: float = 120.0
GRIPPER_DAMPING_RATIO: float = 1.0

##
# Molmo approach thresholds & velocity.
##

MOLMO_APPROACH_REQUERY: bool = True  # False = query once then execute, True = re-query when close
# After the approach reaches its target and stabilizes, if the approach
# target's z is at or below the pelvis (e.g. item on a low shelf), squat
# down by Δz = pelvis_z − target_z BEFORE the post-approach requery so the
# camera has a usable angle on the target. Skipped when Δz < min-delta.
MOLMO_APPROACH_SQUAT_BEFORE_REQUERY: bool = True
MOLMO_APPROACH_SQUAT_MIN_DELTA_M: float = 0.5 # mod 0.15  # don't bother for tiny squats
MOLMO_APPROACH_SQUAT_TOLERANCE_M: float = 0.02  # ema convergence band
# Tolerance band for the post-squat-pick "fully stood up" check used by the
# molmo_node._maybe_start_query gate. Compared against
# |_output_cmd_height - MOLMO_STAND_HEIGHT| to confirm the cascaded
# squat→target_height→output EMA has converged before re-grounding the
# next prompt's camera frame.
MOLMO_POST_SQUAT_STAND_TOLERANCE_M: float = 0.02
# Per-step (pick) squat gate: only commit to the full crouch when the
# target sits this far below the standing hand-reach floor. Smaller
# deficits aren't worth a full squat — the hand is already close enough
# that the residual gap is absorbed by the soft reach clamp. Place steps
# never squat regardless (placing onto a low surface from a stand is
# fine; stooping risks bumping the placed item).
MOLMO_PICK_SQUAT_MIN_DEFICIT_M: float = 0.15
# Z offset applied to the Molmo-pointed grasp target on squat picks. Molmo's
# pixel projection biases high under the squat camera angle (the depth ray
# clips the object's top face rather than its center), so the gripper ends
# up above the actual grasp point. Subtracted from raw_body[2] before
# _resolve_hand_target, so it propagates through to target_body and the
# pick-approach descent. Standing picks are unaffected.
MOLMO_PICK_SQUAT_DOWN_OFFSET_M: float = 0.05
# Half-extent estimate (m) used by molmo_node._project_point_to_body to push
# a single-pixel deprojected pick target along the camera-to-point ray, so the
# resulting 3D position approximates the object's geometric center instead of
# its visible front face. Applied only to the "pick" deprojection caller; ~0.04
# matches a typical 8 cm ball. Tune per target if needed. Adapts to camera
# yaw/pitch automatically (correction is along the actual viewing ray), unlike
# the deprecated pelvis-frame Y_OFFSET below. When FFS / SAM2 is healthy and
# returns an OBB, ffs_obb.center already gives the right XY and this knob is
# bypassed; this exists for the FFS-disabled / FFS-failing path.
MOLMO_PICK_RAY_CORRECTION_M: float = 0.04
# Per-hand pelvis-frame Y bias (m) added to the deprojected pick target after
# ray correction. MOLMO_PICK_RAY_CORRECTION_M only fixes the depth (front-face
# vs center) bias along the camera ray; if Molmo's 2D point is also offset
# from the silhouette center, there is a residual Y component perpendicular
# to the ray that the scalar radius cannot recover. This bias compensates
# for that. Applied in the single-hand pick path (not bimanual, not place).
# Sign matches the hand's nominal Y direction (+Y for left, -Y for right);
# tune small (~few cm) — large values overshoot the object.
MOLMO_PICK_Y_BIAS_L: float = 0.02
MOLMO_PICK_Y_BIAS_R: float = -0.02
# Per-hand waypoint offsets used by molmo_node._guided_waypoints in the
# squat-pick branch (squat_delta > 0 or _approach_squat_target_delta > 0).
# Shape the three-waypoint "spread y, change z, then go in on x+y" path so
# the wrist clears the squatted thigh on its side before descending. Right
# arm needs a wider/back-set spread because its ROM clips earlier under
# deep crouch (see MOLMO_SQUAT_LEFT_HAND_ONLY note above).
MOLMO_SQUAT_PICK_Y_SPREAD_L: float = 0.2
MOLMO_SQUAT_PICK_Y_SPREAD_R: float = -0.5
# Final-approach pelvis-frame goal offsets (Y/X). Superseded by
# MOLMO_PICK_RAY_CORRECTION_M, which fixes the surface-vs-center bias at
# deprojection time and propagates to the goal (these only ever shaped wp3
# of the trajectory, not the appended goal). Zeroed; left in place because
# _guided_waypoints still references the names. Re-tune only if you actually
# need to bias the post-spread approach in a fixed pelvis direction.
MOLMO_SQUAT_PICK_Y_OFFSET_L: float = 0.04
MOLMO_SQUAT_PICK_Y_OFFSET_R: float = -0.04
MOLMO_SQUAT_PICK_X_OFFSET_L: float = -0.05
MOLMO_SQUAT_PICK_X_OFFSET_R: float = 0.0
MOLMO_SQUAT_PICK_X_SPREAD_L: float = -0.05
MOLMO_SQUAT_PICK_X_SPREAD_R: float = -0.5
MOLMO_SQUAT_PICK_Z_OFFSET_L: float = 0.02
MOLMO_SQUAT_PICK_Z_OFFSET_R: float = 0.2
# Single-hand pick/place hand chooser. Values:
#   "auto"  -> legacy target-body-y chooser (left for +y, right for -y)
#   "left"  -> force all single-hand pick/place steps to the left hand
#   "right" -> force all single-hand pick/place steps to the right hand
# Bimanual prompts still use the bimanual path regardless of this setting.
MOLMO_SINGLE_HAND_MODE: str = "left"
# Force the left hand for any single-hand pick (and the follow-on place)
# whenever the robot is currently in an approach-time squat hold. Right-arm
# squat reaches were observed to be flakier under the deep crouch — the
# right shoulder's ROM clips earlier and the policy's pitch tends to lift
# the gripper above the target. Only affects the squat path; standing picks
# still pick the side from target's body-y. Bimanual prompts are honored
# as bimanual regardless. Set False to restore the legacy body-y choice.
MOLMO_SQUAT_LEFT_HAND_ONLY: bool = False
# When the pelvis is squatted, raise the nominal hand z by this amount so
# the hands don't drag n
# ear the legs/floor at crouched height. Scaled by
# how engaged the squat currently is (0 → 0, full squat → full lift).
MOLMO_SQUAT_HAND_LIFT_M: float = 0.05
# When squatted, also extend the nominal hand x (forward in body frame) by
# this amount so the arms reach over the knees instead of folding into the
# thighs. Scaled by squat engagement just like the z-lift; 0 disables.
# Disabled (0.0) under the deep MOLMO_SQUAT_MIN_HEIGHT=0.28 m crouch — the
# pelvis sits low enough that the default hand pose already clears the
# knees, and any non-zero forward bias here pushes the squat-state hand
# further out than typical pick targets, forcing the wrist to retract
# (visible as "hand goes back as the gripper opens" at the start of the
# pick step). Re-enable if you raise MOLMO_SQUAT_MIN_HEIGHT and the arms
# start folding into the thighs again.
MOLMO_SQUAT_HAND_REACH_X_M: float = 0.05
# When squatted, also spread the hands laterally outward by this amount in
# the MOLMO_BIMANUAL_HOVER_SPREAD_DY convention). Scaled by squat engagement
# like the x-reach and z-lift; 0 disables. Currently 0.0 — same caveat as the
# x-reach: at deep MOLMO_SQUAT_MIN_HEIGHT=0.28 m the default hand pose already
# clears the squatted thighs/knees laterally. Plumbed in so a future
# squat-pose tweak can enable it without a code change.
MOLMO_SQUAT_HAND_REACH_Y_M: float = 0.05
# If the post-approach requery still returns a point outside the same
# reach/bearing gate used for the initial approach, allow at most this many
# extra loco re-approaches before falling through to hand execution.
MOLMO_APPROACH_REAPPROACH_LIMIT: int = 1
# Pelvis-frame xy half-width for the loco→grasp transition gate. The target
# clipped to the union of L+R hand reach envelopes (see reach_clamp.py) must
# satisfy |snapped_x| ≤ this and |snapped_y| ≤ this before the robot stops
# walking and executes pick/place.
MOLMO_APPROACH_REACH_HALFWIDTH: float = 0.55
# Fraction of MOLMO_APPROACH_REACH_HALFWIDTH used to offset the planner's
# terminal place waypoint from the target along the pelvis→target vector.
# 0 = plan to the target itself; 1 = plan to the reach-box boundary. Picks
# use MOLMO_APPROACH_HAND_REACH_DIST_M directly instead.
MOLMO_PLACE_APPROACH_STOP_RATIO: float = 0.5# mod 0.9
MOLMO_BIMANUAL_PLACE_APPROACH_STOP_RATIO: float = 0.4
# Pelvis-to-path-tail xy tolerance for the loco→grasp transition. The loco
# tracker hands off to grasp when pelvis is within this radius of the final
# waypoint on /planner/loco_path (which is the planner's snapped free-cell
# if the raw goal sat inside an inflated obstacle). Larger → earlier handoff.
# Pick uses a wider tolerance than place so the robot stops further from
# obstacles (bins) and relies on arm reach; place can afford to walk closer
# because the stop ratio is already generous.
MOLMO_APPROACH_WAYPOINT_TOLERANCE: float = 0.15 #mod 0.4
MOLMO_PLACE_APPROACH_WAYPOINT_TOLERANCE: float = 0.15 #mod 0.4
# Lateral hand-align: offset the approach stop position perpendicular to
# the approach direction so the target lands between body-center and the
# grasping hand's nominal y-position. Ratio 0.0 = no offset (target at
# body y=0 → hand reaches inward ~0.23m to grasp). Ratio 1.0 = target
# exactly at hand-nominal y (no lateral reach, but approach drift can
# push into outward-reach territory). 0.5 biases toward inward-reach,
# keeping any drift on the safe side since reaching inward is natural
# and reaching outward over-extends the arm.
MOLMO_APPROACH_HAND_Y_ALIGN_RATIO: float = 0.2
# Stand-off between pelvis and target along the approach ray for picks:
# the loco approach waypoint is published this many metres back from the
# pick target. Caps the hand-reach budget at the waypoint. Place actions
# use MOLMO_PLACE_APPROACH_STOP_RATIO / MOLMO_BIMANUAL_PLACE_APPROACH_STOP_RATIO
# instead.
MOLMO_APPROACH_HAND_REACH_DIST_M: float = 0.15
# Extra pull-back applied when the upcoming approach will commit to a
# squat (pick whose target_z sits below pelvis_z by at least
# MOLMO_APPROACH_SQUAT_MIN_DELTA_M). Compensates for the descended
# pelvis eating into the hand-reach envelope.
MOLMO_APPROACH_SQUAT_EXTRA_BACKOFF_M: float = 0.05
MOLMO_APPROACH_FAR_DIST: float = 0.7
MOLMO_APPROACH_MAX_VX: float = 0.4 #mod 0.0
MOLMO_APPROACH_FAR_MAX_VX: float = 0.6 #mod 0.5 0.0
MOLMO_APPROACH_MAX_VY: float = 0.4 #mod 0.3 0.0
MOLMO_APPROACH_MAX_YAW: float = 0.6 #mod 0.3 0.0
# Half of the head-camera FOV (~60° total). When the target's heading
# error is within this band the robot doesn't yaw at all — it walks
# forward with whatever lateral/cross-track correction the path tracker
# commands. Yaw only kicks in when the target would otherwise leave the
# camera's view, so the gripper still lines up at the final approach.
MOLMO_APPROACH_ALIGN_YAW_THRESH: float = 0.6  # ~50° in radians
# When the robot is badly misaligned with the path tangent, keep vy=0
# but still allow a small forward "escape" creep if the lookahead sits
# meaningfully in front of the pelvis. This helps it clear side-near
# obstacles before finishing the turn.
MOLMO_APPROACH_ALIGN_ESCAPE_MIN_X: float = 0.15
MOLMO_APPROACH_ALIGN_ESCAPE_MAX_VX: float = 0.25
MOLMO_APPROACH_KP_X: float = 0.8
MOLMO_APPROACH_KP_Y: float = 0.3
MOLMO_APPROACH_KP_YAW: float = 0.8
MOLMO_APPROACH_YAW_DONE_THRESH: float = 0.5   # rad (~8.6°): exit yaw phase
# Lateral-phase exit threshold. Single-hand picks/places need tight lateral
# alignment so the arm reaches inward instead of over-extending outward.
# Bimanual picks straddle the target with both arms, so the lateral budget is
# wider — selected per hand mode in _update_approach.
MOLMO_APPROACH_Y_DONE_THRESH: float = 0.15      # m: exit lateral phase (single hand)
MOLMO_APPROACH_Y_DONE_THRESH_BIMANUAL: float = 0.05  # m: exit lateral phase (hand == "b")
MOLMO_APPROACH_REQUERY_SEC: float = 0.5
# Measured-speed threshold below which the robot is treated as still
# during the requery dwell. Speed = max(|pelvis_lin_vel|,
# MOLMO_APPROACH_STATIONARY_YAW_RATE_W * |pelvis_yaw_rate|) — i.e. we
# fold angular rate into an equivalent linear speed via a weighting
# factor so a single threshold covers both. Any tick where the measured
# speed exceeds this resets the stationary dwell timer so we only
# requery Molmo once the robot has actually come to rest.
MOLMO_APPROACH_STATIONARY_SPEED: float = 0.05
MOLMO_APPROACH_STATIONARY_YAW_RATE_W: float = 0.2
MOLMO_APPROACH_STATIONARY_DWELL: float = 0.5
MOLMO_APPROACH_TIMEOUT_SEC: float = 60.0
# Minimum magnitude for any non-zero vx/vy/yaw the molmo approach
# controller publishes. Commands below the policy's STAND_VEL_THRESHOLD
# (0.1 m/s) get quantised to standing, so we snap any small computed
# command up to ±this value to keep the robot actually walking.
MOLMO_APPROACH_MIN_VEL: float = 0.3
# Exponential moving average alpha applied to vx/vy/yaw before they are
# written into the unified command message. At 50 Hz publish rate, a
# smaller alpha smooths harder: alpha=0.05 ≈ 400 ms time constant
# (~1.2 s to 95 % of a step); alpha=0.2 ≈ 100 ms. Matches the pattern
# dds_xr_node uses for VR teleop.
MOLMO_APPROACH_VEL_EMA_ALPHA: float = 0.5

##
# Molmo search-turn when target is not found.
##

MOLMO_SEARCH_RETRY_LIMIT: int = 1         # re-query in place before waist search
# Waist search: first two are waist-only turns, third is whole-body yaw turn.
# [+70 waist, -70 waist, +180 body yaw]
MOLMO_SEARCH_WAIST_SEQUENCE: tuple[float, ...] = (50.0, -50.0)  # waist-only angles
MOLMO_SEARCH_BODY_TURN_DEG: float = 180.0   # whole-body yaw turn after waist search fails
MOLMO_SEARCH_WAIST_RAMP_SEC: float = 3.0    # time to ramp waist to target angle
MOLMO_SEARCH_WAIST_SETTLE_SEC: float = 1.5  # wait after ramp before re-query
MOLMO_SEARCH_BODY_TURN_SPEED: float = 0.3   # yaw rate when turning body
# Zero-velocity settle hold inserted before each "turn" prompt so the
# EMA-smoothed _output_v* decay to zero and the loco policy stops cleanly
# before the spin begins. Open-loop time-based.
MOLMO_TURN_PRE_PAUSE_SEC: float = 0.5
# Position hold during body_turn: P-controller on world-frame drift,
# rotated into pelvis body frame and emitted as vx/vy. Gain is m/s per
# m of world error; max clips each body-frame axis to avoid aggressive
# counter-steering mid-yaw. Captures the anchor on the first tick of
# body_turn and releases on phase exit.
MOLMO_SEARCH_POS_HOLD_KP: float = 0.3
MOLMO_SEARCH_POS_HOLD_MAX_V: float = 0.2

MOLMO_SEARCH_MODE: str = "quadrant"            # "quadrant" (default) or "waist"
MOLMO_SEARCH_QUADRANT_TURN_DEG: float = 90.0   # degrees per body turn in quadrant mode
MOLMO_SEARCH_QUADRANT_COUNT: int = 4           # number of quadrants to search

##
# Hand-command PD (deploy/common/hand_command_pd.py).
##
# Closed-loop shaping of the pelvis-frame hand target published by Molmo.
# When a raw Molmo point is available, pinocchio FK measures the physical
# gripper midpoint and applies an XY-only correction so that midpoint aligns
# with the raw point while preserving the staged wrist-target path. The policy
# sees the shaped target as its own observation, so its learned arm+body
# coupling is preserved — this is purely observation-side feedback, no joint
# residuals. When no raw Molmo point is published, the controller falls back
# to the original wrist-position shaping behavior.
MOLMO_HAND_CMD_PD_ENABLE: bool = False
MOLMO_HAND_CMD_PD_KP: float = 0.6
MOLMO_HAND_CMD_PD_KD: float = 0.1
MOLMO_HAND_CMD_PD_KP_X_MULT: float = 1.5
MOLMO_HAND_CMD_PD_KP_Y_MULT: float = 1.0
MOLMO_HAND_CMD_PD_KD_X_MULT: float = 1.5
MOLMO_HAND_CMD_PD_KD_Y_MULT: float = 1.0
MOLMO_HAND_CMD_PD_DEADBAND_M: float = 0.01
MOLMO_HAND_CMD_PD_MAX_SHAPE_M: float = 0.1

# When True, the approach-gate reach envelope's body-Z range is anchored
# at the FFS bbox center and sized at MOLMO_REACH_ENVELOPE_BBOX_Z_RATIO ×
# bbox-extent-along-body-Z, instead of the fixed nominal-relative bounds
# (init_z_min .. HAND_POS_LIMIT_XYZ[2]). On FFS miss, the gate falls back
# to the fixed bounds so a flaky tracker can't stall the approach loop.
# Affects only the "should we walk first?" gate (decide_query_approach
# inputs + reachable_by_hand probe + fused-commit overshoot); the runtime
# hand-IK clamp continues to use the fixed envelope every tick.
MOLMO_REACH_ENVELOPE_USE_BBOX: bool = False
# Multiplier applied to the bbox z-extent when MOLMO_REACH_ENVELOPE_USE_BBOX
# is True. 1.0 = the gate's z-window equals the bbox height; >1 makes the
# gate more permissive (fires "execute" sooner for a given target); <1
# forces an approach more often.
MOLMO_REACH_ENVELOPE_BBOX_Z_RATIO: float = 0.2

##
# HY-Embodied-0.5-X VLM backend (deploy/controller/hy_client.py).
##
# Drop-in replacement for the Molmo TCP server: Hunyuan exposes an
# OpenAI-compatible /v1/chat/completions endpoint and emits coordinates
# normalized to (0, 1000) per axis. The client denormalizes to pixel
# space before handing results back to the Molmo-era callers.
HY_HOST: str = "127.0.0.1"
# 8080 is taken on this workstation (viser dev server), so the HY OpenAI
# endpoint moves to 8180. Run `PORT=8180 bash scripts/run_server.sh` on
# the HY side to match.
HY_PORT: int = 8180
# Thinking mode wraps output in <think>...</think><answer>...</answer>.
# OFF by default for grounding: the CoT has a ~30-50% empty-emission rate
# on pointing prompts (model reasons the right pixel but runs out of
# output budget before emitting the JSON answer), and the relation-aware
# grounding prompts we added for left/right placement are tight enough
# that they don't need CoT to land the right pixel. Flip back to True if
# you see "refuses ambiguous prompts" or "hallucinates points when the
# object is absent" regressions. Planner keeps CoT — see HY_PLANNER_THINKING.
HY_THINKING: bool = False
# Planner-stage thinking (task decomposition for compound semantic prompts
# like "put the edible items on the plate"). Reasoning is load-bearing
# here — without CoT the model refuses or emits bad decompositions. Keep
# True unless you have a very good reason.
HY_PLANNER_THINKING: bool = True
# Min interval between Hunyuan queries (s). Matches the Molmo thermal
# guard that protects the Jetson hosting the VLM.
HY_CLIENT_MIN_INTERVAL_S: float = 1.0

##
# VLM provider selection (deploy/controller/hy_client.py).
##
# "hunyuan" routes to the local HY-Embodied OpenAI-compatible server on
# HY_HOST:HY_PORT. "openai" routes to any OpenAI-compatible endpoint
# (default: local cliproxyapi at 127.0.0.1:8317 proxying ChatGPT/GPT-5).
# Override at runtime with the WBC_MJLAB_VLM_PROVIDER env var.
VLM_PROVIDER: str = "openai"

# OpenAI-compatible backend used when VLM_PROVIDER == "openai". cliproxyapi
# is a local proxy that forwards to ChatGPT (GPT-5 family) over the same
# /v1/chat/completions wire format Hunyuan exposes, so the swap is wire-level.
OPENAI_VLM_HOST: str = "127.0.0.1"
OPENAI_VLM_PORT: int = 8317
# Bearer key for the proxy. Resolved in this order: WBC_MJLAB_OPENAI_VLM_API_KEY
# env var → this constant → first `sk-...` entry parsed from the cliproxyapi
# config at OPENAI_VLM_CONFIG_FALLBACK_PATH. The auto-read is what lets the
# toggle work with no env-var setup on this workstation. Leave the constant
# empty so the secret stays out of source control.
OPENAI_VLM_API_KEY: str = ""
# cliproxyapi config to scrape for an api key when both the env var and the
# constant above are empty. Set to "" to disable the fallback.
OPENAI_VLM_CONFIG_FALLBACK_PATH: str = "~/cliproxyapi/config.yaml"
# Model id sent in the chat-completions payload. cliproxyapi exposes
# gpt-5.5 / gpt-5.4 / gpt-5.3-codex / gpt-image-2.
OPENAI_VLM_MODEL_ID: str = "gpt-5.5"

# Reasoning effort for the GPT-5 family. Mirrors HY_THINKING /
# HY_PLANNER_THINKING: grounding is latency-sensitive (every request blocks
# the manipulation pipeline), so default it to "low"; planner +
# search-direction get more headroom because their decompositions are
# load-bearing for correctness.
#
# Tested against cliproxyapi → ChatGPT (gpt-5.5): the accepted set on
# this codex-protocol path is {"low", "medium", "high", "xhigh"}. The
# proxy explicitly rejects "minimal" with HTTP 400. Empty string omits
# the field entirely. Override at runtime with
# WBC_MJLAB_OPENAI_VLM_REASONING_EFFORT and
# WBC_MJLAB_OPENAI_VLM_PLANNER_REASONING_EFFORT.
OPENAI_VLM_REASONING_EFFORT: str = "low"
OPENAI_VLM_PLANNER_REASONING_EFFORT: str = "medium"

##
# Fast-FoundationStereoPose / SAM2 tracker (deploy/controller/ffs_node.py).
##
# Streaming SAM2 tracker seeded from the hunyuan 2D pointing target. Produces
# a temporally-smoothed 6D OBB over ZED's stereo-fused XYZ depth, republished
# in pelvis body frame so molmo_node can use the mask-averaged 3D center as
# a noise-robust grasp target. See plan at
# .claude/plans/i-cloned-git-github-com-lzyang2000-fast-mossy-map.md.

# Absolute path to the cloned repo. SAM2 lives under $FFS_REPO_PATH/SAM2_streaming
# and is imported via sys.path insert at node startup (the repo has no setup.py).
# Resolved relative to this file so it works on any user/clone path.
_WBC_MJLAB_ROOT: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FFS_REPO_PATH: str = os.path.join(_WBC_MJLAB_ROOT, "deploy/fast_foundation_stereo_pose")
# Config + checkpoint handed to build_sam2_camera_predictor. Config is a
# package-relative path (resolved by SAM2's hydra loader); checkpoint is absolute.
FFS_SAM2_CFG: str = "sam2.1/sam2.1_hiera_b+.yaml"
FFS_SAM2_CKPT: str = os.path.join(
    FFS_REPO_PATH, "SAM2_streaming/checkpoints/sam2.1/sam2.1_hiera_base_plus.pt"
)
# OBB fitting: filter outliers beyond the Nth distance-percentile and refuse
# to fit if fewer than MIN_OBJ_PTS survive. Copied from the demo.deploy/assets/g1/g1_sim2sim_29dof_gripper.xml
FFS_MIN_OBJ_PTS: int = 10
FFS_OUTLIER_PCT: float = 90.0
# Temporal smoothing (from combined_sam2_stereo.py).
FFS_OBB_SMOOTH: float = 0.75
FFS_EXTENT_ALPHA_INIT: float = 0.4
FFS_EXTENT_ALPHA_MIN: float = 0.02
FFS_EXTENT_ALPHA_DECAY: float = 0.92
FFS_EXTENT_MAX_CHANGE_RATE: float = 0.05
# RGB and XYZ cloud stamps must agree within this tolerance, else we drop
# the pair — mixing frames across exposures corrupts the mask×XYZ overlay.
FFS_FRAME_STAMP_TOL_MS: float = 50.0
# Prompt service picks the latest buffered frame within this window of the
# requested stamp. Wider than the RGB↔XYZ tolerance because the hunyuan
# response takes time and the frame we used for hunyuan may be slightly stale
# by the time the prompt gets published.
FFS_PROMPT_FRAME_TOL_MS: float = 200.0
# State machine: N consecutive empty masks → LOST; M coasted frames (too-few
# points under mask) → stop publishing. Tuned for ~10 Hz tracking.
FFS_EMPTY_MASK_LIMIT: int = 15
FFS_COAST_LIMIT: int = 30
# Master toggle: True turns the whole SAM2 + OBB pipeline off. ffs_node
# comes up but skips predictor init (no SAM2 weights on GPU), and FfsClient
# short-circuits set_prompt / latest_obb_body to None so molmo_node
# transparently falls back to the single-pixel _project_point_to_body
# (3D waypoint) path with no latency hit per query. Useful for A/B
# against the pre-FFS behavior without restarting or rebuilding.
FFS_DISABLE: bool = True
# FfsClient: ignore cached OBB older than this; set_prompt waits this long
# for a fresh OBB before giving up and returning None.
FFS_LATEST_MAX_AGE_S: float = 0.5
FFS_SET_PROMPT_TIMEOUT_S: float = 1.5
# Frame ring depth on ffs_node — must cover the Molmo/HY query latency so
# the exact prompt frame is still available when set_prompt arrives. At 30
# frames the previous default routinely rolled off after ~2 s and forced a
# stale-fallback path that often produced empty masks. 120 ≈ 4 s @ 30 Hz
# or ~8 s @ 15 Hz. Memory cost is ~2 MB/frame.
FFS_FRAME_RING_SIZE: int = 120
# When set_prompt returns None (transient timeout, ffs_node not yet warm,
# stale stamp), retry up to this many additional times before giving up.
# Each retry blocks for up to FFS_SET_PROMPT_TIMEOUT_S, so worst-case
# total stall = (1 + retries) * FFS_SET_PROMPT_TIMEOUT_S. 0 = no retries.
FFS_SET_PROMPT_RETRIES: int = 0
# Topic names.
FFS_PROMPT_TOPIC: str = "/molmo/ffs/prompt"
FFS_OBB_POSE_TOPIC: str = "/molmo/ffs/obb_pose"
FFS_OBB_EXTENT_TOPIC: str = "/molmo/ffs/obb_extent"
FFS_STATUS_TOPIC: str = "/molmo/ffs/status"
# Per-frame mono8 SAM2 mask, native RGB resolution, stamped with the frame
# it segmented. Subscribed by sim_policy_node to paint a tinted overlay on
# /molmo/visor/overlay (debug-only — the grounding RGB stream is unchanged
# so the VLM is not biased by the colored region on the next prompt).
FFS_MASK_TOPIC: str = "/molmo/ffs/mask"
MOLMO_VISOR_OVERLAY_TOPIC: str = "/molmo/visor/overlay"
# Camera name — must match the zed_bridge configured head_camera name. Used
# to build subscription topics: /molmo/camera/<name>/rgb etc.
FFS_CAMERA_NAME: str = "head_camera"
# Odom topic for pelvis pose. Matches molmo_node._odom_topic.
FFS_ODOM_TOPIC: str = "/g1/odom"

##
# Dex1-1 parallel-jaw gripper (1 DOF per side, single Unitree M4010 motor).
# Driven via deploy/real/dex1_1_service over rt/dex1/{left,right}/{cmd,state}.
# These are the *motor-side* numbers — the policy command bus uses the
# unitless [GRIPPER_OPEN_CMD..GRIPPER_CLOSED_CMD] scalar above, which the
# hardware node maps linearly onto [DEX1_Q_OPEN..DEX1_Q_CLOSED].
##
DEX1_Q_OPEN: float = 5.0       # rad — motor angle at fully-open jaws (software limit)
DEX1_Q_CLOSED: float = 0.0     # rad — motor angle at fully-closed jaws (calibrated zero)
DEX1_KP: float = 5.0           # M4010 stiffness — matches dex1_1_service test client default
DEX1_KD: float = 0.05          # M4010 damping — matches dex1_1_service test client default
DEX1_STROKE_M: float = MOLMO_GRIPPER_MAX_WIDTH_M  # 120 mm full jaw opening; each URDF prismatic joint travels half.
DEX1_TAU_LIMIT: float = 10.0   # N·m — hold position when |tau_est| exceeds this (override with --dex1-tau-limit)

# dds_xr teleop trigger → gripper open-length mapping. The VR controller
# trigger commands a physical jaw opening in [0, DDS_XR_GRIPPER_MAX_OPEN_M]
# (trigger released = max open, fully pressed = closed). That width is then
# converted to the unitless [GRIPPER_OPEN_CMD..GRIPPER_CLOSED_CMD] bus scalar.
DDS_XR_GRIPPER_MAX_OPEN_M: float = 0.09  # 9 cm — max teleop jaw opening


def gripper_open_m_to_cmd(open_m: float) -> float:
  """Convert a physical jaw opening (m) to the [GRIPPER_OPEN_CMD,
  GRIPPER_CLOSED_CMD] bus scalar. Inverse of the hardware/sim relation
  full_width = DEX1_STROKE_M * (1 - cmd): an opening of DEX1_STROKE_M maps to
  GRIPPER_OPEN_CMD (fully open) and 0 maps to GRIPPER_CLOSED_CMD (closed)."""
  span = DEX1_STROKE_M if DEX1_STROKE_M > 1e-6 else 1e-6
  open_frac = min(max(float(open_m) / span, 0.0), 1.0)
  return float(GRIPPER_CLOSED_CMD - open_frac * (GRIPPER_CLOSED_CMD - GRIPPER_OPEN_CMD))
