"""Robot-safe G1 constants copied from the pinned mjlab revision.

These values mirror:
https://github.com/mujocolab/mjlab/blob/60eca4afee7fd6c2c5da55f6f1943bb4dd41b292/src/mjlab/asset_zoo/robots/unitree_g1/g1_constants.py

They are kept here so the real-hardware deploy path can run without importing
mjlab (which imports warp at module import time).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InitialStateCfg:
    pos: tuple[float, float, float]
    joint_pos: dict[str, float]
    joint_vel: dict[str, float]


KNEES_BENT_KEYFRAME = InitialStateCfg(
    pos=(0.0, 0.0, 0.76),
    joint_pos={
        ".*_hip_pitch_joint": -0.312,
        ".*_knee_joint": 0.669,
        ".*_ankle_pitch_joint": -0.363,
        ".*_elbow_joint": 0.6,
        "left_shoulder_roll_joint": 0.2,
        "left_shoulder_pitch_joint": 0.2,
        "right_shoulder_roll_joint": -0.2,
        "right_shoulder_pitch_joint": 0.2,
    },
    joint_vel={".*": 0.0},
)

# Physics-derived PD gains copied from mjlab/docs and the pinned upstream source.
STIFFNESS_5020 = 14.25062309787429
DAMPING_5020 = 0.907222843292423

STIFFNESS_7520_14 = 40.17923863450712
DAMPING_7520_14 = 2.557889775413375

STIFFNESS_7520_22 = 99.09842777666111
DAMPING_7520_22 = 6.308801853496639

STIFFNESS_4010 = 16.77832748089279
DAMPING_4010 = 1.06814150219

G1_ACTION_SCALE: dict[str, float] = {
    ".*_elbow_joint": 0.43857731392336724,
    ".*_shoulder_pitch_joint": 0.43857731392336724,
    ".*_shoulder_roll_joint": 0.43857731392336724,
    ".*_shoulder_yaw_joint": 0.43857731392336724,
    ".*_wrist_roll_joint": 0.43857731392336724,
    ".*_hip_pitch_joint": 0.5475464629911068,
    ".*_hip_yaw_joint": 0.5475464629911068,
    "waist_yaw_joint": 0.5475464629911068,
    ".*_hip_roll_joint": 0.35066146637882434,
    ".*_knee_joint": 0.35066146637882434,
    ".*_wrist_pitch_joint": 0.07450087032950714,
    ".*_wrist_yaw_joint": 0.07450087032950714,
    "waist_pitch_joint": 0.43857731392336724,
    "waist_roll_joint": 0.43857731392336724,
    ".*_ankle_pitch_joint": 0.43857731392336724,
    ".*_ankle_roll_joint": 0.43857731392336724,
}

# -------------------------------------------------------------------------
# Head camera (ZED Mini) mount.
#
# The XML now places the camera at the ZED Mini left-lens offset (in torso
# frame, since head_camera_mount is parented to torso_link):
#   deploy/assets/g1/g1_sim2sim_29dof_gripper.xml:169-171
#     <body name="head_camera_mount" pos="0.105927 0.0315 0.339">
#       <camera name="head_camera" pos="0 0 0"
#               xyaxes="0 -1 0  0.3420201 0 0.9396926" fovy="57"/>
#
# But everything ZedBridge publishes (sl.VIEW.LEFT image, left_cal intrinsics,
# MEASURE.XYZ, and get_position() positional tracking) is expressed relative
# to the LEFT camera — the ZED SDK's reference origin. So the pelvis→camera
# constants below must target the left lens, not the housing center, for
# hardware_node's world→pelvis reconstruction and viz_node's pelvis→head_camera
# TF to be consistent with what ZedBridge emits.
#
# Expressed relative to the pelvis (IMU) frame, assuming all waist joints
# are at their default (neutral) values.
# -------------------------------------------------------------------------

# Foot sole sits ~3 cm below the URDF's ankle_roll_link origin on the
# physical G1 (ankle frame is above the sole). FK through
# ``ankle_roll_link`` returns the link-frame z; subtract this offset to
# get the true ground-contact z when computing pelvis_height / floor_z.
FOOT_SOLE_BELOW_ANKLE_M: float = 0.04

HEAD_CAMERA_NAME: str = "head_camera"

# Translation: pelvis-frame metres from the pelvis origin to the ZED's left
# lens. The XML mount is on the robot centerline (y=0); the ZED Mini baseline
# is 63 mm, so the left lens sits +0.0315 m in pelvis +Y from the housing
# center.
HEAD_CAMERA_POS_IN_PELVIS: tuple[float, float, float] = (0.1019635, 0.0315, 0.383)

# Rotation: camera frame (ZED RIGHT_HANDED_Z_UP_X_FWD: X forward, Y left,
# Z up) in pelvis frame as (roll, pitch, yaw) degrees, ZYX intrinsic
# convention (= extrinsic XYZ: first roll around X, then pitch around Y,
# then yaw around Z), matching _rpy_deg_to_mat in viz_node/hardware_node.
#
# Head-mounted ZED on the real robot pitches down 20° (measured on the
# physical mount). No roll, no yaw. The sim XML mount matches at 20° —
# update both together if the physical tilt changes.
HEAD_CAMERA_RPY_IN_PELVIS: tuple[float, float, float] = (0.0, 24, 0.0)

# ZED publish cadence. Keep ≤ HEAD_CAMERA_ZED_FPS.
HEAD_CAMERA_PUBLISH_HZ: float = 10.0

# ZED SDK initialisation. Strings are resolved to sl.RESOLUTION / sl.DEPTH_MODE
# enums in zed_bridge.py; valid options are listed in those dicts.
HEAD_CAMERA_ZED_RESOLUTION: str = "VGA"   # HD720 | HD1080 | VGA
HEAD_CAMERA_ZED_FPS: int = 15
HEAD_CAMERA_ZED_DEPTH_MODE: str = "NEURAL"  # PERFORMANCE | QUALITY | NEURAL | NEURAL_LIGHT | NEURAL_PLUS | ULTRA
HEAD_CAMERA_ZED_MIN_DEPTH_M: float = 0.1
HEAD_CAMERA_ZED_MAX_DEPTH_M: float = 8.0
# RuntimeParameters knobs. Default is 95, which aggressively masks near-range
# pixels on the ZED Mini's 6.3 cm baseline (smaller baseline → noisier
# disparity than the 2i) — lower to keep more near-field points at the cost
# of some noise.
HEAD_CAMERA_ZED_CONFIDENCE_THRESHOLD: int = 50

# AEC/AGC stays on by default — outdoor light swings make a fixed exposure
# unworkable. The ROI below restricts metering to the lower-center band so
# ceiling lights / sky / windows in the upper part of the frame don't drag
# the exposure up and wash out the floor (where the policy actually looks).
HEAD_CAMERA_ZED_AEC_AGC_ENABLED: bool = True

# AEC/AGC ROI as normalized fractions of the resolved camera resolution:
# (x_frac, y_frac, w_frac, h_frac), origin top-left, y growing down.
# Default keeps the middle 80% width and the lower 65% height (skips the top
# 30% — sky/ceiling/lights — and trims 10% off each side for door/edge clutter).
# Set to None to disable the ROI (full-frame metering).
HEAD_CAMERA_ZED_AEC_AGC_ROI: tuple[float, float, float, float] | None = (0.10, 0.30, 0.80, 0.65)

# Manual exposure / gain overrides (0–100). Setting either disables the
# matching auto-control on the ZED side. Leave as None to keep AEC/AGC active;
# pin a value only for stable indoor shoots.
HEAD_CAMERA_ZED_EXPOSURE: int | None = None
HEAD_CAMERA_ZED_GAIN: int | None = None

# FOV / render size used by sim_node and sim_policy_node so the simulated
# head camera matches what the real ZED Mini produces. ZED Mini datasheet
# spec is 102° H × 57° V × 118° D (max, raw lens); we run at VGA which
# preserves close to that wide FOV after rectification. We pin the
# sim VFOV to 57°; at the 16:9 render aspect this gives HFOV ≈ 88°, which
# is the closest pinhole match to the datasheet without distortion. The
# sim renders at 320×180 to keep the renderer cheap while preserving the
# 16:9 aspect.
HEAD_CAMERA_SIM_FOVY_DEG: float = 57.0
HEAD_CAMERA_SIM_RENDER_H: int = 180
HEAD_CAMERA_SIM_RENDER_W: int = 320
