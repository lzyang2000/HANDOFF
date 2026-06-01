"""Live pelvis↔camera transform on real G1 hardware.

The ZED → pelvis odometry in hardware_node reconstructs world→pelvis from
the ZED's world→camera pose using a pelvis↔camera mount transform. The
mount constants in g1_robot_constants are baked at neutral waist, but the
real chain is

    pelvis → waist_yaw → waist_roll → torso → (head_camera_mount) → lens

so when the three waist joints are nonzero the static transform is wrong.
This module runs the waist-chain FK each tick via pinocchio (same URDF
WristLeveller / RobotSelfMask use) and rebuilds the camera→pelvis transform
live. The camera itself is not in the URDF, but it is rigidly attached to
torso_link, so we derive a one-shot T_torso_camera at construction from the
existing neutral-waist constants and compose it with the live
T_pelvis_torso(q_waist).
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pinocchio as pin

_TORSO_FRAME = "torso_link"
_DEFAULT_URDF = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "g1_29dof_rev_1_0_with_payloads_and_gripper.urdf"
)


def _rpy_deg_to_mat(rpy_deg: Sequence[float]) -> np.ndarray:
    r, p, y = (float(a) * np.pi / 180.0 for a in rpy_deg)
    return pin.rpy.rpyToMatrix(r, p, y)


class PelvisCameraFK:
    """Computes camera→pelvis (R_cp, t_cp) given current joint positions.

    Pelvis is the URDF root (fixed-base model), so per-frame placements
    returned by pinocchio are expressed in pelvis frame directly.
    """

    def __init__(
        self,
        head_camera_pos_in_pelvis: Sequence[float],
        head_camera_rpy_in_pelvis_deg: Sequence[float],
        policy_joint_names: Sequence[str],
        urdf_path: str | Path = _DEFAULT_URDF,
    ) -> None:
        path = Path(urdf_path)
        if not path.exists():
            raise FileNotFoundError(f"URDF not found for PelvisCameraFK: {path}")
        self._model = pin.buildModelFromUrdf(str(path))
        self._data = self._model.createData()
        self._q_idx = self._build_q_index_map(policy_joint_names)

        torso_fid = self._model.getFrameId(_TORSO_FRAME)
        if torso_fid >= self._model.nframes:
            raise ValueError(f"URDF missing frame: {_TORSO_FRAME}")
        self._torso_fid = torso_fid

        # Static pelvis→camera at neutral waist, from the mount constants.
        R_pc_static = _rpy_deg_to_mat(head_camera_rpy_in_pelvis_deg)
        t_pc_static = np.asarray(head_camera_pos_in_pelvis, dtype=np.float64).reshape(3)

        # One-shot neutral-waist FK to get T_pelvis_torso(0), then back out
        # the truly-static T_torso_camera. The camera is rigidly mounted to
        # torso_link, so this composite stays valid for the lifetime of the
        # process regardless of waist motion.
        q0 = pin.neutral(self._model)
        pin.forwardKinematics(self._model, self._data, q0)
        pin.updateFramePlacements(self._model, self._data)
        oMt_neutral = self._data.oMf[self._torso_fid]
        R_pt0 = np.asarray(oMt_neutral.rotation, dtype=np.float64)
        t_pt0 = np.asarray(oMt_neutral.translation, dtype=np.float64)
        # T_torso_camera = T_pelvis_torso(0)^-1 · T_pelvis_camera_static
        self._R_tc = R_pt0.T @ R_pc_static
        self._t_tc = R_pt0.T @ (t_pc_static - t_pt0)

        # Cache the neutral-waist (R_cp, t_cp) for fallback on bad input.
        R_cp0 = R_pc_static.T
        t_cp0 = -R_cp0 @ t_pc_static
        self._R_cp_neutral = R_cp0
        self._t_cp_neutral = t_cp0
        self._bad_jpos_warned = False

    def _build_q_index_map(self, policy_joint_names: Sequence[str]) -> np.ndarray:
        idx = []
        missing = []
        for name in policy_joint_names:
            if not self._model.existJointName(name):
                missing.append(name)
                continue
            joint = self._model.joints[self._model.getJointId(name)]
            if joint.nq <= 0:
                missing.append(name)
                continue
            idx.append(int(joint.idx_q))
        if missing:
            raise ValueError(f"URDF missing joints for PelvisCameraFK: {missing}")
        return np.asarray(idx, dtype=np.int64)

    def camera_to_pelvis(self, jpos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (R_cp, t_cp) in double precision, live per the current waist joints.

        Falls back to the neutral-waist transform (logged once) if jpos
        contains NaN/Inf.
        """
        jp = np.asarray(jpos, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(jp)):
            if not self._bad_jpos_warned:
                print("[WARN] PelvisCameraFK: non-finite jpos; using neutral-waist fallback.")
                self._bad_jpos_warned = True
            return self._R_cp_neutral.copy(), self._t_cp_neutral.copy()

        q = pin.neutral(self._model)
        q[self._q_idx] = jp
        pin.forwardKinematics(self._model, self._data, q)
        pin.updateFramePlacements(self._model, self._data)
        oMt = self._data.oMf[self._torso_fid]
        R_pt = np.asarray(oMt.rotation, dtype=np.float64)
        t_pt = np.asarray(oMt.translation, dtype=np.float64)

        R_pc = R_pt @ self._R_tc
        t_pc = t_pt + R_pt @ self._t_tc
        R_cp = R_pc.T
        t_cp = -R_cp @ t_pc
        return R_cp, t_cp
