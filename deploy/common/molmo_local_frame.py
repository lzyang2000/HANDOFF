from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


ODOM_FRAME_ID = "odom"
LOCAL_FRAME_ID = "molmo_local"


def rotmat_from_yaw(yaw: float) -> np.ndarray:
    c = math.cos(float(yaw))
    s = math.sin(float(yaw))
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def rotmat_to_yaw(rotmat: np.ndarray) -> float:
    mat = np.asarray(rotmat, dtype=np.float64).reshape(3, 3)
    return float(math.atan2(mat[1, 0], mat[0, 0]))


def quat_xyzw_from_yaw(yaw: float) -> np.ndarray:
    half = 0.5 * float(yaw)
    return np.array([0.0, 0.0, math.sin(half), math.cos(half)], dtype=np.float64)


def quat_xyzw_to_yaw(quat_xyzw: np.ndarray) -> float:
    x, y, z, w = [float(v) for v in np.asarray(quat_xyzw, dtype=np.float64).reshape(4)]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(math.atan2(siny_cosp, cosy_cosp))


@dataclass(frozen=True)
class LocalFrameAnchor:
    world_origin: np.ndarray
    world_yaw: float

    def rotation_world_from_local(self) -> np.ndarray:
        return rotmat_from_yaw(self.world_yaw)


def make_local_frame_anchor(
    world_pos: np.ndarray,
    world_rotmat: np.ndarray,
    *,
    floor_z: float = 0.0,
) -> LocalFrameAnchor:
    """Create a local-frame anchor from a world pose.

    ``floor_z`` is the world-frame z of the ground plane beneath the
    robot.  The anchor's z-origin is set to ``floor_z`` so that the
    local frame preserves ``local_z ≡ world_z`` when the floor is at 0
    (hardware) and correctly offsets in simulation.  Callers that know
    the pelvis height (e.g. via FK) should pass
    ``floor_z = pelvis_world_z - pelvis_height_from_fk``.
    """
    origin = np.asarray(world_pos, dtype=np.float64).reshape(3).copy()
    origin[2] = float(floor_z)
    return LocalFrameAnchor(
        world_origin=origin,
        world_yaw=rotmat_to_yaw(world_rotmat),
    )


def make_identity_local_frame_anchor() -> LocalFrameAnchor:
    return LocalFrameAnchor(
        world_origin=np.zeros(3, dtype=np.float64),
        world_yaw=0.0,
    )


def anchors_close(
    lhs: LocalFrameAnchor,
    rhs: LocalFrameAnchor,
    *,
    pos_tol: float = 1e-6,
    yaw_tol: float = 1e-6,
) -> bool:
    if lhs is rhs:
        return True
    if lhs is None or rhs is None:
        return False
    pos_close = np.linalg.norm(lhs.world_origin - rhs.world_origin) <= float(pos_tol)
    yaw_err = math.atan2(
        math.sin(lhs.world_yaw - rhs.world_yaw),
        math.cos(lhs.world_yaw - rhs.world_yaw),
    )
    return bool(pos_close and abs(yaw_err) <= float(yaw_tol))


def world_points_to_local(world_points: np.ndarray, anchor: LocalFrameAnchor) -> np.ndarray:
    pts = np.asarray(world_points, dtype=np.float64)
    flat = pts.reshape(-1, 3)
    rot_world_from_local = anchor.rotation_world_from_local()
    flat_local = (flat - anchor.world_origin.reshape(1, 3)) @ rot_world_from_local
    return flat_local.reshape(pts.shape)


def local_points_to_world(local_points: np.ndarray, anchor: LocalFrameAnchor) -> np.ndarray:
    pts = np.asarray(local_points, dtype=np.float64)
    flat = pts.reshape(-1, 3)
    rot_world_from_local = anchor.rotation_world_from_local()
    flat_world = flat @ rot_world_from_local.T + anchor.world_origin.reshape(1, 3)
    return flat_world.reshape(pts.shape)


def world_point_to_local(world_point: np.ndarray, anchor: LocalFrameAnchor) -> np.ndarray:
    return world_points_to_local(np.asarray(world_point, dtype=np.float64).reshape(1, 3), anchor)[0]


def local_point_to_world(local_point: np.ndarray, anchor: LocalFrameAnchor) -> np.ndarray:
    return local_points_to_world(np.asarray(local_point, dtype=np.float64).reshape(1, 3), anchor)[0]


def world_xy_to_local_xy(world_xy: np.ndarray, anchor: LocalFrameAnchor) -> np.ndarray:
    pts = np.asarray(world_xy, dtype=np.float64).reshape(-1, 2)
    pts3 = np.zeros((pts.shape[0], 3), dtype=np.float64)
    pts3[:, :2] = pts
    return world_points_to_local(pts3, anchor)[:, :2].reshape(np.asarray(world_xy).shape)


def local_xy_to_world_xy(local_xy: np.ndarray, anchor: LocalFrameAnchor) -> np.ndarray:
    pts = np.asarray(local_xy, dtype=np.float64).reshape(-1, 2)
    pts3 = np.zeros((pts.shape[0], 3), dtype=np.float64)
    pts3[:, :2] = pts
    return local_points_to_world(pts3, anchor)[:, :2].reshape(np.asarray(local_xy).shape)


def world_rotmat_to_local(world_rotmat: np.ndarray, anchor: LocalFrameAnchor) -> np.ndarray:
    rot_world_from_local = anchor.rotation_world_from_local()
    return rot_world_from_local.T @ np.asarray(world_rotmat, dtype=np.float64).reshape(3, 3)


def local_rotmat_to_world(local_rotmat: np.ndarray, anchor: LocalFrameAnchor) -> np.ndarray:
    rot_world_from_local = anchor.rotation_world_from_local()
    return rot_world_from_local @ np.asarray(local_rotmat, dtype=np.float64).reshape(3, 3)
