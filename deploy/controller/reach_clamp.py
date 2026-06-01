"""Shared hand-offset clamp helper and reach-envelope constants.

Owns the single source of truth for the hand reach envelope used by both
molmo_node (for teleop commands) and the manipulation planner. Kept free
of rclpy imports so unit tests can exercise the clamp without a ROS env.
teleop_common.py re-exports these constants for backward compatibility.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from wbc_mjlab.g1_constants_custom import HAND_POS_LIMIT_XYZ as _POS_LIMIT, HAND_NEG_LIMIT_XYZ as _NEG_LIMIT

# ---------------------------------------------------------------------------
# Hand position limits (offsets from nominal body-frame positions)
# ---------------------------------------------------------------------------
HAND_POS_LIMIT_XYZ = np.array(_POS_LIMIT, dtype=np.float32)
HAND_NEG_LIMIT_XYZ = np.array(_NEG_LIMIT, dtype=np.float32)

LEFT_HAND_POS_LIMIT_XYZ = HAND_POS_LIMIT_XYZ.copy()
LEFT_HAND_NEG_LIMIT_XYZ = HAND_NEG_LIMIT_XYZ.copy()
LEFT_HAND_POS_LIMIT_XYZ[1] = -HAND_NEG_LIMIT_XYZ[1]
LEFT_HAND_NEG_LIMIT_XYZ[1] = -HAND_POS_LIMIT_XYZ[1]


def clamp_offset(
    hand: str,
    offset: np.ndarray,
    min_z: float = -0.1,
    max_z: Optional[float] = None,
) -> np.ndarray:
    """Clamp a hand offset (pelvis frame) to the policy's reach envelope.

    Args:
        hand: "l" or "r".
        offset: (3,) xyz in pelvis frame.
        min_z: lower bound on z; overrides the envelope's z floor.
        max_z: optional upper bound on z; when None, uses the envelope's
            fixed ceiling (HAND_POS_LIMIT_XYZ[2]). Set explicitly when the
            caller wants to scale the envelope's z-window with an external
            reference (e.g. a bbox-derived gate range).

    Returns:
        (3,) float32 clamped offset.
    """
    if hand == "l":
        lo = LEFT_HAND_NEG_LIMIT_XYZ.copy()
        hi = LEFT_HAND_POS_LIMIT_XYZ.copy()
    else:
        lo = HAND_NEG_LIMIT_XYZ.copy()
        hi = HAND_POS_LIMIT_XYZ.copy()
    lo[2] = min_z
    if max_z is not None:
        hi[2] = float(max_z)
    return np.clip(offset, lo, hi).astype(np.float32)
