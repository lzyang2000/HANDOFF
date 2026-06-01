"""Unified command message schema for controller → policy communication.

All controller nodes publish a single Float32MultiArray of CMD_SIZE floats
on COMMAND_TOPIC.  Values are **absolute** (not offsets): each controller
starts from NOMINAL_COMMAND and applies its input deltas before publishing.
Policy nodes subscribe and extract the fields they need.

Schema (CMD_* indices, CMD_SIZE, topic names) lives here. The actual
nominal-pose values (NOMINAL_COMMAND, hand-frame anchors) live in
``wbc_mjlab.g1_constants_custom`` so they can be tuned alongside the rest
of the project-wide constants.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Topic
# ---------------------------------------------------------------------------
COMMAND_TOPIC = "/g1/command"

# Latched String topic. Identifies which controller owns COMMAND_TOPIC.
# Publishers (molmo_node, xbox_node, ...) skip their /g1/command write
# unless the active source matches their own name. No message ever published
# (e.g. sim) means "no override" and every controller publishes as before.
CONTROL_SOURCE_TOPIC = "/g1/control_source"

# ---------------------------------------------------------------------------
# Field indices into the flat command array
# ---------------------------------------------------------------------------
CMD_VX = 0
CMD_VY = 1
CMD_YAW_RATE = 2
CMD_PITCH = 3
CMD_HEIGHT = 4
CMD_LEFT_HAND = 5       # 5, 6, 7
CMD_RIGHT_HAND = 8      # 8, 9, 10
CMD_LEFT_WRIST = 11     # 11, 12, 13
CMD_RIGHT_WRIST = 14    # 14, 15, 16
CMD_LEFT_GRIPPER = 17
CMD_RIGHT_GRIPPER = 18
CMD_SIZE = 19


def make_command() -> np.ndarray:
    """Return a fresh copy of the nominal command."""
    from wbc_mjlab.g1_constants_custom import NOMINAL_COMMAND
    return NOMINAL_COMMAND.copy()
