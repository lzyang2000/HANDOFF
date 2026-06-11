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
CMD_HEIGHT = 3
CMD_LEFT_HAND = 4       # 4, 5, 6
CMD_RIGHT_HAND = 7      # 7, 8, 9
CMD_LEFT_WRIST = 10     # 10, 11, 12
CMD_RIGHT_WRIST = 13    # 13, 14, 15
CMD_LEFT_GRIPPER = 16
CMD_RIGHT_GRIPPER = 17
CMD_SIZE = 18


def make_command() -> np.ndarray:
    """Return a fresh copy of the nominal command."""
    from wbc_mjlab.g1_constants_custom import NOMINAL_COMMAND
    return NOMINAL_COMMAND.copy()
