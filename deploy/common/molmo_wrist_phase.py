"""Helpers for Molmo wrist-leveller phase classification."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class MolmoActivePhase:
    action: str
    motion_phase: str
    hand: str
    hand_mode: str
    guided_mode: str = ""


def parse_molmo_active_phase(phase: Optional[str]) -> Optional[MolmoActivePhase]:
    text = str(phase or "").strip()
    if not text or text == "idle":
        return None
    parts = text.split(":")
    if len(parts) == 4:
        return MolmoActivePhase(*parts, "")
    if len(parts) == 5:
        return MolmoActivePhase(*parts)
    return None


def is_single_place_leveller_active(
    phase: Optional[MolmoActivePhase],
    side: str,
) -> bool:
    """True while ``side`` is performing a single-hand place.

    The leveller and hand-PD shaping should stay engaged here even though
    the gripper is closed (carrying), so the held object stays level
    through the descend and lands flat. Place sub-phases that run AFTER
    the gripper opens (home/standup/walkback/...) clear ``holding`` via
    ``update_side_holding``, so the caller will already be in the
    vertical-idle branch and this predicate is naturally scoped to the
    pre-release sub-phases.
    """
    return bool(
        phase is not None
        and phase.action == "place"
        and phase.hand_mode == "single"
        and phase.hand == side
    )


def is_single_pick_leveller_allowed(
    phase: Optional[MolmoActivePhase],
    side: str,
) -> bool:
    if is_single_place_leveller_active(phase, side):
        return True
    return bool(
        phase is not None
        and phase.action == "pick"
        and phase.motion_phase != "ee_tracked_reverse"
        and phase.hand_mode == "single"
        and phase.hand == side
    )


def is_single_pick_yaw_freeze(
    phase: Optional[MolmoActivePhase],
    side: str,
) -> bool:
    """True during pick:grasp for ``side`` — the leveller should hold its
    last yaw instead of chasing the live raw target while the gripper closes.

    Why: with the leveller still active during the 50 ms grasp dwell, even a
    few cm of perception jitter on the raw target translates to several
    degrees of yaw demand at close range, swinging the wrist into the object
    while fingers are inside the closing volume.
    """
    return bool(
        phase is not None
        and phase.action == "pick"
        and phase.motion_phase == "grasp"
        and phase.hand_mode == "single"
        and phase.hand == side
    )


def is_single_pick_walkback(
    phase: Optional[MolmoActivePhase],
    side: str,
) -> bool:
    return bool(
        phase is not None
        and phase.action == "pick"
        and phase.motion_phase == "walkback"
        and phase.hand_mode == "single"
        and phase.hand == side
    )


def is_single_pick_retract(
    phase: Optional[MolmoActivePhase],
    side: str,
) -> bool:
    return bool(
        phase is not None
        and phase.action == "pick"
        and phase.motion_phase in ("carry_back", "carry_down", "ee_tracked_reverse")
        and phase.hand_mode == "single"
        and phase.hand == side
    )


def should_clear_single_pick_walkback_hold(
    phase: Optional[MolmoActivePhase],
    side: str,
) -> bool:
    if phase is None:
        return False
    if (
        phase.action == "pick"
        and phase.motion_phase == "ee_tracked_reverse"
        and phase.hand_mode == "single"
        and phase.hand == side
    ):
        return True
    return not (
        phase.action == "pick"
        and phase.hand_mode == "single"
        and phase.hand == side
    )


def is_bimanual_active(phase: Optional[MolmoActivePhase]) -> bool:
    """True for any ``pick`` or ``place`` sub-phase in bimanual mode.

    Covers the full bimanual sequence (``carry_lift``, ``standup``,
    ``walkback``, ``carry_back``, ``carry_lower``, ``bimanual_release``,
    etc.) so the caller can hold both wrists vertical across the whole
    action. An ``idle``/``None`` phase is not active.
    """
    return bool(
        phase is not None
        and phase.action in ("pick", "place")
        and phase.hand == "b"
        and phase.hand_mode == "bimanual"
    )


def is_single_pick_or_place_for(
    phase: Optional[MolmoActivePhase],
    side: str,
) -> bool:
    """True when ``side`` is the active hand in a single-hand pick/place.

    Used to detect when ``side`` should NOT hold the vertical idle pose —
    it's the one doing the manipulation.
    """
    return bool(
        phase is not None
        and phase.hand_mode == "single"
        and phase.action in ("pick", "place")
        and phase.hand == side
    )


# Single-hand place sub-phases that happen AFTER the gripper has opened.
# ``release`` is treated as still-holding (the gripper is opening *during*
# this phase); the carry latch clears only once the sequence moves on to
# one of these. ``ee_tracked_reverse`` and ``carry_back`` are intentionally
# NOT here for the place flow: both are the post-release retract motion
# (FK-based finger-axis retreat, or body-x lift-and-pull-back) and run
# directly above the just-placed object, so the wrist must stay
# horizontal — snapping to the vertical idle pose here would rotate the
# gripper through the object. ``home_xy`` is also absent so the latch
# survives the carry-z traverse; the latch only clears at ``home`` when
# the wrist starts descending to the home z, well behind/clear of the
# placement.
_PLACE_POST_RELEASE_PHASES = frozenset({
    "home",
    "standup",
    "walkback",
    "carry_lower",
})


def update_side_holding(
    prev_holding: bool,
    phase: Optional[MolmoActivePhase],
    side: str,
) -> bool:
    """Update the per-side 'carrying an object' latch.

    The latch is True from the moment ``side`` picks something up until
    it finishes the matching place. That covers pick (all sub-phases),
    the idle gap between pick and place, place pre-release, the
    ``release`` phase itself, and the FK-based ``ee_tracked_reverse``
    retreat that follows release (the leveller must stay horizontal
    while we walk along the gripper's finger axis). It clears once
    place advances to a post-release sub-phase (``home`` / ``standup``
    / ``walkback`` / ``carry_back`` / ``carry_lower``).

    Bimanual phases clear the latch on both sides — a coordinated
    two-hand grasp supersedes any prior solo carry. Opposite-hand
    single-hand phases and transient ``idle`` / ``None`` ticks preserve
    the latch, so the carry state survives hand-over-hand sequencing.
    """
    if phase is None:
        return prev_holding
    if phase.hand_mode == "bimanual":
        return False
    if phase.hand_mode == "single" and phase.hand == side:
        if phase.action == "pick":
            return True
        if phase.action == "place":
            if phase.motion_phase in _PLACE_POST_RELEASE_PHASES:
                return False
            return prev_holding
    return prev_holding
