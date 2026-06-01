#!/usr/bin/env bash
# Layered ckpt_setup for the fully-stable 3-teacher MoE student.
#
# Sources ckpt_setup_stable_seed.sh first (stable WBC + stable NoBV loco
# teachers), then fills the AMP-teacher slot.
#
# The AMP teacher reuses the ORIGINAL (non-stable) AMP teacher because the
# AMP teacher's reward stack (`_replace_with_amp_rewards` in amp_config.py)
# is independent of `_handoff_regularization_reward_cfg` /
# `_apply_loco_teacher_reward_shaping`, so no -Stable variant exists for it.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/ckpt_setup_stable_seed.sh"

# AMP recovery teacher -- produced by train_amp_teacher_flat.sh with
# WBC_ATTACH_PAYLOADS=1 (default). Mirrors the slot from ckpt_setup_amp.sh.
# Override via env vars at the call site to pin a different run.
export AMP_TEACHER_EXP="${AMP_TEACHER_EXP:-2026-04-30_22-36-53_g1_amp_teacher_flat}"
export AMP_TEACHER_PROJ="${AMP_TEACHER_PROJ:-logs/rsl_rl/g1_amp_teacher_flat}"
export AMP_TEACHER_CKPT="${AMP_TEACHER_CKPT:--1}"
