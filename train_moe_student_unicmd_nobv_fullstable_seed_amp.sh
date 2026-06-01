#!/usr/bin/env bash
# Fully-stable 3-teacher MoE student: env-side stability rewards AND
# distillation from stability-trained teachers (stable WBC + stable NoBV
# loco). AMP teacher slot reuses the original AMP teacher (no -Stable
# variant exists for it; see ckpt_setup_stable_seed_amp.sh).
#
# Differs from train_moe_student_unicmd_nobv_stable_seed_amp.sh (env-only)
# only in which ckpt_setup file is sourced -- this one points at the stable
# teacher checkpoints. Both scripts target the same -Stable-AMP task id.
#
# Prerequisite: stable teachers trained AND timestamps filled into
# ckpt_setup_stable_seed.sh.
set -euo pipefail

# Sim-to-real hardware payloads (Jetson on back + Dex1-1 hands).
# Set WBC_ATTACH_PAYLOADS=0 to disable for this run.
export WBC_ATTACH_PAYLOADS="${WBC_ATTACH_PAYLOADS:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${CKPT_SETUP:-${SCRIPT_DIR}/ckpt_setup_stable_seed_amp.sh}"

GPU_ID="0"
if [[ -n "${1:-}" && "${1:-}" != --* ]]; then
  GPU_ID="${1}"
  shift
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run train Wbc-Hand-Dual-Teacher-Flat-Unitree-G1-MoE-UniCmd-NoBV-AMP-Stable \
  --agent.experiment-name g1_hand_moe_flat_unicmd_nobv_fullstable_seed_amp \
  --agent.run-name g1_hand_moe_flat_unicmd_nobv_fullstable_seed_amp \
  --agent.teacher-experiment-name "${WBC_TEACHER_EXP}" \
  --agent.teacher-proj-name "${WBC_TEACHER_PROJ}" \
  --agent.teacher-checkpoint "${WBC_TEACHER_CKPT}" \
  --agent.loco-teacher-experiment-name "${LOCO_TEACHER_NOBV_EXP}" \
  --agent.loco-teacher-proj-name "${LOCO_TEACHER_NOBV_PROJ}" \
  --agent.loco-teacher-checkpoint "${LOCO_TEACHER_NOBV_CKPT}" \
  --agent.amp-teacher-experiment-name "${AMP_TEACHER_EXP}" \
  --agent.amp-teacher-proj-name "${AMP_TEACHER_PROJ}" \
  --agent.amp-teacher-checkpoint "${AMP_TEACHER_CKPT}" \
  --agent.algorithm.dagger-coef 0.4 \
  --agent.algorithm.dagger-coef-min 0.2 \
  --agent.algorithm.arm-kl-coef 0.1 \
  --agent.algorithm.arm-kl-coef-min 0.05 \
  --agent.algorithm.amp-kl-coef 0.4 \
  --agent.algorithm.amp-kl-coef-min 0.2 \
  --env.commands.motion.motion-file /home/yangl/handoff/seed_g1_cbf_standing_payload/seed_dataset_filtered.yaml \
  --env.scene.num-envs 4096 \
  --video True \
  --video-interval 48000 \
  --video-length 500 \
  "$@"
