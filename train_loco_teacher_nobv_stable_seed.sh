#!/usr/bin/env bash
# Non-privileged (NoBV) loco teacher with whole-body stability rewards added.
# Mirrors train_loco_teacher_nobv_seed.sh but trains against the -Stable task
# (com_in_support_polygon, capture_point_in_support_polygon, ankle_hip_step,
# linear & angular momentum change penalties layered on top of the standard
# loco-teacher reward shaping).
set -euo pipefail

# Sim-to-real hardware payloads (Jetson on back + Dex1-1 hands).
# Set WBC_ATTACH_PAYLOADS=0 to disable for this run.
export WBC_ATTACH_PAYLOADS="${WBC_ATTACH_PAYLOADS:-1}"

GPU_ID="0"
if [[ -n "${1:-}" && "${1:-}" != --* ]]; then
  GPU_ID="${1}"
  shift
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run train Loco-Teacher-Flat-Unitree-G1-NoBV-Stable \
  --agent.experiment-name g1_loco_teacher_nobv_stable_seed \
  --agent.run-name g1_loco_teacher_nobv_stable_seed \
  --env.commands.motion.motion-file /home/yangl/handoff/seed_g1_cbf_standing_payload/seed_dataset_filtered.yaml \
  --env.scene.num-envs 4096 \
  --video True \
  --video-interval 48000 \
  --video-length 500 \
  "$@"
