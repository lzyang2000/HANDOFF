#!/usr/bin/env bash
set -euo pipefail

# Sim-to-real hardware payloads (Jetson on back + Dex1-1 hands).
# Set WBC_ATTACH_PAYLOADS=0 to disable for this run.
export WBC_ATTACH_PAYLOADS="${WBC_ATTACH_PAYLOADS:-1}"

# First positional arg = GPU id (default 0). Anything else is forwarded.
GPU_ID="0"
if [[ -n "${1:-}" && "${1:-}" != --* ]]; then
  GPU_ID="${1}"
  shift
fi

# Run from repo root so the top-level ``deploy/`` namespace package is on
# sys.path (g1_constants_custom.py imports it during wbc_mjlab init).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run train Amp-Teacher-Flat-Unitree-G1 \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 100000 \
  --video True \
  --video-interval 48000 \
  --video-length 500 \
  "$@"
