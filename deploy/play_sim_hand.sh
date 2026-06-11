#!/usr/bin/env zsh
# Note: Using zsh to ensure compatibility with ROS 2 setup.zsh
#
# Sim2sim playback of the fully-stable seed+AMP MoE student in MuJoCo, driven by
# an xbox pad or a VR headset (or keyboard / no controller).
#
# Usage:
#   bash deploy/play_sim_hand.sh                     # latest run, prompts for controller
#   bash deploy/play_sim_hand.sh path/to/model.onnx  # explicit ONNX
#   bash deploy/play_sim_hand.sh path/to/model.pt    # auto-export then run
set -e
set -o pipefail

SCRIPT_DIR=${0:A:h}
ROOT_DIR=${SCRIPT_DIR:h}
CLEANED_UP=0

# The one student this repo ships. Matches src/wbc_mjlab/__init__.py and
# train_moe_student_unicmd_nobv_fullstable_seed_amp.sh.
HAND_EXPORT_TASK="Wbc-Hand-Dual-Teacher-Flat-Unitree-G1-MoE-UniCmd-NoBV-AMP-Stable"
HAND_EXPERIMENT_NAME="g1_hand_moe_flat_unicmd_nobv_fullstable_seed_amp"

cleanup() {
  if [[ "${CLEANED_UP}" == "1" ]]; then
    return
  fi
  CLEANED_UP=1
  trap '' SIGINT SIGTERM EXIT
  echo -e '\nStopping...'
  pkill -f "deploy/policy/hand_policy.py"       2>/dev/null || true
  pkill -f "deploy/sim/sim_node.py"             2>/dev/null || true
  pkill -f "deploy/sim/sim_policy_node.py"      2>/dev/null || true
  pkill -f "deploy/controller/keyboard_node.py" 2>/dev/null || true
  pkill -f "deploy/controller/xbox_node.py"     2>/dev/null || true
  pkill -f "deploy/controller/dds_xr_node.py"   2>/dev/null || true
  # Kill any remaining child processes in our process group
  kill -- -$$ 2>/dev/null || true
  # Free UDP ports from any zombie holders
  fuser -k -KILL 9870/udp 2>/dev/null || true
  fuser -k -KILL 9871/udp 2>/dev/null || true
}

select_controller_mode() {
  while true; do
    echo "Select controller mode:"
    echo "  1) none"
    echo "  2) keyboard"
    echo "  3) xbox"
    echo "  4) vr (dds_xr headset teleop)"
    echo "  5) xbox server (browser-based, no local pad)"
    printf "Enter choice [1-5, default 3]: "
    read -r selection
    case "${selection}" in
      "1")        CONTROLLER_MODE="none"        ; return ;;
      "2")        CONTROLLER_MODE="keyboard"    ; return ;;
      ""|"3")     CONTROLLER_MODE="xbox"        ; return ;;
      "4")        CONTROLLER_MODE="dds"         ; return ;;
      "5")        CONTROLLER_MODE="xbox_server" ; return ;;
      *)          echo "Invalid selection: ${selection}" ;;
    esac
  done
}

select_run_mode() {
  while true; do
    echo "Run mode:"
    echo "  1) split  — policy + sim as separate processes"
    echo "  2) sync   — policy + sim in one process, physics-synchronized (default)"
    printf "Enter choice [1-2, default 2]: "
    read -r selection
    case "${selection}" in
      "1")     RUN_MODE="split" ; return ;;
      ""|"2")  RUN_MODE="sync"  ; return ;;
      *)       echo "Invalid selection: ${selection}" ;;
    esac
  done
}

# ---------------------------------------------------------------------------
# 1. Source ROS 2 (auto-detect distro)
# ---------------------------------------------------------------------------
_ros_distros=(rolling jazzy iron humble galactic foxy)
_ros_sourced=0
for _distro in "${_ros_distros[@]}"; do
  if [[ -f "/opt/ros/${_distro}/setup.zsh" ]]; then
    source "/opt/ros/${_distro}/setup.zsh"
    echo "Sourced ROS 2 ${_distro}"
    _ros_sourced=1
    break
  fi
done
if [[ "${_ros_sourced}" == "0" ]]; then
  echo "Error: No ROS 2 installation found in /opt/ros/. Checked: ${_ros_distros[*]}" >&2
  exit 1
fi
case "${_distro}" in
  jazzy|rolling) UV_PYTHON="3.12" ;;
  *)             UV_PYTHON="3.10" ;;
esac
unset _distro _ros_distros _ros_sourced

# Use EGL for headless MuJoCo rendering (no X11/DISPLAY)
if [[ -z "${DISPLAY:-}" ]]; then
  export MUJOCO_GL=egl
fi

# 2. Extract ROS 2 Python paths so uv-managed Python can see rclpy etc.
ROS_PYTHON_PATHS=$(python3 -c "import sys; print(':'.join([p for p in sys.path if 'ros' in p]))")

# ---------------------------------------------------------------------------
# 3. Resolve ONNX model from arg, or export the latest checkpoint of the run.
# ---------------------------------------------------------------------------
ONNX_MODEL="${1:-}"

if [[ -z "${ONNX_MODEL}" ]]; then
  EXPERIMENT_DIR="${ROOT_DIR}/logs/rsl_rl/${HAND_EXPERIMENT_NAME}"
  # (Nom): nullglob + sort by mtime (newest first); no error when nothing matches.
  RUNS=("${EXPERIMENT_DIR}"/*/(Nom))
  LATEST_RUN="${RUNS[1]:-}"
  LATEST_CKPT=""
  if [[ -n "${LATEST_RUN}" ]]; then
    CKPTS=("${LATEST_RUN}"model_*.pt(N))
    if (( ${#CKPTS} )); then
      LATEST_CKPT="$(printf '%s\n' "${CKPTS[@]}" | sed 's/.*model_\([0-9]*\)\.pt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2-)"
    fi
  fi
  if [[ -n "${LATEST_CKPT}" ]]; then
    echo "Exporting ONNX from: ${LATEST_CKPT}"
    uv run --python "${UV_PYTHON}" python deploy/export_onnx.py "${HAND_EXPORT_TASK}" "${LATEST_CKPT}"
    ONNX_MODEL="$(ls -t "${LATEST_RUN}"*.onnx 2>/dev/null | head -1)"
    [[ -z "${ONNX_MODEL}" ]] && { echo "Error: ONNX export failed." >&2; exit 1; }
    echo "Using checkpoint: ${LATEST_CKPT}"
  else
    # No trained checkpoints found; fall back to the bundled policy.
    ONNX_MODEL="${SCRIPT_DIR}/ckpt/policy.onnx"
    if [[ ! -f "${ONNX_MODEL}" ]]; then
      echo "Error: No checkpoints found and no bundled policy at ${ONNX_MODEL}" >&2
      exit 1
    fi
    echo "No trained checkpoints found; using bundled policy: ${ONNX_MODEL}"
  fi
elif [[ "${ONNX_MODEL}" == *.pt ]]; then
  shift
  PT_PATH="$(realpath "${ONNX_MODEL}")"
  echo "Exporting ONNX from: ${PT_PATH} (task: ${HAND_EXPORT_TASK})"
  uv run --python "${UV_PYTHON}" python deploy/export_onnx.py "${HAND_EXPORT_TASK}" "${PT_PATH}"
  PT_DIR="$(dirname "${PT_PATH}")"
  ONNX_MODEL="$(ls -t "${PT_DIR}"/*.onnx 2>/dev/null | head -1)"
  [[ -z "${ONNX_MODEL}" ]] && { echo "Error: ONNX export failed." >&2; exit 1; }
else
  shift
fi

trap cleanup SIGINT SIGTERM EXIT

cd "${ROOT_DIR}"

# ---------------------------------------------------------------------------
# 4. PYTHONPATH: uv-managed env + ROS 2 + project sources
# ---------------------------------------------------------------------------
export PYTHONPATH="${PYTHONPATH:-}:${ROS_PYTHON_PATHS}:${ROOT_DIR}:${ROOT_DIR}/src"

# ---------------------------------------------------------------------------
# 5. Interactive prompts
# ---------------------------------------------------------------------------
select_controller_mode
echo "Selected controller: ${CONTROLLER_MODE}"
select_run_mode
echo "Selected run mode: ${RUN_MODE}"
if [[ "${CONTROLLER_MODE}" == xbox* ]]; then
  export WBC_MJLAB_MOLMO_WRIST_LEVEL_OVERRIDE=0
  export WBC_MJLAB_MOLMO_WRIST_ZERO_OBS=0
  echo "Xbox mode: wrist leveller disabled"
fi
if [[ "${CONTROLLER_MODE}" == "dds" ]]; then
  export WBC_MJLAB_MOLMO_WRIST_LEVEL_OVERRIDE=0
  export WBC_MJLAB_MOLMO_WRIST_ZERO_OBS=0
  export WBC_MJLAB_DDS_XR_WRIST=1
  echo "vr (dds_xr) mode: wrist leveller disabled, VR wrist passthrough enabled"
fi

# ---------------------------------------------------------------------------
# 6. Launch nodes
# ---------------------------------------------------------------------------

# Controller (background); keyboard runs in the foreground at the end.
if [[ "${CONTROLLER_MODE}" == "xbox" ]]; then
  echo "Starting Xbox Controller Node (in background)..."
  uv run --python "${UV_PYTHON}" python deploy/controller/xbox_node.py &
elif [[ "${CONTROLLER_MODE}" == "xbox_server" ]]; then
  echo "Starting Xbox Controller Node in server mode (in background)..."
  echo "  Open http://localhost:${XBOX_BRIDGE_HTTP_PORT:-8765} in a browser with a gamepad."
  uv run --python "${UV_PYTHON}" python deploy/controller/xbox_node.py --source server &
elif [[ "${CONTROLLER_MODE}" == "dds" ]]; then
  echo "Starting DDS XR (VR) Controller Node (in background)..."
  uv run --python "${UV_PYTHON}" python deploy/controller/dds_xr_node.py &
elif [[ "${CONTROLLER_MODE}" == "keyboard" ]]; then
  KEYBOARD_FOREGROUND=1
fi

# Policy + sim launch
if [[ "${RUN_MODE}" == "sync" ]]; then
  # Single process: sim + policy, physics-synchronized.
  if [[ "${KEYBOARD_FOREGROUND:-0}" == "1" ]]; then
    echo "Starting synchronized Simulation+Policy Node (in background)..."
    uv run --python "${UV_PYTHON}" python deploy/sim/sim_policy_node.py "${ONNX_MODEL}" &
    sleep 1.0
    echo "Starting Keyboard Controller Node (in foreground)..."
    uv run --python "${UV_PYTHON}" python deploy/controller/keyboard_node.py
  else
    echo "Starting synchronized Simulation+Policy Node (in foreground)..."
    uv run --python "${UV_PYTHON}" python deploy/sim/sim_policy_node.py "${ONNX_MODEL}"
  fi
else
  # Split: policy and sim as separate processes over UDP.
  echo "Starting Hand Policy Node (in background)..."
  uv run --python "${UV_PYTHON}" python deploy/policy/hand_policy.py "${ONNX_MODEL}" &
  sleep 2.0
  if [[ "${KEYBOARD_FOREGROUND:-0}" == "1" ]]; then
    echo "Starting Simulation Node (in background)..."
    uv run --python "${UV_PYTHON}" python deploy/sim/sim_node.py &
    sleep 1.0
    echo "Starting Keyboard Controller Node (in foreground)..."
    uv run --python "${UV_PYTHON}" python deploy/controller/keyboard_node.py
  else
    echo "Starting Simulation Node (in foreground)..."
    uv run --python "${UV_PYTHON}" python deploy/sim/sim_node.py
  fi
fi
