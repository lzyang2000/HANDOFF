#!/usr/bin/env zsh
# Note: Using zsh to ensure compatibility with ROS 2 setup.zsh
#
# Sim2real deployment of the fully-stable seed+AMP MoE student on the physical
# Unitree G1, driven by an xbox pad or a VR headset (or keyboard / no controller).
#
# Verify deploy/play_sim_hand.sh first. Real hardware exposes any obs/action
# layout bug as a fall, not a soft sim glitch.
#
# Requires the vendored Unitree SDK + Dex1-1 gripper service to be built once:
#   git submodule update --init deploy/real/unitree_sdk2_wrapper deploy/real/dex1_1_service
#   bash deploy/install_unitree_sdk.sh
#
# Usage:
#   bash deploy/play_real_hand.sh                     # latest run, prompts for controller
#   bash deploy/play_real_hand.sh path/to/model.onnx  # explicit ONNX
#   bash deploy/play_real_hand.sh path/to/model.pt    # auto-export then run
./deploy/real/setup_route.sh
set -e
set -o pipefail

SCRIPT_DIR=${0:A:h}
ROOT_DIR=${SCRIPT_DIR:h}
CLEANED_UP=0

HAND_EXPORT_TASK="Wbc-Hand-Dual-Teacher-Flat-Unitree-G1-MoE-UniCmd-NoBV-AMP-Stable"
HAND_EXPERIMENT_NAME="g1_hand_moe_flat_unicmd_nobv_fullstable_seed_amp"

cleanup() {
  if [[ "${CLEANED_UP}" == "1" ]]; then
    return
  fi
  CLEANED_UP=1
  trap '' SIGINT SIGTERM EXIT
  echo -e '\nStopping...'
  pkill -f "dex1_1_gripper_server"               2>/dev/null || true
  pkill -f "deploy/policy/hand_policy.py"        2>/dev/null || true
  pkill -f "deploy/real/hardware_node.py"        2>/dev/null || true
  pkill -f "deploy/controller/keyboard_node.py"  2>/dev/null || true
  pkill -f "deploy/controller/xbox_node.py"      2>/dev/null || true
  pkill -f "deploy/controller/dds_xr_node.py"    2>/dev/null || true
  kill -- -$$ 2>/dev/null || true
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
    printf "Enter choice [1-4, default 3]: "
    read -r selection
    case "${selection}" in
      "1")     CONTROLLER_MODE="none"     ; return ;;
      "2")     CONTROLLER_MODE="keyboard" ; return ;;
      ""|"3")  CONTROLLER_MODE="xbox"     ; return ;;
      "4")     CONTROLLER_MODE="dds"      ; return ;;
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

ROS_PYTHON_PATHS=$(python3 -c "import sys; print(':'.join([p for p in sys.path if 'ros' in p]))")

# ---------------------------------------------------------------------------
# 2. Resolve ONNX model (same logic as play_sim_hand.sh)
# ---------------------------------------------------------------------------
ONNX_MODEL="${1:-}"

if [[ -z "${ONNX_MODEL}" ]]; then
  EXPERIMENT_DIR="${ROOT_DIR}/logs/rsl_rl/${HAND_EXPERIMENT_NAME}"
  LATEST_RUN="$(ls -dt "${EXPERIMENT_DIR}"/*/ 2>/dev/null | head -1)"
  if [[ -z "${LATEST_RUN}" ]]; then
    echo "Error: No runs found in ${EXPERIMENT_DIR}/" >&2
    exit 1
  fi
  LATEST_CKPT="$(ls "${LATEST_RUN}"model_*.pt 2>/dev/null | sed 's/.*model_\([0-9]*\)\.pt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2-)"
  if [[ -z "${LATEST_CKPT}" ]]; then
    echo "Error: No .pt checkpoints found in ${LATEST_RUN}" >&2
    exit 1
  fi
  echo "Exporting ONNX from: ${LATEST_CKPT}"
  uv run --python "${UV_PYTHON}" python deploy/export_onnx.py "${HAND_EXPORT_TASK}" "${LATEST_CKPT}"
  ONNX_MODEL="$(ls -t "${LATEST_RUN}"*.onnx 2>/dev/null | head -1)"
  [[ -z "${ONNX_MODEL}" ]] && { echo "Error: ONNX export failed." >&2; exit 1; }
  echo "Using checkpoint: ${LATEST_CKPT}"
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
# 3. PYTHONPATH
# ---------------------------------------------------------------------------
export PYTHONPATH="${PYTHONPATH:-}:${ROS_PYTHON_PATHS}:${ROOT_DIR}:${ROOT_DIR}/src"

# ---------------------------------------------------------------------------
# 4. Network interface for hardware node
# ---------------------------------------------------------------------------
printf "Network interface for robot DDS [default: enP2p1s0]: "
read -r NET_INTERFACE
NET_INTERFACE="${NET_INTERFACE:-enP2p1s0}"
echo "Using network interface: ${NET_INTERFACE}"

# ---------------------------------------------------------------------------
# 4b. Gripper
# ---------------------------------------------------------------------------
printf "Dex1-1 grippers connected? [Y/n]: "
read -r gripper_sel
if [[ "${gripper_sel}" =~ ^[Nn]$ ]]; then
  USE_GRIPPERS=0
else
  USE_GRIPPERS=1
  sudo -v
fi

# ---------------------------------------------------------------------------
# 5. Controller selection
# ---------------------------------------------------------------------------
select_controller_mode
echo "Selected controller: ${CONTROLLER_MODE}"
if [[ "${CONTROLLER_MODE}" == xbox* ]]; then
  export WBC_MJLAB_MOLMO_WRIST_LEVEL_OVERRIDE=0
  export WBC_MJLAB_MOLMO_WRIST_ZERO_OBS=0
  echo "Xbox mode: wrist leveller disabled"
fi
if [[ "${CONTROLLER_MODE}" == "dds" ]]; then
  export WBC_MJLAB_MOLMO_WRIST_LEVEL_OVERRIDE=0
  export WBC_MJLAB_MOLMO_WRIST_ZERO_OBS=1
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
elif [[ "${CONTROLLER_MODE}" == "dds" ]]; then
  echo "Starting DDS XR (VR) Controller Node (in background)..."
  uv run --python "${UV_PYTHON}" python deploy/controller/dds_xr_node.py &
elif [[ "${CONTROLLER_MODE}" == "keyboard" ]]; then
  KEYBOARD_FOREGROUND=1
fi

# Gripper service (background, optional)
if [[ "${USE_GRIPPERS}" == "1" ]]; then
  echo "Starting Dex1-1 Gripper Service (in background)..."
  sudo deploy/real/dex1_1_service/bin/dex1_1_gripper_server --network "${NET_INTERFACE}" &
fi

# Hand policy (background)
echo "Starting Hand Policy Node (in background)..."
uv run --python "${UV_PYTHON}" python deploy/policy/hand_policy.py "${ONNX_MODEL}" &

sleep 2.0

# Hardware node (foreground, or background if keyboard is foreground)
if [[ "${KEYBOARD_FOREGROUND:-0}" == "1" ]]; then
  echo "Starting Hardware Node (in background)..."
  uv run --python "${UV_PYTHON}" python deploy/real/hardware_node.py --net "${NET_INTERFACE}" &
  sleep 1.0
  echo "Starting Keyboard Controller Node (in foreground)..."
  uv run --python "${UV_PYTHON}" python deploy/controller/keyboard_node.py
else
  echo "Starting Hardware Node (in foreground)..."
  uv run --python "${UV_PYTHON}" python deploy/real/hardware_node.py --net "${NET_INTERFACE}"
fi
