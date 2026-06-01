#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WRAPPER_DIR="$SCRIPT_DIR/real/unitree_sdk2_wrapper"
DEX1_DIR="$SCRIPT_DIR/real/dex1_1_service"

# Python version: match ROS distro (jazzy/rolling → 3.12, everything else → 3.10)
_UV_PYTHON="3.10"
for _distro in rolling jazzy iron humble galactic foxy; do
  if [[ -f "/opt/ros/${_distro}/setup.bash" ]]; then
    case "${_distro}" in
      jazzy|rolling) _UV_PYTHON="3.12" ;;
    esac
    break
  fi
done
unset _distro

UV_PROJECT_ARGS=(--project "$PROJECT_ROOT" --python "${_UV_PYTHON}")

# 1. System deps
sudo apt-get update
sudo apt-get install -y build-essential cmake python3-dev pybind11-dev \
  libserialport-dev libspdlog-dev libboost-all-dev libyaml-cpp-dev libfmt-dev

# 2. Python build deps into the repo uv environment
uv pip install "${UV_PROJECT_ARGS[@]}" pybind11 pybind11-stubgen numpy

# 3. Ensure submodules are populated
git -C "$PROJECT_ROOT" submodule update --init deploy/real/unitree_sdk2_wrapper
git -C "$PROJECT_ROOT" submodule update --init deploy/real/dex1_1_service

# 4. Build
cd "$WRAPPER_DIR/python_binding"
export UNITREE_SDK2_PATH="$(pwd)/.."
uv run "${UV_PROJECT_ARGS[@]}" bash build.sh --sdk-path "$UNITREE_SDK2_PATH"

# 5. Install .so into the repo uv environment site-packages
SITE_PACKAGES=$(uv run "${UV_PROJECT_ARGS[@]}" python -c "import site; print(site.getsitepackages()[0])")
echo "Installing to: $SITE_PACKAGES"
cp build/lib/unitree_interface.cpython-*-linux-gnu.so "$SITE_PACKAGES/unitree_interface.so"

# 6. Build dex1_1_gripper_server
ARCH="$(uname -m)"
DEX1_LIB_FLAG="libUnitreeMotorSDK_Arm64.so"
[[ "$ARCH" == "x86_64" ]] && DEX1_LIB_FLAG="libUnitreeMotorSDK_Linux64.so"

rm -rf "$DEX1_DIR/build"
mkdir -p "$DEX1_DIR/build"
cmake -S "$DEX1_DIR" -B "$DEX1_DIR/build" \
  -DCMAKE_CXX_FLAGS="-I${WRAPPER_DIR}/include -I${WRAPPER_DIR}/thirdparty/include/ddscxx -I${WRAPPER_DIR}/thirdparty/include" \
  -DCMAKE_EXE_LINKER_FLAGS="-L${WRAPPER_DIR}/lib/${ARCH} -L${WRAPPER_DIR}/thirdparty/lib/${ARCH} -Wl,-rpath,${WRAPPER_DIR}/thirdparty/lib/${ARCH}"
make -C "$DEX1_DIR/build" -j"$(nproc)"

# 7. Install motor SDK .so system-wide so the binary can find it at runtime
sudo cp "$DEX1_DIR/lib/${DEX1_LIB_FLAG}" /usr/local/lib/
sudo ldconfig

# 8. Verify
uv run "${UV_PROJECT_ARGS[@]}" python -c "import unitree_interface; print('unitree_interface installed OK')"
echo "dex1_1_gripper_server built at: $DEX1_DIR/bin/dex1_1_gripper_server"
