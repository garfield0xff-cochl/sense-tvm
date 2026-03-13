#!/bin/sh

set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build}"
CONFIG_FILE="$BUILD_DIR/config.cmake"
BUILD_TYPE="${BUILD_TYPE:-RelWithDebInfo}"
JOBS="${JOBS:-}"
CC="${CC:-/usr/bin/clang}"
CXX="${CXX:-/usr/bin/clang++}"

section() {
  printf '\n==> %s\n' "$1"
}

info() {
  printf '  %s\n' "$1"
}

show_env() {
  section "Build configuration"
  info "ROOT_DIR=$ROOT_DIR"
  info "BUILD_DIR=$BUILD_DIR"
  info "BUILD_TYPE=$BUILD_TYPE"
  info "CC=$CC"
  info "CXX=$CXX"
  if [ -n "$JOBS" ]; then
    info "JOBS=$JOBS"
  else
    info "JOBS=auto"
  fi
}

show_env

section "Preparing build directory"
mkdir -p "$BUILD_DIR"
cp "$ROOT_DIR/cmake/config.cmake" "$CONFIG_FILE"

LLVM_CONFIG="${LLVM_CONFIG:-}"
ORIG_DYLD_LIBRARY_PATH="${DYLD_LIBRARY_PATH:-}"

if [ -z "$LLVM_CONFIG" ]; then
  for candidate in llvm-config-17 llvm-config-18; do
    if command -v "$candidate" >/dev/null 2>&1; then
      LLVM_CONFIG=$candidate
      break
    fi
  done
fi

if [ -z "$LLVM_CONFIG" ] || ! command -v "$LLVM_CONFIG" >/dev/null 2>&1; then
  cat <<'EOF'
llvm-config-17 or llvm-config-18 was not found in PATH.
This build_sense.sh is pinned to LLVM 17/18 for compatibility.

Install LLVM 17 or LLVM 18 first, or point LLVM_CONFIG to a matching binary.

Example:
  LLVM_CONFIG=/path/to/llvm-config-17 ./build_sense.sh
  LLVM_CONFIG=/path/to/llvm-config-18 ./build_sense.sh

EOF
  exit 1
fi

section "Using LLVM"
info "LLVM_CONFIG=$LLVM_CONFIG"
if [ -n "$ORIG_DYLD_LIBRARY_PATH" ]; then
  info "Ignoring inherited DYLD_LIBRARY_PATH during build/install steps"
fi

{
  printf 'set(CMAKE_BUILD_TYPE %s)\n' "$BUILD_TYPE"
  printf 'set(USE_LLVM "%s --link-static")\n' "$LLVM_CONFIG"
  printf 'set(USE_CCACHE AUTO)\n'
  printf 'set(SUMMARIZE ON)\n'
} >>"$CONFIG_FILE"

GENERATOR_ARGS=
if command -v ninja >/dev/null 2>&1; then
  GENERATOR_ARGS='-G Ninja'
fi

section "Configuring TVM"
info "config.cmake=$CONFIG_FILE"
if [ -n "$GENERATOR_ARGS" ]; then
  info "generator=Ninja"
  env -u DYLD_LIBRARY_PATH CC="$CC" CXX="$CXX" cmake -S "$ROOT_DIR" -B "$BUILD_DIR" -G Ninja
else
  info "generator=default"
  env -u DYLD_LIBRARY_PATH CC="$CC" CXX="$CXX" cmake -S "$ROOT_DIR" -B "$BUILD_DIR"
fi

section "Building TVM"
if [ -n "$JOBS" ]; then
  info "parallel jobs=$JOBS"
  env -u DYLD_LIBRARY_PATH cmake --build "$BUILD_DIR" --parallel "$JOBS"
else
  info "parallel jobs=auto"
  env -u DYLD_LIBRARY_PATH cmake --build "$BUILD_DIR" --parallel
fi

section "Exporting runtime environment"
export TVM_HOME="$ROOT_DIR"
export TVM_LIBRARY_PATH="$BUILD_DIR"
export PYTHONPATH="$ROOT_DIR/python"
export DYLD_LIBRARY_PATH="$BUILD_DIR${ORIG_DYLD_LIBRARY_PATH:+:$ORIG_DYLD_LIBRARY_PATH}"
 
info "TVM_HOME=$TVM_HOME"
info "TVM_LIBRARY_PATH=$TVM_LIBRARY_PATH"
info "PYTHONPATH=$PYTHONPATH"
info "DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH"

section "Reinstalling editable python packages"
(cd "$ROOT_DIR/3rdparty/tvm-ffi" && env -u DYLD_LIBRARY_PATH python --version && env -u DYLD_LIBRARY_PATH CC="$CC" CXX="$CXX" LLVM_CONFIG="$LLVM_CONFIG" python -m pip install -e .)
(cd "$ROOT_DIR" && env -u DYLD_LIBRARY_PATH python --version && env -u DYLD_LIBRARY_PATH CC="$CC" CXX="$CXX" LLVM_CONFIG="$LLVM_CONFIG" python -m pip install -e .)

section "Synchronizing version metadata"
(cd "$ROOT_DIR" && env -u DYLD_LIBRARY_PATH python version.py)

section "Install Sense Required Dependencies"
(cd "$ROOT_DIR" && env -u DYLD_LIBRARY_PATH python -m pip install numpy psutil cloudpickle xgboost ml_dtypes onnx onnxruntime)

section "Completed"
info "RUN THIS CODE!!!!!"
info "export TVM_HOME=$ROOT_DIR"
info "export TVM_LIBRARY_PATH=$BUILD_DIR"
info "export PYTHONPATH=$ROOT_DIR/python"
info "export DYLD_LIBRARY_PATH=$BUILD_DIR\${DYLD_LIBRARY_PATH:+:\$DYLD_LIBRARY_PATH}"

section "Run Sense"
info "cd sense"
info "python main.py --config=settings/rpi2.json --validate"
