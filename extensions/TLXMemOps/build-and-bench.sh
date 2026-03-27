#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRITON_EXT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE="${WORKSPACE:-$(cd "$TRITON_EXT_ROOT/.." && pwd)}"

# --- Configuration (override via environment) ---
TRITON_SOURCE_DIR="${TRITON_SOURCE_DIR:-$WORKSPACE/triton}"
TRITON_BUILD_DIR="${TRITON_BUILD_DIR:-$TRITON_SOURCE_DIR/build/cmake.linux-x86_64-cpython-3.11}"
LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-$TRITON_SOURCE_DIR/.llvm-project/build}"
VENV="${VENV:-$TRITON_SOURCE_DIR/.venv}"
BUILD_DIR="${BUILD_DIR:-$TRITON_EXT_ROOT/build}"

# --- Activate venv ---
echo "==> Activating venv: $VENV"
source "$VENV/bin/activate"

# --- Configure ---
echo "==> Configuring triton-ext (build dir: $BUILD_DIR)"
cmake -GNinja -B "$BUILD_DIR" -S "$TRITON_EXT_ROOT" \
  -DTRITON_SOURCE_DIR="$TRITON_SOURCE_DIR" \
  -DTRITON_BUILD_DIR="$TRITON_BUILD_DIR" \
  -DLLVM_BUILD_DIR="$LLVM_BUILD_DIR"

# --- Build ---
echo "==> Building triton-ext"
cmake --build "$BUILD_DIR"

# --- Verify plugin ---
PLUGIN_PATH="$BUILD_DIR/lib/libtlx_mem_ops.so"
if [ ! -f "$PLUGIN_PATH" ]; then
  echo "ERROR: Plugin not found at $PLUGIN_PATH"
  exit 1
fi
echo "==> Plugin built: $PLUGIN_PATH"

# --- Run benchmark ---
echo "==> Running AMD GEMM pipelined benchmark"
TRITON_PLUGIN_PATHS="$PLUGIN_PATH" \
  python "$SCRIPT_DIR/python/benchmarks/amd-gemm-pipelined.py"

# --- Run tests ---
echo "==> Running tests"
TRITON_PLUGIN_PATHS="$PLUGIN_PATH" \
  python -m pytest "$SCRIPT_DIR/test/" -v
