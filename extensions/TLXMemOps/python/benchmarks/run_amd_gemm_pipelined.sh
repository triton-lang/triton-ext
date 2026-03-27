#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TLXMEMOPS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# When built within triton-ext, the .so lives in the triton-ext build lib dir.
# Set TRITON_EXT_BUILD_DIR to override (e.g., /path/to/triton-ext/build).
TRITON_EXT_BUILD_DIR="${TRITON_EXT_BUILD_DIR:-$(cd "$TLXMEMOPS_ROOT/../../../build" && pwd)}"

TRITON_PLUGIN_PATHS="$TRITON_EXT_BUILD_DIR/lib/libtlx_mem_ops.so" \
    python "$SCRIPT_DIR/amd-gemm-pipelined.py" "$@"
