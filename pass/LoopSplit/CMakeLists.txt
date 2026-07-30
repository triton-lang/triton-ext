# ---------------------------------------------------------------------------
# LoopSplit — standalone Triton plugin for the loop-bisection pass.
#
# Builds a single self-contained `libloop_split.so`, bundled into the
# `triton_loop_split` wheel and loaded on import (see
# python/triton_loop_split/__init__.py). It provides the `triton-loop-split`
# pass and registers it with Triton's pass pipeline.
#
# Driven by scikit-build-core (`pip install .`), or configured directly:
#
#   cmake -S pass/LoopSplit -B pass/LoopSplit/build -G Ninja \
#       -DLLVM_INSTALL_DIR=$(python ci/pick_local_artifact.py 'llvm-*') \
#       -DTRITON_WHEEL_DIR=$(python ci/probe_triton_wheel.py)
#   cmake --build pass/LoopSplit/build
# ---------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.20)
project(loop_split LANGUAGES C CXX)
include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/triton-ext.cmake")
configure_triton_extension()

# ---------------------------------------------------------------------------
# TableGen: LoopSplit pass declarations → Passes.h.inc.
# ---------------------------------------------------------------------------
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Extension)
add_public_tablegen_target(MLIRLoopSplitIncGen)
add_dependencies(mlir-generic-headers MLIRLoopSplitIncGen)

# ---------------------------------------------------------------------------
# Plugin library. LoopSplit.cpp includes support/ExportPass.cpp to register
# the pass; support/Export.cpp implements tritonGetPluginInfo().
# ---------------------------------------------------------------------------
add_library(${TRITON_EXT_NAME} SHARED
    src/LoopSplit.cpp
    "${TRITON_EXT_SUPPORT_DIR}/Export.cpp")
add_dependencies(${TRITON_EXT_NAME} MLIRLoopSplitIncGen)

target_compile_definitions(${TRITON_EXT_NAME} PRIVATE
    TRITON_EXT_VERSION="${TRITON_EXT_VERSION}"
    TRITON_EXT_NAME=${TRITON_EXT_NAME}
    TRITON_EXT_CLASS=LoopBisectPass)

# Shared compile flags + link-only-against-libtriton (single source of truth).
target_link_libraries(${TRITON_EXT_NAME} PRIVATE triton_ext::plugin)

set_target_properties(${TRITON_EXT_NAME} PROPERTIES
    PREFIX "lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

install(TARGETS ${TRITON_EXT_NAME} LIBRARY DESTINATION triton_loop_split)
