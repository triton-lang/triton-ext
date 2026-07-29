# ---------------------------------------------------------------------------
# uTLX — standalone Triton plugin for the TLX dialect.
#
# Builds a single self-contained `libutlx.so`, bundled into the `utlx` wheel and
# loaded on import (see python/utlx_plugin/__init__.py). It provides the TLX
# dialect (types, attrs, ops), TLX transform passes, TLX custom ops for the
# TritonOpBuilder, and the ConvertTritonToTritonGPU conversion patterns.
#
# The plugin links ONLY against the Triton shared library shipped inside an
# installed Triton wheel (`triton/_C/libtriton.so`), which re-exports the
# MLIR/LLVM symbols it needs (the wheel must be built with
# `TRITON_EXT_ENABLED=1`). LLVM/MLIR is used only for headers, `mlir-tblgen`,
# and CMake modules — never for linking.
#
# Two inputs are required:
#   - TRITON_WHEEL_DIR  — an installed Triton wheel (`<site-packages>/triton`).
#                         Discovered automatically from the active Python
#                         interpreter, or pass `-DTRITON_WHEEL_DIR=...`.
#   - LLVM_INSTALL_DIR  — an LLVM/MLIR install (headers + mlir-tblgen + CMake
#                         modules). Discovered automatically from a local
#                         `llvm-*` artifact (see cmake/llvm.cmake), or pass
#                         `-DLLVM_INSTALL_DIR=...` / set the environment
#                         variable to override.
#
# This file is normally driven by scikit-build-core (`pip install .`), but can
# also be configured directly:
#
#   cmake -S extensions/utlx -B extensions/utlx/build -G Ninja \
#       -DLLVM_INSTALL_DIR=$(python ci/pick_local_artifact.py 'llvm-*') \
#       -DTRITON_WHEEL_DIR=$(python ci/probe_triton_wheel.py)
#   cmake --build extensions/utlx/build
# ---------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.20)
project(utlx LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

find_package(Python COMPONENTS Interpreter REQUIRED)

# ---------------------------------------------------------------------------
# Extension name/version come from pyproject.toml, passed in by scikit-build-core
# as SKBUILD_PROJECT_NAME/SKBUILD_PROJECT_VERSION. The name drives the output
# library filename (`libutlx.so`). Fall back to the project() values for direct
# `cmake` configures that don't go through scikit-build-core.
# ---------------------------------------------------------------------------
if(DEFINED SKBUILD_PROJECT_NAME)
    set(TRITON_EXT_NAME "${SKBUILD_PROJECT_NAME}")
else()
    set(TRITON_EXT_NAME "${PROJECT_NAME}")
endif()
if(DEFINED SKBUILD_PROJECT_VERSION)
    set(TRITON_EXT_VERSION "${SKBUILD_PROJECT_VERSION}")
else()
    set(TRITON_EXT_VERSION "0.0.0+dev")
endif()
message(STATUS "Building triton-ext extension: ${TRITON_EXT_NAME} "
               "(v${TRITON_EXT_VERSION})")

# ---------------------------------------------------------------------------
# Locate Triton from an installed wheel: headers under `<triton>/include`, the
# shared library under `<triton>/_C`. Discover it (from the environment or the
# active interpreter; see cmake/triton-wheel.cmake) unless already given via
# -DTRITON_WHEEL_DIR.
# ---------------------------------------------------------------------------
get_filename_component(TRITON_EXT_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.."
                       ABSOLUTE)
include("${TRITON_EXT_REPO_ROOT}/cmake/triton-wheel.cmake")
if(NOT DEFINED TRITON_WHEEL_DIR)
    find_triton_wheel(TRITON_WHEEL_DIR)
endif()
message(STATUS "Found Triton wheel at ${TRITON_WHEEL_DIR}")

get_filename_component(TRITON_WHEEL_DIR "${TRITON_WHEEL_DIR}" ABSOLUTE)
set(TRITON_WHEEL_INCLUDE_DIR "${TRITON_WHEEL_DIR}/include")
find_library(TRITON_LIB NAMES triton
PATHS "${TRITON_WHEEL_DIR}/_C" NO_DEFAULT_PATH REQUIRED)
message(STATUS "Found Triton library at ${TRITON_LIB}")

if(NOT EXISTS "${TRITON_WHEEL_INCLUDE_DIR}/triton")
    message(FATAL_ERROR
            "Triton wheel is missing C++ headers at ${TRITON_WHEEL_INCLUDE_DIR}; "
            "it must be built with TRITON_EXT_ENABLED=1.")
endif()

# ---------------------------------------------------------------------------
# Locate LLVM/MLIR for headers, mlir-tblgen, and CMake modules (NOT for linking:
# the required symbols are re-exported by libtriton). Discover it (from the
# environment or a local artifact; see cmake/llvm.cmake) unless already given
# via -DLLVM_INSTALL_DIR.
# ---------------------------------------------------------------------------
include("${TRITON_EXT_REPO_ROOT}/cmake/llvm.cmake")
if(NOT DEFINED LLVM_INSTALL_DIR)
    find_llvm_install_dir(LLVM_INSTALL_DIR)
endif()
get_filename_component(LLVM_INSTALL_DIR "${LLVM_INSTALL_DIR}" ABSOLUTE)
find_package(MLIR REQUIRED CONFIG PATHS "${LLVM_INSTALL_DIR}/lib/cmake/mlir")
message(STATUS "Found MLIR at ${MLIR_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)

# ---------------------------------------------------------------------------
# Include search paths (shared by TableGen and the C++ compile). These must be
# set before `mlir_tablegen` so TableGen can resolve `include "..."` directives.
# ---------------------------------------------------------------------------
set(UTLX_DIALECT_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tlx/dialect/include")
set(UTLX_TABLEGEN_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/tlx/dialect/include")

include_directories(SYSTEM
    ${MLIR_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS}
    "${TRITON_WHEEL_INCLUDE_DIR}")
include_directories(
    # `#include "tlx/dialect/include/IR/Dialect.h"` (source root)
    "${CMAKE_CURRENT_SOURCE_DIR}"
    # `#include "tlx/dialect/include/Transforms/Passes.h.inc"` (generated root)
    "${CMAKE_CURRENT_BINARY_DIR}"
    # `#include "IR/Dialect.h"` (source headers)
    "${UTLX_DIALECT_INCLUDE_DIR}"
    # local TableGen includes, e.g. `include "TLXDialect.td"`
    "${UTLX_DIALECT_INCLUDE_DIR}/IR"
    # `#include "IR/Dialect.cpp.inc"` (generated headers)
    "${UTLX_TABLEGEN_OUTPUT_DIR}")
add_definitions(${LLVM_DEFINITIONS})

# ---------------------------------------------------------------------------
# TableGen the TLX dialect and pass declarations.
# ---------------------------------------------------------------------------
set(LLVM_TARGET_DEFINITIONS "${UTLX_DIALECT_INCLUDE_DIR}/IR/TLXDialect.td")
mlir_tablegen(tlx/dialect/include/IR/Dialect.h.inc -gen-dialect-decls -dialect=tlx)
mlir_tablegen(tlx/dialect/include/IR/Dialect.cpp.inc -gen-dialect-defs -dialect=tlx)

set(LLVM_TARGET_DEFINITIONS "${UTLX_DIALECT_INCLUDE_DIR}/IR/TLXTypes.td")
mlir_tablegen(tlx/dialect/include/IR/TLXTypes.h.inc -gen-typedef-decls)
mlir_tablegen(tlx/dialect/include/IR/TLXTypes.cpp.inc -gen-typedef-defs)
mlir_tablegen(tlx/dialect/include/IR/TLXTypesEnums.h.inc -gen-enum-decls)
mlir_tablegen(tlx/dialect/include/IR/TLXTypesEnums.cpp.inc -gen-enum-defs)

set(LLVM_TARGET_DEFINITIONS "${UTLX_DIALECT_INCLUDE_DIR}/IR/TLXOps.td")
mlir_tablegen(tlx/dialect/include/IR/Ops.h.inc -gen-op-decls)
mlir_tablegen(tlx/dialect/include/IR/Ops.cpp.inc -gen-op-defs)

set(LLVM_TARGET_DEFINITIONS "${UTLX_DIALECT_INCLUDE_DIR}/IR/TLXAttrDefs.td")
mlir_tablegen(tlx/dialect/include/IR/TLXAttrDefs.h.inc -gen-attrdef-decls)
mlir_tablegen(tlx/dialect/include/IR/TLXAttrDefs.cpp.inc -gen-attrdef-defs)

set(LLVM_TARGET_DEFINITIONS "${UTLX_DIALECT_INCLUDE_DIR}/Transforms/Passes.td")
mlir_tablegen(tlx/dialect/include/Transforms/Passes.h.inc -gen-pass-decls)

add_public_tablegen_target(uTLXTableGen)

# ---------------------------------------------------------------------------
# Plugin library.
# ---------------------------------------------------------------------------
add_library(${TRITON_EXT_NAME} SHARED
    uTLXPlugin.cpp
    uTLXConversionPatterns.cpp
    # Dialect IR
    tlx/dialect/lib/IR/Dialect.cpp
    tlx/dialect/lib/IR/Ops.cpp
    tlx/dialect/lib/IR/Traits.cpp
    tlx/dialect/lib/IR/Types.cpp
    # Dialect Transforms
    tlx/dialect/lib/Transforms/Fixup.cpp
    tlx/dialect/lib/Transforms/PropagateLayout.cpp
    tlx/dialect/lib/Transforms/InsertRequireLayout.cpp
    tlx/dialect/lib/Transforms/RewriteLocalAlias.cpp
    tlx/dialect/lib/Transforms/ResolvePlaceholderLayouts.cpp
    tlx/dialect/lib/Transforms/PrintTTGIRToTLX.cpp
    tlx/dialect/lib/Transforms/StorageAliasSizeDefinition.cpp
    tlx/dialect/lib/Transforms/StorageAliasAllocation.cpp
    tlx/dialect/lib/Transforms/StorageAliasLowering.cpp
    tlx/dialect/lib/Transforms/BufferOffsetCalculation.cpp
    # Analysis
    tlx/dialect/lib/Analysis/LayoutPropagation.cpp
    # Runtime op creation
    ops/NewOps.cpp
    # Ported passes - NVIDIA
    passes/nvidia/PruneUnusedBarriers.cpp
    passes/nvidia/PingPong.cpp)
add_dependencies(${TRITON_EXT_NAME} uTLXTableGen)

target_compile_definitions(${TRITON_EXT_NAME} PRIVATE
    TRITON_EXT_VERSION="${TRITON_EXT_VERSION}")

target_compile_options(${TRITON_EXT_NAME} PRIVATE
    -fvisibility=hidden
    -fno-exceptions
    -fno-rtti
    -Wno-deprecated-declarations)

# ---------------------------------------------------------------------------
# Optional AMDGPU support (requires third_party AMD headers in the Triton wheel).
# ---------------------------------------------------------------------------
if(EXISTS "${TRITON_WHEEL_INCLUDE_DIR}/Dialect/TritonAMDGPU/IR/Dialect.h")
    message(STATUS "uTLX: AMDGPU dialect header found — enabling UTLX_HAS_AMDGPU")
    target_compile_definitions(${TRITON_EXT_NAME} PRIVATE UTLX_HAS_AMDGPU)
else()
    message(STATUS "uTLX: AMDGPU dialect header not found — AMD support disabled")
endif()

# ---------------------------------------------------------------------------
# Require LLVM_ENABLE_ABI_BREAKING_CHECKS=1 to match the host libtriton (built as
# an asserts build); see cmake/llvm-abi-breaking.cmake for the rationale.
# ---------------------------------------------------------------------------
include("${TRITON_EXT_REPO_ROOT}/cmake/llvm-abi-breaking.cmake")
require_llvm_abi_breaking_checks("${LLVM_INSTALL_DIR}")

# Link ONLY against libtriton, which re-exports the MLIR/LLVM symbols. Use
# --no-as-needed so the DT_NEEDED entry is retained even though the plugin
# resolves the symbols at load time from the already-loaded libtriton.
target_link_libraries(${TRITON_EXT_NAME} PRIVATE
    "$<$<PLATFORM_ID:Linux>:-Wl,--no-as-needed>"
    ${TRITON_LIB}
    "$<$<PLATFORM_ID:Linux>:-Wl,--as-needed>"
    "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>")

# Emit a plain, unversioned `libutlx.so`.
set_target_properties(${TRITON_EXT_NAME} PROPERTIES
    PREFIX "lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# ---------------------------------------------------------------------------
# Install the library into the `utlx_plugin` package so scikit-build-core bundles
# it into the wheel next to python/utlx_plugin/__init__.py, which loads it.
# ---------------------------------------------------------------------------
install(TARGETS ${TRITON_EXT_NAME} LIBRARY DESTINATION utlx_plugin)
