# ---------------------------------------------------------------------------
# uTLX — standalone Triton plugin for the TLX dialect.
#
# Builds a single self-contained `libutlx.so`, bundled into the `utlx` wheel
# and loaded on import (see python/utlx_plugin/__init__.py). It provides the
# TLX dialect (types, attrs, ops), TLX transform passes, custom TLX ops for
# the TritonOpBuilder, and ConvertTritonToTritonGPU conversion patterns.
#
# Driven by scikit-build-core (`pip install .`), or configured directly:
#
#   cmake -S extensions/utlx -B extensions/utlx/build -G Ninja \
#       -DLLVM_INSTALL_DIR=$(python ci/pick_local_artifact.py 'llvm-*') \
#       -DTRITON_WHEEL_DIR=$(python ci/probe_triton_wheel.py)
#   cmake --build extensions/utlx/build
# ---------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.20)
project(utlx LANGUAGES C CXX)
include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/triton-ext.cmake")
configure_triton_extension()

# ---------------------------------------------------------------------------
# Extension-specific include paths. These must be set before mlir_tablegen so
# TableGen can resolve `include "..."` directives in the .td files.
# ---------------------------------------------------------------------------
set(UTLX_DIALECT_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tlx/dialect/include")
set(UTLX_TABLEGEN_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/tlx/dialect/include")

include_directories(
    # `#include "tlx/dialect/include/IR/Dialect.h"` (source root)
    "${CMAKE_CURRENT_SOURCE_DIR}"
    # `#include "IR/Dialect.h"` (source headers)
    "${UTLX_DIALECT_INCLUDE_DIR}"
    # local TableGen includes, e.g. `include "TLXDialect.td"`
    "${UTLX_DIALECT_INCLUDE_DIR}/IR"
    # `#include "IR/Dialect.cpp.inc"` (generated headers)
    "${UTLX_TABLEGEN_OUTPUT_DIR}")

# ---------------------------------------------------------------------------
# TableGen the TLX dialect and pass declarations.
# ---------------------------------------------------------------------------
set(LLVM_TARGET_DEFINITIONS "${UTLX_DIALECT_INCLUDE_DIR}/IR/TLXDialect.td")
mlir_tablegen(tlx/dialect/include/IR/Dialect.h.inc   -gen-dialect-decls -dialect=tlx)
mlir_tablegen(tlx/dialect/include/IR/Dialect.cpp.inc -gen-dialect-defs  -dialect=tlx)

set(LLVM_TARGET_DEFINITIONS "${UTLX_DIALECT_INCLUDE_DIR}/IR/TLXTypes.td")
mlir_tablegen(tlx/dialect/include/IR/TLXTypes.h.inc       -gen-typedef-decls)
mlir_tablegen(tlx/dialect/include/IR/TLXTypes.cpp.inc     -gen-typedef-defs)
mlir_tablegen(tlx/dialect/include/IR/TLXTypesEnums.h.inc  -gen-enum-decls)
mlir_tablegen(tlx/dialect/include/IR/TLXTypesEnums.cpp.inc -gen-enum-defs)

set(LLVM_TARGET_DEFINITIONS "${UTLX_DIALECT_INCLUDE_DIR}/IR/TLXOps.td")
mlir_tablegen(tlx/dialect/include/IR/Ops.h.inc   -gen-op-decls)
mlir_tablegen(tlx/dialect/include/IR/Ops.cpp.inc -gen-op-defs)

set(LLVM_TARGET_DEFINITIONS "${UTLX_DIALECT_INCLUDE_DIR}/IR/TLXAttrDefs.td")
mlir_tablegen(tlx/dialect/include/IR/TLXAttrDefs.h.inc   -gen-attrdef-decls)
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

# ---------------------------------------------------------------------------
# Optional AMDGPU support (requires third_party AMD headers in the Triton wheel).
# ---------------------------------------------------------------------------
if(EXISTS "${TRITON_WHEEL_INCLUDE_DIR}/Dialect/TritonAMDGPU/IR/Dialect.h")
    message(STATUS "uTLX: AMDGPU dialect header found — enabling UTLX_HAS_AMDGPU")
    target_compile_definitions(${TRITON_EXT_NAME} PRIVATE UTLX_HAS_AMDGPU)
else()
    message(STATUS "uTLX: AMDGPU dialect header not found — AMD support disabled")
endif()

# Shared compile flags + link-only-against-libtriton (single source of truth).
target_link_libraries(${TRITON_EXT_NAME} PRIVATE triton_ext::plugin)
# uTLX additionally disables C++ exceptions.
target_compile_options(${TRITON_EXT_NAME} PRIVATE -fno-exceptions)

set_target_properties(${TRITON_EXT_NAME} PROPERTIES
    PREFIX "lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

install(TARGETS ${TRITON_EXT_NAME} LIBRARY DESTINATION utlx_plugin)
