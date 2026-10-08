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
    # Layout passes
    passes/DotOperandLocalLoad.cpp
    passes/KeepLoadLayout.cpp
    passes/RematDotOperandSlices.cpp
    # Ported passes - NVIDIA
    passes/nvidia/PruneUnusedBarriers.cpp
    passes/nvidia/PingPong.cpp
    # Ported passes - AMD
    passes/amd/FallbackAsyncCopy.cpp)
add_dependencies(${TRITON_EXT_NAME} uTLXTableGen)

target_compile_definitions(${TRITON_EXT_NAME} PRIVATE
    TRITON_EXT_VERSION="${TRITON_EXT_VERSION}")

# ---------------------------------------------------------------------------
# Optional AMDGPU support (requires the TritonAMDGPU dialect headers).
#
# Triton does not install those into the wheel, so also look in the Triton
# source and build trees: TRITON_SOURCE_DIR (default: the checkout an editable
# install's TRITON_WHEEL_DIR, <src>/python/triton, lives in) for the headers and
# TRITON_BUILD_DIR (default: <src>/build/cmake.*) for their generated .inc
# files. Both may be set as CMake variables or environment variables.
# ---------------------------------------------------------------------------
set(_utlx_amd_header "Dialect/TritonAMDGPU/IR/Dialect.h")
set(_utlx_amd_gen_header "amd/include/Dialect/TritonAMDGPU/IR/Ops.h.inc")
if(EXISTS "${TRITON_WHEEL_INCLUDE_DIR}/${_utlx_amd_header}")
    message(STATUS "uTLX: AMDGPU dialect header found — enabling UTLX_HAS_AMDGPU")
    target_compile_definitions(${TRITON_EXT_NAME} PRIVATE UTLX_HAS_AMDGPU)
else()
    if(NOT DEFINED TRITON_SOURCE_DIR AND DEFINED ENV{TRITON_SOURCE_DIR})
        set(TRITON_SOURCE_DIR "$ENV{TRITON_SOURCE_DIR}")
    endif()
    if(NOT DEFINED TRITON_SOURCE_DIR)
        get_filename_component(TRITON_SOURCE_DIR "${TRITON_WHEEL_DIR}/../.." ABSOLUTE)
    endif()
    if(NOT DEFINED TRITON_BUILD_DIR AND DEFINED ENV{TRITON_BUILD_DIR})
        set(TRITON_BUILD_DIR "$ENV{TRITON_BUILD_DIR}")
    endif()
    if(NOT DEFINED TRITON_BUILD_DIR)
        file(GLOB _utlx_triton_build_dirs LIST_DIRECTORIES true
             "${TRITON_SOURCE_DIR}/build/cmake.*")
        foreach(_dir IN LISTS _utlx_triton_build_dirs)
            if(EXISTS "${_dir}/third_party/${_utlx_amd_gen_header}")
                set(TRITON_BUILD_DIR "${_dir}")
                break()
            endif()
        endforeach()
    endif()

    if(EXISTS "${TRITON_SOURCE_DIR}/third_party/amd/include/${_utlx_amd_header}"
       AND EXISTS "${TRITON_BUILD_DIR}/third_party/${_utlx_amd_gen_header}")
        message(STATUS "uTLX: AMDGPU dialect headers found in ${TRITON_SOURCE_DIR} "
                       "(generated: ${TRITON_BUILD_DIR}) — enabling UTLX_HAS_AMDGPU")
        target_include_directories(${TRITON_EXT_NAME} SYSTEM PRIVATE
            "${TRITON_SOURCE_DIR}/third_party/amd/include"
            "${TRITON_BUILD_DIR}/third_party")
        target_compile_definitions(${TRITON_EXT_NAME} PRIVATE UTLX_HAS_AMDGPU)
    else()
        message(STATUS "uTLX: AMDGPU dialect headers not found (set TRITON_SOURCE_DIR "
                       "/ TRITON_BUILD_DIR) — AMD support disabled")
    endif()
endif()

# ---------------------------------------------------------------------------
# Triton release/3.8 compatibility: PointerType::get takes a plain int address
# space there, rather than the PtrAddrSpace enum (defined in TritonTypeEnums.td)
# on Triton main. Both report TRITON_VERSION "3.8.0", so probe the header.
# ---------------------------------------------------------------------------
if(NOT EXISTS "${TRITON_WHEEL_INCLUDE_DIR}/triton/Dialect/Triton/IR/TritonTypeEnums.td")
    message(STATUS "uTLX: PtrAddrSpace not found — enabling UTLX_TRITON_INT_ADDR_SPACE")
    target_compile_definitions(${TRITON_EXT_NAME} PRIVATE UTLX_TRITON_INT_ADDR_SPACE)
endif()

# Likewise, TMEMSubSliceOp's derived builder gained a trailing `int dim` on
# Triton main; release/3.8's is (alloc, offset, size) and slices the last dim.
set(_utlx_ttng_ops_td
    "${TRITON_WHEEL_INCLUDE_DIR}/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOps.td")
if(EXISTS "${_utlx_ttng_ops_td}")
    file(READ "${_utlx_ttng_ops_td}" _utlx_ttng_ops_td_content)
    string(FIND "${_utlx_ttng_ops_td_content}" "\"int\":$size, \"int\":$dim"
           _utlx_tmem_subslice_dim_pos)
    if(_utlx_tmem_subslice_dim_pos EQUAL -1)
        message(STATUS "uTLX: TMEMSubSliceOp has no dim builder — enabling UTLX_TRITON_TMEM_SUBSLICE_NO_DIM")
        target_compile_definitions(${TRITON_EXT_NAME} PRIVATE UTLX_TRITON_TMEM_SUBSLICE_NO_DIM)
    endif()
endif()

# Shared compile flags + link-only-against-libtriton (single source of truth).
target_link_libraries(${TRITON_EXT_NAME} PRIVATE triton_ext::plugin)
# uTLX additionally disables C++ exceptions.
target_compile_options(${TRITON_EXT_NAME} PRIVATE -fno-exceptions)

set_target_properties(${TRITON_EXT_NAME} PROPERTIES
    PREFIX "lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

install(TARGETS ${TRITON_EXT_NAME} LIBRARY DESTINATION utlx_plugin)

# These land at the site-packages root rather than inside the package: site.py
# runs the .pth at interpreter startup, the only point early enough for a Triton
# that reads TRITON_PLUGIN_PATHS while importing (see _utlx_autoregister.py).
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/python/utlx_plugin.pth"
    "${CMAKE_CURRENT_SOURCE_DIR}/python/_utlx_autoregister.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/python/_utlx_fbtriton.py"
    DESTINATION .)
# The pytest plugin goes there too, so importing it does not import utlx_plugin
# (and with it Triton) in every pytest run.
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/python/_utlx_pytest.py" DESTINATION .)
