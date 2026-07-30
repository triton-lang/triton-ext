# ---------------------------------------------------------------------------
# Example — standalone Triton plugin for an out-of-tree MLIR dialect.
#
# Builds a single self-contained `libexample.so`, bundled into the `example`
# wheel and loaded on import (see python/triton_example/__init__.py). It provides
# the `example` dialect (types and ops) and registers it with Triton.
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
#   cmake -S dialect/Example -B dialect/Example/build -G Ninja \
#       -DLLVM_INSTALL_DIR=$(python ci/pick_local_artifact.py 'llvm-*') \
#       -DTRITON_WHEEL_DIR=$(python ci/probe_triton_wheel.py)
#   cmake --build dialect/Example/build
# ---------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.20)
project(example LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

find_package(Python COMPONENTS Interpreter REQUIRED)

# ---------------------------------------------------------------------------
# Extension name/version come from pyproject.toml, passed in by scikit-build-core
# as SKBUILD_PROJECT_NAME/SKBUILD_PROJECT_VERSION. The name drives the output
# library filename (`libexample.so`). Fall back to the project() values for
# direct `cmake` configures that don't go through scikit-build-core.
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
# The C++ dialect class registered with Triton (see support/ExportDialect.cpp).
set(TRITON_EXT_CLASS ExampleDialect)
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
# Include search paths (shared by TableGen and the C++ compile). The support
# directory provides the plugin export glue (Export.cpp / ExportDialect.cpp).
# ---------------------------------------------------------------------------
set(TRITON_EXT_SUPPORT_DIR "${TRITON_EXT_REPO_ROOT}/support")

include_directories(SYSTEM
    ${MLIR_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS}
    "${TRITON_WHEEL_INCLUDE_DIR}")
include_directories(
    # source headers, e.g. `#include "Example.h"`
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    # generated TableGen headers, e.g. `#include "ExampleDialect.h.inc"`
    "${CMAKE_CURRENT_BINARY_DIR}"
    # plugin export glue, e.g. `#include "ExportDialect.cpp"` / `"Export.h"`
    "${TRITON_EXT_SUPPORT_DIR}")
add_definitions(${LLVM_DEFINITIONS})

# ---------------------------------------------------------------------------
# TableGen the Example dialect (creates the `MLIRExampleIncGen` target and emits
# ExampleDialect.{h,cpp}.inc, Example.{h,cpp}.inc, ExampleTypes.{h,cpp}.inc).
# ---------------------------------------------------------------------------
set(LLVM_TARGET_DEFINITIONS src/Example.td)
mlir_tablegen(ExampleDialect.h.inc -gen-dialect-decls -dialect=example)
mlir_tablegen(ExampleDialect.cpp.inc -gen-dialect-defs -dialect=example)
mlir_tablegen(Example.h.inc -gen-op-decls)
mlir_tablegen(Example.cpp.inc -gen-op-defs)
mlir_tablegen(ExampleTypes.h.inc -gen-typedef-decls -typedefs-dialect=example)
mlir_tablegen(ExampleTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=example)
add_public_tablegen_target(MLIRExampleIncGen)
add_dependencies(mlir-generic-headers MLIRExampleIncGen)

# ---------------------------------------------------------------------------
# Plugin library. Compiles the dialect together with the support export glue
# (support/Export.cpp implements tritonGetPluginInfo(); src/Example.cpp includes
# support/ExportDialect.cpp to register the dialect).
# ---------------------------------------------------------------------------
add_library(${TRITON_EXT_NAME} SHARED
    src/Example.cpp
    "${TRITON_EXT_SUPPORT_DIR}/Export.cpp")
add_dependencies(${TRITON_EXT_NAME} MLIRExampleIncGen)

target_compile_definitions(${TRITON_EXT_NAME} PRIVATE
    TRITON_EXT_VERSION="${TRITON_EXT_VERSION}"
    TRITON_EXT_NAME=${TRITON_EXT_NAME}
    TRITON_EXT_CLASS=${TRITON_EXT_CLASS})

target_compile_options(${TRITON_EXT_NAME} PRIVATE
    -fvisibility=hidden
    -fno-rtti
    -Wno-deprecated-declarations)

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

# Emit a plain, unversioned `libexample.so`.
set_target_properties(${TRITON_EXT_NAME} PROPERTIES
    PREFIX "lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# ---------------------------------------------------------------------------
# Install the library into the `triton_example` package so scikit-build-core
# bundles it into the wheel next to python/triton_example/__init__.py, which
# loads it.
# ---------------------------------------------------------------------------
install(TARGETS ${TRITON_EXT_NAME} LIBRARY DESTINATION triton_example)
