# ---------------------------------------------------------------------------
# Example — standalone Triton plugin for an out-of-tree MLIR dialect.
#
# Builds a single self-contained shared library, bundled into the
# `triton_example` wheel and loaded on import (see
# python/triton_example/__init__.py). It provides the `example` dialect
# (types and ops) and registers it with Triton.
#
# Driven by scikit-build-core (`pip install .`), or configured directly:
#
#   cmake -S dialect/Example -B dialect/Example/build -G Ninja \
#       -DLLVM_INSTALL_DIR=$(python ci/pick_local_artifact.py 'llvm-*') \
#       -DTRITON_WHEEL_DIR=$(python ci/probe_triton_wheel.py)
#   cmake --build dialect/Example/build
# ---------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.20)
project(example LANGUAGES C CXX)
include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/triton-ext.cmake")
configure_triton_extension()

# Extension-specific include: source headers, e.g. `#include "Example.h"`.
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src")

# ---------------------------------------------------------------------------
# TableGen: Example dialect declarations and definitions.
# ---------------------------------------------------------------------------
set(LLVM_TARGET_DEFINITIONS src/Example.td)
mlir_tablegen(ExampleDialect.h.inc   -gen-dialect-decls    -dialect=example)
mlir_tablegen(ExampleDialect.cpp.inc -gen-dialect-defs     -dialect=example)
mlir_tablegen(Example.h.inc          -gen-op-decls)
mlir_tablegen(Example.cpp.inc        -gen-op-defs)
mlir_tablegen(ExampleTypes.h.inc     -gen-typedef-decls    -typedefs-dialect=example)
mlir_tablegen(ExampleTypes.cpp.inc   -gen-typedef-defs     -typedefs-dialect=example)
add_public_tablegen_target(MLIRExampleIncGen)
add_dependencies(mlir-generic-headers MLIRExampleIncGen)

# ---------------------------------------------------------------------------
# Plugin library. src/Example.cpp includes support/ExportDialect.cpp to
# register the dialect; support/Export.cpp implements tritonGetPluginInfo().
# ---------------------------------------------------------------------------
add_library(${TRITON_EXT_NAME} SHARED
    src/Example.cpp
    "${TRITON_EXT_SUPPORT_DIR}/Export.cpp")
add_dependencies(${TRITON_EXT_NAME} MLIRExampleIncGen)

target_compile_definitions(${TRITON_EXT_NAME} PRIVATE
    TRITON_EXT_VERSION="${TRITON_EXT_VERSION}"
    TRITON_EXT_NAME=${TRITON_EXT_NAME}
    TRITON_EXT_CLASS=ExampleDialect)

# Shared compile flags + link-only-against-libtriton (single source of truth).
target_link_libraries(${TRITON_EXT_NAME} PRIVATE triton_ext::plugin)

set_target_properties(${TRITON_EXT_NAME} PROPERTIES
    PREFIX "lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

install(TARGETS ${TRITON_EXT_NAME} LIBRARY DESTINATION triton_example)
