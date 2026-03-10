cmake_minimum_required(VERSION 3.20)
project(triton-ext DESCRIPTION "Extensions for the Triton compiler" LANGUAGES CXX)

include(cmake/utils.cmake)

set(TRITON_EXT_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(TRITON_EXT_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(TRITON_EXT_LIBRARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/lib")
set(TRITON_EXT_SUPPORT_DIR "${TRITON_EXT_SOURCE_DIR}/support")

# Locate Triton: we expect TRITON_INSTALL_DIR to contain libraries and headers,
# including the TableGen-generated files.
set(TRITON_INSTALL_DIR "$ENV{TRITON_INSTALL_DIR}" CACHE PATH "Path to Triton installation" FORCE)
if(NOT IS_ABSOLUTE "${TRITON_INSTALL_DIR}")
    get_filename_component(TRITON_INSTALL_DIR "${TRITON_INSTALL_DIR}" ABSOLUTE)
endif()
find_library(TRITON_LIB triton PATHS "${TRITON_INSTALL_DIR}/lib" "${TRITON_INSTALL_DIR}/lib64")
message(STATUS "Found Triton library at: ${TRITON_LIB}")
# Include Triton headers.
include_directories("${TRITON_INSTALL_DIR}/include")

# Locate LLVM.
set(LLVM_INSTALL_DIR "$ENV{LLVM_INSTALL_DIR}" CACHE PATH "Path to LLVM installation" FORCE)
if(NOT IS_ABSOLUTE "${LLVM_INSTALL_DIR}")
    get_filename_component(LLVM_INSTALL_DIR "${LLVM_INSTALL_DIR}" ABSOLUTE)
endif()
message(STATUS "Found LLVM installation at: ${LLVM_INSTALL_DIR}")
find_package(MLIR REQUIRED CONFIG PATHS "${LLVM_INSTALL_DIR}/lib/cmake/mlir")
# Include LLVM headers.
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})
# Setup the LLVM CMake modules (e.g., for `add_mlir_library`).
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen) # required by AddMLIR
include(AddLLVM)
include(AddMLIR)

# Build common infrastructure files.
add_subdirectory(${TRITON_EXT_SUPPORT_DIR})

# Build extensions (ordered by typical dependencies).
set(ALL_SUBDIRECTORIES dialect pass backend language)
foreach(subdir ${ALL_SUBDIRECTORIES})
    add_subdirectory("${TRITON_EXT_SOURCE_DIR}/${subdir}"
                     "${TRITON_EXT_BINARY_DIR}/${subdir}")
endforeach()

# Add tests.
add_subdirectory("${TRITON_EXT_SOURCE_DIR}/testing" "${TRITON_EXT_BINARY_DIR}/testing")

# We create a meta-target to track when all extensions are built. All extension targets add themselves to the global
# `TRITON_EXT_BUILT_TARGETS` during setup (e.g., `triton_ext_pass_setup`)
add_custom_target(triton-ext-all-extensions)
get_property(extension_targets GLOBAL PROPERTY TRITON_EXT_BUILT_TARGETS)
add_dependencies(triton-ext-all-extensions ${extension_targets})
