cmake_minimum_required(VERSION 3.20)
project(triton-ext DESCRIPTION "Extensions for the Triton compiler" LANGUAGES C CXX)

include(cmake/utils.cmake)

set(TRITON_EXT_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(TRITON_EXT_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(TRITON_EXT_LIBRARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/lib")
set(TRITON_EXT_SUPPORT_DIR "${TRITON_EXT_SOURCE_DIR}/support")

# Locate Triton: supports two modes:
#   1. TRITON_INSTALL_DIR — path to a Triton installation (headers + lib)
#   2. TRITON_SOURCE_DIR + TRITON_BUILD_DIR — path to Triton source + build trees
set(TRITON_INSTALL_DIR "$ENV{TRITON_INSTALL_DIR}" CACHE PATH "Path to Triton installation" FORCE)
if(NOT TRITON_SOURCE_DIR AND DEFINED ENV{TRITON_SOURCE_DIR})
    set(TRITON_SOURCE_DIR "$ENV{TRITON_SOURCE_DIR}" CACHE PATH "Path to Triton source tree")
endif()
if(NOT TRITON_BUILD_DIR AND DEFINED ENV{TRITON_BUILD_DIR})
    set(TRITON_BUILD_DIR "$ENV{TRITON_BUILD_DIR}" CACHE PATH "Path to Triton CMake build directory")
endif()

if(TRITON_INSTALL_DIR)
    if(NOT IS_ABSOLUTE "${TRITON_INSTALL_DIR}")
        get_filename_component(TRITON_INSTALL_DIR "${TRITON_INSTALL_DIR}" ABSOLUTE)
    endif()
    find_library(TRITON_LIB triton PATHS "${TRITON_INSTALL_DIR}/lib" "${TRITON_INSTALL_DIR}/lib64")
    include_directories("${TRITON_INSTALL_DIR}/include")
elseif(TRITON_SOURCE_DIR)
    find_library(TRITON_LIB triton PATHS "${TRITON_SOURCE_DIR}/python/triton/_C" NO_DEFAULT_PATH)
    include_directories("${TRITON_SOURCE_DIR}/include")
    include_directories("${TRITON_SOURCE_DIR}")
    include_directories("${TRITON_SOURCE_DIR}/third_party/amd/include")
    include_directories("${TRITON_SOURCE_DIR}/third_party")
    if(TRITON_BUILD_DIR)
        include_directories("${TRITON_BUILD_DIR}/include")
        include_directories("${TRITON_BUILD_DIR}/third_party")
    endif()
else()
    message(FATAL_ERROR "Set TRITON_INSTALL_DIR or TRITON_SOURCE_DIR (+ TRITON_BUILD_DIR)")
endif()
message(STATUS "Found Triton library at: ${TRITON_LIB}")

# Locate LLVM: supports LLVM_INSTALL_DIR or LLVM_BUILD_DIR.
set(LLVM_INSTALL_DIR "$ENV{LLVM_INSTALL_DIR}" CACHE PATH "Path to LLVM installation" FORCE)
if(NOT LLVM_BUILD_DIR AND DEFINED ENV{LLVM_BUILD_DIR})
    set(LLVM_BUILD_DIR "$ENV{LLVM_BUILD_DIR}" CACHE PATH "Path to LLVM/MLIR build directory")
endif()

set(_MLIR_SEARCH_PATHS "")
if(LLVM_INSTALL_DIR)
    if(NOT IS_ABSOLUTE "${LLVM_INSTALL_DIR}")
        get_filename_component(LLVM_INSTALL_DIR "${LLVM_INSTALL_DIR}" ABSOLUTE)
    endif()
    message(STATUS "Found LLVM installation at: ${LLVM_INSTALL_DIR}")
    list(APPEND _MLIR_SEARCH_PATHS "${LLVM_INSTALL_DIR}/lib/cmake/mlir")
endif()
if(LLVM_BUILD_DIR)
    message(STATUS "Found LLVM build at: ${LLVM_BUILD_DIR}")
    list(APPEND _MLIR_SEARCH_PATHS "${LLVM_BUILD_DIR}/lib/cmake/mlir")
endif()
find_package(MLIR REQUIRED CONFIG PATHS ${_MLIR_SEARCH_PATHS})
# Include LLVM headers.
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})
# Setup the LLVM CMake modules (e.g., for `add_mlir_library`).
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen) # required by AddMLIR
include(AddLLVM)
include(AddMLIR)

# Temporary workaround for missing header (see `support/python/src/ir.h`).
include_directories(${TRITON_EXT_SUPPORT_DIR})

# Build common infrastructure files.
add_subdirectory(${TRITON_EXT_SUPPORT_DIR})

# Build extensions (ordered by typical dependencies).
set(ALL_SUBDIRECTORIES dialect pass backend language extensions)
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
