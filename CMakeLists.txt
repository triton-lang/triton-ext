cmake_minimum_required(VERSION 3.20)
project(triton-ext DESCRIPTION "Extensions for the Triton compiler" LANGUAGES CXX)

include(cmake/utils.cmake)

set(TRITON_EXT_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(TRITON_EXT_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(TRITON_EXT_LIBRARY_DIR "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/../extensions")
set(TRITON_EXT_INFRA_DIR "${TRITON_EXT_SOURCE_DIR}/infra")
message(STATUS "Triton extension source directory: ${TRITON_EXT_BINARY_DIR}")

# Locate Triton: we need TRITON_DIR for headers and libraries and TRITON_BUILD_DIR for TableGen-generated files.
set(TRITON_DIR "$ENV{TRITON_DIR}" CACHE PATH "Path to Triton installation" FORCE)
if(NOT TRITON_DIR)
    set(TRITON_DIR "${CMAKE_CURRENT_LIST_DIR}/../triton")
endif()
find_library(TRITON_LIB triton PATHS "${TRITON_DIR}/python/triton/_C")
message(STATUS "Found Triton library at: ${TRITON_LIB}")
set(TRITON_BUILD_DIR "$ENV{TRITON_BUILD_DIR}" CACHE PATH "Path to Triton's CMake-generated directory" FORCE)
message(STATUS "Found Triton build directory at: ${TRITON_BUILD_DIR}")
# Include Triton headers.
include_directories("${TRITON_DIR}/include")
include_directories("${TRITON_BUILD_DIR}/include")

# Locate LLVM.
set(LLVM_DIR "$ENV{LLVM_DIR}" CACHE PATH "Path to LLVM installation" FORCE)
message(STATUS "Found LLVM installation at: ${LLVM_DIR}")
find_package(MLIR REQUIRED CONFIG PATHS "${LLVM_DIR}/lib/cmake/mlir")
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
add_subdirectory(${TRITON_EXT_INFRA_DIR})

# Build extensions (ordered by typical dependencies).
set(ALL_SUBDIRECTORIES dialect pass backend language)
foreach(subdir ${ALL_SUBDIRECTORIES})
    add_subdirectory("${TRITON_EXT_SOURCE_DIR}/${subdir}"
                     "${TRITON_EXT_BINARY_DIR}/${subdir}")
endforeach()
