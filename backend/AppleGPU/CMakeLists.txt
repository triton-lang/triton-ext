cmake_minimum_required(VERSION 3.20)
if(NOT DEFINED CMAKE_OSX_DEPLOYMENT_TARGET)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "14.0")
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

project(applegpu_backend LANGUAGES C CXX)
include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/triton-ext.cmake")
configure_triton_extension()

set(APPLE_GPU_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(APPLE_GPU_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}")

include_directories(${TRITON_EXT_SUPPORT_DIR})
include_directories(${APPLE_GPU_INCLUDE_DIR})

set(AGPU_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/agpu/include")
include_directories(${AGPU_INCLUDE_DIR})
include_directories(${APPLE_GPU_GEN_DIR})

file(MAKE_DIRECTORY "${APPLE_GPU_GEN_DIR}/TritonAppleGPUTransforms")

set(LLVM_TARGET_DEFINITIONS
    ${APPLE_GPU_INCLUDE_DIR}/TritonAppleGPUTransforms/Passes.td)
mlir_tablegen(TritonAppleGPUTransforms/Passes.h.inc -gen-pass-decls -name TritonAppleGPU)
add_public_tablegen_target(TritonAppleGPUTransformsPassesIncGen)

set(TO_MSL lib/TritonAppleGPUToMSL)

add_mlir_library(TritonAppleGPUBackend
    ExportAppleGPU.cpp
    "${TRITON_EXT_SUPPORT_DIR}/Export.cpp"

    lib/TritonAppleGPUTransforms/EmitMSL.cpp

    ${TO_MSL}/AgpuEmitter.cpp
    ${TO_MSL}/AgpuHandlersArith.cpp
    ${TO_MSL}/AgpuHandlersMemory.cpp
    ${TO_MSL}/AgpuHandlersValue.cpp

    agpu/src/Printer.cpp

    SHARED
    EXCLUDE_FROM_LIBMLIR

    DEPENDS
    TritonAppleGPUTransformsPassesIncGen

    LINK_LIBS PUBLIC
    ${TRITON_LIB}
)

target_link_libraries(TritonAppleGPUBackend PUBLIC MLIRMathTransforms)
target_link_options(TritonAppleGPUBackend PRIVATE -Wl,-dead_strip)

target_compile_options(TritonAppleGPUBackend PRIVATE
    -Wall -Wextra -Wno-missing-field-initializers)

set_target_properties(TritonAppleGPUBackend PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    OUTPUT_NAME "applegpu_backend"
    CXX_STANDARD 17)

# On macOS with shared LLVM, use flat namespace so the plugin can resolve
# LLVM symbols from any loaded dylib (not just the one it was linked against).
if(APPLE)
    target_link_options(TritonAppleGPUBackend PRIVATE -Wl,-flat_namespace)
endif()

install(TARGETS TritonAppleGPUBackend
        LIBRARY DESTINATION triton_apple_backend)

if(APPLE)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
    list(APPEND CMAKE_PREFIX_PATH "${Python_SITEARCH}/torch/share/cmake")
    find_package(Torch QUIET)
    if(NOT Torch_FOUND)
        message(FATAL_ERROR
            "metal_utils requires torch installed in ${Python_SITEARCH}, or "
            "-DTorch_DIR for a torch elsewhere")
    endif()

    set(METAL_UTILS_PY
        "${CMAKE_CURRENT_SOURCE_DIR}/python/triton_apple_backend")
    python_add_library(metal_utils MODULE "${METAL_UTILS_PY}/metal_utils.m"
                       WITH_SOABI)
    target_compile_options(metal_utils PRIVATE
        -ObjC++ -std=c++20 -O3 -Wno-deprecated-declarations)
    target_link_libraries(metal_utils PRIVATE "${TORCH_LIBRARIES}"
        "-framework Metal" "-framework Foundation")
    target_link_options(metal_utils PRIVATE -undefined dynamic_lookup)
    set_target_properties(metal_utils PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${METAL_UTILS_PY}")
    add_dependencies(TritonAppleGPUBackend metal_utils)
endif()
