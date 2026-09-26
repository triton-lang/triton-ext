cmake_minimum_required(VERSION 3.20)
# Default to the oldest macOS this backend supports, but never override an
# explicit choice: -DCMAKE_OSX_DEPLOYMENT_TARGET and the environment's
# MACOSX_DEPLOYMENT_TARGET both win. Guard on CACHE{} rather than a plain
# `if(NOT DEFINED)`: project() defines the normal variable from the cache, so
# the plain form stopped applying on reconfigure and the same tree built with
# or without the pin depending on how the build directory came to exist.
# Setting it before project() also preempts CMake's own env-var default, which
# is why that is tested here too.
if(NOT DEFINED CACHE{CMAKE_OSX_DEPLOYMENT_TARGET}
   AND NOT DEFINED ENV{MACOSX_DEPLOYMENT_TARGET})
    set(CMAKE_OSX_DEPLOYMENT_TARGET "14.0" CACHE STRING "Minimum macOS version")
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

project(applegpu_backend LANGUAGES C CXX)
include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/triton-ext.cmake")
configure_triton_extension()

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include"
                    "${CMAKE_CURRENT_SOURCE_DIR}/agpu/include")

set(TO_MSL lib/TritonAppleGPUToMSL)

# ExportAppleGPU.cpp writes tritonGetPluginInfo by hand, so support/Export.cpp
# -- which defines the same symbol off its own registries -- is not linked.
add_mlir_library(TritonAppleGPUBackend
    ExportAppleGPU.cpp

    lib/TritonAppleGPUTransforms/EmitMSL.cpp

    ${TO_MSL}/AgpuEmitter.cpp
    ${TO_MSL}/AgpuHandlersArith.cpp
    ${TO_MSL}/AgpuHandlersMemory.cpp
    ${TO_MSL}/AgpuHandlersRebind.cpp
    ${TO_MSL}/AgpuHandlersValue.cpp

    agpu/src/Printer.cpp

    SHARED
    EXCLUDE_FROM_LIBMLIR

    LINK_LIBS PUBLIC
    ${TRITON_LIB}
)

target_compile_options(TritonAppleGPUBackend PRIVATE
    -Wall -Wextra -Wno-missing-field-initializers)

set_target_properties(TritonAppleGPUBackend PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    OUTPUT_NAME "applegpu_backend"
    CXX_STANDARD 17)

install(TARGETS TritonAppleGPUBackend
        LIBRARY DESTINATION triton_apple_backend)

if(APPLE)
    # The plugin's MLIR symbols are resolved at load from libtriton. This
    # leaves only those undefined, where -flat_namespace would put every
    # symbol in the image up for resolution.
    target_link_options(TritonAppleGPUBackend PRIVATE
        -Wl,-dead_strip -Wl,-undefined,dynamic_lookup)

    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
    list(APPEND CMAKE_PREFIX_PATH "${Python_SITEARCH}/torch/share/cmake")
    # Required, not optional: the driver dispatches through torch's MPS stream
    # and has no other runtime. Failing here beats producing a wheel whose
    # backend silently deactivates at import.
    find_package(Torch REQUIRED)
    set(BRIDGE_PY
        "${CMAKE_CURRENT_SOURCE_DIR}/python/triton_apple_backend")

    python_add_library(metal_torch MODULE "${BRIDGE_PY}/metal_torch.mm"
                       WITH_SOABI)
    target_compile_options(metal_torch PRIVATE
        -std=c++20 -O3 -Wno-deprecated-declarations)
    target_link_libraries(metal_torch PRIVATE "${TORCH_LIBRARIES}"
        "-framework Metal" "-framework Foundation")
    target_link_options(metal_torch PRIVATE -undefined dynamic_lookup)
    # THPVariable_* lives in libtorch_python, which TORCH_LIBRARIES omits.
    get_filename_component(TORCH_LIB_DIR "${TORCH_LIBRARY}" DIRECTORY)
    find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_LIB_DIR}"
                 NO_DEFAULT_PATH REQUIRED)
    target_link_libraries(metal_torch PRIVATE "${TORCH_PYTHON_LIBRARY}")
    # This module installs beside torch/ in site-packages.
    set_target_properties(metal_torch PROPERTIES
        INSTALL_RPATH "@loader_path/../torch/lib"
        BUILD_WITH_INSTALL_RPATH TRUE)
    install(TARGETS metal_torch LIBRARY DESTINATION triton_apple_backend)
    add_dependencies(TritonAppleGPUBackend metal_torch)
endif()
