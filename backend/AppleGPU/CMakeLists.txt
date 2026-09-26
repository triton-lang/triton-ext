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

set(APPLE_GPU_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(APPLE_GPU_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}")

include_directories(${TRITON_EXT_SUPPORT_DIR})
include_directories(${APPLE_GPU_INCLUDE_DIR})

set(AGPU_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/agpu/include")
include_directories(${AGPU_INCLUDE_DIR})
include_directories(${APPLE_GPU_GEN_DIR})

list(APPEND LLVM_TABLEGEN_FLAGS
    "-I${APPLE_GPU_INCLUDE_DIR}/Dialect/TritonAppleGPU/IR")

file(MAKE_DIRECTORY "${APPLE_GPU_GEN_DIR}/Dialect/TritonAppleGPU/IR")
file(MAKE_DIRECTORY "${APPLE_GPU_GEN_DIR}/TritonAppleGPUTransforms")

set(LLVM_TARGET_DEFINITIONS
    ${APPLE_GPU_INCLUDE_DIR}/Dialect/TritonAppleGPU/IR/TritonAppleGPUDialect.td)
mlir_tablegen(Dialect/TritonAppleGPU/IR/Dialect.h.inc -gen-dialect-decls -dialect=applegpu)
mlir_tablegen(Dialect/TritonAppleGPU/IR/Dialect.cpp.inc -gen-dialect-defs -dialect=applegpu)
add_public_tablegen_target(TritonAppleGPUDialectIncGen)

set(LLVM_TARGET_DEFINITIONS
    ${APPLE_GPU_INCLUDE_DIR}/Dialect/TritonAppleGPU/IR/TritonAppleGPUAttrDefs.td)
mlir_tablegen(Dialect/TritonAppleGPU/IR/TritonAppleGPUAttrDefs.h.inc -gen-attrdef-decls -attrdefs-dialect=applegpu)
mlir_tablegen(Dialect/TritonAppleGPU/IR/TritonAppleGPUAttrDefs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=applegpu)
add_public_tablegen_target(TritonAppleGPUAttrDefsIncGen)

set(LLVM_TARGET_DEFINITIONS
    ${APPLE_GPU_INCLUDE_DIR}/TritonAppleGPUTransforms/Passes.td)
mlir_tablegen(TritonAppleGPUTransforms/Passes.h.inc -gen-pass-decls -name TritonAppleGPU)
add_public_tablegen_target(TritonAppleGPUTransformsPassesIncGen)

set(TO_MSL lib/TritonAppleGPUToMSL)

add_mlir_library(TritonAppleGPUBackend
    ExportAppleGPU.cpp
    "${TRITON_EXT_SUPPORT_DIR}/Export.cpp"

    lib/Dialect/TritonAppleGPU/IR/Dialect.cpp
    lib/Dialect/TritonAppleGPU/IR/LinearLayoutConversions.cpp

    lib/TritonAppleGPUTransforms/AccelerateAppleMatmul.cpp
    lib/TritonAppleGPUTransforms/AtomicLaneLayout.cpp
    lib/TritonAppleGPUTransforms/EmitMSL.cpp
    lib/TritonAppleGPUTransforms/MaskSelectArmLoads.cpp
    lib/TritonAppleGPUTransforms/PrefetchLoads.cpp
    lib/TritonAppleGPUTransforms/ReduceThroughLayoutChange.cpp
    lib/TritonAppleGPUTransforms/StoreShuffleLayout.cpp

    ${TO_MSL}/AgpuAssume.cpp
    ${TO_MSL}/AgpuCarried.cpp
    ${TO_MSL}/AgpuControl.cpp
    ${TO_MSL}/AgpuDebug.cpp
    ${TO_MSL}/AgpuDeviceFns.cpp
    ${TO_MSL}/AgpuDeviceTile.cpp
    ${TO_MSL}/AgpuDot.cpp
    ${TO_MSL}/AgpuDotChain.cpp
    ${TO_MSL}/AgpuDotDrain.cpp
    ${TO_MSL}/AgpuDotFacts.cpp
    ${TO_MSL}/AgpuDotStage.cpp
    ${TO_MSL}/AgpuEmitter.cpp
    ${TO_MSL}/AgpuHandlersArith.cpp
    ${TO_MSL}/AgpuHandlersAtomic.cpp
    ${TO_MSL}/AgpuHandlersMemDesc.cpp
    ${TO_MSL}/AgpuHandlersMemory.cpp
    ${TO_MSL}/AgpuHandlersRebind.cpp
    ${TO_MSL}/AgpuHandlersTile.cpp
    ${TO_MSL}/AgpuHandlersValue.cpp
    ${TO_MSL}/AgpuLayout.cpp
    ${TO_MSL}/AgpuPool.cpp
    ${TO_MSL}/AgpuPredicate.cpp
    ${TO_MSL}/AgpuRegions.cpp
    ${TO_MSL}/AgpuValues.cpp

    agpu/src/Printer.cpp

    SHARED
    EXCLUDE_FROM_LIBMLIR

    DEPENDS
    TritonAppleGPUDialectIncGen
    TritonAppleGPUAttrDefsIncGen
    TritonAppleGPUTransformsPassesIncGen

    LINK_LIBS PUBLIC
    ${TRITON_LIB}
)

target_link_libraries(TritonAppleGPUBackend PUBLIC MLIRMathTransforms)

target_compile_options(TritonAppleGPUBackend PRIVATE
    -Wall -Wextra -Wno-missing-field-initializers)

set_source_files_properties(lib/Dialect/TritonAppleGPU/IR/Dialect.cpp
    PROPERTIES COMPILE_OPTIONS "-Wno-unused-parameter")

# The plugin resolves its MLIR and LLVM symbols from the already-loaded
# libtriton, so it needs no search path of its own. Linking against LLVM's
# targets would otherwise bake this machine's LLVM directory into the binary,
# which no other machine has.
set_target_properties(TritonAppleGPUBackend PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    OUTPUT_NAME "applegpu_backend"
    CXX_STANDARD 17
    SKIP_BUILD_RPATH TRUE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "")

install(TARGETS TritonAppleGPUBackend
        LIBRARY DESTINATION triton_apple_backend)

# These land at the site-packages root rather than inside the package: site.py
# runs the .pth at interpreter startup, the only point early enough for a
# Triton that reads TRITON_PLUGIN_PATHS while importing.
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/python/applegpu_backend.pth"
    "${CMAKE_CURRENT_SOURCE_DIR}/python/_applegpu_autoregister.py"
    DESTINATION .)

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

    python_add_library(metal_native MODULE "${BRIDGE_PY}/metal_native.m"
                       WITH_SOABI)
    target_compile_options(metal_native PRIVATE
        -ObjC++ -fobjc-arc -std=c++20 -O3 -Wno-deprecated-declarations)
    target_link_libraries(metal_native PRIVATE
        "-framework Metal" "-framework Foundation")
    target_link_options(metal_native PRIVATE -undefined dynamic_lookup)
    set_target_properties(metal_native PROPERTIES
        INSTALL_RPATH "@loader_path/../torch/lib"
        BUILD_WITH_INSTALL_RPATH TRUE)
    install(TARGETS metal_native LIBRARY DESTINATION triton_apple_backend)
    add_dependencies(TritonAppleGPUBackend metal_native)

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
