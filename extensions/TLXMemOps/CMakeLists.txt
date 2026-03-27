triton_extension(TLXMemOpsPlugin)

# ---------------------------------------------------------------------------
# TLXMemOps — standalone Triton plugin for TLX memory operations.
#
# This extension builds libTLXMemOpsPlugin.so, loadable via
# TRITON_PLUGIN_PATHS. It provides its own tritonGetPluginInfo() with both
# custom ops and passes, so it does NOT link against TritonExtensionSupport.
#
# Supports two build modes:
#   1. Within triton-ext (TRITON_INSTALL_DIR / LLVM_INSTALL_DIR set by parent)
#   2. With source/build dirs (TRITON_SOURCE_DIR / TRITON_BUILD_DIR / LLVM_BUILD_DIR)
# ---------------------------------------------------------------------------

# Accept source/build paths from environment if not set via -D
if(NOT TRITON_SOURCE_DIR AND DEFINED ENV{TRITON_SOURCE_DIR})
    set(TRITON_SOURCE_DIR $ENV{TRITON_SOURCE_DIR})
endif()
if(NOT TRITON_BUILD_DIR AND DEFINED ENV{TRITON_BUILD_DIR})
    set(TRITON_BUILD_DIR $ENV{TRITON_BUILD_DIR})
endif()
if(NOT LLVM_BUILD_DIR AND DEFINED ENV{LLVM_BUILD_DIR})
    set(LLVM_BUILD_DIR $ENV{LLVM_BUILD_DIR})
endif()

set(TLX_PLUGIN_SOURCES
    TLXLocalAllocPlugin.cpp
    TLXConversionPatterns.cpp
)

add_library(${PROJECT_NAME} SHARED ${TLX_PLUGIN_SOURCES})

# Determine include directories based on available paths
set(TLX_INCLUDE_DIRS "")

if(TRITON_SOURCE_DIR)
    # Source tree mode: use source + build include paths
    list(APPEND TLX_INCLUDE_DIRS
        "${TRITON_SOURCE_DIR}/include"
        "${TRITON_SOURCE_DIR}"
        "${TRITON_SOURCE_DIR}/third_party/amd/include"
        "${TRITON_SOURCE_DIR}/third_party"
    )
    if(TRITON_BUILD_DIR)
        list(APPEND TLX_INCLUDE_DIRS
            "${TRITON_BUILD_DIR}/include"
            "${TRITON_BUILD_DIR}/third_party"
        )
    endif()

    # Find libtriton.so from source tree
    if(NOT TLX_TRITON_LIB)
        find_library(TLX_TRITON_LIB
            NAMES triton
            PATHS "${TRITON_SOURCE_DIR}/python/triton/_C"
            NO_DEFAULT_PATH
        )
    endif()
elseif(TRITON_INSTALL_DIR)
    # Install dir mode: headers already on include path from parent
    list(APPEND TLX_INCLUDE_DIRS "${TRITON_INSTALL_DIR}/include")
endif()

# LLVM/MLIR include dirs
if(LLVM_BUILD_DIR)
    find_package(MLIR QUIET CONFIG PATHS "${LLVM_BUILD_DIR}/lib/cmake/mlir")
    find_package(LLVM QUIET CONFIG PATHS "${LLVM_BUILD_DIR}/lib/cmake/llvm")
endif()
list(APPEND TLX_INCLUDE_DIRS ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})

target_include_directories(${PROJECT_NAME} PRIVATE ${TLX_INCLUDE_DIRS})

# Link against the correct triton library
if(TLX_TRITON_LIB)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${TLX_TRITON_LIB} MLIRPass)
elseif(TRITON_LIB)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${TRITON_LIB} MLIRPass)
else()
    message(FATAL_ERROR "Could not find libtriton.so. Set TRITON_SOURCE_DIR or TRITON_INSTALL_DIR.")
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES
    CXX_STANDARD 17
    LIBRARY_OUTPUT_DIRECTORY "${TRITON_EXT_LIBRARY_DIR}"
)

target_compile_options(${PROJECT_NAME} PRIVATE
    -fvisibility=hidden
    -fno-exceptions
    -fno-rtti
)
