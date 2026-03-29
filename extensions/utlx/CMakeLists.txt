triton_extension(uTLXPlugin)

# ---------------------------------------------------------------------------
# uTLX — standalone Triton plugin for the TLX dialect.
#
# This extension builds libuTLXPlugin.so, loadable via TRITON_PLUGIN_PATHS.
# It provides:
#   - The TLX dialect (types, attrs, ops)
#   - TLX transform passes
#   - TLX custom ops for the TritonOpBuilder
#   - TLX conversion patterns (ConvertTritonToTritonGPU plugin pass)
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

# ---------------------------------------------------------------------------
# TableGen for the TLX dialect
# ---------------------------------------------------------------------------
set(UTLX_DIALECT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tlx/dialect")
set(UTLX_DIALECT_INCLUDE_DIR "${UTLX_DIALECT_DIR}/include")
set(UTLX_TABLEGEN_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/tlx/dialect/include")

# Ensure the output directories exist
file(MAKE_DIRECTORY "${UTLX_TABLEGEN_OUTPUT_DIR}/IR")
file(MAKE_DIRECTORY "${UTLX_TABLEGEN_OUTPUT_DIR}/Transforms")

# Find mlir-tblgen
if(NOT MLIR_TABLEGEN_EXE)
    if(LLVM_BUILD_DIR)
        find_program(MLIR_TABLEGEN_EXE mlir-tblgen
            PATHS "${LLVM_BUILD_DIR}/bin"
            NO_DEFAULT_PATH)
    endif()
    if(NOT MLIR_TABLEGEN_EXE AND LLVM_INSTALL_DIR)
        find_program(MLIR_TABLEGEN_EXE mlir-tblgen
            PATHS "${LLVM_INSTALL_DIR}/bin"
            NO_DEFAULT_PATH)
    endif()
    if(NOT MLIR_TABLEGEN_EXE)
        find_program(MLIR_TABLEGEN_EXE mlir-tblgen)
    endif()
endif()

if(NOT MLIR_TABLEGEN_EXE)
    message(FATAL_ERROR "Could not find mlir-tblgen. Set LLVM_BUILD_DIR or LLVM_INSTALL_DIR.")
endif()

# Collect tablegen include paths
set(UTLX_TBLGEN_INCLUDES
    "-I${UTLX_DIALECT_INCLUDE_DIR}/IR"
    "-I${UTLX_DIALECT_INCLUDE_DIR}"
)
# MLIR/LLVM tablegen includes
if(LLVM_BUILD_DIR)
    list(APPEND UTLX_TBLGEN_INCLUDES
        "-I${LLVM_BUILD_DIR}/include"
    )
endif()
if(LLVM_INSTALL_DIR)
    list(APPEND UTLX_TBLGEN_INCLUDES
        "-I${LLVM_INSTALL_DIR}/include"
    )
endif()
foreach(dir ${LLVM_INCLUDE_DIRS})
    list(APPEND UTLX_TBLGEN_INCLUDES "-I${dir}")
endforeach()
foreach(dir ${MLIR_INCLUDE_DIRS})
    list(APPEND UTLX_TBLGEN_INCLUDES "-I${dir}")
endforeach()
# Triton tablegen includes
if(TRITON_SOURCE_DIR)
    list(APPEND UTLX_TBLGEN_INCLUDES
        "-I${TRITON_SOURCE_DIR}/include"
        "-I${TRITON_SOURCE_DIR}"
    )
    if(TRITON_BUILD_DIR)
        list(APPEND UTLX_TBLGEN_INCLUDES
            "-I${TRITON_BUILD_DIR}/include"
        )
    endif()
elseif(TRITON_INSTALL_DIR)
    list(APPEND UTLX_TBLGEN_INCLUDES "-I${TRITON_INSTALL_DIR}/include")
endif()

# Helper function for running mlir-tblgen
function(utlx_tablegen input output generator)
    set(extra_args ${ARGN})
    get_filename_component(input_abs "${input}" ABSOLUTE)
    add_custom_command(
        OUTPUT "${output}"
        COMMAND "${MLIR_TABLEGEN_EXE}" ${generator} ${extra_args}
                ${UTLX_TBLGEN_INCLUDES} "${input_abs}"
                -o "${output}"
        DEPENDS "${input_abs}"
        COMMENT "Running mlir-tblgen ${generator} on ${input}"
        VERBATIM
    )
endfunction()

# --- IR TableGen ---
set(IR_TD_DIR "${UTLX_DIALECT_INCLUDE_DIR}/IR")
set(IR_OUT_DIR "${UTLX_TABLEGEN_OUTPUT_DIR}/IR")

# Dialect
utlx_tablegen("${IR_TD_DIR}/TLXDialect.td" "${IR_OUT_DIR}/Dialect.h.inc"
    "-gen-dialect-decls" "-dialect=tlx")
utlx_tablegen("${IR_TD_DIR}/TLXDialect.td" "${IR_OUT_DIR}/Dialect.cpp.inc"
    "-gen-dialect-defs" "-dialect=tlx")

# Types
utlx_tablegen("${IR_TD_DIR}/TLXTypes.td" "${IR_OUT_DIR}/TLXTypes.h.inc"
    "-gen-typedef-decls")
utlx_tablegen("${IR_TD_DIR}/TLXTypes.td" "${IR_OUT_DIR}/TLXTypes.cpp.inc"
    "-gen-typedef-defs")
utlx_tablegen("${IR_TD_DIR}/TLXTypes.td" "${IR_OUT_DIR}/TLXTypesEnums.h.inc"
    "-gen-enum-decls")
utlx_tablegen("${IR_TD_DIR}/TLXTypes.td" "${IR_OUT_DIR}/TLXTypesEnums.cpp.inc"
    "-gen-enum-defs")

# Ops
utlx_tablegen("${IR_TD_DIR}/TLXOps.td" "${IR_OUT_DIR}/Ops.h.inc"
    "-gen-op-decls")
utlx_tablegen("${IR_TD_DIR}/TLXOps.td" "${IR_OUT_DIR}/Ops.cpp.inc"
    "-gen-op-defs")

# AttrDefs
utlx_tablegen("${IR_TD_DIR}/TLXAttrDefs.td" "${IR_OUT_DIR}/TLXAttrDefs.h.inc"
    "-gen-attrdef-decls")
utlx_tablegen("${IR_TD_DIR}/TLXAttrDefs.td" "${IR_OUT_DIR}/TLXAttrDefs.cpp.inc"
    "-gen-attrdef-defs")

# --- Transforms TableGen ---
set(TRANSFORMS_TD_DIR "${UTLX_DIALECT_INCLUDE_DIR}/Transforms")
set(TRANSFORMS_OUT_DIR "${UTLX_TABLEGEN_OUTPUT_DIR}/Transforms")

utlx_tablegen("${TRANSFORMS_TD_DIR}/Passes.td" "${TRANSFORMS_OUT_DIR}/Passes.h.inc"
    "-gen-pass-decls")

# Collect all tablegen outputs into a custom target
set(UTLX_TABLEGEN_OUTPUTS
    "${IR_OUT_DIR}/Dialect.h.inc"
    "${IR_OUT_DIR}/Dialect.cpp.inc"
    "${IR_OUT_DIR}/TLXTypes.h.inc"
    "${IR_OUT_DIR}/TLXTypes.cpp.inc"
    "${IR_OUT_DIR}/TLXTypesEnums.h.inc"
    "${IR_OUT_DIR}/TLXTypesEnums.cpp.inc"
    "${IR_OUT_DIR}/Ops.h.inc"
    "${IR_OUT_DIR}/Ops.cpp.inc"
    "${IR_OUT_DIR}/TLXAttrDefs.h.inc"
    "${IR_OUT_DIR}/TLXAttrDefs.cpp.inc"
    "${TRANSFORMS_OUT_DIR}/Passes.h.inc"
)
add_custom_target(uTLXTableGen DEPENDS ${UTLX_TABLEGEN_OUTPUTS})

# ---------------------------------------------------------------------------
# Plugin sources
# ---------------------------------------------------------------------------
set(UTLX_PLUGIN_SOURCES
    # Plugin entry point
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
    # New/modified op wrappers (runtime op creation)
    ops/NewOps.cpp
    # Ported passes - NVIDIA
    passes/nvidia/PruneUnusedBarriers.cpp
    passes/nvidia/PingPong.cpp
    # Ported passes - AMD
    # NOTE: AMD barrier passes require patched triton (ReadBarrierPhaseOp,
    # ArriveBarrierOp with expectedCount). Uncomment when
    # triton-tlx-core-changes patch is applied:
    # passes/amd/LowerBarrierOps.cpp
    # passes/amd/BarrierOpConversion.cpp
)

add_library(${PROJECT_NAME} SHARED ${UTLX_PLUGIN_SOURCES})
add_dependencies(${PROJECT_NAME} uTLXTableGen)

# ---------------------------------------------------------------------------
# Include directories
# ---------------------------------------------------------------------------
set(UTLX_INCLUDE_DIRS
    # For #include "IR/Dialect.h" (relative from lib/IR/*.cpp)
    # Source headers: found via tlx/dialect/include/
    "${UTLX_DIALECT_INCLUDE_DIR}"
    # Generated headers: found via tablegen output
    "${UTLX_TABLEGEN_OUTPUT_DIR}"
    # For #include "tlx/dialect/include/IR/..." (source headers)
    "${CMAKE_CURRENT_SOURCE_DIR}"
    # For #include "tlx/dialect/include/IR/..." (generated headers)
    "${CMAKE_CURRENT_BINARY_DIR}"
)

if(TRITON_SOURCE_DIR)
    list(APPEND UTLX_INCLUDE_DIRS
        "${TRITON_SOURCE_DIR}/include"
        "${TRITON_SOURCE_DIR}"
        "${TRITON_SOURCE_DIR}/third_party/amd/include"
        "${TRITON_SOURCE_DIR}/third_party"
        "${TRITON_SOURCE_DIR}/third_party/nvidia/include"
    )
    if(TRITON_BUILD_DIR)
        list(APPEND UTLX_INCLUDE_DIRS
            "${TRITON_BUILD_DIR}/include"
            "${TRITON_BUILD_DIR}/third_party"
        )
    endif()

    # Find libtriton.so from source tree
    if(NOT UTLX_TRITON_LIB)
        find_library(UTLX_TRITON_LIB
            NAMES triton
            PATHS "${TRITON_SOURCE_DIR}/python/triton/_C"
            NO_DEFAULT_PATH
        )
    endif()
elseif(TRITON_INSTALL_DIR)
    list(APPEND UTLX_INCLUDE_DIRS "${TRITON_INSTALL_DIR}/include")
endif()

# LLVM/MLIR include dirs
if(LLVM_BUILD_DIR)
    find_package(MLIR QUIET CONFIG PATHS "${LLVM_BUILD_DIR}/lib/cmake/mlir")
    find_package(LLVM QUIET CONFIG PATHS "${LLVM_BUILD_DIR}/lib/cmake/llvm")
endif()
list(APPEND UTLX_INCLUDE_DIRS ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})

target_include_directories(${PROJECT_NAME} PRIVATE ${UTLX_INCLUDE_DIRS})

# ---------------------------------------------------------------------------
# Optional AMDGPU support (requires third_party headers from triton source)
# ---------------------------------------------------------------------------
set(_AMDGPU_HEADER_FOUND FALSE)
foreach(_dir ${UTLX_INCLUDE_DIRS})
    if(EXISTS "${_dir}/Dialect/TritonAMDGPU/IR/Dialect.h")
        set(_AMDGPU_HEADER_FOUND TRUE)
        break()
    endif()
endforeach()
if(_AMDGPU_HEADER_FOUND)
    message(STATUS "AMDGPU dialect header found — enabling UTLX_HAS_AMDGPU")
    target_compile_definitions(${PROJECT_NAME} PRIVATE UTLX_HAS_AMDGPU)
else()
    message(STATUS "AMDGPU dialect header not found — AMD support disabled")
endif()

# ---------------------------------------------------------------------------
# Link libraries
# ---------------------------------------------------------------------------
if(UTLX_TRITON_LIB)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${UTLX_TRITON_LIB})
elseif(TRITON_LIB)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${TRITON_LIB})
else()
    message(FATAL_ERROR "Could not find libtriton.so. Set TRITON_SOURCE_DIR or TRITON_INSTALL_DIR.")
endif()

target_link_libraries(${PROJECT_NAME} PRIVATE
    MLIRPass
    MLIRAnalysis
    MLIRTransforms
    MLIRIR
    MLIRSupport
    LLVMSupport
)

# ---------------------------------------------------------------------------
# Build properties
# ---------------------------------------------------------------------------
set_target_properties(${PROJECT_NAME} PROPERTIES
    CXX_STANDARD 17
    LIBRARY_OUTPUT_DIRECTORY "${TRITON_EXT_LIBRARY_DIR}"
)

target_compile_options(${PROJECT_NAME} PRIVATE
    -fvisibility=hidden
    -fno-exceptions
    -fno-rtti
    -Wno-deprecated-declarations
)
