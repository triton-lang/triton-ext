triton_ext_pass_setup(LoopBisectPass)

# Add pass TableGen.
set(EXT_INCLUDE "TritonExt${TRITON_EXT_CLASS}IncGen")
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Extension)
add_public_tablegen_target(${EXT_INCLUDE})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# Add compile definitions for TritonExtPass generated code
add_compile_definitions(
    TRITON_EXT_NAME=${PROJECT_NAME}
    TRITON_EXT_CLASS=${TRITON_EXT_CLASS}
)

# Build the pass extension library (SHARED)
add_mlir_library(${PROJECT_NAME}
    LoopSplit.cpp

    SHARED

    DEPENDS
    ${EXT_INCLUDE}

    LINK_LIBS PUBLIC
    TritonExtPassInfra
    MLIRPass
    MLIRArithDialect
    MLIRSCFDialect
    LLVMSupport
    MLIRSupport
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY
    "${TRITON_EXT_LIBRARY_DIR}")
