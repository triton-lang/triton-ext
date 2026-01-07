triton_ext_pass_setup(LoopBisectPass)

# Add Pass tablegen
set(EXT_Include "TritonExt${TRITON_EXT_CLASS}IncGen")
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Extension)
add_public_tablegen_target(${EXT_Include})

# Add compile definitions for TritonExtPass generated code
add_compile_definitions(
    TRITON_EXT_NAME=${PROJECT_NAME} 
    TRITON_EXT_CLASS=${TRITON_EXT_CLASS}
)

# Build the pass extension library (SHARED)
add_mlir_library(${PROJECT_NAME}
    LoopSplit.cpp

    SHARED

    ADDITIONAL_HEADER_DIRS
    ${TRITON_EXT_BINARY_DIR}/lib

    DEPENDS
    TritonTableGen
    TritonCanonicalizeIncGen
    TritonExtPassInfra
    ${EXT_Include}

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
