project(loop_split)
set(TRITON_EXT_CLASS LoopBisectPass)

# Only build this plugin if it is enabled in TPAPI_TARGET_NAMES
list(FIND TRITON_EXT_NAMES ${PROJECT_NAME} _index)
list(LENGTH TRITON_EXT_NAMES _size)
if(NOT _index EQUAL -1 OR _size EQUAL 0)
    message(STATUS "TRITON EXT: building ${PROJECT_NAME}")

    # Add Pass tablegen
    set(EXT_Include "TritonExt${TRITON_EXT_CLASS}IncGen")
    set(LLVM_TARGET_DEFINITIONS Passes.td)
    mlir_tablegen(Passes.h.inc -gen-pass-decls -name Extension)
    add_public_tablegen_target(${EXT_Include})

    add_compile_definitions(
        TRITON_EXT_NAME=${PROJECT_NAME} 
        TRITON_EXT_CLASS=${TRITON_EXT_CLASS}
    )

    # Build the pass extension library
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

else()
    message(STATUS "TRITON EXT: skipping ${PROJECT_NAME}")
endif()
