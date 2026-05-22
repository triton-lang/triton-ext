triton_extension(ArithmeticIntensityPass)

# Add pass TableGen.
set(PASS_TABLEGEN_OUTPUT "TritonExt${TRITON_EXT_CLASS}IncGen")
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Extension)
add_public_tablegen_target(${PASS_TABLEGEN_OUTPUT})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# Add compile definitions for generated code.
add_compile_definitions(
    TRITON_EXT_NAME=${PROJECT_NAME}
    TRITON_EXT_CLASS=${TRITON_EXT_CLASS}
)

# Build the pass extension library (SHARED).
add_mlir_library(${PROJECT_NAME}
    ArithmeticIntensity.cpp

    SHARED

    DEPENDS
    ${PASS_TABLEGEN_OUTPUT}

    LINK_LIBS PUBLIC
    TritonExtensionSupport
)

target_link_libraries(${PROJECT_NAME}
  PUBLIC
    TritonExtensionSupport
    -Wl,--no-as-needed
    ${TRITON_LIB}
    -Wl,--as-needed
    # ... MLIR libs ...
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY
    "${TRITON_EXT_LIBRARY_DIR}")
