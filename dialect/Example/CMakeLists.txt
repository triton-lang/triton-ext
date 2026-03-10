triton_extension(ExampleDialect)

# Add dialect TableGen; creates `MLIR${dialect}IncGen` target.
add_mlir_dialect(Example example)
add_mlir_doc(Example Example Dialects/ -gen-dialect-doc -dialect=example)

# Add TableGen files as includes.
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# Add compile definitions for generated code.
add_compile_definitions(
    TRITON_EXT_NAME=${PROJECT_NAME}
    TRITON_EXT_CLASS=${TRITON_EXT_CLASS}
)

# Build the pass extension library (SHARED).
add_mlir_library(${PROJECT_NAME}
    Example.cpp

    SHARED

    DEPENDS
    MLIRExampleIncGen

    LINK_LIBS PUBLIC
    TritonExtensionSupport
    MLIRDialect
    LLVMSupport
    MLIRSupport
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY
    "${TRITON_EXT_LIBRARY_DIR}")
