project(TritonExtensionSupport)

add_mlir_library(${PROJECT_NAME}
    Export.cpp
    PARTIAL_SOURCES_INTENDED

    SHARED

    LINK_LIBS PUBLIC
    MLIRPass
    MLIRDialect
    LLVMSupport
    MLIRSupport
    "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>"
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY
    "${TRITON_EXT_LIBRARY_DIR}")
