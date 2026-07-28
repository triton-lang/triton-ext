triton_extension(TypeIDCast)

add_compile_definitions(
    TRITON_EXT_NAME=${PROJECT_NAME}
    TRITON_EXT_CLASS=${TRITON_EXT_CLASS}
)

add_mlir_library(${PROJECT_NAME}
    TypeIDCast.cpp

    SHARED

    LINK_LIBS PUBLIC
    TritonExtensionSupport
    ${TRITON_LIB}
    MLIRPass
    MLIRIR
    LLVMSupport
    MLIRSupport
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${TRITON_EXT_LIBRARY_DIR}"
    OUTPUT_NAME "typeid_cast")

if(APPLE)
    target_link_options(${PROJECT_NAME} PRIVATE -Wl,-flat_namespace)
endif()
