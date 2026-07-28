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

add_subdirectory(python-stubs)

# triton-ext-opt: an opt driver for extension passes (see triton-ext-opt.cpp).
add_llvm_executable(triton-ext-opt
    triton-ext-opt.cpp
    PARTIAL_SOURCES_INTENDED
)
llvm_update_compile_flags(triton-ext-opt)
target_link_libraries(triton-ext-opt PRIVATE
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRSupport
    MLIRTransforms
    MLIRArithDialect
    MLIRControlFlowDialect
    MLIRFuncDialect
    MLIRGPUDialect
    MLIRMathDialect
    MLIRSCFDialect
    ${TRITON_LIB}
)

# Resolve libtriton's imported CPython symbols (see python-stubs). macOS strips
# DYLD_INSERT_LIBRARIES under SIP, so link the stubs in; Linux preloads them.
if(APPLE)
    target_link_libraries(triton-ext-opt PRIVATE pystubs_obj)
else()
    target_link_options(triton-ext-opt PRIVATE -Wl,--allow-shlib-undefined)
endif()

set_target_properties(triton-ext-opt PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${TRITON_EXT_BINARY_DIR}/bin")
