set(PLUGIN "LoopSplit")
set(PLUGIN_STATUS "experimental")

message(STATUS "TRITON PLUGINS: building ${PLUGIN}")
# Add Pass tablegen
set(PLUGIN_Include "TritonPlugin${PLUGIN}IncGen")
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Plugins)
add_public_tablegen_target(${PLUGIN_Include})

add_mlir_library(${PLUGIN}
    LoopSplit.cpp

    SHARED

    ADDITIONAL_HEADER_DIRS
    ${TRITON_PLUGINS_BINARY_DIR}/lib

    DEPENDS
    TritonTableGen
    TritonCanonicalizeIncGen
    TPAPIPassInfra
    ${PLUGIN_Include}

    LINK_LIBS PUBLIC
    TPAPIPassInfra
    MLIRPass
    LLVMSupport
    MLIRSupport
    "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>"
)   

# CMAKE_LIBRARY_OUTPUT_DIRECTORY is only set during the Python
# build. It is empty if building directly from the root
# CMakeLists.txt file. Therefore if not building from Python just
# use the default CMake shared lib path otherwise this causes a hard
# build error
if(DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set_target_properties(${PLUGIN} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY
        "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/../plugins")
endif(DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)
