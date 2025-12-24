set(PROJ_NAME TPAPIPassInfra)

if(NOT TRITON_PLUGIN_NAME_ONLY_PASS)

    add_mlir_library(${PROJ_NAME}
        PassPlugin.cpp

        SHARED

        LINK_LIBS PUBLIC
        MLIRPass
        LLVMSupport
        MLIRSupport
        "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>"
    )

    if(DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)
        set_target_properties(${PROJ_NAME} PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY
        "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/../plugins")
    endif(DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)

    #target_compile_options(${PROJ_NAME} PRIVATE -fvisibility=hidden)
endif()