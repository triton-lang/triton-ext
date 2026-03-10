# Include the support code for all extensions (see `#include "Export*.cpp"`).
include_directories(${TRITON_EXT_SUPPORT_DIR})

# Build all dialect extensions.
file(GLOB EXT_DIALECT_DIRS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/*")
foreach(EXT_DIALECT_DIR ${EXT_DIALECT_DIRS})
    if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${EXT_DIALECT_DIR}")
        triton_ext_should_build_extension("${CMAKE_CURRENT_SOURCE_DIR}/${EXT_DIALECT_DIR}" _should_build)
        if(${_should_build})
            message(STATUS "Found triton-ext dialect: building ${EXT_DIALECT_DIR}")
            set(EXT_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/${EXT_DIALECT_DIR}")
            add_subdirectory(${EXT_DIALECT_DIR} ${EXT_BUILD_DIR})
        else()
            message(STATUS "Found triton-ext dialect: skipping ${EXT_DIALECT_DIR} (not enabled)")
        endif()
    endif()
endforeach()
