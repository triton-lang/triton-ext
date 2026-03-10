# Include the support code for all extensions (see `#include "Export*.cpp"`).
include_directories(${TRITON_EXT_SUPPORT_DIR})

# Build all pass extensions.
file(GLOB EXT_PASS_DIRS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/*")
foreach(EXT_PASS_DIR ${EXT_PASS_DIRS})
    if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${EXT_PASS_DIR}")
        triton_ext_should_build_extension("${CMAKE_CURRENT_SOURCE_DIR}/${EXT_PASS_DIR}" _should_build)
        if(${_should_build})
            message(STATUS "Found triton-ext pass: building ${EXT_PASS_DIR}")
            set(EXT_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/${EXT_PASS_DIR}")
            add_subdirectory(${EXT_PASS_DIR} ${EXT_BUILD_DIR})
        else()
            message(STATUS "Found triton-ext pass: skipping ${EXT_PASS_DIR} (not enabled)")
        endif()
    endif()
endforeach()
