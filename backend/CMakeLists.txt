# Include the support code for all extensions (see `#include "Export*.cpp"`).
include_directories(${TRITON_EXT_SUPPORT_DIR})

# Build all backend extensions.
file(GLOB EXT_BACKEND_DIRS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/*")
foreach(EXT_BACKEND_DIR ${EXT_BACKEND_DIRS})
    if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${EXT_BACKEND_DIR}")
        triton_ext_should_build_extension("${CMAKE_CURRENT_SOURCE_DIR}/${EXT_BACKEND_DIR}" _should_build)
        if(${_should_build})
            message(STATUS "Found triton-ext backend: building ${EXT_BACKEND_DIR}")
            set(EXT_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/${EXT_BACKEND_DIR}")
            add_subdirectory(${EXT_BACKEND_DIR} ${EXT_BUILD_DIR})
        else()
            message(STATUS "Found triton-ext backend: skipping ${EXT_BACKEND_DIR} (not enabled)")
        endif()
    endif()
endforeach()
