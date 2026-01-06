# Include the pass infra directory for all pass extensions
set(TRITON_EXT_PASS_INFRA_DIR "${TRITON_EXT_INFRA_DIR}/pass")
include_directories(${TRITON_EXT_PASS_INFRA_DIR})

# Build all pass extensions
file(GLOB EXT_PASS_DIRS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/*")
foreach(EXT_PASS_DIR ${EXT_PASS_DIRS})
    if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${EXT_PASS_DIR}")
        triton_ext_should_build_extension("${CMAKE_CURRENT_SOURCE_DIR}/${EXT_PASS_DIR}" _should_build)
        if(${_should_build})
            message(STATUS "TRITON_EXT Pass: building ${EXT_PASS_DIR}")
            set(EXT_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/${EXT_PASS_DIR}")
            add_subdirectory(${EXT_PASS_DIR} ${EXT_BUILD_DIR})
        else()
            message(STATUS "TRITON_EXT Pass: skipping ${EXT_PASS_DIR} (not enabled)")
        endif()
    endif()
endforeach()
