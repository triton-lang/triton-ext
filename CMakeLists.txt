set(TRITON_EXT_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(TRITON_EXT_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(TRITON_EXT_LIBRARY_DIR "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/../extensions")

set(TRITON_EXT_INFRA_DIR "${TRITON_EXT_SOURCE_DIR}/infra")

include(cmake/utils.cmake)

# Build TPPass API infrastructure
add_subdirectory(${TRITON_EXT_INFRA_DIR})

# Build extensions
#  Ordered by typical dependencies
set(ALL_SUBDIRECTORIES dialect pass backend language)

# Add all subdirectories by default
foreach(subdir ${ALL_SUBDIRECTORIES})
    add_subdirectory("${TRITON_EXT_SOURCE_DIR}/${subdir}"
                     "${TRITON_EXT_BINARY_DIR}/${subdir}")
endforeach()
