set(TRITON_PLUGINS_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(TRITON_PLUGINS_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/triton-plugins")

# Reverse order for dependencies
#set(ALL_SUBDIRECTORIES tools backends passes dialects lang-ext)
set(ALL_SUBDIRECTORIES backends passes dialects lang-ext)

# Add all subdirectories by default
foreach(subdir ${ALL_SUBDIRECTORIES})
    add_subdirectory("${TRITON_PLUGINS_SOURCE_DIR}/${subdir}"
                     "${TRITON_PLUGINS_BINARY_DIR}/${subdir}")
endforeach()
