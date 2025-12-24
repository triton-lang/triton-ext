include_directories(${CMAKE_CURRENT_SOURCE_DIR}/api)

# Build TPPass API infra
add_subdirectory(api)


add_subdirectory(LoopSplit)

# Propagate accumulated TRITON_PLUGIN_NAMES from passes/ back to the parent
# (the project that included triton-plugins).
set(TRITON_PLUGIN_NAMES "${TRITON_PLUGIN_NAMES}" PARENT_SCOPE)
