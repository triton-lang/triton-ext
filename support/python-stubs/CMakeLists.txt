add_library(pystubs SHARED PythonStubs.c)

set_target_properties(pystubs PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${TRITON_EXT_LIBRARY_DIR}"
)
