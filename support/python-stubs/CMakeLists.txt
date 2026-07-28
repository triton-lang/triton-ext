# Empty implementations of the CPython C-API symbols libtriton.so imports, so a
# libtriton-linked tool loads without a real interpreter. Generated from
# libtriton by gen-python-stubs.sh. Built as a shared lib (preloaded on Linux)
# and an object lib (linked into triton-ext-opt on macOS).
add_library(pystubs SHARED PythonStubs.c)
set_target_properties(pystubs PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${TRITON_EXT_LIBRARY_DIR}")

add_library(pystubs_obj OBJECT PythonStubs.c)
set_target_properties(pystubs_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)
