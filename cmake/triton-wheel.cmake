# ---------------------------------------------------------------------------
# find_triton_wheel(<out_var>)
#
# Discover an installed Triton wheel and store its path (<site-packages>/triton)
# in <out_var>. Resolution order:
#   1. the TRITON_WHEEL_DIR environment variable;
#   2. ci/probe-triton-wheel.py run against the active Python interpreter.
#
# The wheel must be built with TRITON_EXT_ENABLED=1 so it ships the C++ headers
# under `<triton>/include` and the shared library under `<triton>/_C`.
# ---------------------------------------------------------------------------

get_filename_component(_TRITON_WHEEL_REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/.."
                       ABSOLUTE)
set(_TRITON_WHEEL_PROBE "${_TRITON_WHEEL_REPO_ROOT}/ci/probe_triton_wheel.py")

function(find_triton_wheel out_var)
    if(DEFINED ENV{TRITON_WHEEL_DIR})
        set(${out_var} "$ENV{TRITON_WHEEL_DIR}" PARENT_SCOPE)
        return()
    endif()
    find_package(Python COMPONENTS Interpreter REQUIRED)
    execute_process(
        COMMAND ${Python_EXECUTABLE} "${_TRITON_WHEEL_PROBE}"
        OUTPUT_VARIABLE _dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _rc)
    if(NOT _rc EQUAL 0)
        message(FATAL_ERROR
                "Could not import Triton. Install a Triton wheel built with "
                "TRITON_EXT_ENABLED=1, or pass -DTRITON_WHEEL_DIR=<site-packages>/triton")
    endif()
    set(${out_var} "${_dir}" PARENT_SCOPE)
endfunction()
