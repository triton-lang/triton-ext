# ---------------------------------------------------------------------------
# find_llvm_install_dir(<out_var>)
#
# Discover an LLVM/MLIR install and store its root path in <out_var>.
# Resolution order:
#   1. the LLVM_INSTALL_DIR environment variable;
#   2. ci/pick_local_artifact.py run against the repo root, searching for the
#      youngest directory matching the glob pattern `llvm-*`.
#
# The install must provide C++ headers, mlir-tblgen, and CMake modules under
# the standard layout (<install>/include, <install>/bin, <install>/lib/cmake).
# ---------------------------------------------------------------------------

get_filename_component(_LLVM_REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/.."
                       ABSOLUTE)
set(_LLVM_PICK_ARTIFACT "${_LLVM_REPO_ROOT}/ci/pick_local_artifact.py")

function(find_llvm_install_dir out_var)
    if(DEFINED ENV{LLVM_INSTALL_DIR})
        set(${out_var} "$ENV{LLVM_INSTALL_DIR}" PARENT_SCOPE)
        return()
    endif()
    find_package(Python COMPONENTS Interpreter REQUIRED)
    execute_process(
        COMMAND ${Python_EXECUTABLE} "${_LLVM_PICK_ARTIFACT}" "llvm-*"
        WORKING_DIRECTORY "${_LLVM_REPO_ROOT}"
        OUTPUT_VARIABLE _dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _rc)
    if(NOT _rc EQUAL 0)
        message(FATAL_ERROR
                "Could not find a local LLVM artifact. Download one with "
                "`python ci/download-artifact.py llvm`, or pass "
                "-DLLVM_INSTALL_DIR=<path> to cmake.")
    endif()
    # pick_local_artifact returns a bare directory name relative to the repo
    # root; resolve it to an absolute path.
    set(${out_var} "${_LLVM_REPO_ROOT}/${_dir}" PARENT_SCOPE)
endfunction()
