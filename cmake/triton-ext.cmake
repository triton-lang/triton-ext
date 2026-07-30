# cmake/triton-ext.cmake
# ---------------------------------------------------------------------------
# Shared setup for standalone Triton extension builds (scikit-build-core).
#
# Usage (from each extension's CMakeLists.txt):
#
#   cmake_minimum_required(VERSION 3.20)
#   project(<name> LANGUAGES C CXX)
#   include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/triton-ext.cmake")
#   configure_triton_extension()
#
# Note: cmake_minimum_required() and project() must be literal, direct calls
# at the top of the CMakeLists.txt (CMake policy CMP0000 / CMP0048); they
# cannot be moved into a macro or include.
#
# After the call the following variables are set in the caller's scope:
#   TRITON_EXT_ROOT            repo root
#   TRITON_EXT_SUPPORT_DIR     <repo>/support
#   TRITON_EXT_NAME            extension name
#   TRITON_EXT_VERSION         extension version
#   TRITON_WHEEL_DIR           installed Triton wheel directory
#   TRITON_WHEEL_INCLUDE_DIR   Triton wheel include directory
#   TRITON_LIB                 path to libtriton.so
#   LLVM_INSTALL_DIR           LLVM/MLIR installation directory
#   (and all variables from find_package(MLIR))
#
# After the call the following CMake target is available:
#   triton_ext::plugin         INTERFACE target encoding the invariants every
#                              plugin shared library must satisfy: compile
#                              flags and the link-only-against-libtriton
#                              recipe.  Link it instead of repeating the
#                              recipe:
#                                target_link_libraries(<name> PRIVATE
#                                    triton_ext::plugin)
# ---------------------------------------------------------------------------

# Capture the repo root at include-time: this file lives at <repo>/cmake/, so
# CMAKE_CURRENT_LIST_DIR here is <repo>/cmake/ and its parent is the repo root.
# This is a file-scope assignment; the macro reads it as a closed-over variable.
get_filename_component(_triton_ext_root "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)

# configure_triton_extension()
#
# Sets up C++ standard, finds Python, resolves the extension name/version,
# discovers the Triton wheel and LLVM/MLIR installations, configures shared
# include paths, runs the ABI-breaking check, and defines the
# triton_ext::plugin INTERFACE target.
#
# Reads PROJECT_NAME and PROJECT_VERSION set by the caller's project() call.
macro(configure_triton_extension)
    # -------------------------------------------------------------------------
    # C++ standard — must match Triton and LLVM.
    # -------------------------------------------------------------------------
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

    find_package(Python COMPONENTS Interpreter REQUIRED)

    # -------------------------------------------------------------------------
    # Expose the repo root and support directory to extension authors.
    # -------------------------------------------------------------------------
    set(TRITON_EXT_ROOT "${_triton_ext_root}")
    set(TRITON_EXT_SUPPORT_DIR "${_triton_ext_root}/support")

    # -------------------------------------------------------------------------
    # Extension name: always taken from the CMake project() name so that the
    # built library filename matches what the Python package expects.
    # -------------------------------------------------------------------------
    set(TRITON_EXT_NAME "${PROJECT_NAME}")

    if(DEFINED SKBUILD_PROJECT_VERSION)
        set(TRITON_EXT_VERSION "${SKBUILD_PROJECT_VERSION}")
    else()
        set(TRITON_EXT_VERSION "0.0.0+dev")
    endif()

    message(STATUS "Building triton-ext extension: ${TRITON_EXT_NAME} "
                   "(v${TRITON_EXT_VERSION})")

    # -------------------------------------------------------------------------
    # Locate the Triton wheel (headers + libtriton.so).
    # -------------------------------------------------------------------------
    include("${_triton_ext_root}/cmake/triton-wheel.cmake")
    if(NOT DEFINED TRITON_WHEEL_DIR)
        find_triton_wheel(TRITON_WHEEL_DIR)
    endif()
    message(STATUS "Found Triton wheel at ${TRITON_WHEEL_DIR}")

    get_filename_component(TRITON_WHEEL_DIR "${TRITON_WHEEL_DIR}" ABSOLUTE)
    set(TRITON_WHEEL_INCLUDE_DIR "${TRITON_WHEEL_DIR}/include")
    find_library(TRITON_LIB NAMES triton
        PATHS "${TRITON_WHEEL_DIR}/_C" NO_DEFAULT_PATH REQUIRED)
    message(STATUS "Found Triton library at ${TRITON_LIB}")

    if(NOT EXISTS "${TRITON_WHEEL_INCLUDE_DIR}/triton")
        message(FATAL_ERROR
            "Triton wheel is missing C++ headers at ${TRITON_WHEEL_INCLUDE_DIR}; "
            "it must be built with TRITON_EXT_ENABLED=1.")
    endif()

    # -------------------------------------------------------------------------
    # Locate LLVM/MLIR (headers, mlir-tblgen, CMake modules). NOT linked
    # against — libtriton re-exports all required symbols.
    # -------------------------------------------------------------------------
    include("${_triton_ext_root}/cmake/llvm.cmake")
    if(NOT DEFINED LLVM_INSTALL_DIR)
        find_llvm_install_dir(LLVM_INSTALL_DIR)
    endif()
    get_filename_component(LLVM_INSTALL_DIR "${LLVM_INSTALL_DIR}" ABSOLUTE)
    find_package(MLIR REQUIRED CONFIG PATHS "${LLVM_INSTALL_DIR}/lib/cmake/mlir")
    message(STATUS "Found MLIR at ${MLIR_DIR}")

    list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
    include(TableGen)
    include(AddLLVM)
    include(AddMLIR)

    # -------------------------------------------------------------------------
    # Include paths shared by TableGen invocations and C++ compiles.
    # Extensions may add their own with further include_directories() calls.
    # -------------------------------------------------------------------------
    include_directories(SYSTEM
        ${MLIR_INCLUDE_DIRS}
        ${LLVM_INCLUDE_DIRS}
        "${TRITON_WHEEL_INCLUDE_DIR}")
    include_directories(
        # generated TableGen headers land here
        "${CMAKE_CURRENT_BINARY_DIR}"
        # plugin export glue: Export.h / ExportPass.cpp / ExportDialect.cpp
        "${TRITON_EXT_SUPPORT_DIR}")
    add_definitions(${LLVM_DEFINITIONS})

    # -------------------------------------------------------------------------
    # ABI-breaking check: the extension must match libtriton's assert build.
    # -------------------------------------------------------------------------
    include("${_triton_ext_root}/cmake/llvm-abi-breaking.cmake")
    require_llvm_abi_breaking_checks("${LLVM_INSTALL_DIR}")

    # -------------------------------------------------------------------------
    # triton_ext::plugin — INTERFACE target encoding the compile flags and
    # link recipe that every plugin shared library must use identically:
    #   • -fvisibility=hidden / -fno-rtti / -Wno-deprecated-declarations
    #   • link ONLY against libtriton (which re-exports MLIR/LLVM symbols);
    #     --no-as-needed retains the DT_NEEDED entry even though symbols are
    #     resolved at dlopen() time from the already-loaded libtriton.
    # Extensions may still add extra compile options or definitions on their
    # own target; this target covers only the shared invariants.
    # -------------------------------------------------------------------------
    if(NOT TARGET triton_ext::plugin)
        add_library(triton_ext_plugin INTERFACE)
        add_library(triton_ext::plugin ALIAS triton_ext_plugin)
        target_compile_options(triton_ext_plugin INTERFACE
            -fvisibility=hidden
            -fno-rtti
            -Wno-deprecated-declarations)
        target_link_libraries(triton_ext_plugin INTERFACE
            "$<$<PLATFORM_ID:Linux>:-Wl,--no-as-needed>"
            ${TRITON_LIB}
            "$<$<PLATFORM_ID:Linux>:-Wl,--as-needed>"
            "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>")
    endif()
endmacro(configure_triton_extension)
