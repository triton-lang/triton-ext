# ---------------------------------------------------------------------------
# require_llvm_abi_breaking_checks(<llvm_install_dir>)
#
# Verify that the LLVM install at <llvm_install_dir> was built with
# LLVM_ENABLE_ABI_BREAKING_CHECKS=1, failing configuration otherwise.
#
# That flag toggles ilist sentinel tracking, which changes the mangled type of
# every ilist_iterator (pervasive in MLIR). A plugin compiled against headers
# with a different value than the host libtriton (an asserts build, checks on)
# produces runtime "undefined symbol" load failures. The LLVM install's
# abi-breaking.h hardcodes the value with an unconditional `#define`, so rather
# than patch it we simply parse it and require that it is 1.
# ---------------------------------------------------------------------------

function(require_llvm_abi_breaking_checks llvm_install_dir)
    set(_abi_src "${llvm_install_dir}/include/llvm/Config/abi-breaking.h")
    if(NOT EXISTS "${_abi_src}")
        message(FATAL_ERROR
                "Could not find ${_abi_src}; cannot verify "
                "LLVM_ENABLE_ABI_BREAKING_CHECKS.")
    endif()
    file(READ "${_abi_src}" _abi_contents)
    string(REGEX MATCH
        "#define[ \t]+LLVM_ENABLE_ABI_BREAKING_CHECKS[ \t]+([0-9]+)"
        _abi_match "${_abi_contents}")
    if(NOT _abi_match OR NOT CMAKE_MATCH_1 EQUAL 1)
        message(FATAL_ERROR
            "LLVM_ENABLE_ABI_BREAKING_CHECKS must be 1 in ${_abi_src} "
            "(found '${CMAKE_MATCH_1}'). The plugin links against a libtriton "
            "built as an asserts build (ABI breaking checks on); a mismatch "
            "produces runtime 'undefined symbol' load failures. Use an LLVM "
            "install built with LLVM_ENABLE_ABI_BREAKING_CHECKS=1.")
    endif()
    message(STATUS "Verified LLVM_ENABLE_ABI_BREAKING_CHECKS=1")
endfunction()
