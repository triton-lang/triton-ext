# Register the lit tests for the triton-ext project.

# First, we generate the `lit.site.cfg.py` file from `lit.site.cfg.py.in`; we will use the substituted variables in
# our main `lit.cfg.py`.
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
)

# We expect `lit` to be available, likely pip-installed.
find_program(LLVM_EXTERNAL_LIT
  NAMES lit llvm-lit
  DOC "Path to lit test runner"
  REQUIRED
)
message(STATUS "Found lit: ${LLVM_EXTERNAL_LIT}")

# We also expect `FileCheck` to be available, but it should be provided by the LLVM installation.
find_program(FILECHECK_PATH
  NAMES FileCheck
  HINTS "${LLVM_INSTALL_DIR}/bin"
  DOC "Path to FileCheck"
  NO_DEFAULT_PATH
)
find_program(FILECHECK_PATH
  NAMES FileCheck
  DOC "Path to FileCheck (if not found in LLVM installation)"
  REQUIRED
)
message(STATUS "Found FileCheck: ${FILECHECK_PATH}")

# Now, register all lit tests to this target (see `lit.cfg.py` for the discovery logic).
add_lit_testsuite(check-lit-tests "Running the triton-ext lit tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  ARGS "-Dfilecheck=${FILECHECK_PATH}"
  DEPENDS triton-ext-all-extensions
)
set_target_properties(check-lit-tests PROPERTIES FOLDER "Tests")
