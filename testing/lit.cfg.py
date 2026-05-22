import os
import sys
import lit.formats
from lit.llvm import llvm_config

# Add the testing source directory to `sys.path`` so that `lit.local.cfg.py` files anywhere in the
# tree can import shared utilities (e.g., `utils.py`).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# On macOS, system shells (`/bin/sh`, `/bin/bash`) carry the `__RESTRICT` Mach-O segment, which
# causes dyld to strip all DYLD_* variables from their environment before they start. We need to set
# `DYLD_LIBRARY_PATH` for `triton-opt` to find LLVM's libraries (see `utils.py`). This makes
# `DYLD_LIBRARY_PATH` ineffective for any command run through those shells. `lit`'s internal shell
# runs test commands directly via Python subprocess (no shell in between), so `DYLD_LIBRARY_PATH`
# reaches the process and works correctly.
_use_internal_shell = sys.platform == "darwin" or llvm_config.use_lit_shell

config.name = "TRITON-EXT"
config.test_format = lit.formats.ShTest(not _use_internal_shell)
config.suffixes = [".mlir", ".ll"]
config.test_source_root = config.triton_ext_source_dir
config.test_exec_root = os.path.join(config.triton_ext_binary_dir, "test")
config.excludes = [
    ".git",
    "build",
    "support",
    # It is important that we exclude the directory containing this test configuration: otherwise,
    # `config.test_source_root` above will reload it, but without the `lit.site.cfg.py.in`
    # substitutions, which will cause this test configuration to fail to load. Instead, we load the
    # `lit.site.cfg.py` generated from `lit.site.cfg.py.in` in the `triton-ext/build` directory,
    # which directly calls `load_config` on this file, with substitutions applied.
    "testing",
]

# Also exclude any artifacts downloaded in the top-level directory.
for top in os.listdir(config.test_source_root):
    if top.startswith("llvm-") or top.startswith("triton-"):
        config.excludes.append(top)

# Extend the environment: add Triton and LLVM tools to PATH (e.g., for `triton-opt`, `FileCheck`).
triton_tools_dir = os.path.join(config.triton_install_dir, "bin")
llvm_tools_dir = os.path.join(config.llvm_install_dir, "bin")
tool_dirs = [triton_tools_dir, llvm_tools_dir]
for d in tool_dirs:
    llvm_config.with_environment("PATH", d, append_path=True)

# Extend the environment, as Triton does: "--enable-var-scope is enabled by default in MLIR test. This option avoids
# accidentally reusing variables across a -LABEL match; it can be explicitly opted-in by prefixing the variable name
# with $."
config.environment["FILECHECK_OPTS"] = "--enable-var-scope"
