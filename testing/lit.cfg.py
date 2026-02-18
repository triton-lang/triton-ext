import os
import lit.formats
from lit.llvm import llvm_config

config.name = "TRITON-EXT"
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
config.suffixes = [".mlir", ".ll"]
config.test_source_root = config.triton_ext_source_dir
config.test_exec_root = os.path.join(config.triton_ext_binary_dir, "test")
config.excludes = [
    ".git",
    "build",
    "infra",
    # It is important that we exclude the directory containing this test configuration: otherwise,
    # `config.test_source_root` above will reload it, but without the `lit.site.cfg.py.in` substitutions, which will
    # cause this test configuration to fail to load. Instead, we load the `lit.site.cfg.py` generated from
    # `lit.site.cfg.py.in` in the `triton-ext/build` directory, which directly calls `load_config` on this file, with
    # substitutions applied.
    "testing",
]

# Also exclude any artifacts downloaded in the top-level directory.
for top in os.listdir(config.test_source_root):
    if top.startswith("llvm-") or top.startswith("triton-"):
        config.excludes.append(top)

# Extend the environment: add Triton and LLVM tools to PATH (e.g., for `triton-opt`, `FileCheck`).
triton_tools_dir = os.path.join(config.triton_build_dir, "bin")
llvm_tools_dir = os.path.join(config.llvm_install_dir, "bin")
tool_dirs = [triton_tools_dir, llvm_tools_dir]
for d in tool_dirs:
    llvm_config.with_environment("PATH", d, append_path=True)

# Extend the environment, as Triton does: "--enable-var-scope is enabled by default in MLIR test. This option avoids
# accidentally reusing variables across a -LABEL match; it can be explicitly opted-in by prefixing the variable name
# with $."
config.environment["FILECHECK_OPTS"] = "--enable-var-scope"

# Extend the environment: due to how we link Triton, `triton-opt` will not run unless it can find LLVM's shared
# libraries.
llvm_lib_dir = os.path.join(config.llvm_install_dir, "lib")
llvm_config.with_environment("LD_LIBRARY_PATH", llvm_lib_dir, append_path=True)
