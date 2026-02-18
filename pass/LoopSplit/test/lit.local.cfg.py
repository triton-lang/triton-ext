import os
import sys

# Extend the test environment from `lit-test/lit.cfg.py`: point `triton-opt` at the shared library for this pass. In the
# future, this should calculate an OS-agnostic path, possibly using the CMake target (TODO).
config.environment["TRITON_PASS_PLUGIN_PATH"] = os.path.join(
    config.triton_ext_binary_dir, "lib", "libloop_split.so")
print(
    f"ENV: "
    f"LD_LIBRARY_PATH={config.environment['LD_LIBRARY_PATH']} "
    f"TRITON_PASS_PLUGIN_PATH={config.environment['TRITON_PASS_PLUGIN_PATH']}",
    file=sys.stderr)
