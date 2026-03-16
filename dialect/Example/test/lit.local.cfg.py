import os
import sys

# Extend the test environment from `lit-test/lit.cfg.py`: point `triton-opt` at the shared library for this pass. In the
# future, this should calculate an OS-agnostic path, possibly using the CMake target (TODO).
config.environment["TRITON_PLUGIN_PATHS"] = os.path.join(
    config.triton_ext_binary_dir, "lib", "libexample.so")
print(
    f"ENV: "
    f"LD_LIBRARY_PATH={config.environment['LD_LIBRARY_PATH']} "
    f"TRITON_PLUGIN_PATHS={config.environment['TRITON_PLUGIN_PATHS']}",
    file=sys.stderr)
