import os

config.environment["TRITON_PLUGIN_PATHS"] = os.path.join(
    config.triton_ext_binary_dir, "lib", "libtypeid_cast.so")
