"""
Set up lit test environment--`setup_environment`--in `lit.local.cfg.py` files.

This module is added to sys.path by the top-level `lit.cfg.py` so that any `lit.local.cfg.py` in the
tree can import it directly.
"""

import sys
from pathlib import Path
import logging
from typing import Any


def find_extension_config_path(lit_cfg_path: Path) -> Path:
    """Find the path to the extension's `triton-ext.conf` by searching parent directories."""
    path = lit_cfg_path.parent
    while path != path.parent:
        candidate = path / "triton-ext.conf"
        if candidate.is_file():
            return candidate
        path = path.parent
    raise FileNotFoundError(
        f"Could not find triton-ext.conf in any parent directory of {lit_cfg_path}"
    )


def read_extension_name(ext_cfg_path: Path) -> str:
    """Read the extension name from the given `triton-ext.conf` file."""
    with open(ext_cfg_path) as f:
        line = next(f)
        name, _ = line.split(";", 1)
        return name.strip()
    raise ValueError(f"Could not find extension name in {ext_cfg_path}")


def find_library_path(ext_name: str, search_dirs: list[Path]) -> Path:
    """Find the path to the shared library for the given extension name."""
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    for dir in search_dirs:
        candidate = Path(dir) / f"lib{ext_name}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find shared library for extension {ext_name} in any of {search_dirs}"
    )


def setup_environment(config: Any, lit_cfg_path: str):
    """Set up the environment for running lit tests with a Triton plugin. """
    # Extend the environment: calculate the path to the extension's shared library.
    ext_cfg_path = find_extension_config_path(Path(lit_cfg_path))
    ext_name = read_extension_name(ext_cfg_path)
    lib_dir = Path(config.triton_ext_binary_dir) / "lib"
    config.environment["TRITON_PLUGIN_PATHS"] = find_library_path(
        ext_name, [lib_dir])
    logging.debug(
        f'ENV: TRITON_PLUGIN_PATHS={config.environment["TRITON_PLUGIN_PATHS"]}'
    )

    # Extend the environment: due to how we link Triton, `triton-opt` will not run unless it can
    # find LLVM's shared libraries.
    llvm_lib_dir = Path(config.llvm_install_dir) / "lib"
    sys_lib_var = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
    config.environment[sys_lib_var] = llvm_lib_dir
    logging.debug(f'ENV: {sys_lib_var}={config.environment[sys_lib_var]}')
