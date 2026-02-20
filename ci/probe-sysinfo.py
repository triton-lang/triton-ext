#!/usr/bin/env python3
"""
This script returns the `os` and `arch` fields naming a triton-ext GitHub artifact.
- first, it examines the GitHub environment
  (https://docs.github.com/en/actions/reference/workflows-and-actions/variables): `RUNNER_OS` and `RUNNER_ARCH`
- if these are not set, it falls back to Python's `platform` module.

Usage:
    python probe-sysinfo.py
"""

import sys
import platform
import os


def run() -> tuple[str, str]:
    """Get the current OS and architecture (lowercase)."""
    os_name = os.getenv('RUNNER_OS', sys.platform).lower()
    if os_name.startswith("linux"):
        os_name = "linux"
    elif os_name.startswith("darwin"):
        os_name = "macos"
    elif os_name.startswith("win"):
        os_name = "windows"

    arch = os.getenv('RUNNER_ARCH', platform.machine()).lower()
    if arch in ["x86_64", "amd64"]:
        arch = "x64"
    elif arch in ["aarch64", "arm64"]:
        arch = "arm64"

    return os_name, arch


if __name__ == "__main__":
    os_name, arch = run()
    print(f"{os_name}-{arch}")
