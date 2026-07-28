#!/usr/bin/env python3
"""
This script returns the `os` and `arch` fields naming a triton-ext artifact. -
first, it examines the GitHub environment
  (https://docs.github.com/en/actions/reference/workflows-and-actions/variables):
  `RUNNER_OS` and `RUNNER_ARCH`
- if these are not set, it falls back to Python's `platform` module.

To match upstream Triton's artifact naming, if `REFINE_OS` is configured a Linux
OS name is refined to the distro family rather than the generic `linux`:

- RHEL-compatible distros (RHEL, CentOS, Fedora, etc.): `almalinux`
- everything else (Ubuntu, Debian, etc.): `ubuntu`

Usage:
    [REFINE_OS=1|0] python probe_sysinfo.py
"""

import os
import platform
import sys

import common


def refine_linux_distro() -> str:
    """Return the artifact OS name for the current Linux distribution.

    Reads /etc/os-release via `platform.freedesktop_os_release()` and checks
    `ID` and `ID_LIKE` for known RHEL-family identifiers. Falls back to `ubuntu`
    for unrecognised or non-freedesktop distributions.
    """
    _RHEL_FAMILY = {"rhel", "centos", "fedora", "almalinux", "rocky"}
    info = platform.freedesktop_os_release()
    distro_id = info.get("ID", "").lower()
    id_like = set(info.get("ID_LIKE", "").lower().split())
    if distro_id in _RHEL_FAMILY or _RHEL_FAMILY & id_like:
        return "almalinux"
    else:
        return "ubuntu"


def run(refine_os=False) -> tuple[str, str]:
    """Get the current OS and architecture (lowercase)."""
    os_name = os.getenv('RUNNER_OS', sys.platform).lower()
    if os_name.startswith("linux"):
        os_name = refine_linux_distro() if refine_os else "linux"
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
    refine_os = common.env2bool("REFINE_OS")
    os_name, arch = run(refine_os)
    print(f"{os_name}-{arch}")
