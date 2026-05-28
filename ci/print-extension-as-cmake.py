#!/usr/bin/env python3
"""
Parse a `triton-ext.toml` manifest and print it as CMake commands.

Usage:
    python ci/print-extension-as-cmake.py <path/to/triton-ext.toml>
"""

from __future__ import annotations

import sys
import extension
from pathlib import Path


def main(path: Path) -> int:
    ext = extension.load(path)
    print(f'set(TRITON_EXT_NAME "{ext.name}")')
    print(f'set(TRITON_EXT_STATUS "{ext.status}")')
    print(f'set(TRITON_EXT_ENABLED "{"ON" if ext.enabled else "OFF"}")')
    print(f'set(TRITON_EXT_VERSION "{ext.version}")')
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path/to/triton-ext.toml>",
              file=sys.stderr)
        sys.exit(1)
    sys.exit(main(Path(sys.argv[1])))
