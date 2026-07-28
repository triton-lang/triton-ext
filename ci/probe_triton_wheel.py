#!/usr/bin/env python3
"""
Determine the location of an installed Triton wheel.
"""

import importlib
from pathlib import Path


def run():
    """Return the path to the installed Triton wheel."""
    module = importlib.import_module("triton")
    return Path(module.__file__).parent


if __name__ == "__main__":
    print(run())
