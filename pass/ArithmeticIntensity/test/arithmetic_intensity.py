#!/usr/bin/env python3
"""Opt driver for the ArithmeticIntensity pass.

Runs ``arithmetic-intensity`` over an MLIR input and prints the result.
Importing ``triton_arithmetic_intensity`` registers the pass with Triton; both
Triton and the extension must be installed (``make build install``).

Run by hand with::

    ./arithmetic_intensity.py arithmetic-intensity.mlir
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "testing"))

import triton_arithmetic_intensity  # noqa: E402, F401  registers the plugin
from mlir_runner import run_passes  # noqa: E402
from triton._C.libtriton import passes  # noqa: E402

run_passes([passes.plugin.add_arithmetic_intensity])
