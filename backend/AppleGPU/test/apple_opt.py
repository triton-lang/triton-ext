#!/usr/bin/env python3
"""Opt driver for the Apple GPU passes.

Runs the plugin passes named after the input, in order, over an MLIR file and
prints the result. A pass's arguments follow its name after ``=``::

    ./apple_opt.py test/reduce-through-layout-change.mlir reduce_through_layout_change
    ./apple_opt.py test/prefetch-loads.mlir prefetch_loads=2
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "testing"))

import triton  # noqa: E402, F401  loads libtriton, which the plugin links against
import triton_apple_backend  # noqa: E402, F401  registers the plugin on import
from mlir_runner import run_passes  # noqa: E402
from triton._C.libtriton import passes  # noqa: E402


def adder(spec):
    name, _, args = spec.partition("=")
    add = getattr(passes.plugin, f"add_{name}")
    return (lambda pm: add(pm, args.split(","))) if args else add


run_passes([adder(spec) for spec in sys.argv[2:]])
