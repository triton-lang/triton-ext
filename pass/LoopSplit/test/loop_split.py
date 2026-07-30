#!/usr/bin/env python3
"""Opt driver for the LoopSplit pass.

Runs ``triton-loop-split`` followed by ``canonicalize`` over an MLIR input and
prints the result. Importing ``triton_loop_split`` registers the pass with
Triton; both Triton and the extension must be installed (``make build install``).

Run by hand with::

    ./loop_split.py test/loop-split.mlir
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "testing"))

import triton_loop_split  # noqa: E402, F401  registers the plugin on import
from mlir_runner import run_passes  # noqa: E402
from triton._C.libtriton import passes  # noqa: E402

run_passes([passes.plugin.add_loop_split, passes.common.add_canonicalizer])
