#!/usr/bin/env python3
"""Opt driver for the Example dialect.

Parses an MLIR input using the Example dialect and runs ``canonicalize`` to
verify that ``example`` ops round-trip and are not folded away. Importing
``triton_example`` registers the dialect with Triton; both Triton and the
extension must be installed (``make build install``).

Run by hand with::

    ./example.py test/zero.mlir
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "testing"))

import triton_example  # noqa: E402, F401  registers the dialect on import
from triton._C.libtriton import passes  # noqa: E402
from mlir_runner import run_passes  # noqa: E402

run_passes([passes.common.add_canonicalizer])
