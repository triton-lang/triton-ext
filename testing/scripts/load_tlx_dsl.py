#!/usr/bin/env python3
"""Check that utlx registers the `triton.language.extra.tlx` DSL.

Loading the utlx .so alone is not enough: importing `utlx_plugin` is what
inserts itself into `sys.modules` as `triton.language.extra.tlx`. Run manually:

    TRITON_PLUGIN_PATHS=/path/to/libutlx.so \
    PYTHONPATH=extensions/utlx/python \
    python testing/scripts/load_tlx_dsl.py
"""

import sys

import triton  # noqa: F401
import utlx_plugin  # noqa: F401  (registers triton.language.extra.tlx)
from triton.language.extra import tlx


def main() -> int:
    for n in ("local_alloc", "local_view", "local_store", "local_load"):
        assert hasattr(tlx, n), f"missing tlx.{n}"
    return 0


if __name__ == "__main__":
    sys.exit(main())
