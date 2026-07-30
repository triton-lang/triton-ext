"""
An example out-of-tree Triton dialect registered as an extension.
"""

from pathlib import Path

import triton._C.libtriton as _libtriton

# Register the Example extension library with Triton.
PLUGIN_DIR = Path(__file__).resolve().parent
PLUGIN_LIBRARY = PLUGIN_DIR / "libexample.so"
_libtriton.ir.extend_dialects_with(str(PLUGIN_LIBRARY))  # adds dialects
