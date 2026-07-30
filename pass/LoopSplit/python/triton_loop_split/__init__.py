"""
The LoopSplit pass for Triton, registered as an extension.
"""

from pathlib import Path

import triton._C.libtriton as _libtriton

# Register the LoopSplit extension library with Triton.
PLUGIN_DIR = Path(__file__).resolve().parent
PLUGIN_LIBRARY = PLUGIN_DIR / "libloop_split.so"
_libtriton.passes.plugin.extend_with(str(PLUGIN_LIBRARY))  # adds passes
