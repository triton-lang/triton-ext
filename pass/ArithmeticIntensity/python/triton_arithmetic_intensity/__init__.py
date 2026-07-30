"""Register the arithmetic-intensity pass as a Triton plugin.

Importing this package loads the compiled plugin library that is bundled
alongside this file and hands it to Triton's plugin API. Triton must already be
imported (so ``libtriton`` is loaded); the plugin resolves its MLIR/LLVM symbols
from that already-loaded library, so no ``LD_LIBRARY_PATH`` or
``TRITON_PLUGIN_PATHS`` is required.
"""

from __future__ import annotations

from pathlib import Path

import triton._C.libtriton as libtriton

PLUGIN_DIR = Path(__file__).resolve().parent
PLUGIN_LIBRARY = PLUGIN_DIR / "libarithmetic_intensity.so"
libtriton.passes.plugin.extend_with(str(PLUGIN_LIBRARY))  # adds passes
