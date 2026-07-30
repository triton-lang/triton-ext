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

_HERE = Path(__file__).resolve().parent
# The library filename is platform dependent (`.so`/`.dylib`/`.dll`).
_PATTERNS = ("libarithmetic_intensity*.so", "libarithmetic_intensity*.dylib",
             "arithmetic_intensity*.dll", "libarithmetic_intensity*.dll")


def _find_plugin_library() -> Path:
    for pattern in _PATTERNS:
        matches = sorted(_HERE.glob(pattern))
        if matches:
            return matches[0]
    raise ImportError(
        f"arithmetic_intensity plugin library not found next to {_HERE}; "
        "was the package built (e.g. `pip install .`)?")


PLUGIN_LIBRARY = _find_plugin_library()
libtriton.passes.plugin.extend_with(str(PLUGIN_LIBRARY))  # adds passes
