"""
An example out-of-tree Triton dialect registered as an extension.
"""

from pathlib import Path

import triton._C.libtriton as _libtriton

_HERE = Path(__file__).resolve().parent
_PATTERNS = ("lib*example*.so", "lib*example*.dylib", "*example*.dll")


def _find_plugin_library() -> Path:
    for pattern in _PATTERNS:
        matches = sorted(_HERE.glob(pattern))
        if matches:
            return matches[0]
    raise ImportError(f"example plugin library not found next to {_HERE}; "
                      "was the package built (e.g. `pip install .`)?")


# Register the Example extension library with Triton.
PLUGIN_LIBRARY = _find_plugin_library()
_libtriton.ir.extend_dialects_with(str(PLUGIN_LIBRARY))  # adds dialects
