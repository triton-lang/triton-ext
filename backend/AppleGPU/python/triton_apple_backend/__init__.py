"""Register the Apple GPU backend as a Triton plugin.

Importing this package loads the compiled plugin library that is bundled
alongside this file and hands it to Triton's plugin API. Triton must already be
imported (so ``libtriton`` is loaded); the plugin resolves its MLIR/LLVM symbols
from that already-loaded library, so no ``LD_LIBRARY_PATH`` or
``TRITON_PLUGIN_PATHS`` is required.

An editable install serves this file from the source tree while cmake installs
the library into site-packages, so both are searched.
"""

from __future__ import annotations

import sysconfig
from pathlib import Path

import triton._C.libtriton as libtriton

PLUGIN_DIR = Path(__file__).resolve().parent
PLUGIN_NAME = "libapplegpu_backend.dylib"
PLUGIN_LIBRARY = next(
    (p for p in (PLUGIN_DIR / PLUGIN_NAME,
                 Path(sysconfig.get_paths()["purelib"]) / PLUGIN_DIR.name /
                 PLUGIN_NAME) if p.exists()), None)

if PLUGIN_LIBRARY is not None:
    libtriton.passes.plugin.extend_with(str(PLUGIN_LIBRARY))  # adds passes
