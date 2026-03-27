"""triton.language.extra.utlx — re-export from utlx_plugin.

This package can be symlinked into triton/python/triton/language/extra/utlx
so that `import triton.language.extra.utlx as tlx` resolves to the uTLX
plugin's Python DSL.
"""

import os as _os
import sys as _sys

# Ensure the uTLX plugin python dir is on sys.path so utlx_plugin is importable
_plugin_python_dir = _os.path.dirname(_os.path.dirname(_os.path.realpath(__file__)))
if _plugin_python_dir not in _sys.path:
    _sys.path.insert(0, _plugin_python_dir)

from utlx_plugin import *  # noqa: F401,F403
from utlx_plugin import __all__  # noqa: F401
