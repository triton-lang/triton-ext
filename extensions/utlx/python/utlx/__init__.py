"""triton.language.extra.tlx — re-export from utlx_plugin.

This package can be symlinked into triton/python/triton/language/extra/tlx
so that `import triton.language.extra.tlx as tlx` resolves to the uTLX
plugin's Python DSL.
"""

import os as _os
import sys as _sys

# Ensure the uTLX plugin python dir is on sys.path so utlx_plugin is importable
_plugin_python_dir = _os.path.dirname(_os.path.dirname(_os.path.realpath(__file__)))
if _plugin_python_dir not in _sys.path:
    _sys.path.insert(0, _plugin_python_dir)

# Use lazy re-export to avoid circular import during triton bootstrap.
# utlx_plugin imports triton.language.core, which triggers triton.language.extra
# to discover and load this package.
import importlib as _importlib


def __getattr__(name):
    _mod = _importlib.import_module("utlx_plugin")
    try:
        return getattr(_mod, name)
    except AttributeError:
        raise AttributeError(f"module 'triton.language.extra.tlx' has no attribute {name!r}")


def __dir__():
    _mod = _importlib.import_module("utlx_plugin")
    return dir(_mod)
