# triton-apple: Apple GPU backend for Triton
# Discovered by Triton via entry_points (pyproject.toml)

import sys as _sys
import sysconfig as _sysconfig
import importlib.abc as _importlib_abc
import importlib.util as _importlib_util
from pathlib import Path as _Path

# An editable install serves this file from the source tree while cmake
# installs the library into site-packages, so both are searched.
PLUGIN_DIR = _Path(__file__).resolve().parent
PLUGIN_NAME = "libapplegpu_backend.dylib"
PLUGIN_LIBRARY = next(
    (p for p in (PLUGIN_DIR / PLUGIN_NAME,
                 _Path(_sysconfig.get_paths()["purelib"]) / PLUGIN_DIR.name /
                 PLUGIN_NAME) if p.exists()), None)

# An in-tree backend ships triton/language/extra/<name>/ and that package's own
# scan finds it (see cuda/ and hip/). An out-of-tree plugin cannot put files
# there so the stubs go in through an import hook. It must run before user code
# caches a reference to the module.


class _LibdevicePatchFinder(_importlib_abc.MetaPathFinder):

    def find_spec(self, fullname, path, target=None):
        if fullname != 'triton.language.extra.libdevice':
            return None
        _sys.meta_path.remove(self)
        spec = _importlib_util.find_spec(fullname)
        if spec is None:
            return None
        orig_loader = spec.loader

        class _PatchingLoader:

            @staticmethod
            def create_module(spec):
                return (orig_loader.create_module(spec) if hasattr(
                    orig_loader, 'create_module') else None)

            @staticmethod
            def exec_module(module):
                orig_loader.exec_module(module)
                import triton.language as tl
                from triton_apple_backend.libdevice_stubs import ALL_STUBS
                for name, fn in ALL_STUBS.items():
                    if hasattr(module, name):
                        setattr(module, name, fn)
                    if not hasattr(tl.math, name):
                        setattr(tl.math, name, fn)

        spec.loader = _PatchingLoader()
        return spec


_sys.meta_path.insert(0, _LibdevicePatchFinder())

if PLUGIN_LIBRARY is not None:
    import triton._C.libtriton as _libtriton
    _libtriton.passes.plugin.extend_with(str(PLUGIN_LIBRARY))
