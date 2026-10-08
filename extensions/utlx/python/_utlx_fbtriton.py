"""Load an fbtriton checkout's test helpers into stock Triton, in every process.

fbtriton's tests import helpers from ``triton._internal_testing`` that only the
fork defines (e.g. ``swizzle_scale_to_5d``). When ``UTLX_FBTRITON_ROOT`` names
an fbtriton checkout, stock Triton's module is backfilled on first import with
the functions and classes the checkout's own ``python/triton/_internal_testing.py``
defines and stock Triton lacks; helpers stock Triton has keep their stock
versions.

``utlx_plugin.pth`` imports this at interpreter startup, so it also reaches the
processes tests start themselves -- ``run_in_process``'s forkserver children
re-import the test module without any pytest plugin. ``_utlx_pytest`` sets the
variable when the tests live in an fbtriton checkout. Unset, this does nothing.

PYTEST_DONT_REWRITE: the .pth always imports this before pytest can.
"""
import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys

ROOT_ENV = "UTLX_FBTRITON_ROOT"
_NAME = "triton._internal_testing"
_PRIVATE = "_utlx_fbtriton_internal_testing"


def _backfill(module, root):
    from triton.backends.compiler import GPUTarget

    # The fork's helpers call this fork-only GPUTarget method.
    if not hasattr(GPUTarget, "is_cuda_backend"):
        GPUTarget.is_cuda_backend = lambda self: self.backend in ("cuda",
                                                                  "tileir")

    spec = importlib.util.spec_from_file_location(
        _PRIVATE, os.path.join(root, "python", "triton",
                               "_internal_testing.py"))
    fbtriton = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fbtriton)
    for name, value in vars(fbtriton).items():
        if (not name.startswith("_") and not hasattr(module, name)
                and getattr(value, "__module__", None) == _PRIVATE):
            setattr(module, name, value)


class _Loader(importlib.abc.Loader):

    def __init__(self, inner, root):
        self.inner, self.root = inner, root

    def create_module(self, spec):
        return self.inner.create_module(spec)

    def exec_module(self, module):
        self.inner.exec_module(module)
        _backfill(module, self.root)


class _Finder(importlib.abc.MetaPathFinder):

    def __init__(self, root):
        self.root = root

    def find_spec(self, fullname, path=None, target=None):
        if fullname != _NAME:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _Loader(spec.loader, self.root)
        return spec


def install(root):
    """Backfill from *root* now if the module is loaded, else on first import."""
    if _NAME in sys.modules:
        _backfill(sys.modules[_NAME], root)
    elif not any(isinstance(f, _Finder) for f in sys.meta_path):
        # First, so it sees the import before the PathFinder loads it unwrapped.
        sys.meta_path.insert(0, _Finder(root))


if os.environ.get(ROOT_ENV):
    install(os.environ[ROOT_ENV])
