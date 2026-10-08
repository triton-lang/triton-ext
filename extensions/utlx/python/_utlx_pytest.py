"""pytest plugin that runs fbtriton's TLX op tests against fbtriton's own code.

fbtriton's TLX op tests (``python/test/unit/tlx_ops``) exercise the TLX op
library (``triton.tlx``) and import test helpers from
``triton._internal_testing`` that only Meta's fork defines. When the tests
pytest is given live in an fbtriton checkout, this plugin takes both from that
checkout, so everything the tests touch comes from fbtriton except the TLX
implementation under test, uTLX itself:

- ``triton.tlx`` is served from ``<checkout>/third_party/tlx`` by setting
  ``UTLX_TLX_OPS_ROOT`` (see ``_utlx_autoregister``). As an environment
  variable it reaches the subprocesses some tests start too. An explicitly set
  ``UTLX_TLX_OPS_ROOT`` is left alone.
- Helpers stock Triton's ``triton._internal_testing`` lacks are copied in from
  ``<checkout>/python/triton/_internal_testing.py`` when the module is first
  imported. Helpers stock Triton defines keep their stock versions.

The ``pytest11`` entry point loads this into every pytest run in an
environment with uTLX installed; without an fbtriton checkout among the test
paths it does nothing. Disable it with ``-p no:utlx``.

pytest imports plugins at startup, so this lives outside ``utlx_plugin``
(whose import loads Triton) and imports nothing heavy.
"""

import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys

import pytest

INTERNAL_TESTING = "triton._internal_testing"
OPS_ROOT_ENV = "UTLX_TLX_OPS_ROOT"  # read by _utlx_autoregister
_PRIVATE_NAME = "_utlx_fbtriton_internal_testing"


def find_fbtriton_root(paths):
    """Return the first fbtriton checkout containing one of *paths*, or None.

    A checkout is recognized by its TLX op library and its
    ``triton/_internal_testing.py``.
    """
    for path in paths:
        path = os.path.abspath(path)
        while True:
            if (os.path.isfile(
                    os.path.join(path, "third_party", "tlx", "ops",
                                 "__init__.py")) and os.path.isfile(
                                     os.path.join(path, "python", "triton",
                                                  "_internal_testing.py"))):
                return path
            parent = os.path.dirname(path)
            if parent == path:
                break
            path = parent
    return None


def backfill(module, fbtriton_root):
    """Copy into *module* the fbtriton helpers it lacks.

    Only functions and classes defined in fbtriton's file are copied, not names
    it merely imports.
    """
    from triton.backends.compiler import GPUTarget

    # The fork's helpers call this fork-only GPUTarget method.
    if not hasattr(GPUTarget, "is_cuda_backend"):
        GPUTarget.is_cuda_backend = lambda self: self.backend in ("cuda",
                                                                  "tileir")

    spec = importlib.util.spec_from_file_location(
        _PRIVATE_NAME,
        os.path.join(fbtriton_root, "python", "triton",
                     "_internal_testing.py"))
    fbtriton = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fbtriton)
    for name, value in vars(fbtriton).items():
        if (not name.startswith("_") and not hasattr(module, name)
                and getattr(value, "__module__", None) == _PRIVATE_NAME):
            setattr(module, name, value)


class _BackfillLoader(importlib.abc.Loader):
    """Run the real loader, then backfill the module it produced."""

    def __init__(self, inner, fbtriton_root):
        self.inner = inner
        self.fbtriton_root = fbtriton_root

    def create_module(self, spec):
        return self.inner.create_module(spec)

    def exec_module(self, module):
        self.inner.exec_module(module)
        backfill(module, self.fbtriton_root)


class _BackfillFinder(importlib.abc.MetaPathFinder):
    """Wrap the loader of ``triton._internal_testing`` in ``_BackfillLoader``."""

    def __init__(self, fbtriton_root):
        self.fbtriton_root = fbtriton_root

    def find_spec(self, fullname, path=None, target=None):
        if fullname != INTERNAL_TESTING:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _BackfillLoader(spec.loader, self.fbtriton_root)
        return spec


@pytest.hookimpl(tryfirst=True)
def pytest_load_initial_conftests(early_config, parser, args):
    # Before any conftest, so nothing has imported triton.tlx or
    # triton._internal_testing yet.
    params = early_config.invocation_params
    paths = [
        os.path.join(params.dir,
                     arg.split("::")[0]) for arg in params.args
        if not arg.startswith("-")
        and os.path.exists(os.path.join(params.dir,
                                        arg.split("::")[0]))
    ]
    root = find_fbtriton_root(paths or [str(params.dir)])
    if root is None:
        return
    os.environ.setdefault(OPS_ROOT_ENV, os.path.join(root, "third_party",
                                                     "tlx"))
    if INTERNAL_TESTING in sys.modules:
        backfill(sys.modules[INTERNAL_TESTING], root)
    else:
        sys.meta_path.insert(0, _BackfillFinder(root))
