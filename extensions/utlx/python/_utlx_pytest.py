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
  ``<checkout>/python/triton/_internal_testing.py`` by setting
  ``UTLX_FBTRITON_ROOT`` (see ``_utlx_fbtriton``), which also reaches test
  subprocesses. An explicitly set ``UTLX_FBTRITON_ROOT`` is left alone.

The ``pytest11`` entry point loads this into every pytest run in an
environment with uTLX installed; without an fbtriton checkout among the test
paths it does nothing. Disable it with ``-p no:utlx``.

pytest imports plugins at startup, so this lives outside ``utlx_plugin``
(whose import loads Triton) and imports nothing heavy.
"""

import os

import pytest

import _utlx_fbtriton

OPS_ROOT_ENV = "UTLX_TLX_OPS_ROOT"  # read by _utlx_autoregister


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
    if not os.environ.get(_utlx_fbtriton.ROOT_ENV):  # empty counts as unset
        os.environ[_utlx_fbtriton.ROOT_ENV] = root
    _utlx_fbtriton.install(os.environ[_utlx_fbtriton.ROOT_ENV])
