"""uTLX registration test, owned by the utlx extension.

This test used to live in the repository-wide ``testing/test_plugins.py`` and
skipped itself on ``ImportError``. That skip cannot distinguish "this job did not
build utlx" from "utlx is built and broken", so a real breakage reported green.
It is moved next to the extension it tests -- where it runs as part of
``make -C extensions/utlx test`` -- and the import is no longer swallowed.
"""
from __future__ import annotations

import importlib

import pytest


def test_utlx_registers_tlx_dsl() -> None:
    """Importing the plugin package registers ``triton.language.extra.tlx``.

    The namespace is set up by ``extensions/utlx/python/utlx_plugin/__init__.py``
    when the package is imported. A failure here is a failure, not a skip: the
    utlx test suite is only supposed to run once utlx is installed.
    """
    importlib.import_module("utlx_plugin")

    import triton.language.extra as extra

    assert hasattr(extra, "tlx"), "triton.language.extra.tlx not registered"

    tlx = importlib.import_module("triton.language.extra.tlx")
    for attr in ("local_alloc", "local_view", "local_store", "local_load"):
        assert hasattr(tlx, attr), f"tlx.{attr} missing after importing utlx_plugin"


@pytest.mark.parametrize("attr", ["local_alloc", "local_view", "local_store", "local_load"])
def test_tlx_operations_are_callable(attr: str) -> None:
    """The registered names must be the real DSL entry points, not placeholders.

    ``hasattr`` alone is satisfied by any module attribute -- including one left
    behind by a partially imported module. Requiring the attribute to be callable
    is the cheapest check that the namespace really exposes the ops.
    """
    import triton.language.extra.tlx as tlx

    assert callable(getattr(tlx, attr)), f"tlx.{attr} is not callable"
