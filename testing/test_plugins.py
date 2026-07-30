"""Plugin integration tests.

Auto-discovers every extension declared by a ``pyproject.toml`` carrying a
``[tool.triton-ext]`` stanza and exercises it via a direct import.  Each
enabled extension is imported using ``import <package>``, which loads and
registers the compiled plugin library. No ``TRITON_PLUGIN_PATHS``,
``PYTHONPATH``, or ``LD_LIBRARY_PATH`` overrides are needed; the extensions
must be installed beforehand (e.g. with ``make build && make install``).

Tests:
  - test_plugins_discovered                -- guard: at least one plugin exists.
  - test_plugin_loads[<name>]              -- ``import <package>`` succeeds.
  - test_plugin_compiles_kernel[<name>]    -- JIT-decorate and lower a basic
                                             kernel through the plugin pipeline.
  - test_utlx_registers_tlx_dsl           -- utlx registers
                                             ``triton.language.extra.tlx``.

Adding a new plugin: drop a ``pyproject.toml`` with ``[tool.triton-ext]``;
both parametrized tests pick it up automatically.  To exempt a plugin from a
parametrized test mark it at parametrize time with
``pytest.param(..., marks=pytest.mark.skip(...))``.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from _pytest.mark.structures import ParameterSet

REPO_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(REPO_ROOT / "ci"))
import extension  # noqa: E402  (ci/ is added to sys.path above)

# Map from extension *name* (as in triton-ext.toml / pyproject.toml) to the
# importable Python package name.
_PACKAGE_MAP: dict[str, str] = {
    "arithmetic_intensity": "triton_arithmetic_intensity",
    "triton_arithmetic_intensity": "triton_arithmetic_intensity",
    "loop_split": "triton_loop_split",
    "triton_loop_split": "triton_loop_split",
    "example": "triton_example",
    "triton-example": "triton_example",
    "utlx": "utlx_plugin",
}


def _package_name(ext_name: str) -> str:
    """Return the importable Python package name for an extension."""
    if ext_name in _PACKAGE_MAP:
        return _PACKAGE_MAP[ext_name]
    # Fall back: replace hyphens/spaces with underscores
    return ext_name.replace("-", "_").replace(" ", "_")


def _discover_plugins() -> list[ParameterSet]:
    plugins: list[ParameterSet] = []
    for cfg in extension.discover():
        if cfg.enabled:
            plugins.append(pytest.param(cfg.name, id=cfg.name))
    plugins.sort(key=lambda p: p.id)
    return plugins


PLUGINS = _discover_plugins()

# ---------------------------------------------------------------------------
# Generic per-plugin tests (auto-discovered)
# ---------------------------------------------------------------------------


def test_plugins_discovered() -> None:
    """Guard against silently testing nothing if discovery breaks."""
    assert PLUGINS, f"No triton-ext extensions found under {REPO_ROOT}"


@pytest.mark.parametrize("name", PLUGINS)
def test_plugin_loads(name: str) -> None:
    """Smoke: ``import <package>`` succeeds with the plugin registered."""
    pkg = _package_name(name)
    try:
        importlib.import_module(pkg)
    except ImportError as exc:
        pytest.skip(
            f"Package {pkg!r} not installed (run `make build && make install`): {exc}"
        )


# example dialect is scaffolding-only — its Dialect::initialize() doesn't
# register StringAttr, so kernel compile aborts with an LLVM storage-uniquer
# error.  Tag it as skip at parametrize time.
_COMPILE_PLUGINS = [
    pytest.param(p.values[0],
                 marks=pytest.mark.skip(reason="scaffolding-only dialect"),
                 id=p.id) if p.id == "example" else p for p in PLUGINS
]


@pytest.mark.parametrize("name", _COMPILE_PLUGINS)
def test_plugin_compiles_kernel(name: str) -> None:
    """User scenario: with the plugin imported, JIT-decorate and lower a basic kernel."""
    pkg = _package_name(name)
    try:
        importlib.import_module(pkg)
    except ImportError as exc:
        pytest.skip(
            f"Package {pkg!r} not installed (run `make build && make install`): {exc}"
        )

    import triton
    import triton.language as tl

    @triton.jit
    def _kernel(x_ptr, y_ptr, n: tl.constexpr):
        offs = tl.arange(0, n)
        x = tl.load(x_ptr + offs)
        tl.store(y_ptr + offs, x)

    # Compiling the kernel (lowering to PTX/HSACO) requires a GPU device.
    # Skip gracefully if no GPU is available rather than failing.
    try:
        import torch
        if not torch.cuda.is_available():
            pytest.skip(
                "No CUDA device available; skipping kernel compile test")
        device = torch.device("cuda")
        n = 32
        x = torch.ones(n, device=device)
        y = torch.zeros(n, device=device)
        _kernel[(1, )](x, y, n)
        assert torch.allclose(x, y), "kernel output mismatch"
    except ImportError:
        pytest.skip("torch not installed; skipping kernel compile test")


# ---------------------------------------------------------------------------
# Plugin-specific tests
# ---------------------------------------------------------------------------


def test_utlx_registers_tlx_dsl() -> None:
    """utlx registers ``triton.language.extra.tlx`` with local_alloc/view/store/load.

    The namespace is set up by ``extensions/utlx/python/utlx_plugin/__init__.py``
    when the package is imported.
    """
    try:
        import utlx_plugin  # noqa: F401
    except ImportError:
        pytest.skip(
            "utlx_plugin not installed (run `make build && make install`)")

    import triton.language.extra as extra
    assert hasattr(extra, "tlx"), "triton.language.extra.tlx not registered"

    import triton.language.extra.tlx as tlx
    for attr in ("local_alloc", "local_view", "local_store", "local_load"):
        assert hasattr(tlx,
                       attr), f"tlx.{attr} missing after import utlx_plugin"
