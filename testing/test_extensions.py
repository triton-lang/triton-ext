"""Discover installed extensions, import their packages, and compile a minimal kernel."""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(REPO_ROOT / "ci"))
import extension  # noqa: E402

EXTENSIONS = [cfg for cfg in extension.discover() if cfg.enabled]
INSTALLED_EXTENSIONS = [
    pytest.param(cfg.package, id=cfg.name) for cfg in EXTENSIONS
    if importlib.util.find_spec(cfg.package) is not None
]


def test_extension_list_is_not_empty() -> None:
    """Require enabled manifests and at least one installed extension."""
    assert EXTENSIONS, "no enabled extensions discovered; check the pyproject manifests"
    assert INSTALLED_EXTENSIONS, (
        "no extension is installed in this environment; build at least one "
        "(make build && make install) before running the integration tests")


@pytest.mark.parametrize("package", INSTALLED_EXTENSIONS)
def test_extension_loads(package: str) -> None:
    """Import each installed extension package and expose native load errors."""
    importlib.import_module(package)


@pytest.mark.parametrize("package", INSTALLED_EXTENSIONS)
def test_extension_compiles_kernel(package: str) -> None:
    """Compile a minimal kernel with an explicit CUDA target and the extension loaded."""
    import triton
    import triton.language as tl
    from triton.backends.compiler import GPUTarget

    if not hasattr(triton, "compile"):
        pytest.skip("this triton exposes no compile() entry point")

    importlib.import_module(package)

    @triton.jit
    def _identity(x_ptr, y_ptr, n: tl.constexpr):
        offs = tl.arange(0, n)
        tl.store(y_ptr + offs, tl.load(x_ptr + offs))

    src = triton.compiler.ASTSource(
        fn=_identity,
        signature={
            "x_ptr": "*fp32",
            "y_ptr": "*fp32",
            "n": "constexpr"
        },
        constexprs={"n": 32},
    )
    compiled = triton.compile(src, target=GPUTarget("cuda", 89, 32))
    assert compiled.asm, "extension compile produced no assembly"


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
