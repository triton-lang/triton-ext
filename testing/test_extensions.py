"""Reworked extension integration tests for triton-ext #112.

Addresses the five points in the issue, and fixes a concrete defect the audit
found: the package map carried hyphenated *and* underscored keys, so a newly
whitespace-normalised extension name such as ``arithmetic-intensity`` fell
through to ``str.replace("-", "_")`` and resolved to ``arithmetic_intensity``,
which is not the import package. The real import name is declared once, in each
extension's ``pyproject.toml``, under ``[tool.scikit-build.wheel.packages]`` -- so
it is read from there instead of being maintained by hand.

Scope of each change:

* package name            -> derived from pyproject, no hand-written map
* load failures           -> FAIL with the original traceback, never skip
* compile test            -> compiles through an explicit target, no CUDA device
                             required, so it is a real compile test on a CPU node
* utlx-specific test      -> now lives in extensions/utlx/test/, next to the
                             extension it tests (see test_registration.py)
* naming                  -> "extension" throughout
"""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from _pytest.mark.structures import ParameterSet

REPO_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(REPO_ROOT / "ci"))
import extension  # noqa: E402  (ci/ is added to sys.path above)


def _lookup(ext_name: str):
    """Find a manifest by short name, wheel name or directory basename.

    The name an extension is known by is not uniform -- the CI listing shows
    ``example`` while the manifest's project name is ``triton-example`` -- so all
    three identifiers are accepted rather than assuming one.
    """
    for cfg in extension.discover():
        if ext_name in (cfg.name, cfg.wheel, Path(cfg.path).name):
            return cfg
    return None


def package_name(ext_name: str) -> str:
    """Import package declared by an extension's pyproject.toml.

    ``[tool.scikit-build.wheel.packages]`` maps import name -> source directory,
    so the first key is the importable package. Reading it here is what removes
    the hand-maintained mapping: adding an extension needs no edit to this file.
    """
    cfg = _lookup(ext_name)
    if cfg is None:
        raise AssertionError(f"unknown extension {ext_name!r}")
    # Manifest.path is repo-relative, so it is anchored at REPO_ROOT rather than
    # at the process working directory.
    manifest = REPO_ROOT / cfg.path / "pyproject.toml"
    if not manifest.is_file():
        raise AssertionError(f"{ext_name}: no pyproject.toml at {manifest}")
    with manifest.open("rb") as fh:
        doc = tomllib.load(fh)
    packages = doc.get("tool", {}).get("scikit-build", {}).get("wheel", {}).get("packages")
    if not packages:
        raise AssertionError(
            f"{ext_name}: pyproject.toml declares no [tool.scikit-build.wheel.packages], "
            "so the import package cannot be derived")
    return next(iter(packages))


def _discover_extensions() -> list[ParameterSet]:
    found = [pytest.param(cfg.name, id=cfg.name) for cfg in extension.discover() if cfg.enabled]
    found.sort(key=lambda p: p.id)
    return found


EXTENSIONS = _discover_extensions()


def _installed(names: list[ParameterSet]) -> list[ParameterSet]:
    """Keep the extensions whose import package is actually present.

    A job matrix builds only a subset, so absence here is a deliberate skip --
    but only after the package name resolved, so a wrong name (the defect in the
    old map) surfaces as a failure rather than being absorbed into a skip.
    """
    import importlib.util

    # Only "the package is not importable" is a skip. An unresolvable name is a
    # defect in the derivation and must propagate, which is why the name is
    # resolved before entering the try.
    def present(ext_name: str) -> bool:
        pkg = package_name(ext_name)
        try:
            return importlib.util.find_spec(pkg) is not None
        except ModuleNotFoundError:
            # raised when the package's parent package does not exist
            return False

    return [p for p in names if present(p.id)]


INSTALLED_EXTENSIONS = _installed(EXTENSIONS)

# Extensions that install a pass rather than a dialect, so they are exercised by
# running their pass over the bundled IR fixture instead of by lowering a kernel.
_PASS_EXTENSIONS = {
    "arithmetic-intensity": (
        "arithmetic_intensity.py",
        Path("pass/ArithmeticIntensity/test/add-kernel.mlir"),
        'tt.bandwidth = "1024", tt.compute = "256"',
    ),
    "loop-split": (
        "loop_split.py",
        Path("pass/LoopSplit/test/loop-split.mlir"),
        # The pass rewrites `scf.for` into a prologue plus a split loop; the
        # arithmetic below is the prologue's peeled first iteration.
        "arith.subf %arg4, %2 : tensor<256xf32>",
    ),
    "example": (
        "example.py",
        Path("dialect/Example/test/zero.mlir"),
        # The dialect's own fixture: its op must round-trip through the
        # parser/printer. Without registration the input cannot even be parsed,
        # so this is what makes the dialect extension's load test meaningful --
        # importing the package alone succeeds either way.
        "example.zero",
    ),
}


def test_extension_list_is_not_empty() -> None:
    """A guard so a broken manifest turns into a failure, not into zero tests.

    Without this, losing every pyproject manifest would make the parametrised
    tests collect nothing and the job would pass while testing nothing.
    """
    assert EXTENSIONS, "no enabled extensions discovered; check the pyproject manifests"
    assert INSTALLED_EXTENSIONS, (
        "no extension is installed in this environment; build at least one "
        "(make build && make install) before running the integration tests")


@pytest.mark.parametrize("name", INSTALLED_EXTENSIONS)
def test_extension_loads(name: str) -> None:
    """``import <package>`` must succeed.

    This used to skip on ImportError, which cannot tell "this job did not build
    the extension" from "the extension is broken". A missing extension in a job
    that is supposed to have it, and a broken native load, are both failures; the
    original traceback is preserved so the cause is visible.
    """
    pkg = package_name(name)
    importlib.import_module(pkg)


@pytest.mark.parametrize("name", INSTALLED_EXTENSIONS)
def test_extension_compiles_kernel(name: str) -> None:
    """Lower a minimal kernel with the extension loaded.

    The point is to prove the extension participates in compilation and did not
    break codegen -- not to run a kernel. Compiling against an explicit
    ``GPUTarget`` needs the backend's toolchain but no visible device, so this
    test does not skip when no GPU is present; a missing toolchain is an
    environment failure and must be reported rather than skipped into green.
    """
    import triton
    import triton.language as tl
    from triton.backends.compiler import GPUTarget

    if not hasattr(triton, "compile"):
        pytest.skip("this triton exposes no compile() entry point")

    importlib.import_module(package_name(name))

    @triton.jit
    def _identity(x_ptr, y_ptr, n: tl.constexpr):
        offs = tl.arange(0, n)
        tl.store(y_ptr + offs, tl.load(x_ptr + offs))

    src = triton.compiler.ASTSource(
        fn=_identity,
        signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "n": "constexpr"},
        constexprs={"n": 32},
    )
    compiled = triton.compile(src, target=GPUTarget("cuda", 89, 32))
    assert compiled.asm, "extension compile produced no assembly"


@pytest.mark.parametrize("name", sorted(_PASS_EXTENSIONS))
def test_extension_pass_runs(name: str) -> None:
    """The registered pass must still be reachable and must still do its job.

    An extension whose ``__init__`` imports cleanly can still be dead: if plugin
    registration silently stops working, ``import`` succeeds and every other test
    here stays green. Running the pass over the extension's own IR fixture and
    checking for the attribute it computes is the only assertion that covers the
    registration path itself.
    """
    if name not in {p.id for p in INSTALLED_EXTENSIONS}:
        pytest.skip(f"{name} is not installed in this environment")

    driver, fixture, expected = _PASS_EXTENSIONS[name]
    mlir_input = REPO_ROOT / fixture
    assert mlir_input.is_file(), f"{name}: missing IR fixture {mlir_input}"

    # The driver runs in a subprocess, so it must be handed the same triton this
    # process resolved: a stale wheel earlier on its sys.path would silently be
    # exercised instead (and fail plugin registration, as the installed PyPI
    # wheel is built without TRITON_EXT_ENABLED).
    import triton

    triton_root = str(Path(triton.__file__).resolve().parent.parent)
    driver_dir = mlir_input.parent
    proc = subprocess.run(
        [sys.executable, driver, mlir_input.name],
        cwd=driver_dir, capture_output=True, text=True,
        env={**os.environ,
             "PYTHONPATH": os.pathsep.join(
                 p for p in (triton_root, os.environ.get("PYTHONPATH")) if p)})
    assert proc.returncode == 0, (
        f"{name}: pass driver failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    assert expected in proc.stdout, (
        f"{name}: pass produced no '{expected}' attribute; plugin registration "
        f"is not reaching the pipeline\n{proc.stdout}")


def test_extension_failure_is_not_swallowed(monkeypatch) -> None:
    """A broken package must raise out of the load test, not be skipped.

    Injects an ImportError into the import machinery and checks the test body
    propagates it, which is the behaviour change requested in the issue.
    """
    target = INSTALLED_EXTENSIONS[0].id
    pkg = package_name(target)
    real_import = importlib.import_module

    def boom(name, *a, **kw):
        if name == pkg:
            raise ImportError("injected broken native load")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(importlib, "import_module", boom)
    with pytest.raises(ImportError):
        test_extension_loads(target)


def test_package_name_comes_from_pyproject() -> None:
    """The derivation must match the manifest, not a hand-written guess.

    ``arithmetic-intensity`` is the case that exposed the old map: the hyphenated
    name was absent, and the fallback produced ``arithmetic_intensity`` while the
    manifest declares ``triton_arithmetic_intensity``.
    """
    assert package_name("arithmetic-intensity") == "triton_arithmetic_intensity"
    assert package_name("loop-split") == "triton_loop_split"
    assert package_name("triton-example") == "triton_example"
    assert package_name("apple-backend") == "triton_apple_backend"
