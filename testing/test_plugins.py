"""Plugin integration tests.

Auto-discovers every plugin declared by a `triton-ext.conf` and exercises
its `lib<name>.so` from a fresh Python interpreter with `TRITON_PLUGIN_PATHS`
set. Each plugin runs in its own subprocess so failures isolate cleanly.

Tests:
  - test_plugin_loads[<name>]            -- plugin static-init: `import triton`
                                            succeeds with the .so loaded.
  - test_plugin_compiles_kernel[<name>]  -- end-to-end: JIT-decorate and lower
                                            a basic kernel through the plugin's
                                            pipeline.
  - test_<plugin>_<feature>              -- plugin-specific scenarios, gated
                                            with `@pytest.mark.skipif` on the
                                            plugin's .so existence.

Adding a new plugin: drop a `triton-ext.conf`; both parametrized tests pick
it up. To exempt a plugin from a parametrized test, mark it at parametrize
time with `pytest.param(..., marks=pytest.mark.skip(...))` -- see
`_COMPILE_PLUGINS` for an example.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = Path(os.environ.get("TRITON_EXT_BUILD_DIR", REPO_ROOT / "build"))
PLUGIN_LIB_DIR = BUILD_DIR / "lib"


def _discover_plugins() -> list[pytest.ParameterSet]:
    plugins: list[pytest.ParameterSet] = []
    for conf in REPO_ROOT.rglob("triton-ext.conf"):
        rel_parts = conf.relative_to(REPO_ROOT).parts
        if rel_parts[0].startswith(("triton-", "llvm-", "build")):
            continue
        text = conf.read_text().strip()
        if not text:
            continue
        # Format is `name;status[;hash]` (CMake list); we only need the name.
        name = text.split(";", 1)[0].strip()
        if not name:
            continue
        plugins.append(pytest.param(name, id=name))
    plugins.sort(key=lambda p: p.id)
    return plugins


PLUGINS = _discover_plugins()


def _plugin_path(name: str) -> Path:
    return PLUGIN_LIB_DIR / f"lib{name}.so"


def _run_with_plugin(plugin_path: Path, script: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "TRITON_PLUGIN_PATHS": str(plugin_path)}
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


# ---------------------------------------------------------------------------
# Generic per-plugin tests (auto-discovered)
# ---------------------------------------------------------------------------


def test_plugins_discovered() -> None:
    """Guard against silently testing nothing if discovery breaks."""
    assert PLUGINS, f"No triton-ext.conf files found under {REPO_ROOT}"


@pytest.mark.parametrize("name", PLUGINS)
def test_plugin_loads(name: str) -> None:
    """Smoke: `import triton` succeeds with the plugin loaded."""
    path = _plugin_path(name)
    if not path.is_file():
        pytest.skip(f"Plugin not built at {path} (extension may be disabled)")
    result = _run_with_plugin(path, "import triton")
    assert result.returncode == 0, (
        f"Loading plugin {name} from {path} failed:\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )


# example dialect is scaffolding-only -- its Dialect::initialize() doesn't
# register StringAttr, so kernel compile aborts with an LLVM storage-uniquer
# error. Tag it as skip at parametrize time.
_COMPILE_PLUGINS = [
    pytest.param(
        p.values[0], marks=pytest.mark.skip(reason="scaffolding-only dialect"), id=p.id
    )
    if p.id == "example"
    else p
    for p in PLUGINS
]


@pytest.mark.parametrize("name", _COMPILE_PLUGINS)
def test_plugin_compiles_kernel(name: str) -> None:
    """User scenario: with the plugin loaded, JIT-decorate and lower a basic kernel."""
    path = _plugin_path(name)
    if not path.is_file():
        pytest.skip(f"Plugin not built at {path} (extension may be disabled)")
    script = """
        import sys
        import triton
        import triton.language as tl

        @triton.jit
        def kernel(in_ptr, out_ptr, BLOCK: tl.constexpr):
            offs = tl.arange(0, BLOCK)
            tl.store(out_ptr + offs, tl.load(in_ptr + offs))

        try:
            target = triton.runtime.driver.active.get_current_target()
        except Exception as e:
            print(f"No target ({type(e).__name__}: {e}); skipping compile.")
            sys.exit(0)
        src = triton.compiler.ASTSource(
            fn=kernel,
            signature={"in_ptr": "*fp32", "out_ptr": "*fp32"},
            constexprs={"BLOCK": 128},
        )
        triton.compile(src, target=target)
        """
    result = _run_with_plugin(path, script)
    assert result.returncode == 0, (
        f"Plugin {name} broke kernel compile:\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# Plugin-specific tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _plugin_path("utlx").is_file(), reason="utlx plugin not built")
def test_utlx_registers_tlx_dsl() -> None:
    """utlx registers `triton.language.extra.tlx` with local_alloc/view/store/load.

    The Python namespace is set up by `extensions/utlx/python/utlx_plugin/__init__.py`
    when imported -- it inserts itself into `sys.modules` as
    `triton.language.extra.tlx`. Loading the .so alone is not enough.
    """
    plugin_path = _plugin_path("utlx")
    utlx_python = REPO_ROOT / "extensions" / "utlx" / "python"
    env = {
        **os.environ,
        "TRITON_PLUGIN_PATHS": str(plugin_path),
        "PYTHONPATH": f"{utlx_python}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
    }
    script = """
        import triton  # noqa: F401
        import utlx_plugin  # noqa: F401  (registers triton.language.extra.tlx)
        from triton.language.extra import tlx
        for n in ("local_alloc", "local_view", "local_store", "local_load"):
            assert hasattr(tlx, n), f"missing tlx.{n}"
        """
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"utlx tlx-DSL check failed:\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
