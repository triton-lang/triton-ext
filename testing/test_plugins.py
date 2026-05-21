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

The kernel-compile and tlx-DSL scenarios live as standalone scripts under
`testing/scripts/` so they can be run by hand to debug a plugin, e.g.
`TRITON_PLUGIN_PATHS=build/lib/lib<name>.so python testing/scripts/compile_kernel.py`.
On failure each test prints the exact command to reproduce it.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = Path(os.environ.get("TRITON_EXT_BUILD_DIR", REPO_ROOT / "build"))
PLUGIN_LIB_DIR = BUILD_DIR / "lib"
SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"


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


def _format_command(env_overrides: dict[str, str], args: list[str]) -> str:
    """Render a copy-pasteable shell command for manual debugging."""
    prefix = " ".join(f"{k}={shlex.quote(v)}"
                      for k, v in env_overrides.items())
    cmd = " ".join(shlex.quote(a) for a in args)
    return f"{prefix} {cmd}".strip()


def _run(env_overrides: dict[str, str],
         args: list[str]) -> tuple[subprocess.CompletedProcess, str]:
    """Run a subprocess and return it along with its debug command string."""
    env = {**os.environ, **env_overrides}
    command = _format_command(env_overrides, args)
    result = subprocess.run(
        args,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return result, command


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
    result, command = _run({"TRITON_PLUGIN_PATHS": str(path)},
                           [sys.executable, "-c", "import triton"])
    assert result.returncode == 0, (
        f"Loading plugin {name} failed. Reproduce with:\n  {command}\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}")


# example dialect is scaffolding-only -- its Dialect::initialize() doesn't
# register StringAttr, so kernel compile aborts with an LLVM storage-uniquer
# error. Tag it as skip at parametrize time.
_COMPILE_PLUGINS = [
    pytest.param(p.values[0],
                 marks=pytest.mark.skip(reason="scaffolding-only dialect"),
                 id=p.id) if p.id == "example" else p for p in PLUGINS
]


@pytest.mark.parametrize("name", _COMPILE_PLUGINS)
def test_plugin_compiles_kernel(name: str) -> None:
    """User scenario: with the plugin loaded, JIT-decorate and lower a basic kernel."""
    path = _plugin_path(name)
    if not path.is_file():
        pytest.skip(f"Plugin not built at {path} (extension may be disabled)")
    result, command = _run(
        {"TRITON_PLUGIN_PATHS": str(path)},
        [sys.executable,
         str(SCRIPTS_DIR / "compile_kernel.py")])
    assert result.returncode == 0, (
        f"Plugin {name} broke kernel compile. Reproduce with:\n  {command}\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}")


# ---------------------------------------------------------------------------
# Plugin-specific tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _plugin_path("utlx").is_file(),
                    reason="utlx plugin not built")
def test_utlx_registers_tlx_dsl() -> None:
    """utlx registers `triton.language.extra.tlx` with local_alloc/view/store/load.

    The Python namespace is set up by `extensions/utlx/python/utlx_plugin/__init__.py`
    when imported -- it inserts itself into `sys.modules` as
    `triton.language.extra.tlx`. Loading the .so alone is not enough.
    """
    plugin_path = _plugin_path("utlx")
    utlx_python = REPO_ROOT / "extensions" / "utlx" / "python"
    pythonpath = f"{utlx_python}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"
    result, command = _run(
        {
            "TRITON_PLUGIN_PATHS": str(plugin_path),
            "PYTHONPATH": pythonpath,
        },
        [sys.executable, str(SCRIPTS_DIR / "load_tlx_dsl.py")])
    assert result.returncode == 0, (
        f"utlx tlx-DSL check failed. Reproduce with:\n  {command}\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}")
