"""Integration test: every plugin registered in this repo loads into Python.

`make test` otherwise runs only lit/FileCheck tests, which exercise MLIR passes
but never load the plugin shared libraries into Python. That hid a regression
in the plugin's static init path (PluginInfo::tritonVersion left null) for five
Triton-pin bumps.

For each `triton-ext.conf` in the source tree we resolve the corresponding
`lib<name>.so` in the build dir and, in a fresh interpreter, run
`import triton` with `TRITON_PLUGIN_PATHS` pointed at the .so. Each plugin
runs in its own subprocess so a failure isolates to a single plugin.

This catches PluginInfo-level regressions on Triton builds that load plugins
eagerly at import time — the source-built pin used in CI does. Older release
wheels load lazily and may pass even for a broken plugin; CI is the canonical
environment for this signal.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = Path(os.environ.get("TRITON_EXT_BUILD_DIR", REPO_ROOT / "build"))
PLUGIN_LIB_DIR = BUILD_DIR / "lib"


def _discover_plugins() -> list[pytest.ParameterSet]:
    plugins: list[pytest.ParameterSet] = []
    for conf in REPO_ROOT.rglob("triton-ext.conf"):
        # Skip downloaded artifact trees (triton-<hash>-..., llvm-<hash>-...) and
        # the build dir itself.
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


def test_plugins_discovered() -> None:
    """Guard against silently testing nothing if discovery breaks."""
    assert PLUGINS, f"No triton-ext.conf files found under {REPO_ROOT}"


@pytest.mark.parametrize("name", PLUGINS)
def test_plugin_loads(name: str) -> None:
    plugin_path = PLUGIN_LIB_DIR / f"lib{name}.so"
    if not plugin_path.is_file():
        pytest.skip(
            f"Plugin not built at {plugin_path} (extension may be disabled)")

    env = {**os.environ, "TRITON_PLUGIN_PATHS": str(plugin_path)}
    result = subprocess.run(
        [sys.executable, "-c", "import triton"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"Loading plugin {name} from {plugin_path} failed:\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}")
