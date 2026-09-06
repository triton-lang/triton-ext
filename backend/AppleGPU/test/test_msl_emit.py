"""Asserts on the MSL emitted for checked-in TTGIR fixtures.

Run: pytest backend/AppleGPU/test/test_msl_emit.py
Needs TRITON_PLUGIN_PATHS pointing at the built libapplegpu_backend dylib.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent
DRIVER = HERE / "msl_driver.py"
MSL = HERE / "msl"


def emit(fixture: str) -> str:
    src = MSL / fixture
    if not src.exists():
        pytest.skip(f"missing fixture {src}")
    proc = subprocess.run(
        [sys.executable, str(DRIVER), str(src)],
        capture_output=True,
        text=True)
    if proc.returncode != 0:
        if "plugin not loaded" in proc.stderr:
            pytest.skip("AppleGPU plugin not loaded (set TRITON_PLUGIN_PATHS)")
        pytest.fail(f"msl_driver failed:\n{proc.stderr}")
    return proc.stdout


@pytest.fixture(scope="module")
def packed16() -> str:
    return emit("atomic_packed16.mlir")


def test_packed16_emits_cas_loop(packed16: str) -> None:
    assert "__agpu_atomic_rmw_packed16" in packed16
    assert "atomic_compare_exchange_weak_explicit" in packed16


def test_packed16_instantiates_the_narrowing_helper(packed16: str) -> None:
    assert "__agpu_atomic_rmw_packed16<half>" in packed16
    assert "__agpu_narrow16<half>" in packed16
    assert "__agpu_rtne_int_half" in packed16


def test_packed16_passes_the_add_opcode(packed16: str) -> None:
    calls = [
        ln for ln in packed16.splitlines()
        if "__agpu_atomic_rmw_packed16<" in ln and "inline" not in ln
    ]
    assert calls, "no packed16 call sites emitted"
    assert all(ln.rstrip().endswith(", 0);") for ln in calls), calls


def test_no_raw_ast_nodes(packed16: str) -> None:
    assert "ulong2(0, 0)" not in packed16
