"""KernelAbi.h derives scalar offsets from the bit width; driver.py packs the
same buffer from the `_SCALAR_PACK_INFO` table. These pin the two together.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

# Loaded by path to avoid `import triton_apple_backend.driver`, which runs
# Triton's backend discovery and needs a built plugin. The ABI tables are
# plain data and need none of that.
_PKG = Path(__file__).resolve().parents[1] / "python" / "triton_apple_backend"


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pytest.importorskip("torch", reason="driver.py imports torch at module scope")

driver = _load_by_path("_agpu_driver_abi", _PKG / "driver.py")
hw_constants = _load_by_path("_agpu_hw_constants", _PKG / "hw_constants.py")

_SCALAR_PACK_INFO = driver._SCALAR_PACK_INFO
_TY_TO_CPP = driver._TY_TO_CPP
_compute_scalar_layout = driver._compute_scalar_layout
ty_to_cpp = driver.ty_to_cpp

# What the C++ side derives from the MLIR type. i1 is the documented special
# case (1 bit, stored as 1 byte); fp64 occupies 8 bytes though MSL reads it as
# a 4-byte float, so the cursor must still advance by 8.
_BITS = {
    "i1": 1,
    "u1": 1,
    "i8": 8,
    "u8": 8,
    "i16": 16,
    "u16": 16,
    "fp16": 16,
    "bf16": 16,
    "i32": 32,
    "u32": 32,
    "fp32": 32,
    "i64": 64,
    "u64": 64,
    "fp64": 64,
}


def _cxx_size(ty: str) -> int:
    bits = _BITS[ty]
    return 1 if bits == 1 else bits // 8


@pytest.mark.parametrize("ty", sorted(_SCALAR_PACK_INFO))
def test_size_matches_the_cxx_bitwidth_rule(ty):
    _, size, _ = _SCALAR_PACK_INFO[ty]
    assert size == _cxx_size(ty)


@pytest.mark.parametrize("ty", sorted(_SCALAR_PACK_INFO))
def test_alignment_equals_size(ty):
    _, size, align = _SCALAR_PACK_INFO[ty]
    assert align == size


def test_both_tables_cover_the_same_types():
    assert set(_TY_TO_CPP) == set(_SCALAR_PACK_INFO)


@pytest.mark.parametrize("ty", sorted(_SCALAR_PACK_INFO))
def test_every_packable_type_has_a_cpp_name(ty):
    assert ty_to_cpp(ty)


def test_ty_to_cpp_names_the_drift_it_hit():
    with pytest.raises(KeyError, match="SCALAR_PACK_INFO"):
        ty_to_cpp("fp8e4nv")


def test_offsets_follow_natural_alignment():
    # i8 at 0, then i32 must skip to 4, i64 to 8 and the total rounds nothing
    # up past the last member.
    total, offsets = _compute_scalar_layout(["i8", "i32", "i64"])
    assert offsets == [0, 4, 8]
    assert total == 16


def test_fp64_occupies_eight_bytes():
    total, offsets = _compute_scalar_layout(["fp64", "i8"])
    assert offsets == [0, 8]
    assert total == 9


def _header_constant(name: str) -> int:
    hdr = (Path(__file__).resolve().parents[1] / "agpu" / "include" / "agpu" /
           "core" / "Units.h").read_text()
    m = re.search(rf"{name}\s*=\s*(\d+)", hdr)
    assert m, f"{name} not found in Units.h"
    return int(m.group(1))


@pytest.mark.parametrize("py_name,cxx_name", [
    ("WARP_SIZE", "kWarpSize"),
    ("SG_FRAG_DIM", "kSgFragDim"),
    ("TG_BUDGET_BYTES", "kTGResidentBudgetBytes"),
])
def test_hardware_constants_match_the_cxx_owner(py_name, cxx_name):
    assert getattr(hw_constants, py_name) == _header_constant(cxx_name)
