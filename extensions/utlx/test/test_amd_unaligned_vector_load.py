"""Unaligned global vector loads on gfx942/gfx950.

These targets execute vector loads from addresses aligned only to the element
size. uTLX's utlx_widen_unaligned_loads and utlx_widen_unaligned_buffer_loads
passes let a load use the full contiguity axis analysis proves, where stock
Triton clamps it to the proven base alignment.
"""

import re

import pytest
import torch
import triton
import triton.language as tl

from conftest import DEVICE, is_hip

BLOCK = 2048


def _is_unaligned_vector_load_target():
    if not is_hip():
        return False
    try:
        return triton.runtime.driver.active.get_current_target().arch in (
            "gfx942", "gfx950")
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _is_unaligned_vector_load_target(),
                                reason="Requires gfx942 or gfx950")


def _buffer_load_contiguity(compiled):
    loads = [
        line for line in compiled.asm["ttgir"].splitlines()
        if "amdg.buffer_load " in line
    ]
    assert loads, "expected a buffer_load"
    return [
        int(m.group(1)) if (m := re.search(r"contiguity = (\d+)", line)) else 1
        for line in loads
    ]


@triton.jit
def _offset_copy(x_ptr, y_ptr, OFFSET: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    tl.store(y_ptr + offsets, tl.load(x_ptr + OFFSET + offsets))


@pytest.mark.parametrize("offset", [1, 2, 3, 7])
def test_plain_unaligned_load(offset):
    x = torch.arange(BLOCK + 8, device=DEVICE, dtype=torch.float16)
    y = torch.empty(BLOCK, device=DEVICE, dtype=torch.float16)
    compiled = _offset_copy[(1, )](x,
                                   y,
                                   OFFSET=offset,
                                   BLOCK_SIZE=BLOCK,
                                   num_warps=4)
    torch.testing.assert_close(y, x[offset:offset + BLOCK], atol=0, rtol=0)
    assert _buffer_load_contiguity(compiled) == [8]
    assert "buffer_load_dwordx4" in compiled.asm["amdgcn"]


@triton.jit
def _masked_offset_copy(x_ptr, y_ptr, OFFSET: tl.constexpr,
                        BLOCK_SIZE: tl.constexpr, VALID: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < VALID
    tl.store(y_ptr + offsets,
             tl.load(x_ptr + OFFSET + offsets, mask=mask, other=0))


@pytest.mark.parametrize("offset", [1, 2, 3, 7])
def test_masked_unaligned_load(offset):
    """A mask constant over 8-element groups permits 16-byte loads."""
    valid = BLOCK - 8
    x = torch.arange(BLOCK + 1, device=DEVICE, dtype=torch.float16)
    y = torch.empty(BLOCK, device=DEVICE, dtype=torch.float16)
    compiled = _masked_offset_copy[(1, )](x,
                                          y,
                                          OFFSET=offset,
                                          BLOCK_SIZE=BLOCK,
                                          VALID=valid,
                                          num_warps=4)
    expected = torch.zeros_like(y)
    expected[:valid] = x[offset:offset + valid]
    torch.testing.assert_close(y, expected, atol=0, rtol=0)
    assert _buffer_load_contiguity(compiled) == [8]
    assert "buffer_load_dwordx4" in compiled.asm["amdgcn"]


@triton.jit
def _odd_mask_copy(x_ptr, y_ptr, OFFSET: tl.constexpr,
                   BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets % 2 == 0
    tl.store(y_ptr + offsets,
             tl.load(x_ptr + OFFSET + offsets, mask=mask, other=0))


def test_fine_grained_mask_stays_narrow():
    """Every element in one vector must share a predicate."""
    x = torch.arange(BLOCK + 1, device=DEVICE, dtype=torch.float16)
    y = torch.empty(BLOCK, device=DEVICE, dtype=torch.float16)
    compiled = _odd_mask_copy[(1, )](x,
                                     y,
                                     OFFSET=1,
                                     BLOCK_SIZE=BLOCK,
                                     num_warps=4)
    expected = x[1:1 + BLOCK].clone()
    expected[1::2] = 0
    torch.testing.assert_close(y, expected, atol=0, rtol=0)
    assert _buffer_load_contiguity(compiled) == [1]


@triton.jit
def _offset_copy_both(x_ptr, y_ptr, OFFSET: tl.constexpr,
                      BLOCK_SIZE: tl.constexpr):
    offsets = OFFSET + tl.arange(0, BLOCK_SIZE)
    tl.store(y_ptr + offsets, tl.load(x_ptr + offsets))


def test_unaligned_load_with_unaligned_store():
    """With no aligned access in its slice, only the load's own relaxed width
    can widen its layout; the store keeps its aligned width."""
    x = torch.arange(BLOCK + 8, device=DEVICE, dtype=torch.float16)
    y = torch.zeros(BLOCK + 8, device=DEVICE, dtype=torch.float16)
    compiled = _offset_copy_both[(1, )](x,
                                        y,
                                        OFFSET=1,
                                        BLOCK_SIZE=BLOCK,
                                        num_warps=4)
    expected = torch.zeros_like(y)
    expected[1:1 + BLOCK] = x[1:1 + BLOCK]
    torch.testing.assert_close(y, expected, atol=0, rtol=0)
    assert _buffer_load_contiguity(compiled) == [8]
    assert "buffer_load_dwordx4" in compiled.asm["amdgcn"]


@triton.jit
def _row_copy(x_ptr, y_ptr, STRIDE: tl.constexpr, M: tl.constexpr,
              N: tl.constexpr):
    rows = tl.arange(0, M)[:, None]
    cols = tl.arange(0, N)[None, :]
    tl.store(y_ptr + rows * N + cols, tl.load(x_ptr + rows * STRIDE + cols))


def test_unaligned_row_stride():
    """An odd row stride leaves each f16 row only 2-byte aligned."""
    M, N, stride = 64, 64, 69
    x = torch.randn((M, stride), device=DEVICE, dtype=torch.float16)
    y = torch.empty((M, N), device=DEVICE, dtype=torch.float16)
    compiled = _row_copy[(1, )](x, y, STRIDE=stride, M=M, N=N, num_warps=4)
    torch.testing.assert_close(y, x[:, :N], atol=0, rtol=0)
    assert _buffer_load_contiguity(compiled) == [8]
