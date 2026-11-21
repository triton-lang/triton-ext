"""tlx.buffer_load / buffer_store / buffer_load_to_local on AMD.

These lower to the amdg buffer ops directly, as in the fork, so the trusted
``contiguity`` reaches the backend and the offsets keep the register layout
the kernel chose, whatever it is.
"""

import re

import pytest
import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

from conftest import DEVICE, is_hip

pytestmark = pytest.mark.skipif(not is_hip(), reason="Requires AMD")

ROWS, COLS = 16, 256

# [16, 256] bytes over 4 waves of 64: each thread owns 16 consecutive bytes of
# a row (value bits along the row), lanes and waves take the rest.
ROW_RUNS: tl.constexpr = tlx.layout(
    shape=((2, ) * 8, (2, ) * 4),
    stride=((16, 32, 64, 128, 256, 512, 1024, 2048), (1, 2, 4, 8)),
)
# The same tile with each thread's 16 bytes spread over 16 rows instead.
COLUMN_RUNS: tl.constexpr = tlx.layout(
    shape=((2, ) * 8, (2, ) * 4),
    stride=((1, 2, 4, 8, 16, 32, 64, 128), (256, 512, 1024, 2048)),
)
# Every wave repeats wave 0's addresses: a non-injective (generic) layout.
BROADCAST_WAVES: tl.constexpr = tlx.layout(
    shape=((2, ) * 8, (2, ) * 6),
    stride=((16, 32, 64, 128, 256, 512, 0, 0), (1, 2, 4, 8, 1024, 2048)),
)


def _shuffled_offsets():
    # Swapping 16-byte blocks keeps every 16-byte run but hides it from axis
    # analysis, so only the trusted contiguity can vectorize the load.
    r = torch.arange(ROWS, device=DEVICE)[:, None]
    c = torch.arange(COLS, device=DEVICE)[None, :]
    return r * COLS + (c ^ 16)


@triton.jit
def _offsets(ROWS: tl.constexpr, COLS: tl.constexpr):
    r = tl.arange(0, ROWS)[:, None]
    c = tl.arange(0, COLS)[None, :]
    return r * COLS + (c ^ 16), r * COLS + c


@triton.jit
def _shuffled_load(x_ptr, y_ptr, ROWS: tl.constexpr, COLS: tl.constexpr,
                   LAYOUT: tl.constexpr, CONTIGUITY: tl.constexpr):
    src, dst = _offsets(ROWS, COLS)
    x = tlx.buffer_load(x_ptr,
                        tlx.require_layout(src, LAYOUT),
                        contiguity=CONTIGUITY)
    tl.store(y_ptr + dst, x)


def _buffer_ops(compiled, name):
    return [
        line for line in compiled.asm["ttgir"].splitlines()
        if f"amdg.{name} " in line
    ]


def test_buffer_load_trusted_contiguity():
    x = torch.randint(0,
                      255, (ROWS * COLS, ),
                      device=DEVICE,
                      dtype=torch.uint8)
    y = torch.empty_like(x)
    compiled = _shuffled_load[(1, )](x,
                                     y,
                                     ROWS,
                                     COLS,
                                     ROW_RUNS,
                                     16,
                                     num_warps=4)
    torch.testing.assert_close(y, x[_shuffled_offsets().flatten()])
    loads = _buffer_ops(compiled, "buffer_load")
    assert loads and all("contiguity = 16" in line for line in loads), loads
    assert "buffer_load_dwordx4" in compiled.asm["amdgcn"]
    assert "buffer_load_ubyte" not in compiled.asm["amdgcn"]


@triton.jit
def _relaid_load(x_ptr, y_ptr, ROWS: tl.constexpr, COLS: tl.constexpr,
                 LAYOUT: tl.constexpr, OTHER: tl.constexpr):
    src, dst = _offsets(ROWS, COLS)
    x = tlx.buffer_load(x_ptr, tlx.require_layout(src, LAYOUT), contiguity=16)
    tlx.buffer_store(tlx.require_layout(x, OTHER), y_ptr,
                     tlx.require_layout(dst, OTHER))


def test_buffer_load_keeps_its_layout():
    """Layout passes may move the load to the store's layout, where the
    contiguity it was given does not hold; it must stay where it was."""
    x = torch.randint(0,
                      255, (ROWS * COLS, ),
                      device=DEVICE,
                      dtype=torch.uint8)
    y = torch.empty_like(x)
    compiled = _relaid_load[(1, )](x,
                                   y,
                                   ROWS,
                                   COLS,
                                   ROW_RUNS,
                                   COLUMN_RUNS,
                                   num_warps=4)
    torch.testing.assert_close(y, x[_shuffled_offsets().flatten()])
    for line in _buffer_ops(compiled, "buffer_load"):
        pinned = re.search(r"utlx\.buffer_layout = (#\w+)", line)
        assert pinned and line.rstrip().split("->")[-1].find(
            pinned.group(1)) >= 0, line


@triton.jit
def _masked_rows(x_ptr, y_ptr, VALID, ROWS: tl.constexpr, COLS: tl.constexpr,
                 LAYOUT: tl.constexpr):
    src, dst = _offsets(ROWS, COLS)
    rows = tl.arange(0, ROWS)[:, None] < VALID
    x = tlx.buffer_load(x_ptr,
                        tlx.require_layout(src, LAYOUT),
                        mask=rows,
                        other=7,
                        contiguity=16)
    # A [ROWS, 1] mask, broadcast to the offsets by buffer_store.
    tlx.buffer_store(x,
                     y_ptr,
                     tlx.require_layout(dst, LAYOUT),
                     mask=rows,
                     contiguity=16)


def test_buffer_ops_broadcast_mask_and_other():
    valid = 5
    x = torch.randint(0,
                      255, (ROWS * COLS, ),
                      device=DEVICE,
                      dtype=torch.uint8)
    y = torch.zeros_like(x)
    _masked_rows[(1, )](x, y, valid, ROWS, COLS, ROW_RUNS, num_warps=4)
    expected = x[_shuffled_offsets().flatten()].view(ROWS, COLS).clone()
    expected[valid:] = 0
    torch.testing.assert_close(y.view(ROWS, COLS), expected)


@triton.jit
def _load_to_local(x_ptr, y_ptr, ROWS: tl.constexpr, COLS: tl.constexpr,
                   LAYOUT: tl.constexpr):
    src, dst = _offsets(ROWS, COLS)
    buf = tlx.local_alloc((ROWS, COLS), tl.uint8, 1)
    tlx.buffer_load_to_local(buf[0],
                             x_ptr,
                             tlx.require_layout(src, LAYOUT),
                             contiguity=16)
    tlx.async_load_commit_group()
    tlx.async_load_wait_group(0)
    tl.store(y_ptr + dst, tlx.local_load(buf[0]))


@pytest.mark.parametrize("layout", ["ROW_RUNS", "BROADCAST_WAVES"])
def test_buffer_load_to_local(layout):
    """Any register layout is accepted, including a non-injective one, which
    ttg.async_copy_global_to_local's verifier rejects."""
    if triton.runtime.driver.active.get_current_target().arch != "gfx950":
        pytest.skip("direct-to-LDS 128-bit loads need gfx950")
    x = torch.randint(0,
                      255, (ROWS * COLS, ),
                      device=DEVICE,
                      dtype=torch.uint8)
    y = torch.empty_like(x)
    compiled = _load_to_local[(1, )](x,
                                     y,
                                     ROWS,
                                     COLS,
                                     globals()[layout],
                                     num_warps=4)
    torch.testing.assert_close(y, x[_shuffled_offsets().flatten()])
    assert _buffer_ops(compiled, "buffer_load_to_local")
