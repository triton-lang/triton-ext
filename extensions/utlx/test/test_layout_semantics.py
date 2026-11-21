"""Explicit register layouts through ordinary Triton ops."""

import pytest
import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.runtime.jit import constexpr_function

from conftest import DEVICE, is_hip

pytestmark = pytest.mark.skipif(not is_hip(), reason="Requires AMD")

ROWS, COLS = 16, 64

# [16, 64] over 4 waves of 64 lanes, two consecutive values per thread.
TILE: tl.constexpr = tlx.layout(
    shape=((2, ) * 8, (2, ) * 2),
    stride=((4, 8, 16, 32, 64, 128, 256, 512), (1, 2)),
)


@triton.jit
def _where_mixed_layouts(x_ptr, y_ptr, ROWS: tl.constexpr, COLS: tl.constexpr,
                         LAYOUT: tl.constexpr):
    offs = tl.arange(0, ROWS)[:, None] * COLS + tl.arange(0, COLS)[None, :]
    x = tlx.require_layout(tl.load(x_ptr + offs), LAYOUT)
    best = tl.max(x, axis=1)
    slots = tl.arange(0, COLS)[None, :]
    # A plain condition and a broadcast arm with a layout from the reduction.
    y = tl.where(slots == 0, best[:, None], tl.zeros((ROWS, COLS), tl.float32))
    # A condition and an arm whose layouts differ.
    y = tl.where(x == best[:, None], y + 1, y)
    tl.store(y_ptr + offs, y)


def test_where_mixed_layouts():
    x = torch.randn(ROWS, COLS, device=DEVICE)
    y = torch.empty_like(x)
    _where_mixed_layouts[(1, )](x, y, ROWS, COLS, TILE, num_warps=4)
    best = x.max(dim=1, keepdim=True).values
    expected = torch.zeros_like(x)
    expected[:, :1] = best
    expected += (x == best).float()
    torch.testing.assert_close(y, expected)


@constexpr_function
def _row_slice(parent, rows, cols):
    """TILE with its columns sliced away, supplying the parent's shape."""

    class _Slice(tlx.slice_layout):

        def to_ir(self, builder, shape=None, element_type=None):
            return builder.make_slice_encoding_attr(
                1, self.parent.to_ir(builder, (rows, cols), element_type))

    return _Slice(parent, 1)


@triton.jit
def _expand_sliced(y_ptr, ROWS: tl.constexpr, COLS: tl.constexpr,
                   LAYOUT: tl.constexpr):
    rows = tlx.require_layout(
        tl.arange(0, ROWS).to(tl.float32), _row_slice(LAYOUT, ROWS, COLS))
    y = tl.broadcast_to(tl.expand_dims(rows, 1), (ROWS, COLS))
    offs = tl.arange(0, ROWS)[:, None] * COLS + tl.arange(0, COLS)[None, :]
    tl.store(y_ptr + offs, y)


def test_slice_layout_subclass():
    """tlx.slice_layout is a class kernels can subclass, as in the fork."""
    y = torch.empty(ROWS, COLS, device=DEVICE)
    compiled = _expand_sliced[(1, )](y, ROWS, COLS, TILE, num_warps=4)
    expected = torch.arange(ROWS, device=DEVICE,
                            dtype=torch.float32)[:, None].expand(ROWS, COLS)
    torch.testing.assert_close(y, expected)
    assert "#ttg.slice<{dim = 1" in compiled.asm["ttgir"]
