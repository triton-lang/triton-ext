"""Explicit register layouts through ordinary Triton ops."""

import pytest
import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

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
