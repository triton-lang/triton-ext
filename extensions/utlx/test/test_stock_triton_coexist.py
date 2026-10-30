"""Stock Triton kernels in a process that has uTLX loaded.

conftest imports utlx_plugin, which routes every Triton kernel through uTLX's
TTIR->TTGIR conversion, so that conversion has to handle what Triton's own one
does.
"""
import torch

import triton
import triton.language as tl

from conftest import DEVICE


@triton.jit(noinline=True)
def _load_block(ptr, BLOCK: tl.constexpr):
    return tl.load(ptr + tl.arange(0, BLOCK))


@triton.jit
def _add_through_noinline(x_ptr, y_ptr, z_ptr, BLOCK: tl.constexpr):
    # tt.func, tt.call and tt.return all carry the returned tensor's type.
    z = _load_block(x_ptr, BLOCK) + _load_block(y_ptr, BLOCK)
    tl.store(z_ptr + tl.arange(0, BLOCK), z)


def test_noinline_function_returning_tensor():
    x = torch.randn(128, device=DEVICE)
    y = torch.randn(128, device=DEVICE)
    z = torch.empty_like(x)
    _add_through_noinline[(1, )](x, y, z, BLOCK=128, num_warps=1)
    torch.testing.assert_close(z, x + y)


@triton.jit
def _cat_can_reorder(x_ptr, y_ptr, z_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    z = tl.cat(tl.load(x_ptr + offs), tl.load(y_ptr + offs), can_reorder=True)
    tl.store(z_ptr + tl.arange(0, 2 * N), z)


def test_cat_can_reorder():
    x = torch.arange(128, dtype=torch.int32, device=DEVICE)
    y = torch.arange(128, 256, dtype=torch.int32, device=DEVICE)
    z = torch.empty(256, dtype=torch.int32, device=DEVICE)
    _cat_can_reorder[(1, )](x, y, z, N=128, num_warps=4)
    # can_reorder=True leaves the order of the result unspecified.
    torch.testing.assert_close(torch.sort(z).values, torch.cat([x, y]))
