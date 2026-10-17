"""Gluon kernels in a process that has uTLX loaded.

conftest imports utlx_plugin, which on Triton releases without
TritonSemantic.make_tensor patches tl.tensor process-wide, and installs uTLX's
add_stages hook for every kernel. Gluon kernels carry their own layouts and have
their own ttgir stage, so neither may change them.
"""
import pytest
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from conftest import DEVICE, is_hip_cdna4

pytestmark = pytest.mark.skipif(
    not is_hip_cdna4(),
    reason="the Gluon AMD buffer ops used here need gfx950")


@gluon.jit
def _buffer_copy_2d(x_ptr, y_ptr, M: gl.constexpr, N: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [4, 1], [1, 0])
    # expand_dims of a sliced distributed value, which fails if uTLX re-types
    # Gluon's distributed_type as a plain block type.
    rows = gl.arange(0, M, layout=gl.SliceLayout(1, layout))[:, None]
    cols = gl.arange(0, N, layout=gl.SliceLayout(0, layout))[None, :]
    offsets = rows * N + cols
    # amdg buffer ops, which uTLX's TTIR->TTGIR conversion marks illegal.
    x = gl.amd.cdna4.buffer_load(x_ptr, offsets)
    gl.amd.cdna4.buffer_store(x, y_ptr, offsets)


def test_gluon_kernel_with_utlx_loaded():
    M, N = 32, 64
    x = torch.randn(M, N, device=DEVICE)
    y = torch.empty_like(x)
    _buffer_copy_2d[(1, )](x, y, M, N, num_warps=4)
    torch.testing.assert_close(y, x)
