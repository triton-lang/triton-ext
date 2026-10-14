"""Tests for uTLX's Hopper ``async_dot_wait``.

The accumulator of a pipelined wgmma loop leaves the loop through
``tlx.release_layout``, i.e. with no encoding, so the ``async_dot_wait(0, acc)``
after the loop waits on an unencoded tensor. TLXConvertTritonToTritonGPU must
convert that ``ttng.warp_group_dot_wait`` like any other op; leaving it as is
failed with "failed to legalize unresolved materialization".
"""

import re

import pytest
import torch

import triton
import triton.language as tl
from conftest import tlx, DEVICE, is_hopper


@triton.jit
def _pipelined_mm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * BLOCK_N + offs_n[None, :]

    a_bufs = tlx.local_alloc((BLOCK_M, BLOCK_K), tlx.dtype_of(a_ptr), 2)
    b_bufs = tlx.local_alloc((BLOCK_K, BLOCK_N), tlx.dtype_of(b_ptr), 2)

    tlx.local_store(tlx.local_view(a_bufs, 0), tl.load(a_ptrs))
    tlx.local_store(tlx.local_view(b_bufs, 0), tl.load(b_ptrs))
    tlx.fence_async_shared()
    acc = tlx.async_dot(tlx.local_view(a_bufs, 0), tlx.local_view(b_bufs, 0))

    for k in range(1, K // BLOCK_K):
        # The dot that last read this buffer finished at the previous wait(1).
        buf = k % 2
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * BLOCK_N
        tlx.local_store(tlx.local_view(a_bufs, buf), tl.load(a_ptrs))
        tlx.local_store(tlx.local_view(b_bufs, buf), tl.load(b_ptrs))
        tlx.fence_async_shared()
        acc = tlx.async_dot(tlx.local_view(a_bufs, buf),
                            tlx.local_view(b_bufs, buf), acc)
        acc = tlx.async_dot_wait(1, acc)

    # Waits on the loop-carried accumulator, which has no encoding here.
    acc = tlx.async_dot_wait(0, acc)

    c_ptrs = c_ptr + offs_m[:, None] * BLOCK_N + offs_n[None, :]
    tl.store(c_ptrs, acc.to(tlx.dtype_of(c_ptr)))


@pytest.mark.skipif(not is_hopper(), reason="Requires Hopper GPU")
def test_async_dot_wait_on_loop_result():
    M, N, K = 64, 64, 256
    torch.manual_seed(0)
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    c = torch.empty((M, N), device=DEVICE, dtype=torch.float16)

    kernel = _pipelined_mm_kernel[(1, )](a,
                                         b,
                                         c,
                                         K=K,
                                         BLOCK_M=M,
                                         BLOCK_N=N,
                                         BLOCK_K=32)

    torch.testing.assert_close(c, torch.matmul(a, b), atol=1e-2, rtol=1e-2)
    # The wait after the loop sees the accumulator in the wgmma layout, not
    # converted out of it while the last dot is still in flight.
    ttgir = kernel.asm["ttgir"]
    assert re.search(
        r"ttng\.warp_group_dot_wait %[\w#]+ \{pendings = 0 : i32\} : "
        r"tensor<64x64xf32, #mma>", ttgir), ttgir
