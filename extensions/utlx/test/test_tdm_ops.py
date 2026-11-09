"""GPU tests for uTLX AMD TDM (Tensor Data Mover) ops.

Covers the TLX spellings of upstream Triton's Gluon cdna5.tdm builtins
(tlx.tdm_async_load / tdm_async_store / tdm_async_gather / tdm_async_scatter /
tdm_prefetch / tdm_async_wait / tdm_update_tensor_descriptor), which lower
through the portable utlx_async_tdm_* plugin ops to amdg.* MLIR ops.
TDM requires a CDNA5 (gfx1250) target; every test skips elsewhere.
"""

import pytest
import torch

import triton
import triton.language as tl
from conftest import tlx, DEVICE, is_hip_cdna5

pytestmark = pytest.mark.skipif(not is_hip_cdna5(),
                                reason="TDM needs CDNA5 (gfx1250)")

M, N = 32, 128
BM, BN = 16, 64

# ---------------------------------------------------------------------------
# async_load -> local_load -> store, then async_store back
# ---------------------------------------------------------------------------


def test_tdm_load_store_round_trip():

    @triton.jit
    def kernel(in_ptr, out_ptr):
        desc = tlx.make_tensor_descriptor(None, in_ptr, [M, N], [N, 1],
                                          [BM, BN])
        buf = tlx.local_alloc((BM, BN), tl.float16, 1)
        tlx.tdm_async_load(desc, buf[0], offsets=[0, 0])
        tlx.tdm_async_wait(0)
        x = tlx.local_load(buf[0])
        offs = tl.arange(0, BM)[:, None] * N + tl.arange(0, BN)[None, :]
        tl.store(out_ptr + offs, x)

    x = torch.randn((M, N), device=DEVICE, dtype=torch.float16)
    out = torch.empty((BM, BN), device=DEVICE, dtype=torch.float16)
    kernel[(1, )](x, out)
    torch.testing.assert_close(out, x[:BM, :BN])


def test_tdm_async_store_round_trip():

    @triton.jit
    def kernel(in_ptr, out_ptr):
        desc_in = tlx.make_tensor_descriptor(None, in_ptr, [M, N], [N, 1],
                                             [BM, BN])
        desc_out = tlx.make_tensor_descriptor(None, out_ptr, [M, N], [N, 1],
                                              [BM, BN])
        buf = tlx.local_alloc((BM, BN), tl.float16, 1)
        tlx.tdm_async_load(desc_in, buf[0], offsets=[16, 64])
        tlx.tdm_async_wait(0)
        tlx.tdm_async_store(desc_out, buf[0], offsets=[0, 0])
        tlx.tdm_async_wait(0)

    x = torch.randn((M, N), device=DEVICE, dtype=torch.float16)
    out = torch.zeros((M, N), device=DEVICE, dtype=torch.float16)
    kernel[(1, )](x, out)
    torch.testing.assert_close(out[:BM, :BN], x[16:16 + BM, 64:64 + BN])


# ---------------------------------------------------------------------------
# update_tensor_descriptor + prefetch
# ---------------------------------------------------------------------------


def test_tdm_update_and_prefetch():

    @triton.jit
    def kernel(in_ptr, out_ptr):
        desc = tlx.make_tensor_descriptor(None, in_ptr, [64, 64], [64, 1],
                                          [32, 64])
        tlx.tdm_prefetch(desc, [32, 0])
        desc2 = tlx.tdm_update_tensor_descriptor(desc,
                                                 add_offsets=[32, 0],
                                                 clamp_bounds=True)
        buf = tlx.local_alloc((32, 64), tl.float16, 1)
        tlx.tdm_async_load(desc2, buf[0])
        tlx.tdm_async_wait(0)
        x = tlx.local_load(buf[0])
        offs = tl.arange(0, 32)[:, None] * 64 + tl.arange(0, 64)[None, :]
        tl.store(out_ptr + offs, x)

    x = torch.randn((64, 64), device=DEVICE, dtype=torch.float16)
    out = torch.empty((32, 64), device=DEVICE, dtype=torch.float16)
    kernel[(1, )](x, out)
    torch.testing.assert_close(out, x[32:, :])


# ---------------------------------------------------------------------------
# gather / scatter
# ---------------------------------------------------------------------------


def test_tdm_gather_scatter():

    @triton.jit
    def kernel(in_ptr, out_ptr):
        desc_in = tlx.make_tensor_descriptor(None, in_ptr, [M, N], [N, 1],
                                             [BM, BN])
        desc_out = tlx.make_tensor_descriptor(None, out_ptr, [M, N], [N, 1],
                                              [BM, BN])
        rows = (BM - 1) - tl.arange(0, BM)
        buf = tlx.local_alloc((BM, BN), tl.float16, 1)
        tlx.tdm_async_gather(desc_in, rows, buf[0])
        tlx.tdm_async_wait(0)
        tlx.tdm_async_scatter(desc_out, rows, buf[0])
        tlx.tdm_async_wait(0)

    x = torch.randn((M, N), device=DEVICE, dtype=torch.float16)
    out = torch.zeros((M, N), device=DEVICE, dtype=torch.float16)
    kernel[(1, )](x, out)
    # Gather reversed rows then scatter them back through the same reversed
    # index set restores the original tile.
    torch.testing.assert_close(out[:BM, :BN], x[:BM, :BN])


# ---------------------------------------------------------------------------
# fused load
# ---------------------------------------------------------------------------


def test_tdm_load_fused():

    @triton.jit
    def kernel(in_ptr, out_ptr):
        desc = tlx.make_tensor_descriptor(None, in_ptr, [M, N], [N, 1],
                                          [BM, BN])
        buf_a = tlx.local_alloc((BM, BN), tl.float16, 1)
        buf_b = tlx.local_alloc((BM, BN), tl.float16, 1)
        desc_b = tlx.tdm_update_tensor_descriptor(desc, add_offsets=[16, 0])
        tlx.tdm_async_load_fused([(desc, buf_a[0], 0b0011),
                                  (desc_b, buf_b[0], 0b1100)])
        tlx.tdm_async_wait(0)
        x = tlx.local_load(buf_a[0]) + tlx.local_load(buf_b[0])
        offs = tl.arange(0, BM)[:, None] * N + tl.arange(0, BN)[None, :]
        tl.store(out_ptr + offs, x)

    x = torch.randn((M, N), device=DEVICE, dtype=torch.float16)
    out = torch.empty((BM, BN), device=DEVICE, dtype=torch.float16)
    kernel[(1, )](x, out, num_warps=4)
    torch.testing.assert_close(out, x[:BM, :BN] + x[16:16 + BM, :BN])


# ---------------------------------------------------------------------------
# cache_policy / scope attributes reach the IR
# ---------------------------------------------------------------------------


def test_tdm_cache_policy_and_scope_in_ir():

    @triton.jit
    def kernel(in_ptr):
        desc = tlx.make_tensor_descriptor(None, in_ptr, [M, N], [N, 1],
                                          [BM, BN])
        buf = tlx.local_alloc((BM, BN), tl.float16, 1)
        tlx.tdm_async_load(desc,
                           buf[0],
                           offsets=[0, 0],
                           scope="gpu",
                           cache_policy="high")
        tlx.tdm_async_wait(0)

    x = torch.randn((M, N), device=DEVICE, dtype=torch.float16)
    compiled = kernel[(1, )](x)
    ttgir = compiled.asm["ttgir"]
    assert "amdg.async_tdm_copy_global_to_local" in ttgir
    assert "cachePolicy" in ttgir
    assert "scope" in ttgir


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
