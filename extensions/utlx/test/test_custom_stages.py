"""Tests for uTLX custom compilation stages.

Covers:
  - inspect_stages_hook key/hash generation
  - AMD ttgir stage replacement (utlx_convert_triton_to_tritongpu)
  - Layout propagation pass integration (utlx_insert_and_propagate_layout)
"""

import pytest
import torch

import triton
import triton.language as tl
from conftest import tlx, DEVICE, is_hip
from utlx_plugin import custom_stages

# ---------------------------------------------------------------------------
# inspect_stages_hook: key/hash generation
# ---------------------------------------------------------------------------


def test_inspect_stages_hook_returns_key_hash():
    """When called with no args, returns (key, hash) tuple."""
    key, hash_val = custom_stages.inspect_stages_hook()
    assert isinstance(key, str)
    assert isinstance(hash_val, str)
    assert len(hash_val) == 64  # SHA-256 hex digest


def test_inspect_stages_hook_key_is_file_content():
    """Key should be the source code of custom_stages.py."""
    import pathlib
    key, _ = custom_stages.inspect_stages_hook()
    expected = pathlib.Path(custom_stages.__file__).read_text()
    assert key == expected


def test_inspect_stages_hook_hash_deterministic():
    """Hash should be deterministic across calls."""
    _, h1 = custom_stages.inspect_stages_hook()
    _, h2 = custom_stages.inspect_stages_hook()
    assert h1 == h2


# ---------------------------------------------------------------------------
# Full compilation with custom stages: AMD GEMM
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not is_hip(), reason="AMD-specific test")
def test_amd_custom_stages_gemm():
    """Verify full compilation pipeline works with uTLX custom stages on AMD."""

    @triton.jit
    def gemm_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        offs_am = tl.arange(0, BLOCK_M)
        offs_bn = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + (offs_am[:, None] * stride_am +
                          offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk +
                          offs_bn[None, :] * stride_bn)

        bufs_A = tlx.local_alloc((BLOCK_M, BLOCK_K), tlx.dtype_of(a_ptr), 1)
        bufs_B = tlx.local_alloc((BLOCK_K, BLOCK_N), tlx.dtype_of(b_ptr), 1)

        a_view = tlx.local_view(bufs_A, 0)
        b_view = tlx.local_view(bufs_B, 0)

        a_reg = tl.load(a_ptrs)
        b_reg = tl.load(b_ptrs)

        tlx.local_store(a_view, a_reg)
        tlx.local_store(b_view, b_reg)

        a_tile = tlx.local_load(a_view)
        b_tile = tlx.local_load(b_view)

        acc = tl.dot(a_tile, b_tile)

        c = acc.to(tlx.dtype_of(c_ptr))
        offs_cm = tl.arange(0, BLOCK_M)
        offs_cn = tl.arange(0, BLOCK_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[
            None, :]
        tl.store(c_ptrs, c)

    M, N, K = 64, 64, 64
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    c = torch.zeros((M, N), device=DEVICE, dtype=torch.float16)

    gemm_kernel[(1, )](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=M,
        BLOCK_N=N,
        BLOCK_K=K,
    )
    c_ref = torch.matmul(a, b)
    torch.testing.assert_close(c, c_ref)


# ---------------------------------------------------------------------------
# Compilation with larger sizes (multiple blocks)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not is_hip(), reason="AMD-specific test")
def test_amd_custom_stages_multi_block():
    """Multi-block GEMM to verify grid launch with custom stages."""

    @triton.jit
    def gemm_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

        offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + (offs_am[:, None] * stride_am +
                          offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk +
                          offs_bn[None, :] * stride_bn)

        bufs_A = tlx.local_alloc((BLOCK_M, BLOCK_K), tlx.dtype_of(a_ptr), 1)
        bufs_B = tlx.local_alloc((BLOCK_K, BLOCK_N), tlx.dtype_of(b_ptr), 1)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        K_ITERS = tl.cdiv(K, BLOCK_K)
        for _ in tl.range(0, K_ITERS, num_stages=0):
            a_view = tlx.local_view(bufs_A, 0)
            b_view = tlx.local_view(bufs_B, 0)

            a_reg = tl.load(a_ptrs, mask=offs_k[None, :] < K)
            b_reg = tl.load(b_ptrs, mask=offs_k[:, None] < K)

            tlx.local_store(a_view, a_reg)
            tlx.local_store(b_view, b_reg)

            a_tile = tlx.local_load(a_view)
            b_tile = tlx.local_load(b_view)
            acc = tl.dot(a_tile, b_tile, acc)

            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        c = acc.to(tlx.dtype_of(c_ptr))
        offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[
            None, :]
        tl.store(c_ptrs, c, mask=c_mask)

    M, N, K = 256, 256, 128
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    c = torch.zeros((M, N), device=DEVICE, dtype=torch.float16)

    grid = ((M // BLOCK_M) * (N // BLOCK_N), )
    gemm_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    c_ref = torch.matmul(a, b)
    torch.testing.assert_close(c, c_ref, atol=1e-1, rtol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
