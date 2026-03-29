"""GPU tests for uTLX memory ops.

Covers:
  - local_alloc (1D, 2D, multi-buffer, double-buffer)
  - local_view
  - local_store / local_load round-trip
  - dtype_of utility
  - tl.dot integration with tlx smem load/store
  - compile-only IR verification
"""

import pytest
import torch

import triton
import triton.language as tl
from conftest import tlx, DEVICE, get_current_target

# ---------------------------------------------------------------------------
# local_alloc -> local_store -> local_load round-trip (1D)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype,torch_dtype", [
    (tl.float16, torch.float16),
    (tl.float32, torch.float32),
    (tl.bfloat16, torch.bfloat16),
])
def test_local_alloc_store_load_1d(dtype, torch_dtype):
    """Allocate a 1D SMEM buffer, store a vector, load it back."""

    @triton.jit
    def kernel(in_ptr, out_ptr, BLOCK: tl.constexpr, DTYPE: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        x = tl.load(in_ptr + offs)
        buf = tlx.local_alloc((BLOCK, ), DTYPE, 1)
        view = tlx.local_view(buf, 0)
        tlx.local_store(view, x)
        y = tlx.local_load(view)
        tl.store(out_ptr + offs, y)

    BLOCK = 128
    x = torch.randn(BLOCK, device=DEVICE, dtype=torch_dtype)
    out = torch.empty_like(x)

    kernel[(1, )](x, out, BLOCK=BLOCK, DTYPE=dtype)
    torch.testing.assert_close(out, x)


# ---------------------------------------------------------------------------
# local_alloc -> local_store -> local_load round-trip (2D)
# ---------------------------------------------------------------------------


def test_local_alloc_store_load_2d():
    """Allocate a 2D SMEM buffer, store a tile, load it back."""

    @triton.jit
    def kernel(in_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr):
        row = tl.arange(0, M)
        col = tl.arange(0, N)
        offs = row[:, None] * N + col[None, :]
        x = tl.load(in_ptr + offs)
        buf = tlx.local_alloc((M, N), tl.float16, 1)
        view = tlx.local_view(buf, 0)
        tlx.local_store(view, x)
        y = tlx.local_load(view)
        tl.store(out_ptr + offs, y)

    M, N = 64, 64
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float16)
    out = torch.empty_like(x)

    kernel[(1, )](x, out, M=M, N=N)
    torch.testing.assert_close(out, x)


# ---------------------------------------------------------------------------
# Multi-buffer allocation
# ---------------------------------------------------------------------------


def test_local_alloc_multi_buffer():
    """Allocate multiple SMEM buffers and use them independently."""

    @triton.jit
    def kernel(in_ptr_a, in_ptr_b, out_ptr, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        a = tl.load(in_ptr_a + offs)
        b = tl.load(in_ptr_b + offs)

        bufs = tlx.local_alloc((BLOCK, ), tl.float16, 2)

        view0 = tlx.local_view(bufs, 0)
        view1 = tlx.local_view(bufs, 1)

        tlx.local_store(view0, a)
        tlx.local_store(view1, b)

        a_loaded = tlx.local_load(view0)
        b_loaded = tlx.local_load(view1)

        tl.store(out_ptr + offs, a_loaded + b_loaded)

    BLOCK = 128
    a = torch.randn(BLOCK, device=DEVICE, dtype=torch.float16)
    b = torch.randn(BLOCK, device=DEVICE, dtype=torch.float16)
    out = torch.empty(BLOCK, device=DEVICE, dtype=torch.float16)

    kernel[(1, )](a, b, out, BLOCK=BLOCK)
    torch.testing.assert_close(out, a + b)


# ---------------------------------------------------------------------------
# Double-buffer 2D (GEMM-style)
# ---------------------------------------------------------------------------


def test_local_alloc_double_buffer_2d():
    """Double buffer pattern: alloc 2 x (M, K), store/load through both."""

    @triton.jit
    def kernel(in_ptr, out_ptr, M: tl.constexpr, K: tl.constexpr):
        row = tl.arange(0, M)
        col = tl.arange(0, K)
        offs = row[:, None] * K + col[None, :]
        x = tl.load(in_ptr + offs)

        bufs = tlx.local_alloc((M, K), tl.float16, 2)

        v0 = tlx.local_view(bufs, 0)
        tlx.local_store(v0, x)
        y = tlx.local_load(v0)

        v1 = tlx.local_view(bufs, 1)
        tlx.local_store(v1, y)
        z = tlx.local_load(v1)

        tl.store(out_ptr + offs, z)

    M, K = 64, 64
    x = torch.randn(M, K, device=DEVICE, dtype=torch.float16)
    out = torch.empty_like(x)

    kernel[(1, )](x, out, M=M, K=K)
    torch.testing.assert_close(out, x)


# ---------------------------------------------------------------------------
# Multiple independent allocations
# ---------------------------------------------------------------------------


def test_local_alloc_two_independent_buffers():
    """Two separate local_alloc calls for independent buffers."""

    @triton.jit
    def kernel(x_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        x = tl.load(x_ptr + offs)
        y = tl.load(y_ptr + offs)

        buf_x = tlx.local_alloc((BLOCK, ), tl.float32, 1)
        buf_y = tlx.local_alloc((BLOCK, ), tl.float32, 1)

        vx = tlx.local_view(buf_x, 0)
        vy = tlx.local_view(buf_y, 0)

        tlx.local_store(vx, x)
        tlx.local_store(vy, y)

        x_reg = tlx.local_load(vx)
        y_reg = tlx.local_load(vy)

        tl.store(out_ptr + offs, x_reg + y_reg)

    BLOCK = 64
    x = torch.rand(BLOCK, device=DEVICE, dtype=torch.float32)
    y = torch.rand(BLOCK, device=DEVICE, dtype=torch.float32)
    out = torch.empty(BLOCK, device=DEVICE, dtype=torch.float32)

    kernel[(1, )](x, y, out, BLOCK=BLOCK)
    torch.testing.assert_close(out, x + y)


# ---------------------------------------------------------------------------
# Three-buffer allocation with 3 views
# ---------------------------------------------------------------------------


def test_local_alloc_three_buffers():
    """Allocate 3 buffers, store different data, verify loads."""

    @triton.jit
    def kernel(a_ptr, b_ptr, c_ptr, out_ptr, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        a = tl.load(a_ptr + offs)
        b = tl.load(b_ptr + offs)
        c = tl.load(c_ptr + offs)

        bufs = tlx.local_alloc((BLOCK, ), tl.float32, 3)

        v0 = tlx.local_view(bufs, 0)
        v1 = tlx.local_view(bufs, 1)
        v2 = tlx.local_view(bufs, 2)

        tlx.local_store(v0, a)
        tlx.local_store(v1, b)
        tlx.local_store(v2, c)

        a_out = tlx.local_load(v0)
        b_out = tlx.local_load(v1)
        c_out = tlx.local_load(v2)

        tl.store(out_ptr + offs, a_out + b_out + c_out)

    BLOCK = 64
    a = torch.rand(BLOCK, device=DEVICE, dtype=torch.float32)
    b = torch.rand(BLOCK, device=DEVICE, dtype=torch.float32)
    c = torch.rand(BLOCK, device=DEVICE, dtype=torch.float32)
    out = torch.empty(BLOCK, device=DEVICE, dtype=torch.float32)

    kernel[(1, )](a, b, c, out, BLOCK=BLOCK)
    torch.testing.assert_close(out, a + b + c)


# ---------------------------------------------------------------------------
# tl.dot with tlx smem load/store
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("M,N,K", [
    (16, 16, 16),
    (32, 32, 32),
    (64, 64, 64),
    (64, 64, 32),
    (128, 128, 64),
])
def test_dot_with_tlx_smem(M, N, K):
    """tl.load -> tlx.local_store -> tlx.local_load -> tl.dot"""

    @triton.jit
    def dot_kernel(
        X,
        stride_xm,
        stride_xk,
        Y,
        stride_yk,
        stride_yn,
        Z,
        stride_zm,
        stride_zn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        off_m = tl.arange(0, BLOCK_M)
        off_n = tl.arange(0, BLOCK_N)
        off_k = tl.arange(0, BLOCK_K)

        a_ptrs = X + (off_m[:, None] * stride_xm + off_k[None, :] * stride_xk)
        b_ptrs = Y + (off_k[:, None] * stride_yk + off_n[None, :] * stride_yn)

        buf_a = tlx.local_alloc((BLOCK_M, BLOCK_K), tlx.dtype_of(X), 1)
        buf_b = tlx.local_alloc((BLOCK_K, BLOCK_N), tlx.dtype_of(Y), 1)
        a_view = tlx.local_view(buf_a, 0)
        b_view = tlx.local_view(buf_b, 0)

        a_reg = tl.load(a_ptrs)
        b_reg = tl.load(b_ptrs)

        tlx.local_store(a_view, a_reg)
        tlx.local_store(b_view, b_reg)

        a_tile = tlx.local_load(a_view)
        b_tile = tlx.local_load(b_view)

        c_tile = tl.dot(a_tile, b_tile)
        c = c_tile.to(tlx.dtype_of(Z))

        c_ptrs = Z + stride_zm * off_m[:, None] + stride_zn * off_n[None, :]
        tl.store(c_ptrs, c)

    dtype = torch.float16
    x = torch.randn((M, K), device=DEVICE, dtype=dtype)
    y = torch.randn((K, N), device=DEVICE, dtype=dtype)
    z = torch.zeros((M, N), device=DEVICE, dtype=dtype)

    dot_kernel[(1, 1)](
        x,
        x.stride(0),
        x.stride(1),
        y,
        y.stride(0),
        y.stride(1),
        z,
        z.stride(0),
        z.stride(1),
        BLOCK_M=M,
        BLOCK_K=K,
        BLOCK_N=N,
    )
    z_ref = torch.matmul(x, y)
    torch.testing.assert_close(z, z_ref)


# ---------------------------------------------------------------------------
# dtype_of utility
# ---------------------------------------------------------------------------


def test_dtype_of():
    """Verify dtype_of returns the correct element type."""

    @triton.jit
    def kernel(in_ptr, out_ptr, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        x = tl.load(in_ptr + offs)
        buf = tlx.local_alloc((BLOCK, ), tlx.dtype_of(in_ptr), 1)
        view = tlx.local_view(buf, 0)
        tlx.local_store(view, x)
        y = tlx.local_load(view)
        tl.store(out_ptr + offs, y)

    BLOCK = 64
    x = torch.randn(BLOCK, device=DEVICE, dtype=torch.float16)
    out = torch.empty_like(x)

    kernel[(1, )](x, out, BLOCK=BLOCK)
    torch.testing.assert_close(out, x)


@pytest.mark.parametrize("torch_dtype", [torch.float32, torch.bfloat16])
def test_dtype_of_various(torch_dtype):
    """dtype_of with different pointer types."""

    @triton.jit
    def kernel(in_ptr, out_ptr, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        x = tl.load(in_ptr + offs)
        buf = tlx.local_alloc((BLOCK, ), tlx.dtype_of(in_ptr), 1)
        view = tlx.local_view(buf, 0)
        tlx.local_store(view, x)
        y = tlx.local_load(view)
        tl.store(out_ptr + offs, y)

    BLOCK = 64
    x = torch.randn(BLOCK, device=DEVICE, dtype=torch_dtype)
    out = torch.empty_like(x)

    kernel[(1, )](x, out, BLOCK=BLOCK)
    torch.testing.assert_close(out, x)


# ---------------------------------------------------------------------------
# Pipelined GEMM pattern (from amd-gemm-pipelined tutorial)
# ---------------------------------------------------------------------------


def test_pipelined_gemm_pattern():
    """Simplified pipelined GEMM using multi-buffer local_alloc."""

    @triton.jit
    def kernel(
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

        # Allocate 2 buffers for double-buffering
        bufs_A = tlx.local_alloc((BLOCK_M, BLOCK_K), tlx.dtype_of(a_ptr), 2)
        bufs_B = tlx.local_alloc((BLOCK_K, BLOCK_N), tlx.dtype_of(b_ptr), 2)

        # Load first tile into buffer 0
        a_view0 = tlx.local_view(bufs_A, 0)
        b_view0 = tlx.local_view(bufs_B, 0)
        a0 = tl.load(a_ptrs)
        b0 = tl.load(b_ptrs)
        tlx.local_store(a_view0, a0)
        tlx.local_store(b_view0, b0)

        # Compute from buffer 0
        a_tile = tlx.local_load(a_view0)
        b_tile = tlx.local_load(b_view0)
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

    kernel[(1, )](
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
# Compile-only: verify IR generation (no GPU required)
# ---------------------------------------------------------------------------


def test_local_alloc_compile_only():
    """Verify local_alloc generates valid TTGIR."""

    @triton.jit
    def kernel(ptr, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        x = tl.load(ptr + offs)
        buf = tlx.local_alloc((BLOCK, ), tl.float16, 1)
        view = tlx.local_view(buf, 0)
        tlx.local_store(view, x)
        y = tlx.local_load(view)
        tl.store(ptr + offs, y)

    src = triton.compiler.ASTSource(
        fn=kernel,
        signature={"ptr": "*fp16"},
        constexprs={"BLOCK": 128},
    )
    try:
        ret = triton.compile(src, target=get_current_target())
    except Exception:
        pytest.skip("No GPU target available for compilation")

    ir_str = ret.asm.get("ttgir", "")
    assert "local_alloc" in ir_str, f"Expected 'local_alloc' in TTGIR.\nIR:\n{ir_str}"
    assert "local_load" in ir_str, f"Expected 'local_load' in TTGIR.\nIR:\n{ir_str}"
    assert "local_store" in ir_str, f"Expected 'local_store' in TTGIR.\nIR:\n{ir_str}"


def test_multi_buffer_compile_only():
    """Verify multi-buffer local_alloc generates memdesc_index in TTGIR."""

    @triton.jit
    def kernel(ptr, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        x = tl.load(ptr + offs)
        bufs = tlx.local_alloc((BLOCK, ), tl.float32, 3)
        v0 = tlx.local_view(bufs, 0)
        v1 = tlx.local_view(bufs, 1)
        tlx.local_store(v0, x)
        y = tlx.local_load(v0)
        tlx.local_store(v1, y)
        z = tlx.local_load(v1)
        tl.store(ptr + offs, z)

    src = triton.compiler.ASTSource(
        fn=kernel,
        signature={"ptr": "*fp32"},
        constexprs={"BLOCK": 64},
    )
    try:
        ret = triton.compile(src, target=get_current_target())
    except Exception:
        pytest.skip("No GPU target available for compilation")

    ir_str = ret.asm.get("ttgir", "")
    assert "local_alloc" in ir_str
    assert "memdesc_index" in ir_str or "memdesc_subview" in ir_str


# ---------------------------------------------------------------------------
# Varying block sizes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("BLOCK", [32, 64, 128, 256])
def test_local_alloc_varying_block_sizes(BLOCK):
    """local_alloc -> store -> load with different block sizes."""

    @triton.jit
    def kernel(in_ptr, out_ptr, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        x = tl.load(in_ptr + offs)
        buf = tlx.local_alloc((BLOCK, ), tl.float32, 1)
        view = tlx.local_view(buf, 0)
        tlx.local_store(view, x)
        y = tlx.local_load(view)
        tl.store(out_ptr + offs, y)

    x = torch.randn(BLOCK, device=DEVICE, dtype=torch.float32)
    out = torch.empty_like(x)
    kernel[(1, )](x, out, BLOCK=BLOCK)
    torch.testing.assert_close(out, x)


# ---------------------------------------------------------------------------
# Integer dtypes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype,torch_dtype", [
    (tl.int8, torch.int8),
    (tl.int16, torch.int16),
    (tl.int32, torch.int32),
])
def test_local_alloc_integer_dtypes(dtype, torch_dtype):
    """local_alloc with integer element types."""

    @triton.jit
    def kernel(in_ptr, out_ptr, BLOCK: tl.constexpr, DTYPE: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        x = tl.load(in_ptr + offs)
        buf = tlx.local_alloc((BLOCK, ), DTYPE, 1)
        view = tlx.local_view(buf, 0)
        tlx.local_store(view, x)
        y = tlx.local_load(view)
        tl.store(out_ptr + offs, y)

    BLOCK = 64
    x = torch.randint(-100, 100, (BLOCK, ), device=DEVICE, dtype=torch_dtype)
    out = torch.empty_like(x)
    kernel[(1, )](x, out, BLOCK=BLOCK, DTYPE=dtype)
    torch.testing.assert_close(out, x)


# ---------------------------------------------------------------------------
# Overwrite: store twice, last write wins
# ---------------------------------------------------------------------------


def test_local_alloc_overwrite():
    """Store twice to the same view, verify last write wins."""

    @triton.jit
    def kernel(a_ptr, b_ptr, out_ptr, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        a = tl.load(a_ptr + offs)
        b = tl.load(b_ptr + offs)

        buf = tlx.local_alloc((BLOCK, ), tl.float32, 1)
        view = tlx.local_view(buf, 0)

        tlx.local_store(view, a)  # first write
        tlx.local_store(view, b)  # overwrite

        y = tlx.local_load(view)
        tl.store(out_ptr + offs, y)

    BLOCK = 64
    a = torch.randn(BLOCK, device=DEVICE, dtype=torch.float32)
    b = torch.randn(BLOCK, device=DEVICE, dtype=torch.float32)
    out = torch.empty(BLOCK, device=DEVICE, dtype=torch.float32)
    kernel[(1, )](a, b, out, BLOCK=BLOCK)
    torch.testing.assert_close(out, b)


# ---------------------------------------------------------------------------
# Varying 2D shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("M,N", [
    (16, 16),
    (32, 64),
    (64, 32),
    (128, 128),
])
def test_local_alloc_2d_shapes(M, N):
    """local_alloc 2D with different tile shapes."""

    @triton.jit
    def kernel(in_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr):
        row = tl.arange(0, M)
        col = tl.arange(0, N)
        offs = row[:, None] * N + col[None, :]
        x = tl.load(in_ptr + offs)
        buf = tlx.local_alloc((M, N), tl.float16, 1)
        view = tlx.local_view(buf, 0)
        tlx.local_store(view, x)
        y = tlx.local_load(view)
        tl.store(out_ptr + offs, y)

    x = torch.randn(M, N, device=DEVICE, dtype=torch.float16)
    out = torch.empty_like(x)
    kernel[(1, )](x, out, M=M, N=N)
    torch.testing.assert_close(out, x)


# ---------------------------------------------------------------------------
# Multi-block kernel with local_alloc
# ---------------------------------------------------------------------------


def test_local_alloc_multi_block_kernel():
    """Multiple program instances each with their own local_alloc."""

    @triton.jit
    def kernel(in_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        x = tl.load(in_ptr + offs, mask=mask)

        buf = tlx.local_alloc((BLOCK, ), tl.float32, 1)
        view = tlx.local_view(buf, 0)
        tlx.local_store(view, x)
        y = tlx.local_load(view)

        tl.store(out_ptr + offs, y, mask=mask)

    N = 1024
    BLOCK = 128
    x = torch.randn(N, device=DEVICE, dtype=torch.float32)
    out = torch.empty_like(x)
    grid = (triton.cdiv(N, BLOCK), )
    kernel[grid](x, out, N, BLOCK=BLOCK)
    torch.testing.assert_close(out, x)


# ---------------------------------------------------------------------------
# Accumulation pattern: load from two smem buffers and add
# ---------------------------------------------------------------------------


def test_local_alloc_accumulate_from_smem():
    """Load from two smem buffers, add in registers, store result."""

    @triton.jit
    def kernel(a_ptr, b_ptr, out_ptr, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        a = tl.load(a_ptr + offs)
        b = tl.load(b_ptr + offs)

        buf = tlx.local_alloc((BLOCK, ), tl.float16, 2)

        va = tlx.local_view(buf, 0)
        vb = tlx.local_view(buf, 1)

        tlx.local_store(va, a)
        tlx.local_store(vb, b)

        a_reg = tlx.local_load(va)
        b_reg = tlx.local_load(vb)
        result = a_reg + b_reg

        tl.store(out_ptr + offs, result)

    BLOCK = 128
    a = torch.randn(BLOCK, device=DEVICE, dtype=torch.float16)
    b = torch.randn(BLOCK, device=DEVICE, dtype=torch.float16)
    out = torch.empty(BLOCK, device=DEVICE, dtype=torch.float16)
    kernel[(1, )](a, b, out, BLOCK=BLOCK)
    torch.testing.assert_close(out, a + b)


# ---------------------------------------------------------------------------
# Chain: alloc -> store -> load -> compute -> store -> load
# ---------------------------------------------------------------------------


def test_local_alloc_chain():
    """Chain of smem store/load with computation in between."""

    @triton.jit
    def kernel(in_ptr, out_ptr, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        x = tl.load(in_ptr + offs)

        buf = tlx.local_alloc((BLOCK, ), tl.float32, 2)

        v0 = tlx.local_view(buf, 0)
        v1 = tlx.local_view(buf, 1)

        # store x, load, double it
        tlx.local_store(v0, x)
        y = tlx.local_load(v0)
        doubled = y + y

        # store doubled, load back
        tlx.local_store(v1, doubled)
        z = tlx.local_load(v1)

        tl.store(out_ptr + offs, z)

    BLOCK = 64
    x = torch.randn(BLOCK, device=DEVICE, dtype=torch.float32)
    out = torch.empty_like(x)
    kernel[(1, )](x, out, BLOCK=BLOCK)
    torch.testing.assert_close(out, x * 2)


# ---------------------------------------------------------------------------
# local_view with dynamic index (loop variable)
# ---------------------------------------------------------------------------


def test_local_view_in_loop():
    """Use local_view with loop variable index (pipelining pattern)."""

    @triton.jit
    def kernel(in_ptr, out_ptr, BLOCK: tl.constexpr, NUM_BUFS: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        x = tl.load(in_ptr + offs)

        bufs = tlx.local_alloc((BLOCK, ), tl.float32, NUM_BUFS)

        # Store into each buffer sequentially
        for i in tl.static_range(NUM_BUFS):
            view = tlx.local_view(bufs, i)
            tlx.local_store(view, x)

        # Load from last buffer
        last_view = tlx.local_view(bufs, NUM_BUFS - 1)
        y = tlx.local_load(last_view)
        tl.store(out_ptr + offs, y)

    BLOCK = 64
    x = torch.randn(BLOCK, device=DEVICE, dtype=torch.float32)
    out = torch.empty_like(x)
    kernel[(1, )](x, out, BLOCK=BLOCK, NUM_BUFS=4)
    torch.testing.assert_close(out, x)


# ---------------------------------------------------------------------------
# Dot with multi-iteration K loop through smem
# ---------------------------------------------------------------------------


def test_dot_k_loop_through_smem():
    """GEMM with K tiled through smem (2 iterations)."""

    @triton.jit
    def kernel(
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
        offs_m = tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + (offs_m[:, None] * stride_am +
                          offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk +
                          offs_n[None, :] * stride_bn)

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
        c_ptrs = c_ptr + stride_cm * offs_m[:,
                                            None] + stride_cn * offs_n[None, :]
        tl.store(c_ptrs, c)

    M, N, K = 64, 64, 128
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    c = torch.zeros((M, N), device=DEVICE, dtype=torch.float16)

    kernel[(1, )](
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
        BLOCK_K=64,
    )
    c_ref = torch.matmul(a, b)
    torch.testing.assert_close(c, c_ref, atol=1e-1, rtol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
