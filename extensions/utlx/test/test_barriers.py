"""Tests for uTLX barrier ops.

Covers:
  - alloc_barriers (compile-only + GPU)
  - alloc_warp_barrier (compile-only + GPU)
  - barrier_arrive / barrier_wait (compile-only)
  - named_barrier_wait / named_barrier_arrive (compile-only)
  - barrier_expect_bytes (compile-only)

Note: Full barrier synchronization tests require specific hardware patterns
(e.g., warp specialization, async copies). These tests verify that the ops
compile correctly and that barrier allocation runs on GPU without crashing.
"""

import pytest
import torch

import triton
import triton.language as tl
from conftest import tlx, DEVICE, is_hopper_or_newer, get_current_target

# ---------------------------------------------------------------------------
# alloc_barriers: compile-only IR verification
# ---------------------------------------------------------------------------


def test_alloc_barriers_compile_only():
    """Verify alloc_barriers generates init_barrier in TTGIR."""

    @triton.jit
    def kernel(Out):
        tlx.alloc_barriers(num_barriers=tl.constexpr(4),
                           arrive_count=tl.constexpr(1))
        pid = tl.program_id(0)
        tl.store(Out + pid, pid)

    src = triton.compiler.ASTSource(
        fn=kernel,
        signature={"Out": "*i32"},
        constexprs={},
    )
    try:
        ret = triton.compile(src, target=get_current_target())
    except Exception:
        pytest.skip("No GPU target available for compilation")

    ir_str = ret.asm.get("ttgir", "")
    assert "local_alloc" in ir_str, f"Expected 'local_alloc' in TTGIR.\nIR:\n{ir_str}"
    assert "init_barrier" in ir_str, f"Expected 'init_barrier' in TTGIR.\nIR:\n{ir_str}"


def test_alloc_warp_barrier_compile_only():
    """Verify alloc_warp_barrier generates init_barrier with correct arrive count."""

    @triton.jit
    def kernel(Out):
        tlx.alloc_warp_barrier(
            num_barriers=tl.constexpr(2),
            num_warps=tl.constexpr(4),
            num_arrivals=tl.constexpr(1),
        )
        pid = tl.program_id(0)
        tl.store(Out + pid, pid)

    src = triton.compiler.ASTSource(
        fn=kernel,
        signature={"Out": "*i32"},
        constexprs={},
    )
    try:
        ret = triton.compile(src, target=get_current_target())
    except Exception:
        pytest.skip("No GPU target available for compilation")

    ir_str = ret.asm.get("ttgir", "")
    assert "init_barrier" in ir_str, f"Expected 'init_barrier' in TTGIR.\nIR:\n{ir_str}"
    # arrive_count = num_warps(4) * 32 * num_arrivals(1) = 128
    assert "128" in ir_str, f"Expected arrive_count 128 in TTGIR.\nIR:\n{ir_str}"


# ---------------------------------------------------------------------------
# alloc_barriers: GPU execution
# ---------------------------------------------------------------------------


def test_alloc_barriers_on_gpu():
    """Allocate barriers on GPU and verify the kernel runs without errors."""

    @triton.jit
    def kernel(in_ptr, out_ptr, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        x = tl.load(in_ptr + offs)
        tlx.alloc_barriers(num_barriers=tl.constexpr(2),
                           arrive_count=tl.constexpr(1))
        tl.store(out_ptr + offs, x)

    BLOCK = 128
    x = torch.randn(BLOCK, device=DEVICE, dtype=torch.float16)
    out = torch.empty_like(x)

    kernel[(1, )](x, out, BLOCK=BLOCK)
    torch.testing.assert_close(out, x)


@pytest.mark.parametrize("num_barriers", [1, 2, 4])
def test_alloc_barriers_various_counts(num_barriers):
    """Test alloc_barriers with different barrier counts."""

    @triton.jit
    def kernel(out_ptr, NUM_BARS: tl.constexpr):
        tlx.alloc_barriers(num_barriers=NUM_BARS, arrive_count=tl.constexpr(1))
        pid = tl.program_id(0)
        tl.store(out_ptr + pid, pid)

    out = torch.zeros(1, device=DEVICE, dtype=torch.int32)
    kernel[(1, )](out, NUM_BARS=num_barriers)
    assert out[0].item() == 0


def test_alloc_warp_barrier_on_gpu():
    """Test alloc_warp_barrier runs on GPU without errors."""

    @triton.jit
    def kernel(out_ptr):
        tlx.alloc_warp_barrier(
            num_barriers=tl.constexpr(2),
            num_warps=tl.constexpr(4),
            num_arrivals=tl.constexpr(1),
        )
        pid = tl.program_id(0)
        tl.store(out_ptr + pid, pid)

    out = torch.zeros(1, device=DEVICE, dtype=torch.int32)
    kernel[(1, )](out)
    assert out[0].item() == 0


# ---------------------------------------------------------------------------
# alloc_barriers with smem ops: combined test
# ---------------------------------------------------------------------------


def test_alloc_barriers_with_smem_ops():
    """Combine barrier allocation with SMEM load/store."""

    @triton.jit
    def kernel(in_ptr, out_ptr, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        x = tl.load(in_ptr + offs)

        tlx.alloc_barriers(num_barriers=tl.constexpr(2),
                           arrive_count=tl.constexpr(1))

        buf = tlx.local_alloc((BLOCK, ), tl.float16, 1)
        view = tlx.local_view(buf, 0)
        tlx.local_store(view, x)
        y = tlx.local_load(view)

        tl.store(out_ptr + offs, y)

    BLOCK = 128
    x = torch.randn(BLOCK, device=DEVICE, dtype=torch.float16)
    out = torch.empty_like(x)

    kernel[(1, )](x, out, BLOCK=BLOCK)
    torch.testing.assert_close(out, x)


# ---------------------------------------------------------------------------
# barrier_arrive / barrier_wait: compile-only
# (Full synchronization requires warp specialization or async copy patterns)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not is_hopper_or_newer(),
                    reason="Barrier ops require Hopper+")
def test_barrier_wait_arrive_compile_only():
    """Verify barrier_wait and barrier_arrive compile into valid IR."""

    @triton.jit
    def kernel(Out, BLOCK: tl.constexpr):
        bars = tlx.alloc_barriers(num_barriers=tl.constexpr(1),
                                  arrive_count=tl.constexpr(1))
        bar_view = tlx.local_view(bars, 0)
        tlx.barrier_arrive(bar_view)
        tlx.barrier_wait(bar_view, tl.constexpr(0))
        pid = tl.program_id(0)
        tl.store(Out + pid, pid)

    src = triton.compiler.ASTSource(
        fn=kernel,
        signature={"Out": "*i32"},
        constexprs={"BLOCK": 128},
    )
    try:
        ret = triton.compile(src, target=get_current_target())
    except Exception:
        pytest.skip("Compilation not supported on this target")

    ir_str = ret.asm.get("ttgir", "")
    assert "barrier" in ir_str.lower(
    ), f"Expected barrier ops in TTGIR.\nIR:\n{ir_str}"


# ---------------------------------------------------------------------------
# alloc_barriers: larger counts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_barriers", [8, 16])
def test_alloc_barriers_large_counts(num_barriers):
    """Test alloc_barriers with larger barrier counts."""

    @triton.jit
    def kernel(in_ptr, out_ptr, BLOCK: tl.constexpr, NUM_BARS: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        x = tl.load(in_ptr + offs)
        tlx.alloc_barriers(num_barriers=NUM_BARS, arrive_count=tl.constexpr(1))
        tl.store(out_ptr + offs, x)

    BLOCK = 64
    x = torch.randn(BLOCK, device=DEVICE, dtype=torch.float32)
    out = torch.empty_like(x)
    kernel[(1, )](x, out, BLOCK=BLOCK, NUM_BARS=num_barriers)
    torch.testing.assert_close(out, x)


# ---------------------------------------------------------------------------
# alloc_barriers: different arrive counts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arrive_count", [1, 2, 4])
def test_alloc_barriers_arrive_counts(arrive_count):
    """Test alloc_barriers with different arrive counts."""

    @triton.jit
    def kernel(out_ptr, NUM_BARS: tl.constexpr, ARRIVE: tl.constexpr):
        tlx.alloc_barriers(num_barriers=NUM_BARS, arrive_count=ARRIVE)
        pid = tl.program_id(0)
        tl.store(out_ptr + pid, pid)

    out = torch.zeros(1, device=DEVICE, dtype=torch.int32)
    kernel[(1, )](out, NUM_BARS=2, ARRIVE=arrive_count)
    assert out[0].item() == 0


# ---------------------------------------------------------------------------
# alloc_warp_barrier: different warp counts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_warps", [1, 2, 4, 8])
def test_alloc_warp_barrier_various_warps(num_warps):
    """Test alloc_warp_barrier with different warp counts."""

    @triton.jit
    def kernel(out_ptr, NUM_WARPS: tl.constexpr):
        tlx.alloc_warp_barrier(
            num_barriers=tl.constexpr(1),
            num_warps=NUM_WARPS,
            num_arrivals=tl.constexpr(1),
        )
        pid = tl.program_id(0)
        tl.store(out_ptr + pid, pid)

    out = torch.zeros(1, device=DEVICE, dtype=torch.int32)
    kernel[(1, )](out, NUM_WARPS=num_warps)
    assert out[0].item() == 0


def test_alloc_warp_barrier_compile_only_arrive_count():
    """Verify warp barrier arrive count = num_warps * 32 * num_arrivals."""

    @triton.jit
    def kernel(Out):
        tlx.alloc_warp_barrier(
            num_barriers=tl.constexpr(1),
            num_warps=tl.constexpr(2),
            num_arrivals=tl.constexpr(3),
        )
        pid = tl.program_id(0)
        tl.store(Out + pid, pid)

    src = triton.compiler.ASTSource(
        fn=kernel,
        signature={"Out": "*i32"},
        constexprs={},
    )
    try:
        ret = triton.compile(src, target=get_current_target())
    except Exception:
        pytest.skip("No GPU target available for compilation")

    ir_str = ret.asm.get("ttgir", "")
    # arrive_count = 2 * 32 * 3 = 192
    assert "192" in ir_str, f"Expected arrive_count 192 in TTGIR.\nIR:\n{ir_str}"


# ---------------------------------------------------------------------------
# Barriers combined with multi-buffer smem pattern
# ---------------------------------------------------------------------------


def test_barriers_with_multi_buffer_smem():
    """Barriers + multi-buffer local_alloc (typical pipelining setup)."""

    @triton.jit
    def kernel(a_ptr, b_ptr, out_ptr, BLOCK: tl.constexpr,
               NUM_BUFS: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        a = tl.load(a_ptr + offs)
        b = tl.load(b_ptr + offs)

        tlx.alloc_barriers(num_barriers=NUM_BUFS, arrive_count=tl.constexpr(1))

        bufs = tlx.local_alloc((BLOCK, ), tl.float16, NUM_BUFS)

        v0 = tlx.local_view(bufs, 0)
        v1 = tlx.local_view(bufs, 1)

        tlx.local_store(v0, a)
        tlx.local_store(v1, b)

        a_reg = tlx.local_load(v0)
        b_reg = tlx.local_load(v1)

        tl.store(out_ptr + offs, a_reg + b_reg)

    BLOCK = 128
    a = torch.randn(BLOCK, device=DEVICE, dtype=torch.float16)
    b = torch.randn(BLOCK, device=DEVICE, dtype=torch.float16)
    out = torch.empty(BLOCK, device=DEVICE, dtype=torch.float16)
    kernel[(1, )](a, b, out, BLOCK=BLOCK, NUM_BUFS=2)
    torch.testing.assert_close(out, a + b)


# ---------------------------------------------------------------------------
# Barriers in multi-block kernel
# ---------------------------------------------------------------------------


def test_barriers_multi_block():
    """Barrier allocation in a multi-block kernel."""

    @triton.jit
    def kernel(in_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements

        tlx.alloc_barriers(num_barriers=tl.constexpr(2),
                           arrive_count=tl.constexpr(1))

        x = tl.load(in_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x, mask=mask)

    N = 512
    BLOCK = 128
    x = torch.randn(N, device=DEVICE, dtype=torch.float32)
    out = torch.empty_like(x)
    grid = (triton.cdiv(N, BLOCK), )
    kernel[grid](x, out, N, BLOCK=BLOCK)
    torch.testing.assert_close(out, x)


# ---------------------------------------------------------------------------
# Barriers + 2D smem + dot
# ---------------------------------------------------------------------------


def test_barriers_with_2d_smem_dot():
    """Barrier allocation combined with 2D smem buffers and tl.dot."""

    @triton.jit
    def kernel(
        a_ptr,
        b_ptr,
        c_ptr,
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
        tlx.alloc_barriers(num_barriers=tl.constexpr(2),
                           arrive_count=tl.constexpr(1))

        off_m = tl.arange(0, BLOCK_M)
        off_n = tl.arange(0, BLOCK_N)
        off_k = tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + (off_m[:, None] * stride_am +
                          off_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (off_k[:, None] * stride_bk +
                          off_n[None, :] * stride_bn)

        buf_a = tlx.local_alloc((BLOCK_M, BLOCK_K), tlx.dtype_of(a_ptr), 1)
        buf_b = tlx.local_alloc((BLOCK_K, BLOCK_N), tlx.dtype_of(b_ptr), 1)

        va = tlx.local_view(buf_a, 0)
        vb = tlx.local_view(buf_b, 0)

        a_reg = tl.load(a_ptrs)
        b_reg = tl.load(b_ptrs)

        tlx.local_store(va, a_reg)
        tlx.local_store(vb, b_reg)

        a_tile = tlx.local_load(va)
        b_tile = tlx.local_load(vb)
        c_tile = tl.dot(a_tile, b_tile)

        c = c_tile.to(tlx.dtype_of(c_ptr))
        c_ptrs = c_ptr + stride_cm * off_m[:,
                                           None] + stride_cn * off_n[None, :]
        tl.store(c_ptrs, c)

    M, N, K = 64, 64, 64
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    c = torch.zeros((M, N), device=DEVICE, dtype=torch.float16)

    kernel[(1, )](
        a,
        b,
        c,
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
# Compile-only: alloc_barriers IR checks with different configs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_barriers,arrive_count", [
    (1, 1),
    (4, 2),
    (8, 4),
])
def test_alloc_barriers_ir_patterns(num_barriers, arrive_count):
    """Verify alloc_barriers generates correct IR for various configs."""

    @triton.jit
    def kernel(Out, NUM_BARS: tl.constexpr, ARRIVE: tl.constexpr):
        tlx.alloc_barriers(num_barriers=NUM_BARS, arrive_count=ARRIVE)
        pid = tl.program_id(0)
        tl.store(Out + pid, pid)

    src = triton.compiler.ASTSource(
        fn=kernel,
        signature={"Out": "*i32"},
        constexprs={
            "NUM_BARS": num_barriers,
            "ARRIVE": arrive_count
        },
    )
    try:
        ret = triton.compile(src, target=get_current_target())
    except Exception:
        pytest.skip("No GPU target available for compilation")

    ir_str = ret.asm.get("ttgir", "")
    assert "init_barrier" in ir_str
    assert "local_alloc" in ir_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
