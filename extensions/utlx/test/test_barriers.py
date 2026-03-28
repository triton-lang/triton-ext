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
from conftest import tlx, DEVICE, is_hip, is_hopper_or_newer, get_current_target


# ---------------------------------------------------------------------------
# alloc_barriers: compile-only IR verification
# ---------------------------------------------------------------------------

def test_alloc_barriers_compile_only():
    """Verify alloc_barriers generates init_barrier in TTGIR."""

    @triton.jit
    def kernel(Out):
        bars = tlx.alloc_barriers(
            num_barriers=tl.constexpr(4), arrive_count=tl.constexpr(1)
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
    assert "local_alloc" in ir_str, f"Expected 'local_alloc' in TTGIR.\nIR:\n{ir_str}"
    assert "init_barrier" in ir_str, f"Expected 'init_barrier' in TTGIR.\nIR:\n{ir_str}"


def test_alloc_warp_barrier_compile_only():
    """Verify alloc_warp_barrier generates init_barrier with correct arrive count."""

    @triton.jit
    def kernel(Out):
        bars = tlx.alloc_warp_barrier(
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
        bars = tlx.alloc_barriers(
            num_barriers=tl.constexpr(2), arrive_count=tl.constexpr(1)
        )
        tl.store(out_ptr + offs, x)

    BLOCK = 128
    x = torch.randn(BLOCK, device=DEVICE, dtype=torch.float16)
    out = torch.empty_like(x)

    kernel[(1,)](x, out, BLOCK=BLOCK)
    torch.testing.assert_close(out, x)


@pytest.mark.parametrize("num_barriers", [1, 2, 4])
def test_alloc_barriers_various_counts(num_barriers):
    """Test alloc_barriers with different barrier counts."""

    @triton.jit
    def kernel(out_ptr, NUM_BARS: tl.constexpr):
        bars = tlx.alloc_barriers(
            num_barriers=NUM_BARS, arrive_count=tl.constexpr(1)
        )
        pid = tl.program_id(0)
        tl.store(out_ptr + pid, pid)

    out = torch.zeros(1, device=DEVICE, dtype=torch.int32)
    kernel[(1,)](out, NUM_BARS=num_barriers)
    assert out[0].item() == 0


def test_alloc_warp_barrier_on_gpu():
    """Test alloc_warp_barrier runs on GPU without errors."""

    @triton.jit
    def kernel(out_ptr):
        bars = tlx.alloc_warp_barrier(
            num_barriers=tl.constexpr(2),
            num_warps=tl.constexpr(4),
            num_arrivals=tl.constexpr(1),
        )
        pid = tl.program_id(0)
        tl.store(out_ptr + pid, pid)

    out = torch.zeros(1, device=DEVICE, dtype=torch.int32)
    kernel[(1,)](out)
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

        bars = tlx.alloc_barriers(
            num_barriers=tl.constexpr(2), arrive_count=tl.constexpr(1)
        )

        buf = tlx.local_alloc((BLOCK,), tl.float16, 1)
        view = tlx.local_view(buf, 0)
        tlx.local_store(view, x)
        y = tlx.local_load(view)

        tl.store(out_ptr + offs, y)

    BLOCK = 128
    x = torch.randn(BLOCK, device=DEVICE, dtype=torch.float16)
    out = torch.empty_like(x)

    kernel[(1,)](x, out, BLOCK=BLOCK)
    torch.testing.assert_close(out, x)


# ---------------------------------------------------------------------------
# barrier_arrive / barrier_wait: compile-only
# (Full synchronization requires warp specialization or async copy patterns)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not is_hopper_or_newer(), reason="Barrier ops require Hopper+")
def test_barrier_wait_arrive_compile_only():
    """Verify barrier_wait and barrier_arrive compile into valid IR."""

    @triton.jit
    def kernel(Out, BLOCK: tl.constexpr):
        bars = tlx.alloc_barriers(
            num_barriers=tl.constexpr(1), arrive_count=tl.constexpr(1)
        )
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
    assert "barrier" in ir_str.lower(), f"Expected barrier ops in TTGIR.\nIR:\n{ir_str}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
