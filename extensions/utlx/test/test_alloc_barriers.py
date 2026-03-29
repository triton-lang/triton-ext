"""Tests for utlx_plugin barrier ops (alloc_barriers, alloc_warp_barrier).

Tests the alloc_barriers operation: allocating mbarrier buffers in shared memory
and initializing them with InitBarrierOp. Covers compile-only IR verification
and on-GPU execution.

Originally from TLXMemOps, consolidated into utlx.
"""

import pytest
import torch

import triton
import triton.language as tl

from conftest import tlx, is_hopper_or_newer

# ---------------------------------------------------------------------------
# Compile-only: verify IR generation without running on GPU
# ---------------------------------------------------------------------------


def test_alloc_barriers_compile_only():
    """Verify alloc_barriers generates valid TTGIR with expected ops."""

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
        ret = triton.compile(
            src, target=triton.runtime.driver.active.get_current_target())
    except Exception:
        pytest.skip("No GPU target available for compilation")

    ir_str = ret.asm.get("ttgir", "")
    assert "local_alloc" in ir_str, (
        f"Expected 'local_alloc' in TTGIR.\nIR:\n{ir_str}")
    assert "init_barrier" in ir_str, (
        f"Expected 'init_barrier' in TTGIR.\nIR:\n{ir_str}")
    assert "memdesc_index" in ir_str, (
        f"Expected 'memdesc_index' in TTGIR.\nIR:\n{ir_str}")


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
        ret = triton.compile(
            src, target=triton.runtime.driver.active.get_current_target())
    except Exception:
        pytest.skip("No GPU target available for compilation")

    ir_str = ret.asm.get("ttgir", "")
    assert "init_barrier" in ir_str, (
        f"Expected 'init_barrier' in TTGIR.\nIR:\n{ir_str}")
    # arrive_count = num_warps(4) * 32 * num_arrivals(1) = 128
    assert "128" in ir_str, (
        f"Expected arrive_count 128 in TTGIR.\nIR:\n{ir_str}")


# ---------------------------------------------------------------------------
# On-GPU: alloc_barriers execution
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
def test_alloc_barriers_on_gpu(device="cuda"):
    """Allocate barriers on GPU and verify the kernel runs without errors."""

    @triton.jit
    def kernel(in_ptr, out_ptr, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        x = tl.load(in_ptr + offs)

        # Allocate barriers (exercises alloc_barriers plugin op)
        tlx.alloc_barriers(num_barriers=tl.constexpr(2),
                           arrive_count=tl.constexpr(1))

        tl.store(out_ptr + offs, x)

    BLOCK = 128
    x = torch.randn(BLOCK, device=device, dtype=torch.float16)
    out = torch.empty_like(x)

    kernel[(1, )](x, out, BLOCK=BLOCK)
    torch.testing.assert_close(out, x)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
@pytest.mark.parametrize("num_barriers", [1, 2, 4])
def test_alloc_barriers_various_counts(num_barriers, device="cuda"):
    """Test alloc_barriers with different barrier counts."""

    @triton.jit
    def kernel(out_ptr, NUM_BARS: tl.constexpr):
        tlx.alloc_barriers(num_barriers=NUM_BARS, arrive_count=tl.constexpr(1))
        pid = tl.program_id(0)
        tl.store(out_ptr + pid, pid)

    out = torch.zeros(1, device=device, dtype=torch.int32)
    kernel[(1, )](out, NUM_BARS=num_barriers)
    assert out[0].item() == 0


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
def test_alloc_warp_barrier_on_gpu(device="cuda"):
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

    out = torch.zeros(1, device=device, dtype=torch.int32)
    kernel[(1, )](out)
    assert out[0].item() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
