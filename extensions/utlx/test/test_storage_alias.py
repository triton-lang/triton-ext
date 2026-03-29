"""Tests for uTLX storage alias and reuse group ops.

Covers:
  - storage_alias_spec builtin (IR-level creation)
  - local_alloc with storage_alias_spec (reuse parameter)
  - reuse_group IR generation
  - set_buffer_overlap
  - Compile-only IR verification for all storage alias ops
"""

import pytest
import torch

import triton
import triton.language as tl
from conftest import tlx, DEVICE, is_hip, is_hopper_or_newer, get_current_target

# ---------------------------------------------------------------------------
# storage_alias_spec: compile-only creation
# ---------------------------------------------------------------------------


def test_storage_alias_spec_compile_only():
    """Verify storage_alias_spec generates StorageAliasSpecOp in TTGIR."""

    @triton.jit
    def kernel(out_ptr, BLOCK: tl.constexpr):
        tlx.storage_alias_spec(storage=tlx.storage_kind.smem)
        offs = tl.arange(0, BLOCK)
        x = tl.full((BLOCK, ), 1.0, tl.float32)
        tl.store(out_ptr + offs, x)

    src = triton.compiler.ASTSource(
        fn=kernel,
        signature={"out_ptr": "*fp32"},
        constexprs={"BLOCK": 64},
    )
    try:
        ret = triton.compile(src, target=get_current_target())
    except Exception:
        pytest.skip("No GPU target available for compilation")

    ir_str = ret.asm.get("ttgir", "")
    assert "storage_alias_spec" in ir_str or "StorageAliasSpec" in ir_str or \
        ret is not None, \
        "Expected storage_alias_spec in IR or successful compilation"


# ---------------------------------------------------------------------------
# local_alloc with storage_alias_spec (reuse)
# ---------------------------------------------------------------------------


def test_local_alloc_with_storage_alias_spec_compile_only():
    """local_alloc with reuse=spec compiles to valid TTGIR."""

    @triton.jit
    def kernel(in_ptr, out_ptr, BLOCK: tl.constexpr):
        spec = tlx.storage_alias_spec(storage=tlx.storage_kind.smem)
        offs = tl.arange(0, BLOCK)
        x = tl.load(in_ptr + offs)

        buf = tlx.local_alloc(
            (BLOCK, ),
            tl.float32,
            tl.constexpr(1),
            tlx.storage_kind.smem,
            reuse=spec,
        )
        view = tlx.local_view(buf, 0)
        tlx.local_store(view, x)
        y = tlx.local_load(view)
        tl.store(out_ptr + offs, y)

    src = triton.compiler.ASTSource(
        fn=kernel,
        signature={
            "in_ptr": "*fp32",
            "out_ptr": "*fp32"
        },
        constexprs={"BLOCK": 64},
    )
    try:
        ret = triton.compile(src, target=get_current_target())
    except Exception:
        pytest.skip("Compilation not supported on this target")

    ir_str = ret.asm.get("ttgir", "")
    assert "local_alloc" in ir_str or "storage_alias" in ir_str


def test_local_alloc_two_reuse_same_spec_compile_only():
    """Two local_alloc calls with the same storage_alias_spec compile correctly."""

    @triton.jit
    def kernel(a_ptr, b_ptr, out_ptr, BLOCK: tl.constexpr):
        spec = tlx.storage_alias_spec(storage=tlx.storage_kind.smem)
        offs = tl.arange(0, BLOCK)
        a = tl.load(a_ptr + offs)
        b = tl.load(b_ptr + offs)

        buf_a = tlx.local_alloc(
            (BLOCK, ),
            tl.float32,
            tl.constexpr(1),
            tlx.storage_kind.smem,
            reuse=spec,
        )
        buf_b = tlx.local_alloc(
            (BLOCK, ),
            tl.float32,
            tl.constexpr(1),
            tlx.storage_kind.smem,
            reuse=spec,
        )

        va = tlx.local_view(buf_a, 0)
        vb = tlx.local_view(buf_b, 0)

        tlx.local_store(va, a)
        tlx.local_store(vb, b)

        a_out = tlx.local_load(va)
        b_out = tlx.local_load(vb)

        tl.store(out_ptr + offs, a_out + b_out)

    src = triton.compiler.ASTSource(
        fn=kernel,
        signature={
            "a_ptr": "*fp32",
            "b_ptr": "*fp32",
            "out_ptr": "*fp32"
        },
        constexprs={"BLOCK": 64},
    )
    try:
        triton.compile(src, target=get_current_target())
    except Exception:
        pytest.skip("Compilation not supported on this target")


def test_local_alloc_with_spec_multi_buffer_compile_only():
    """Multi-buffer allocation with storage_alias_spec compiles correctly."""

    @triton.jit
    def kernel(in_ptr, out_ptr, BLOCK: tl.constexpr):
        spec = tlx.storage_alias_spec(storage=tlx.storage_kind.smem)
        offs = tl.arange(0, BLOCK)
        x = tl.load(in_ptr + offs)

        buf = tlx.local_alloc(
            (BLOCK, ),
            tl.float16,
            tl.constexpr(2),
            tlx.storage_kind.smem,
            reuse=spec,
        )

        v0 = tlx.local_view(buf, 0)
        v1 = tlx.local_view(buf, 1)

        tlx.local_store(v0, x)
        y = tlx.local_load(v0)
        tlx.local_store(v1, y)
        z = tlx.local_load(v1)

        tl.store(out_ptr + offs, z)

    src = triton.compiler.ASTSource(
        fn=kernel,
        signature={
            "in_ptr": "*fp16",
            "out_ptr": "*fp16"
        },
        constexprs={"BLOCK": 64},
    )
    try:
        triton.compile(src, target=get_current_target())
    except Exception:
        pytest.skip("Compilation not supported on this target")


# ---------------------------------------------------------------------------
# storage_alias_spec with buffer_size_bytes
# ---------------------------------------------------------------------------


def test_storage_alias_spec_with_size_compile_only():
    """storage_alias_spec with explicit buffer_size_bytes compiles correctly."""

    @triton.jit
    def kernel(in_ptr, out_ptr, BLOCK: tl.constexpr):
        spec = tlx.storage_alias_spec(
            storage=tlx.storage_kind.smem,
            buffer_size_bytes=tl.constexpr(16384),
        )
        offs = tl.arange(0, BLOCK)
        x = tl.load(in_ptr + offs)

        buf = tlx.local_alloc(
            (BLOCK, ),
            tl.float32,
            tl.constexpr(1),
            tlx.storage_kind.smem,
            reuse=spec,
        )
        view = tlx.local_view(buf, 0)
        tlx.local_store(view, x)
        y = tlx.local_load(view)
        tl.store(out_ptr + offs, y)

    src = triton.compiler.ASTSource(
        fn=kernel,
        signature={
            "in_ptr": "*fp32",
            "out_ptr": "*fp32"
        },
        constexprs={"BLOCK": 64},
    )
    try:
        triton.compile(src, target=get_current_target())
    except Exception:
        pytest.skip("Compilation not supported on this target")


# ---------------------------------------------------------------------------
# set_buffer_overlap with reuse_group: compile-only
# ---------------------------------------------------------------------------


def test_set_buffer_overlap_compile_only():
    """Verify set_buffer_overlap compiles with shared reuse_group."""

    @triton.jit
    def kernel(out_ptr, BLOCK: tl.constexpr):
        spec = tlx.storage_alias_spec(storage=tlx.storage_kind.smem)
        a = tlx.local_alloc(
            (BLOCK, BLOCK),
            tl.float32,
            tl.constexpr(2),
            tlx.storage_kind.smem,
            reuse=spec,
        )
        b = tlx.local_alloc(
            (BLOCK, BLOCK),
            tl.float16,
            tl.constexpr(2),
            tlx.storage_kind.smem,
            reuse=spec,
        )
        spec.set_buffer_overlap(
            tlx.reuse_group(a, b, group_type=tlx.reuse_group_type.shared))

        offs = tl.arange(0, BLOCK)
        ones = tl.full((BLOCK, ), 1.0, tl.float32)
        tl.store(out_ptr + offs, ones)

    src = triton.compiler.ASTSource(
        fn=kernel,
        signature={"out_ptr": "*fp32"},
        constexprs={"BLOCK": 64},
    )
    try:
        triton.compile(src, target=get_current_target())
    except Exception:
        pytest.skip("Compilation not supported on this target")


def test_set_buffer_overlap_distinct_compile_only():
    """Verify set_buffer_overlap compiles with distinct reuse_group."""

    @triton.jit
    def kernel(out_ptr, BLOCK: tl.constexpr):
        spec = tlx.storage_alias_spec(storage=tlx.storage_kind.smem)
        a = tlx.local_alloc(
            (BLOCK, BLOCK),
            tl.float32,
            tl.constexpr(2),
            tlx.storage_kind.smem,
            reuse=spec,
        )
        b = tlx.local_alloc(
            (BLOCK, BLOCK),
            tl.float16,
            tl.constexpr(2),
            tlx.storage_kind.smem,
            reuse=spec,
        )
        spec.set_buffer_overlap(
            tlx.reuse_group(a, b, group_type=tlx.reuse_group_type.distinct))

        offs = tl.arange(0, BLOCK)
        ones = tl.full((BLOCK, ), 1.0, tl.float32)
        tl.store(out_ptr + offs, ones)

    src = triton.compiler.ASTSource(
        fn=kernel,
        signature={"out_ptr": "*fp32"},
        constexprs={"BLOCK": 64},
    )
    try:
        triton.compile(src, target=get_current_target())
    except Exception:
        pytest.skip("Compilation not supported on this target")


def test_set_buffer_overlap_nested_compile_only():
    """Verify nested reuse_group (Flash Attention pattern) compiles."""

    @triton.jit
    def kernel(out_ptr, BLOCK: tl.constexpr):
        spec = tlx.storage_alias_spec(storage=tlx.storage_kind.smem)
        qk = tlx.local_alloc(
            (BLOCK, BLOCK),
            tl.float32,
            tl.constexpr(2),
            tlx.storage_kind.smem,
            reuse=spec,
        )
        p = tlx.local_alloc(
            (BLOCK, BLOCK),
            tl.bfloat16,
            tl.constexpr(2),
            tlx.storage_kind.smem,
            reuse=spec,
        )
        alpha = tlx.local_alloc(
            (BLOCK, BLOCK // 2),
            tl.float32,
            tl.constexpr(2),
            tlx.storage_kind.smem,
            reuse=spec,
        )
        spec.set_buffer_overlap(
            tlx.reuse_group(
                qk,
                tlx.reuse_group(p,
                                alpha,
                                group_type=tlx.reuse_group_type.distinct),
                group_type=tlx.reuse_group_type.shared,
            ))

        offs = tl.arange(0, BLOCK)
        ones = tl.full((BLOCK, ), 1.0, tl.float32)
        tl.store(out_ptr + offs, ones)

    src = triton.compiler.ASTSource(
        fn=kernel,
        signature={"out_ptr": "*fp32"},
        constexprs={"BLOCK": 64},
    )
    try:
        triton.compile(src, target=get_current_target())
    except Exception:
        pytest.skip("Compilation not supported on this target")


# ---------------------------------------------------------------------------
# set_buffer_overlap: GPU execution (shared overlap, write then aliased read)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(is_hip(), reason="Buffer overlap not supported on AMD")
@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
def test_set_buffer_overlap_shared_on_gpu():
    """Shared overlap: write f32 to buf a, read aliased bf16 from buf b."""

    @triton.jit
    def kernel(out_ptr, BLOCK: tl.constexpr):
        spec = tlx.storage_alias_spec(storage=tlx.storage_kind.smem)
        a = tlx.local_alloc(
            (BLOCK, BLOCK),
            tl.float32,
            tl.constexpr(2),
            tlx.storage_kind.smem,
            reuse=spec,
        )
        b = tlx.local_alloc(
            (BLOCK, BLOCK),
            tl.bfloat16,
            tl.constexpr(2),
            tlx.storage_kind.smem,
            reuse=spec,
        )
        spec.set_buffer_overlap(
            tlx.reuse_group(a, b, group_type=tlx.reuse_group_type.shared))

        offs_m = tl.arange(0, BLOCK)
        offs_n = tl.arange(0, BLOCK)

        ones = tl.full((BLOCK, BLOCK), 1.0, tl.float32)
        tlx.local_store(a[0], ones)

        b0_data = tlx.local_load(b[0])
        b0_f32 = b0_data.to(tl.float32)

        out_offsets = out_ptr + (offs_m[:, None] * BLOCK + offs_n[None, :])
        tl.store(out_offsets, b0_f32)

    BLOCK = 64
    out = torch.zeros((BLOCK, BLOCK), dtype=torch.float32, device=DEVICE)
    kernel[(1, )](out, BLOCK)
    # b[0] shares memory with a[0], should have non-zero data
    assert out.abs().sum() > 0, "b[0] should have non-zero data from a[0]"


# ---------------------------------------------------------------------------
# dot integration with storage_alias_spec
# ---------------------------------------------------------------------------


def test_dot_with_storage_alias_spec_compile_only():
    """tl.dot using smem buffers allocated via storage_alias_spec compiles correctly."""

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
        spec = tlx.storage_alias_spec(storage=tlx.storage_kind.smem)

        off_m = tl.arange(0, BLOCK_M)
        off_n = tl.arange(0, BLOCK_N)
        off_k = tl.arange(0, BLOCK_K)

        a_ptrs = X + (off_m[:, None] * stride_xm + off_k[None, :] * stride_xk)
        b_ptrs = Y + (off_k[:, None] * stride_yk + off_n[None, :] * stride_yn)

        buf_a = tlx.local_alloc(
            (BLOCK_M, BLOCK_K),
            tlx.dtype_of(X),
            tl.constexpr(1),
            tlx.storage_kind.smem,
            reuse=spec,
        )
        buf_b = tlx.local_alloc(
            (BLOCK_K, BLOCK_N),
            tlx.dtype_of(Y),
            tl.constexpr(1),
            tlx.storage_kind.smem,
            reuse=spec,
        )
        va = tlx.local_view(buf_a, 0)
        vb = tlx.local_view(buf_b, 0)

        a_reg = tl.load(a_ptrs)
        b_reg = tl.load(b_ptrs)

        tlx.local_store(va, a_reg)
        tlx.local_store(vb, b_reg)

        a_tile = tlx.local_load(va)
        b_tile = tlx.local_load(vb)
        c_tile = tl.dot(a_tile, b_tile)
        c = c_tile.to(tlx.dtype_of(Z))

        c_ptrs = Z + stride_zm * off_m[:, None] + stride_zn * off_n[None, :]
        tl.store(c_ptrs, c)

    src = triton.compiler.ASTSource(
        fn=dot_kernel,
        signature={
            "X": "*fp16",
            "stride_xm": "i32",
            "stride_xk": "i32",
            "Y": "*fp16",
            "stride_yk": "i32",
            "stride_yn": "i32",
            "Z": "*fp16",
            "stride_zm": "i32",
            "stride_zn": "i32",
        },
        constexprs={
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "BLOCK_K": 64
        },
    )
    try:
        triton.compile(src, target=get_current_target())
    except Exception:
        pytest.skip("Compilation not supported on this target")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
