"""uTLX MMA operations.

These ops use gluon builder methods for warp group dot,
tcgen05 dot, and layout operations.
"""

import triton.language.core as tl

from . import types as tlx

import re


def _cuda_parse_arch(arch):
    pattern = r"^sm(\d+)$"
    match = re.fullmatch(pattern, arch)
    if not match:
        raise ValueError(f"arch must have the form {pattern}")
    return int(match.group(1))


def require_nv_mma_shared_layout(x, swizzled, _builder=None, fp4Padded=False):
    assert isinstance(
        x.type.layout,
        tlx.shared_layout_encoding), "input must be a shared tensor"
    rank = len(x.shape)
    layout = tlx.nv_mma_shared_layout_encoding(
        shape=x.shape,
        order=x.type.layout.order,
        elemType=x.dtype,
        numCTAsPerCGA=[1] * rank,
        numCTASplit=[1] * rank,
        numCTAOrder=[1] * rank,
        fp4Padded=fp4Padded,
        swizzled=swizzled,
    )
    # Combined custom op: creates encoding + RequireLayoutOp in one step
    result = _builder.utlx_require_nv_mma_shared_layout(
        [x.handle] + [_builder.get_int32(int(s)) for s in layout.shape] +
        [_builder.get_int32(o) for o in layout.order] + [
            _builder.get_int32(1 if layout.fp4Padded else 0),
            _builder.get_int32(1 if layout.swizzled else 0)
        ])
    if result is not None:
        return result
    return x.handle


def require_dot_operand_layout(opnd, opIdx, parent_value, _builder=None):
    # parent_value is a Value whose type carries the parent NvidiaMmaEncoding
    result = _builder.utlx_require_dot_operand_layout(
        [opnd.handle, _builder.get_int32(opIdx), parent_value])
    if result is not None:
        return result
    return opnd.handle


def require_tmem_layout_col_stride(src, col_stride, _builder=None):
    assert (isinstance(src, tlx.buffered_tensor)
            and src.type.storage == tlx.storage_kind.tmem
            and isinstance(src.type.layout, tlx.tensor_memory_layout_encoding))
    old_layout = src.type.layout
    if old_layout.colStride != col_stride:
        result = _builder.utlx_require_tensor_memory_layout([
            src.handle,
            _builder.get_int32(old_layout.blockM),
            _builder.get_int32(old_layout.blockN),
            _builder.get_int32(col_stride),
            _builder.get_int32(old_layout.CTASplitM),
            _builder.get_int32(old_layout.CTASplitN)
        ])
        if result is not None:
            return result
    return src.handle


def require_tmem_scales_layout(src, _builder=None):
    assert isinstance(
        src, tlx.buffered_tensor) and src.type.storage == tlx.storage_kind.tmem
    result = _builder.utlx_require_tensor_memory_scales_layout([src.handle])
    if result is not None:
        return result
    return src.handle


@tl.builtin
def async_dot(
    A,
    B,
    acc=None,
    use_acc=None,
    pred=None,
    mBarriers=None,
    two_ctas=False,
    force_async=False,
    input_precision=None,
    out_dtype=tl.float32,
    _semantic=None,
) -> tl.tensor:
    """Warp-group matrix multiply-accumulate (Hopper wgmma / Blackwell tcgen05.mma)."""
    if mBarriers is None:
        mBarriers = []

    (A, B, acc_handle, input_precision, max_num_imprecise_acc,
     ret_ty) = _semantic.dot_precheck(A, B, acc, input_precision, None, None,
                                      out_dtype, two_ctas)

    assert A.shape[0] >= 64, "M must be at least 64"
    assert A.shape[1] >= 16, "K must be at least 16"
    assert B.shape[1] >= 32, "N must be at least 32"

    cuda_compute_capability = _cuda_parse_arch(_semantic.builder.options.arch)
    version = 5 if cuda_compute_capability >= 100 else 3

    if isinstance(
            A,
            tlx.buffered_tensor) and A.type.storage == tlx.storage_kind.smem:
        A_handle = require_nv_mma_shared_layout(A, True, _semantic.builder)
    elif isinstance(A, tl.tensor):
        assert cuda_compute_capability < 100, "register operand is not supported on Blackwell"
        A_handle = A.handle
    else:
        A_handle = require_tmem_layout_col_stride(A, 1, _semantic.builder)

    B_handle = require_nv_mma_shared_layout(B, True, _semantic.builder)

    if version == 5:
        assert isinstance(A, tlx.buffered_tensor)
        acc_handle = require_tmem_layout_col_stride(acc, 1, _semantic.builder)
        handles = [t.handle for t in mBarriers]
        force_async or len(handles) > 0
        use_acc_handle = None
        if use_acc is not None:
            if isinstance(use_acc, tl.tensor):
                use_acc_handle = use_acc.handle
            else:
                use_acc_handle = _semantic.builder.get_int1(use_acc.value)
        # Use gluon: create_tcgen05_mma(a, b, acc, useAcc, pred, mbarriers, mbarrier_preds, two_ctas, multicast)
        _semantic.builder.create_tcgen05_mma(A_handle, B_handle, acc_handle,
                                             use_acc_handle, pred, handles, [],
                                             two_ctas, False)
        return tl.tensor(acc_handle, tl.void)
    else:
        # Create NvidiaMma encoding and apply it to acc via combined custom op
        acc_with_layout = _semantic.builder.utlx_require_nv_mma_layout([
            A_handle, acc_handle,
            _semantic.builder.get_int32(version),
            _semantic.builder.get_int32(0),
            _semantic.builder.get_int32(_semantic.builder.options.num_warps)
        ])
        if isinstance(A, tl.tensor):
            # Apply dot operand encoding to A; pass acc_with_layout as parent carrier
            A_handle = require_dot_operand_layout(A, 0, acc_with_layout,
                                                  _semantic.builder)
        # Use gluon: create_warpgroup_mma(a, b, acc, useAcc, precision, maxNumImpreciseAcc, isAsync)
        output = _semantic.builder.create_warpgroup_mma(
            A_handle, B_handle, acc_with_layout, None, input_precision,
            max_num_imprecise_acc, True)
        output = _semantic.builder.utlx_release_layout([output])
        return tl.tensor(output, ret_ty)


@tl.builtin
def async_dot_scaled(
    A,
    B,
    acc,
    A_scale,
    A_format,
    B_scale,
    B_format,
    use_acc=None,
    pred=None,
    mBarriers=None,
    two_ctas=False,
    force_async=False,
    out_dtype=tl.float32,
    _semantic=None,
) -> tl.tensor:
    """Scaled warp-group MMA using Blackwell tcgen05.mma."""
    if mBarriers is None:
        mBarriers = []

    assert A.shape[0] >= 64
    assert A.shape[1] >= 16
    assert B.shape[1] >= 32

    cuda_compute_capability = _cuda_parse_arch(_semantic.builder.options.arch)
    assert cuda_compute_capability >= 100, "async_dot_scaled is only available on Blackwell"

    supported_formats = {"e2m1", "e4m3", "e5m2"}
    A_format = tl._unwrap_if_constexpr(A_format)
    B_format = tl._unwrap_if_constexpr(B_format)
    assert A_format in supported_formats
    assert B_format in supported_formats
    A_type = _semantic._str_to_fp_type(A_format)
    B_type = _semantic._str_to_fp_type(B_format)

    is_A_fp4 = A_format == "e2m1"
    is_B_fp4 = B_format == "e2m1"
    is_mixed = A_format != B_format
    A_handle = require_nv_mma_shared_layout(A,
                                            True,
                                            _semantic.builder,
                                            fp4Padded=is_A_fp4 and is_mixed)
    B_handle = require_nv_mma_shared_layout(B,
                                            True,
                                            _semantic.builder,
                                            fp4Padded=is_B_fp4 and is_mixed)

    if A_scale.type.storage == tlx.storage_kind.tmem:
        A_scale_handle = require_tmem_scales_layout(A_scale, _semantic.builder)
    else:
        A_scale_handle = require_nv_mma_shared_layout(A_scale, False,
                                                      _semantic.builder)

    if B_scale.type.storage == tlx.storage_kind.tmem:
        B_scale_handle = require_tmem_scales_layout(B_scale, _semantic.builder)
    else:
        B_scale_handle = require_nv_mma_shared_layout(B_scale, False,
                                                      _semantic.builder)

    acc_handle = require_tmem_layout_col_stride(acc, 1, _semantic.builder)
    bar_handles = [t.handle for t in mBarriers]
    use_acc_handle = None
    if use_acc is not None:
        if isinstance(use_acc, tl.tensor):
            use_acc_handle = use_acc.handle
        else:
            use_acc_handle = _semantic.builder.get_int1(use_acc.value)
    # Use gluon: create_tcgen05_mma_scaled
    _semantic.builder.create_tcgen05_mma_scaled(A_handle, B_handle, acc_handle,
                                                A_scale_handle, B_scale_handle,
                                                A_type, B_type, use_acc_handle,
                                                pred, bar_handles, [],
                                                two_ctas)
    return tl.tensor(acc_handle, tl.void)


@tl.builtin
def async_dot_wait(pendings: tl.constexpr,
                   inp: tl.tensor,
                   _semantic=None) -> tl.tensor:
    """Wait for completion of prior asynchronous dot operations."""
    pendings = tl._unwrap_if_constexpr(pendings)
    # Use custom op that handles ReleaseLayoutOp unwrap/rewire
    result = _semantic.builder.utlx_warp_group_dot_wait(
        [inp.handle, _semantic.builder.get_int32(pendings)])
    return tl.tensor(result, inp.type)


@tl.builtin
def tcgen05_commit(mBarrier, two_ctas=False, _semantic=None) -> tl.tensor:
    """Make the mbarrier track completion of all prior tcgen5 operations."""
    if not two_ctas:
        pred_handle = _semantic.builder.get_int1(True)
    else:
        cta_rank = _semantic.builder.utlx_cluster_cta_rank([])
        mod_result = _semantic.builder.create_urem(
            cta_rank, _semantic.builder.get_int32(2))
        pred_handle = _semantic.builder.create_icmpEQ(
            mod_result, _semantic.builder.get_int32(0))
    # Use gluon: create_tcgen05_commit(barrier, pred, descs)
    _semantic.builder.create_tcgen05_commit(mBarrier.handle, pred_handle, [])
    return tl.tensor(mBarrier.handle, tl.void)
