"""uTLX MMA operations.

These ops use standard triton builder methods for warp group dot,
tcgen05 dot, and layout operations. They require the triton build
to have the corresponding pybind methods.
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
    assert isinstance(x.type.layout, tlx.shared_layout_encoding), "input must be a shared tensor"
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
    layout_handle = _builder.make_nv_mma_shared_encoding_attr(
        [int(x) for x in layout.shape],
        layout.order,
        layout.elemType.to_ir(_builder),
        layout.numCTAsPerCGA,
        layout.numCTASplit,
        layout.numCTAOrder,
        layout.fp4Padded,
        layout.swizzled,
    )
    return _builder.create_require_layout(x.handle, layout_handle)


def require_dot_operand_layout(opnd, opIdx, parent_layout, _builder=None):
    layout_handle = _builder.make_dot_operand_encoding_attr(opnd.handle, opIdx, parent_layout)
    return _builder.create_require_layout(opnd.handle, layout_handle)


def require_tmem_layout_col_stride(src, col_stride, _builder=None):
    assert (isinstance(src, tlx.buffered_tensor) and src.type.storage == tlx.storage_kind.tmem
            and isinstance(src.type.layout, tlx.tensor_memory_layout_encoding))
    old_layout = src.type.layout
    if old_layout.colStride != col_stride:
        layout_handle = _builder.make_tensor_memory_encoding_attr(
            old_layout.blockM, old_layout.blockN, col_stride,
            old_layout.CTASplitM, old_layout.CTASplitN)
        return _builder.create_require_layout(src.handle, layout_handle)
    return src.handle


def require_tmem_scales_layout(src, _builder=None):
    assert isinstance(src, tlx.buffered_tensor) and src.type.storage == tlx.storage_kind.tmem
    layout = tlx.tensor_memory_scales_layout_encoding.make_default()
    layout_handle = layout.to_ir(_builder)
    return _builder.create_require_layout(src.handle, layout_handle)


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
     ret_ty) = _semantic.dot_precheck(A, B, acc, input_precision, None, None, out_dtype, two_ctas)

    assert A.shape[0] >= 64, "M must be at least 64"
    assert A.shape[1] >= 16, "K must be at least 16"
    assert B.shape[1] >= 32, "N must be at least 32"

    cuda_compute_capability = _cuda_parse_arch(_semantic.builder.options.arch)
    version = 5 if cuda_compute_capability >= 100 else 3

    if isinstance(A, tlx.buffered_tensor) and A.type.storage == tlx.storage_kind.smem:
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
        is_async = force_async or len(handles) > 0
        use_acc_handle = None
        if use_acc is not None:
            if isinstance(use_acc, tl.tensor):
                use_acc_handle = use_acc.handle
            else:
                use_acc_handle = _semantic.builder.get_int1(use_acc.value)
        output = _semantic.builder.create_tcgen5_dot(
            A_handle, B_handle, acc_handle, use_acc_handle, pred, two_ctas, handles, is_async)
        return tl.tensor(output, tl.void)
    else:
        mma_layout = _semantic.builder.make_nv_mma_encoding_attr(
            A_handle, acc_handle, version, 0, _semantic.builder.options.num_warps)
        acc = _semantic.builder.create_require_layout(acc_handle, mma_layout)
        if isinstance(A, tl.tensor):
            A_handle = require_dot_operand_layout(A, 0, mma_layout, _semantic.builder)
        output = _semantic.builder.create_warp_group_dot(
            A_handle, B_handle, acc, input_precision, max_num_imprecise_acc, True)
        output = _semantic.builder.create_release_layout(output)
        return tl.tensor(output, ret_ty)


@tl.builtin
def async_dot_scaled(
    A, B, acc, A_scale, A_format, B_scale, B_format,
    use_acc=None, pred=None, mBarriers=None, two_ctas=False,
    force_async=False, out_dtype=tl.float32, _semantic=None,
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
    A_handle = require_nv_mma_shared_layout(A, True, _semantic.builder, fp4Padded=is_A_fp4 and is_mixed)
    B_handle = require_nv_mma_shared_layout(B, True, _semantic.builder, fp4Padded=is_B_fp4 and is_mixed)

    if A_scale.type.storage == tlx.storage_kind.tmem:
        A_scale_handle = require_tmem_scales_layout(A_scale, _semantic.builder)
    else:
        A_scale_handle = require_nv_mma_shared_layout(A_scale, False, _semantic.builder)

    if B_scale.type.storage == tlx.storage_kind.tmem:
        B_scale_handle = require_tmem_scales_layout(B_scale, _semantic.builder)
    else:
        B_scale_handle = require_nv_mma_shared_layout(B_scale, False, _semantic.builder)

    acc_handle = require_tmem_layout_col_stride(acc, 1, _semantic.builder)
    bar_handles = [t.handle for t in mBarriers]
    is_async = force_async or len(bar_handles) > 0
    use_acc_handle = None
    if use_acc is not None:
        if isinstance(use_acc, tl.tensor):
            use_acc_handle = use_acc.handle
        else:
            use_acc_handle = _semantic.builder.get_int1(use_acc.value)
    output = _semantic.builder.create_tcgen5_dot_scaled(
        A_handle, B_handle, acc_handle, A_scale_handle, B_scale_handle,
        A_type, B_type, use_acc_handle, pred, two_ctas, bar_handles, is_async)
    return tl.tensor(output, tl.void)


@tl.builtin
def async_dot_wait(pendings: tl.constexpr, inp: tl.tensor, _semantic=None) -> tl.tensor:
    """Wait for completion of prior asynchronous dot operations."""
    pendings = tl._unwrap_if_constexpr(pendings)
    return tl.tensor(_semantic.builder.create_warp_group_dot_wait([inp.handle], pendings)[0], inp.type)


@tl.builtin
def tcgen05_commit(mBarrier, two_ctas=False, _semantic=None) -> tl.tensor:
    """Make the mbarrier track completion of all prior tcgen5 operations."""
    if not two_ctas:
        pred_handle = _semantic.builder.get_int1(True)
    else:
        cta_rank = _semantic.builder.create_cluster_cta_rank()
        mod_result = _semantic.builder.create_urem(cta_rank, _semantic.builder.get_int32(2))
        pred_handle = _semantic.builder.create_icmpEQ(mod_result, _semantic.builder.get_int32(0))
    return tl.tensor(_semantic.builder.create_tcgen05_commit(mBarrier.handle, pred_handle), tl.void)
