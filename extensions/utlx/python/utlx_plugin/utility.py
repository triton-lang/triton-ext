"""uTLX utility functions."""

import os
import re
import sys

import triton.language.core as tl


def ensure_plugin_on_path():
    """Add the uTLX plugin python directory to sys.path."""
    plugin_python_dir = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    if plugin_python_dir not in sys.path:
        sys.path.insert(0, plugin_python_dir)


def is_hip():
    try:
        from triton.runtime import driver
        target = driver.active.get_current_target()
        return target.backend == "hip"
    except Exception:
        return False


def cuda_parse_arch(arch):
    pattern = r"^sm(\d+)$"
    match = re.fullmatch(pattern, arch)
    if not match:
        raise ValueError(f"arch must have the form {pattern}")
    return int(match.group(1))


@tl.builtin
def cluster_cta_rank(_semantic=None):
    return tl.tensor(_semantic.builder.utlx_cluster_cta_rank([]), tl.int32)


@tl.builtin
def cluster_size_1d(_semantic=None):
    return tl.tensor(_semantic.builder.utlx_cluster_size_1d([]), tl.int32)


@tl.builtin
def thread_id(axis, _semantic=None):
    axis = tl._unwrap_if_constexpr(axis)
    if axis not in (0, 1, 2):
        raise ValueError(f"thread_id axis must be 0, 1, or 2 but got {axis}")
    return tl.tensor(
        _semantic.builder.utlx_thread_id([_semantic.builder.get_int32(axis)]),
        tl.int32)


@tl.builtin
def async_task_replica_id(_semantic=None):
    from utlx_plugin.compiler.code_generator import _get_region_replica_id_stack

    region_replica_id_stack = _get_region_replica_id_stack()
    assert len(region_replica_id_stack) > 0, (
        "async_task_replica_id must be called inside an async region where the stack must be non-empty"
    )
    return tl.constexpr(region_replica_id_stack[-1])


@tl.builtin
def dtype_of(v, _semantic=None) -> tl.dtype:
    if isinstance(v, tl.tensor):
        dtype = v.type.element_ty
        if dtype.is_ptr():
            dtype = dtype.element_ty
        return dtype
    elif isinstance(v, tl.tensor_descriptor_base):
        return v.dtype
    else:
        src_ty = getattr(v, 'type', None)
        if src_ty is not None and isinstance(src_ty, tl.pointer_type):
            return tl.constexpr(src_ty.element_ty)
        raise ValueError(
            f"dtype_of only works on tensors and tensor descriptors, but got {v}"
        )


@tl.builtin
def size_of(dtype: tl.dtype, _semantic=None) -> tl.constexpr:
    dtype = tl._unwrap_if_constexpr(dtype)
    assert isinstance(
        dtype, tl.dtype), f"size_of expects a dtype, but got {type(dtype)}"
    return tl.constexpr(dtype.primitive_bitwidth // 8)


@tl.builtin
def get_fp8_format_name(dtype: tl.dtype, _semantic=None) -> tl.constexpr:
    dtype = tl._unwrap_if_constexpr(dtype)
    assert isinstance(
        dtype, tl.dtype
    ), f"get_fp8_format_name expects a dtype, but got {type(dtype)}"
    if dtype == tl.float8e5:
        return tl.constexpr("e5m2")
    elif dtype == tl.float8e4nv:
        return tl.constexpr("e4m3")
    else:
        raise AssertionError(
            f"get_fp8_format_name only supports tl.float8e5 (e5m2) and tl.float8e4nv (e4m3), "
            f"but got {dtype}")


@tl.builtin
def clock64(_semantic=None):
    return tl.tensor(_semantic.builder.utlx_clock64([]), tl.int64)


@tl.builtin
def stoch_round(
    src: tl.tensor,
    dst_ty: tl.dtype,
    rand_bits: tl.tensor,
    _semantic=None,
) -> tl.tensor:
    capability = int(cuda_parse_arch(_semantic.builder.options.arch))
    assert capability >= 100, (
        f"stoch_round requires compute capability >= 100 (Blackwell GPU), "
        f"current capability: {capability}")
    src_ty = src.type
    src_sca_ty = src_ty.scalar

    assert src_sca_ty == tl.float32, (
        f"Stochastic rounding only supports fp32 source, got {src_sca_ty}.")
    assert dst_ty in [
        tl.float8e5, tl.float8e4nv, tl.float16, tl.bfloat16
    ], (f"Stochastic rounding only supports fp8/fp16/bf16 destination, got {dst_ty}."
        )

    rbits_ty = rand_bits.type
    if src_ty.is_block() and rbits_ty.is_block():
        assert src_ty.shape == rbits_ty.shape, f"rand_bits shape {rbits_ty.shape} must match src shape {src_ty.shape}"

    if src_sca_ty == dst_ty:
        return src

    if src_ty.is_block():
        result_ty = src_ty.with_element_ty(dst_ty)
        dst_ir_ty = result_ty.to_ir(_semantic.builder)
    else:
        result_ty = dst_ty
        dst_ir_ty = dst_ty.to_ir(_semantic.builder)
    # Use fp_to_fp with rounding=RS (stochastic), mode=2
    dst = _semantic.builder.utlx_fp_to_fp_with_rbits([
        _semantic.builder.get_null_value(dst_ir_ty), src.handle,
        rand_bits.handle,
        _semantic.builder.get_int32(2)
    ])
    return tl.tensor(dst, result_ty)
