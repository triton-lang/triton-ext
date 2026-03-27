"""uTLX Plugin memory ops: local_alloc, local_view, local_store, local_load, storage_alias_spec.

These ops call plugin custom ops registered by uTLXPlugin.cpp:
  - utlx_local_alloc(type_carrier, *shape_dims, target_hint) -> MemDesc Value
  - utlx_local_view(alloc, buffer_idx) -> MemDesc Value
  - utlx_local_store(dst, src) -> void
  - utlx_local_load(subView) -> tensor Value
  - utlx_storage_alias_spec(storage_kind, buffer_size_bytes) -> StorageAliasSpec
  - utlx_storage_alias_local_alloc(spec, type_carrier, *shape_dims, storage_hint) -> MemDesc
"""

from typing import Optional

import triton.language.core as tl

from . import types as tlx


# Map triton dtype to builder method name for creating a type-carrier constant
_DTYPE_TO_BUILDER_METHOD = {
    tl.float16: "get_fp16",
    tl.bfloat16: "get_bf16",
    tl.float32: "get_fp32",
    tl.float64: "get_fp64",
    tl.int8: "get_int8",
    tl.int16: "get_int16",
    tl.int32: "get_int32",
    tl.int64: "get_int64",
    tl.uint8: "get_uint8",
    tl.uint16: "get_uint16",
    tl.uint32: "get_uint32",
    tl.uint64: "get_uint64",
}


def _make_type_carrier(builder, dtype):
    """Create a type-carrier scalar constant of the desired element type."""
    builder_method = _DTYPE_TO_BUILDER_METHOD.get(dtype)
    if builder_method is None:
        raise ValueError(f"Unsupported dtype: {dtype}")

    if builder_method.startswith("get_fp") or builder_method.startswith("get_bf"):
        return getattr(builder, builder_method)(0.0)
    else:
        return getattr(builder, builder_method)(0)


def _detect_amd(builder):
    """Detect AMD target for SMEM encoding selection."""
    arch = getattr(getattr(builder, 'options', None), 'arch', '')
    return isinstance(arch, str) and arch.startswith('gfx')


@tl.builtin
def local_alloc(
    shape: tuple,
    dtype: tl.dtype,
    num: tl.constexpr,
    storage: tlx.storage_kind = tlx.storage_kind.smem,
    layout=None,
    reuse=None,
    _semantic=None,
) -> tlx.buffered_tensor:
    """
    Allocate buffers in shared/tensor memory and return a buffered_tensor.

    Args:
        shape: Shape of each buffer (excluding the num dimension).
        dtype: Data type of the buffer elements.
        num: Number of buffers to allocate (compile-time constant).
        storage: Storage kind (smem or tmem).
        layout: Optional memory layout encoding.
        reuse: Optional storage_alias_spec for buffer sharing.

    Returns:
        A buffered_tensor representing the allocated buffers.
    """
    unwrapped_shape = [tl._unwrap_if_constexpr(dim) for dim in shape]
    unwrapped_num = tl._unwrap_if_constexpr(num)
    full_shape = [unwrapped_num] + unwrapped_shape
    dtype = tl._unwrap_if_constexpr(dtype)

    # If reuse is a storage_alias_spec, use storage_alias_local_alloc
    if reuse is not None and isinstance(reuse, tlx.storage_alias_spec):
        return _local_alloc_with_storage_alias(
            _semantic, reuse, dtype, full_shape, unwrapped_shape, unwrapped_num, storage
        )

    if storage == tlx.storage_kind.tmem:
        raise NotImplementedError(
            "uTLX plugin tmem local_alloc without storage_alias_spec "
            "requires the full dialect plugin. Use storage_alias_spec for tmem."
        )

    type_carrier = _make_type_carrier(_semantic.builder, dtype)
    shape_values = [_semantic.builder.get_int32(int(dim)) for dim in full_shape]
    is_amd = _detect_amd(_semantic.builder)
    target_hint = _semantic.builder.get_int32(1 if is_amd else 0)

    args = [type_carrier] + shape_values + [target_hint]
    tensor_handle = _semantic.builder.utlx_local_alloc(args)

    if len(unwrapped_shape) == 1 or is_amd:
        py_layout = tlx.swizzled_shared_layout_encoding.make_default(
            rank=len(unwrapped_shape)
        )
    else:
        py_layout = tlx.nv_mma_shared_layout_encoding.make_default(
            unwrapped_shape, dtype
        )

    return tlx.buffered_tensor(
        tensor_handle, dtype, unwrapped_shape, unwrapped_num, storage, py_layout
    )


def _local_alloc_with_storage_alias(semantic, spec, dtype, full_shape,
                                    unwrapped_shape, unwrapped_num, storage):
    """Allocate via storage_alias_local_alloc custom op."""
    type_carrier = _make_type_carrier(semantic.builder, dtype)
    shape_values = [semantic.builder.get_int32(int(dim)) for dim in full_shape]
    is_tmem = storage == tlx.storage_kind.tmem
    storage_hint = semantic.builder.get_int32(1 if is_tmem else 0)

    args = [spec.handle, type_carrier] + shape_values + [storage_hint]
    tensor_handle = semantic.builder.utlx_storage_alias_local_alloc(args)

    if is_tmem:
        py_layout = tlx.tensor_memory_layout_encoding.make_default(unwrapped_shape)
    elif len(unwrapped_shape) == 1 or _detect_amd(semantic.builder):
        py_layout = tlx.swizzled_shared_layout_encoding.make_default(
            rank=len(unwrapped_shape)
        )
    else:
        py_layout = tlx.nv_mma_shared_layout_encoding.make_default(
            unwrapped_shape, dtype
        )

    return tlx.buffered_tensor(
        tensor_handle, dtype, unwrapped_shape, unwrapped_num, storage, py_layout
    )


@tl.builtin
def local_view(
    local_allocated_buffers: tlx.buffered_tensor,
    buffer_idx: int,
    _semantic=None,
) -> tlx.buffered_tensor:
    """Returns a subview of the buffer at the given index."""
    buffer_idx = _semantic._convert_elem_to_ir_value(buffer_idx, require_i64=False)
    view_handle = _semantic.builder.utlx_local_view(
        [local_allocated_buffers.handle, buffer_idx]
    )

    original_shape = local_allocated_buffers.shape
    if local_allocated_buffers.type.num == 0:
        if len(original_shape) == 1:
            new_shape = [1]
        else:
            new_shape = original_shape[1:]
    else:
        new_shape = original_shape

    return tlx.buffered_tensor(
        view_handle,
        local_allocated_buffers.dtype,
        new_shape,
        0,
        local_allocated_buffers.type.storage,
        local_allocated_buffers.type.layout,
    )


@tl.builtin
def local_load(
    src: tlx.buffered_tensor,
    _semantic=None,
) -> tl.tensor:
    """Load from SMEM buffer into a register tensor."""
    block_type = tl.block_type(src.type.element_ty, src.type.shape)
    output = _semantic.builder.utlx_local_load([src.handle])
    return tl.tensor(output, block_type)


@tl.builtin
def local_store(
    dst: tlx.buffered_tensor,
    src: tl.tensor,
    _semantic=None,
) -> tl.tensor:
    """Store a register tensor into an SMEM buffer."""
    _semantic.builder.utlx_local_store([dst.handle, src.handle])
    return tl.tensor(src.handle, tl.void)


@tl.builtin
def storage_alias_spec(
    storage: tlx.storage_kind = tlx.storage_kind.smem,
    buffer_size_bytes=None,
    _semantic=None,
) -> tlx.storage_alias_spec:
    """
    Create a storage alias specification for buffer sharing.

    Args:
        storage: Storage kind (smem or tmem).
        buffer_size_bytes: Optional explicit buffer size in bytes.

    Returns:
        A storage_alias_spec that can be passed to local_alloc via `reuse`.
    """
    storage_val = 0 if storage == tlx.storage_kind.smem else 1
    storage_ir = _semantic.builder.get_int32(storage_val)

    size_val = -1
    if buffer_size_bytes is not None:
        size_val = tl._unwrap_if_constexpr(buffer_size_bytes)
    size_ir = _semantic.builder.get_int64(size_val)

    handle = _semantic.builder.utlx_storage_alias_spec(
        [storage_ir, size_ir]
    )

    return tlx.storage_alias_spec(
        handle,
        storage,
        buffer_size_bytes=size_val if size_val >= 0 else None,
    )
