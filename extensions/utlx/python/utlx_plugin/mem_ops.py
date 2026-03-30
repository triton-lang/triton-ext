"""uTLX Plugin memory ops.

These ops call plugin custom ops registered by uTLXPlugin.cpp for
operations that require the TLX dialect, and standard triton builder
methods for operations that exist in upstream triton.
"""

from typing import Optional, Tuple, Union

import triton.language.core as tl
from triton._C.libtriton import ir

from . import types as tlx
from .types import storage_kind
from .utility import cuda_parse_arch

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

    if builder_method.startswith("get_fp") or builder_method.startswith(
            "get_bf"):
        return getattr(builder, builder_method)(0.0)
    else:
        return getattr(builder, builder_method)(0)


def _detect_amd(builder):
    """Detect AMD target for SMEM encoding selection."""
    arch = getattr(getattr(builder, 'options', None), 'arch', '')
    return isinstance(arch, str) and arch.startswith('gfx')


def _assert_blackwell_for_tmem(arch):
    capability = int(cuda_parse_arch(arch))
    assert capability >= 100, "tmem is only available on Blackwell"


def _create_tmem_compatible_tensor_layout(builder,
                                          tensor: tlx.buffered_tensor):
    """Create a DummyRegisterLayout encoding for TMEM-compatible register layout."""
    return builder.utlx_make_dummy_register_layout(
        [builder.get_int32(int(s)) for s in tensor.shape] +
        [_make_type_carrier(builder, tensor.dtype),
         builder.get_int32(1)])  # tmemCompatible=True


def _get_remote_cta_rank_handle(remote_cta_rank, _semantic):
    if isinstance(remote_cta_rank, tl.constexpr) or isinstance(
            remote_cta_rank, int):
        return _semantic._convert_elem_to_ir_value(
            tl._unwrap_if_constexpr(remote_cta_rank), require_i64=False)
    else:
        assert isinstance(remote_cta_rank, tl.tensor), (
            f"`remote_cta_rank` must be tl.tensor or tl.constexpr, got {type(remote_cta_rank)}"
        )
        return remote_cta_rank.handle


@tl.builtin
def storage_alias_spec(
    storage: tlx.storage_kind = tlx.storage_kind.smem,
    buffer_size_bytes=None,
    _semantic=None,
) -> tlx.storage_alias_spec:
    """Create a storage alias specification for buffer sharing."""
    storage_val = 0 if storage == tlx.storage_kind.smem else 1
    storage_ir = _semantic.builder.get_int32(storage_val)

    size_val = -1
    if buffer_size_bytes is not None:
        size_val = tl._unwrap_if_constexpr(buffer_size_bytes)
    size_ir = _semantic.builder.get_int64(size_val)

    handle = _semantic.builder.utlx_storage_alias_spec([storage_ir, size_ir])

    return tlx.storage_alias_spec(
        handle,
        storage,
        buffer_size_bytes=size_val if size_val >= 0 else None,
    )


@tl.builtin
def local_alloc(
    shape: tuple,
    dtype: tl.dtype,
    num: tl.constexpr,
    storage: tlx.storage_kind = tlx.storage_kind.smem,
    layout=None,
    reuse: Optional[Union[tlx.buffered_tensor, tlx.storage_alias_spec]] = None,
    _semantic=None,
) -> tlx.buffered_tensor:
    """Allocate buffers in shared/tensor memory and return a buffered_tensor."""
    unwrapped_shape = [tl._unwrap_if_constexpr(dim) for dim in shape]
    unwrapped_num = tl._unwrap_if_constexpr(num)
    full_shape = [unwrapped_num] + unwrapped_shape
    dtype = tl._unwrap_if_constexpr(dtype)

    # If reuse is a storage_alias_spec, use storage_alias_local_alloc
    if reuse is not None and isinstance(reuse, tlx.storage_alias_spec):
        return _local_alloc_with_storage_alias(_semantic, reuse, dtype,
                                               full_shape, unwrapped_shape,
                                               unwrapped_num, storage)

    # If reuse is a buffered_tensor, use local_alias (share memory with existing buffer)
    if reuse is not None and isinstance(reuse, tlx.buffered_tensor):
        return _local_alloc_with_alias(_semantic, reuse, dtype, full_shape,
                                       unwrapped_shape, unwrapped_num, storage)

    if storage == tlx.storage_kind.tmem:
        return _local_alloc_tmem(_semantic, dtype, full_shape, unwrapped_shape,
                                 unwrapped_num)

    type_carrier = _make_type_carrier(_semantic.builder, dtype)
    shape_values = [
        _semantic.builder.get_int32(int(dim)) for dim in full_shape
    ]
    is_amd = _detect_amd(_semantic.builder)
    target_hint = _semantic.builder.get_int32(1 if is_amd else 0)

    args = [type_carrier] + shape_values + [target_hint]
    tensor_handle = _semantic.builder.utlx_local_alloc(args)

    if len(unwrapped_shape) == 1 or is_amd:
        py_layout = tlx.swizzled_shared_layout_encoding.make_default(
            rank=len(unwrapped_shape))
    else:
        py_layout = tlx.nv_mma_shared_layout_encoding.make_default(
            unwrapped_shape, dtype)

    return tlx.buffered_tensor(tensor_handle, dtype, unwrapped_shape,
                               unwrapped_num, storage, py_layout)


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
        py_layout = tlx.tensor_memory_layout_encoding.make_default(
            unwrapped_shape)
    elif len(unwrapped_shape) == 1 or _detect_amd(semantic.builder):
        py_layout = tlx.swizzled_shared_layout_encoding.make_default(
            rank=len(unwrapped_shape))
    else:
        py_layout = tlx.nv_mma_shared_layout_encoding.make_default(
            unwrapped_shape, dtype)

    return tlx.buffered_tensor(tensor_handle, dtype, unwrapped_shape,
                               unwrapped_num, storage, py_layout)


def _local_alloc_tmem(semantic, dtype, full_shape, unwrapped_shape,
                      unwrapped_num):
    """Allocate standalone TMEM (without storage_alias_spec)."""
    type_carrier = _make_type_carrier(semantic.builder, dtype)
    shape_values = [semantic.builder.get_int32(int(dim)) for dim in full_shape]

    # Use DummyTMEMLayoutEncoding for sub-16-bit types (int8/uint8 for scales)
    use_dummy = dtype.primitive_bitwidth < 16
    if use_dummy and dtype not in (tl.uint8, tl.int8):
        raise NotImplementedError(
            f"TMEM layouts not supported for {dtype} yet")
    layout_hint = semantic.builder.get_int32(1 if use_dummy else 0)

    args = [type_carrier] + shape_values + [layout_hint]
    tensor_handle = semantic.builder.utlx_local_alloc_tmem(args)

    if use_dummy:
        py_layout = tlx.DummyTMEMLayoutEncoding()
    else:
        py_layout = tlx.tensor_memory_layout_encoding.make_default(
            unwrapped_shape)

    return tlx.buffered_tensor(tensor_handle, dtype, unwrapped_shape,
                               unwrapped_num, storage_kind.tmem, py_layout)


def _local_alloc_with_alias(semantic, reuse_tensor, dtype, full_shape,
                            unwrapped_shape, unwrapped_num, storage):
    """Allocate via utlx_local_alias (share memory with existing buffered_tensor)."""
    if reuse_tensor.type.storage != storage:
        raise ValueError(
            f"reuse tensor has storage {reuse_tensor.type.storage} but "
            f"allocation requests {storage}")
    type_carrier = _make_type_carrier(semantic.builder, dtype)
    shape_values = [semantic.builder.get_int32(int(dim)) for dim in full_shape]
    is_tmem = storage == storage_kind.tmem
    storage_hint = semantic.builder.get_int32(1 if is_tmem else 0)

    args = [reuse_tensor.handle, type_carrier] + shape_values + [storage_hint]
    tensor_handle = semantic.builder.utlx_local_alias(args)

    if is_tmem:
        py_layout = tlx.tensor_memory_layout_encoding.make_default(
            unwrapped_shape)
    elif len(unwrapped_shape) == 1 or _detect_amd(semantic.builder):
        py_layout = tlx.swizzled_shared_layout_encoding.make_default(
            rank=len(unwrapped_shape))
    else:
        py_layout = tlx.nv_mma_shared_layout_encoding.make_default(
            unwrapped_shape, dtype)

    return tlx.buffered_tensor(tensor_handle, dtype, unwrapped_shape,
                               unwrapped_num, storage, py_layout)


@tl.builtin
def local_view(
    local_allocated_buffers,
    buffer_idx: int,
    _semantic=None,
):
    """Returns a subview of the buffer at the given index."""
    buffer_idx = _semantic._convert_elem_to_ir_value(buffer_idx,
                                                     require_i64=False)
    view_handle = _semantic.builder.utlx_local_view(
        [local_allocated_buffers.handle, buffer_idx])

    if isinstance(local_allocated_buffers, tlx.mbarrier):
        return tlx.mbarrier(
            view_handle,
            0,
            local_allocated_buffers.type.layout,
            is_warp_barrier=local_allocated_buffers.is_warp_barrier)
    elif isinstance(local_allocated_buffers, tlx.clc_response):
        return tlx.clc_response(view_handle, 0,
                                local_allocated_buffers.type.layout)
    else:
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
def remote_view(
    local_allocated_buffer: tlx.mbarrier,
    remote_cta_rank,
    _semantic=None,
) -> tlx.mbarrier:
    """Returns a remote view of the buffer in another CTA."""
    assert isinstance(
        local_allocated_buffer,
        tlx.mbarrier), "remote_view only supports barrier for now"
    assert local_allocated_buffer.type.storage == storage_kind.smem, "remote_view requires local smem as input"
    remote_cta_rank_handle = _get_remote_cta_rank_handle(
        remote_cta_rank, _semantic)
    remote_buf_handle = _semantic.builder.utlx_map_to_remote_buffer(
        [local_allocated_buffer.handle, remote_cta_rank_handle])
    return tlx.mbarrier(remote_buf_handle, 0,
                        local_allocated_buffer.type.layout,
                        storage_kind.smemCluster)


@tl.builtin
def remote_shmem_store(
    dst: tlx.buffered_tensor,
    src: tl.tensor,
    remote_cta_rank,
    _semantic=None,
) -> tl.tensor:
    """Store a distributed tensor into remote shared memory."""
    assert dst.type.storage == tlx.storage_kind.smem
    assert remote_cta_rank is not None
    remote_cta_rank_handle = _get_remote_cta_rank_handle(
        remote_cta_rank, _semantic)
    _semantic.builder.utlx_remote_shmem_store(
        [src.handle, dst.handle, remote_cta_rank_handle])
    return tl.tensor(src.handle, tl.void)


@tl.builtin
def async_remote_shmem_store(
    dst: tlx.buffered_tensor,
    src: tl.tensor,
    remote_cta_rank,
    barrier: tlx.mbarrier,
    _semantic=None,
) -> tl.tensor:
    """Asynchronously store a tensor into remote shared memory."""
    assert dst.type.storage == tlx.storage_kind.smem
    assert remote_cta_rank is not None
    assert barrier is not None
    remote_cta_rank_handle = _get_remote_cta_rank_handle(
        remote_cta_rank, _semantic)
    _semantic.builder.utlx_async_remote_shmem_store(
        [src.handle, dst.handle, remote_cta_rank_handle, barrier.handle])
    return tl.tensor(src.handle, tl.void)


@tl.builtin
def subslice(
    local_allocated_buffer: tlx.buffered_tensor,
    offset: int,
    size: int,
    _semantic=None,
) -> tlx.buffered_tensor:
    """Returns a subslice of the buffer (in TMEM) along the innermost dimension."""
    assert local_allocated_buffer.type.storage == tlx.storage_kind.tmem, "subslice is only supported for tmem"
    subslice_shape = [dim for dim in local_allocated_buffer.type.shape[:-1]
                      ] + [size]
    return tlx.buffered_tensor(
        _semantic.builder.create_tmem_subslice(local_allocated_buffer.handle,
                                               offset, size),
        local_allocated_buffer.type.element_ty,
        subslice_shape,
        local_allocated_buffer.type.num,
        local_allocated_buffer.type.storage,
        local_allocated_buffer.type.layout,
    )


@tl.builtin
def local_slice(
    buffer: tlx.buffered_tensor,
    offset: list,
    shape: list,
    _semantic=None,
) -> tlx.buffered_tensor:
    """Returns a slice of the buffer."""
    if buffer.type.storage == tlx.storage_kind.tmem:
        assert len(offset) == 2 and len(shape) == 2
        assert offset[0] == 0
        assert shape[0] == buffer.type.shape[0]
        return subslice(buffer, offset[1], shape[1], _semantic=_semantic)
    else:
        slice_handle = _semantic.builder.create_memdesc_subslice(
            buffer.handle, offset, shape)
        return tlx.buffered_tensor(
            slice_handle,
            buffer.type.scalar,
            shape,
            0,
            buffer.type.storage,
            buffer.type.layout,
        )


@tl.builtin
def local_trans(input: tlx.buffered_tensor,
                dims: Tuple[int, ...] = (1, 0),
                _semantic=None) -> tlx.buffered_tensor:
    """Permutes the dimensions of a buffered tensor."""
    if len(input.type.shape) != len(dims):
        raise ValueError(
            "permute dims must have the same length as input shape")
    if sorted(tl._unwrap_if_constexpr(d)
              for d in dims) != list(range(len(dims))):
        raise ValueError(
            f"permute dims must be a permutation of 0, 1, ..., n-1, but were {dims}"
        )

    permuted_handle = _semantic.builder.create_memdesc_trans(
        input.handle, dims)
    return input.make_permute(permuted_handle, dims)


@tl.builtin
def local_reinterpret(
    src: tlx.buffered_tensor,
    dtype: tl.dtype,
    shape=None,
    _semantic=None,
) -> tlx.buffered_tensor:
    """Reinterpret the dtype and shape of a buffered tensor."""
    if shape is None:
        shape = src.type.shape
    else:
        assert isinstance(
            src, tlx.buffered_tensor
        ) and src.type.storage == tlx.storage_kind.smem, (
            "TLX local_reinterpret with reshaping only supports SMEM")

    reinterpreted_value_handle = _semantic.builder.create_memdesc_reinterpret(
        src.handle, dtype.to_ir(_semantic.builder), shape)
    return tlx.buffered_tensor(reinterpreted_value_handle, dtype, shape,
                               src.type.num, src.type.storage, src.type.layout)


@tl.builtin
def async_load(
    src: tl.tensor,
    result: tlx.buffered_tensor,
    mask=None,
    other=None,
    cache_modifier: str = "",
    eviction_policy: str = "",
    is_volatile: bool = False,
    bulk: bool = False,
    bulk_size=None,
    barrier: Optional[tlx.mbarrier] = None,
    _semantic=None,
) -> tlx.async_token:
    """Loads buffer from global to local memory asynchronously."""
    bulk = tl._unwrap_if_constexpr(bulk)

    if bulk:
        assert len(result.type.shape
                   ) == 1, "bulk async_load requires a 1D result buffer"
        assert barrier is not None, "bulk async_load requires a barrier"
        assert mask is None and other is None

        dest_bytes = result.type.shape[0] * (
            result.type.element_ty.primitive_bitwidth // 8)
        if bulk_size is None:
            bulk_size = dest_bytes

        const_bulk_size = None
        if isinstance(bulk_size, tl.constexpr):
            const_bulk_size = bulk_size.value
            bulk_size_handle = _semantic.builder.get_int32(bulk_size.value)
        elif isinstance(bulk_size, tl.tensor):
            bulk_size_handle = bulk_size.handle
        else:
            const_bulk_size = int(bulk_size)
            bulk_size_handle = _semantic.builder.get_int32(int(bulk_size))
        if const_bulk_size is not None:
            assert const_bulk_size <= dest_bytes, (
                f"bulk_size ({const_bulk_size}) exceeds destination buffer size ({dest_bytes} bytes)"
            )

        _semantic._str_to_load_cache_modifier(cache_modifier)
        _semantic._str_to_eviction_policy(eviction_policy)
        token = _semantic.builder.utlx_async_load([
            src.handle, result.handle, bulk_size_handle, barrier.handle,
            _semantic.builder.get_int32(1)
        ])  # useBulk=1
        return tlx.async_token(token)

    assert bulk_size is None, "bulk_size requires bulk=True"
    assert barrier is None, "barrier requires bulk=True"

    mask = tl._unwrap_if_constexpr(mask)
    other = tl._unwrap_if_constexpr(other)
    if mask is not None:
        mask = _semantic.to_tensor(mask)
    if other is not None:
        other = _semantic.to_tensor(other)

    if src.type.is_ptr() and src.type.element_ty.is_block():
        raise NotImplementedError(
            "async_load by block pointer is not supported yet")
    else:
        _, src, mask, other, _ = _semantic._prepare_legacy_load(
            src, mask, other, None, None)

    _semantic._str_to_load_cache_modifier(cache_modifier)
    _semantic._str_to_eviction_policy(eviction_policy)

    args = [src.handle, result.handle]
    if mask is not None:
        args.append(mask.handle)
    if other is not None:
        args.append(other.handle)
    args.append(_semantic.builder.get_int32(0))  # useBulk=0
    token = _semantic.builder.utlx_async_load(args)
    return tlx.async_token(token)


@tl.builtin
def async_load_commit_group(
    tokens=None,
    _semantic=None,
) -> tlx.async_token:
    """Commits all prior async_load ops into an async group."""
    if tokens is None:
        tokens = []
    handles = [
        t.handle for t in tokens if t is not None and t.handle is not None
    ]
    result = _semantic.builder.utlx_async_commit_group(handles)
    return tlx.async_token(result)


@tl.builtin
def async_load_wait_group(
    pendings: tl.constexpr,
    tokens=None,
    _semantic=None,
) -> tlx.async_token:
    """Wait for completion of prior asynchronous copy operations."""
    pendings = tl._unwrap_if_constexpr(pendings)
    if tokens is None:
        tokens = []
    handles = [
        t.handle for t in tokens if t is not None and t.handle is not None
    ]
    args = [_semantic.builder.get_int32(pendings)] + handles
    _semantic.builder.utlx_async_wait_group(args)
    return tlx.async_token(None)


@tl.builtin
def local_load(
    src: tlx.buffered_tensor,
    token: Optional[tlx.async_token] = None,
    _semantic=None,
) -> tl.tensor:
    """Load from SMEM/TMEM buffer into a register tensor."""
    block_type = tl.block_type(src.type.element_ty, src.type.shape)
    storage = src.type.storage
    if storage == tlx.storage_kind.tmem:
        _assert_blackwell_for_tmem(_semantic.builder.options.arch)
        tmem_layout = _create_tmem_compatible_tensor_layout(
            _semantic.builder, src)
        load_handle = _semantic.builder.create_tmem_load(
            src.handle, tmem_layout, token.handle if token else None)
        output = _semantic.builder.utlx_release_layout([load_handle])
        return tl.tensor(output, block_type)
    else:
        args = [src.handle]
        if token is not None and token.handle is not None:
            args.append(token.handle)
        output = _semantic.builder.utlx_local_load(args)
        return tl.tensor(output, block_type)


@tl.builtin
def local_store(
    dst: tlx.buffered_tensor,
    src: tl.tensor,
    _semantic=None,
) -> tl.tensor:
    """Store a register tensor into an SMEM/TMEM buffer."""
    storage = dst.type.storage
    if storage == tlx.storage_kind.tmem:
        _assert_blackwell_for_tmem(_semantic.builder.options.arch)
        tmem_layout = _create_tmem_compatible_tensor_layout(
            _semantic.builder, dst)
        src_handle = _semantic.builder.utlx_require_with_layout_carrier(
            [src.handle, tmem_layout])
        return tl.tensor(
            _semantic.builder.create_tmem_store(dst.handle, src_handle),
            tl.void)

    _semantic.builder.utlx_local_store([dst.handle, src.handle])
    return tl.tensor(src.handle, tl.void)


@tl.builtin
def tmem_copy(
    src: tlx.buffered_tensor,
    dst: tlx.buffered_tensor,
    _semantic=None,
) -> None:
    """Asynchronous copy from shared memory to tensor memory (tcgen05.cp)."""
    assert isinstance(src,
                      tlx.buffered_tensor), "source must be a buffered tensor"
    assert isinstance(
        dst, tlx.buffered_tensor), "destination must be a buffered tensor"
    assert src.type.storage == tlx.storage_kind.smem, "source must be in shared memory"
    assert dst.type.storage == tlx.storage_kind.tmem, "destination must be in tensor memory"
    _assert_blackwell_for_tmem(_semantic.builder.options.arch)
    _semantic.builder.create_tmem_copy(src.handle, dst.handle)


@tl.builtin
def async_store(
    dst_global_ptr: tl.tensor,
    src_smem: tlx.buffered_tensor,
    size: tl.tensor,
    _semantic=None,
) -> None:
    """Asynchronously copy from shared memory to global memory."""
    if isinstance(size, tl.constexpr):
        size_handle = _semantic._convert_elem_to_ir_value(size.value,
                                                          require_i64=False)
    elif isinstance(size, tl.tensor):
        size_handle = size.handle
    else:
        size_handle = _semantic._convert_elem_to_ir_value(size,
                                                          require_i64=False)
    _semantic.builder.utlx_async_store(
        [src_smem.handle, dst_global_ptr.handle, size_handle])


@tl.builtin
def fence(scope: tl.constexpr, _semantic=None) -> None:
    """Memory fence with the specified scope."""
    scope = tl._unwrap_if_constexpr(scope)
    if scope == "async_shared":
        _semantic.builder.create_fence_async_shared(False)
    elif scope in ("gpu", "sys"):
        scope_val = 0 if scope == "gpu" else 1
        _semantic.builder.utlx_fence([_semantic.builder.get_int32(scope_val)])
    else:
        raise ValueError(
            f"fence scope must be 'gpu', 'sys', or 'async_shared', got '{scope}'"
        )


@tl.builtin
def fence_async_shared(_semantic=None) -> None:
    """Deprecated: use fence('async_shared') instead."""
    _semantic.builder.create_fence_async_shared(False)


@tl.builtin
def allocate_tensor_descriptor(
    num: tl.constexpr,
    _semantic=None,
) -> tlx.tensor_descriptor_ptr:
    """Allocate buffer in global memory for tensor descriptor storage."""
    if not isinstance(num, tl.constexpr):
        raise ValueError("`num` must be a constexpr")

    unwrapped_num = tl._unwrap_if_constexpr(num)
    descriptor_size = 128
    nbytes = descriptor_size * unwrapped_num
    alignment = 128

    tensor_handle = _semantic.builder.utlx_global_scratch_alloc([
        _semantic.builder.get_int32(nbytes),
        _semantic.builder.get_int32(alignment)
    ])
    return tlx.tensor_descriptor_ptr(tensor_handle, unwrapped_num,
                                     descriptor_size)


@tl.builtin
def make_tensor_descriptor(
    desc_ptr,
    base: tl.tensor,
    shape: list,
    strides: list,
    block_shape: list,
    padding_option="zero",
    _semantic=None,
) -> tl.tensor_descriptor_base:
    """Create a TMA descriptor on device."""
    if desc_ptr is not None and not isinstance(desc_ptr,
                                               tlx.tensor_descriptor_ptr):
        raise TypeError(
            f"desc_ptr must be None or tlx.tensor_descriptor_ptr, got {type(desc_ptr)}"
        )
    ndim = len(shape)
    if not (1 <= ndim <= 5):
        raise ValueError(f"Expected 1 <= ndim <= 5 but got {ndim} dimensions")
    if len(strides) != ndim:
        raise ValueError(f"Expected {ndim} strides but got {len(strides)}")
    if len(block_shape) != ndim:
        raise ValueError(
            f"Expected block_shape to have {ndim} dimensions but got {len(block_shape)}"
        )
    assert isinstance(base.dtype, tl.pointer_type)

    elem_size = base.dtype.element_ty.primitive_bitwidth // 8
    contig_dim_size = tl._unwrap_if_constexpr(block_shape[-1])
    if contig_dim_size * elem_size < 16:
        raise ValueError(
            f"Descriptor block shape must have at least 16 bytes in the last dimension, "
            f"but got {contig_dim_size} * {elem_size} = {contig_dim_size * elem_size} bytes"
        )
    last_stride = tl._unwrap_if_constexpr(strides[-1])
    if last_stride != 1:
        raise ValueError(
            f"Tensor descriptor last dim stride must be 1 but got {last_stride}"
        )

    shape = [_semantic.make_scalar(x, tl.int32) for x in shape]
    strides = [
        _semantic.make_scalar(tl._unwrap_if_constexpr(x), tl.int64)
        for x in strides
    ]
    block_shape = tl._unwrap_shape(block_shape)

    assert isinstance(base.type, tl.pointer_type)
    block_type = tl.block_type(base.type.element_ty, block_shape)
    base_handle = base.handle
    is_signed_int = base.type.element_ty.is_int_signed()
    padding = _semantic._str_to_padding_option(padding_option)

    if base.type.element_ty.is_int() and padding == ir.PADDING_OPTION.PAD_NAN:
        raise ValueError(
            "Padding option `nan` is not supported for integer blocks")

    desc_handle = desc_ptr.handle if desc_ptr is not None else None
    if desc_handle:
        handle = _semantic.builder.create_make_tensor_descriptor(
            base_handle, [s.handle for s in shape],
            [s.handle for s in strides], desc_handle, block_shape,
            is_signed_int, padding)
    else:
        handle = _semantic.builder.create_make_tensor_descriptor(
            base_handle, [s.handle for s in shape],
            [s.handle for s in strides], block_shape, is_signed_int, padding)
    return tl.tensor_descriptor(handle, shape, strides, block_type)


@tl.builtin
def reinterpret_tensor_descriptor(
    desc_ptr: tlx.tensor_descriptor_ptr,
    block_shape: list,
    dtype: tl.dtype,
    _semantic=None,
) -> tl.tensor_descriptor_base:
    """Reinterpret a tensor descriptor pointer as a TMA-backed tensor descriptor."""
    if not isinstance(desc_ptr, tlx.tensor_descriptor_ptr):
        raise TypeError(
            f"desc_ptr must be tlx.tensor_descriptor_ptr, got {type(desc_ptr)}"
        )

    ptr_type = tl.pointer_type(tl.int8)
    tensor_wrapper = tl.tensor(desc_ptr.handle, ptr_type)
    block_ty = tl.block_type(tl._unwrap_if_constexpr(dtype), block_shape)
    return _semantic.reinterpret_tensor_descriptor(tensor_wrapper, block_ty)


@tl.builtin
def async_descriptor_load(
    desc: tl.tensor_descriptor_base,
    result: tlx.buffered_tensor,
    offsets: list,
    barrier: tlx.mbarrier,
    pred: tl.tensor = None,
    cache_modifier: str = "",
    eviction_policy: str = "",
    multicast_targets: Optional[list] = None,
    _semantic=None,
) -> None:
    """Asynchronously load a tensor tile from global memory via TMA."""
    from .mma_ops import require_nv_mma_shared_layout
    if multicast_targets is None:
        multicast_targets = []
    eviction_policy = tl._unwrap_if_constexpr(eviction_policy)
    assert eviction_policy in ("", "evict_first", "evict_last"), \
        f"eviction_policy must be '', 'evict_first', or 'evict_last', got '{eviction_policy}'"
    assert isinstance(desc, tl.tensor_descriptor_base)
    ndim = len(desc.block_shape)
    assert len(offsets) == ndim
    result_handle = require_nv_mma_shared_layout(result, True,
                                                 _semantic.builder)
    offsets = _semantic._convert_to_ir_values(offsets, require_i64=False)
    if pred is None:
        pred_handle = _semantic.builder.get_int1(True)
    else:
        pred_handle = pred.handle
    multicast = len(multicast_targets) > 0
    # Use gluon: create_async_tma_copy_global_to_local(desc, coord, barrier, result, pred, multicast, offsets)
    _semantic.builder.create_async_tma_copy_global_to_local(
        desc.handle, offsets, barrier.handle, result_handle, pred_handle,
        multicast, None)


@tl.builtin
def async_descriptor_prefetch_tensor(
    desc: tl.tensor_descriptor_base,
    offsets: list,
    pred: tl.tensor = None,
    eviction_policy: str = "",
    _semantic=None,
) -> None:
    """Hint hardware to prefetch a tensor tile into L2 cache via TMA."""
    eviction_policy = tl._unwrap_if_constexpr(eviction_policy)
    assert eviction_policy in ("", "evict_first", "evict_last"), \
        f"eviction_policy must be '', 'evict_first', or 'evict_last', got '{eviction_policy}'"
    assert isinstance(desc, tl.tensor_descriptor_base)
    ndim = len(desc.block_shape)
    assert len(offsets) == ndim
    offsets = _semantic._convert_to_ir_values(offsets, require_i64=False)
    _semantic._str_to_eviction_policy(eviction_policy)
    if pred is None:
        pred_handle = _semantic.builder.get_int1(True)
    else:
        pred_handle = pred.handle
    _semantic.builder.utlx_async_tma_prefetch([desc.handle] + offsets +
                                              [pred_handle])


@tl.builtin
def async_descriptor_store(
    desc: tl.tensor_descriptor_base,
    source: tlx.buffered_tensor,
    offsets: list,
    eviction_policy: str = "",
    store_reduce: str = "",
    _semantic=None,
) -> None:
    """Asynchronously store data from shared memory to global memory via TMA."""
    from .mma_ops import require_nv_mma_shared_layout
    assert isinstance(desc, tl.tensor_descriptor_base)
    eviction_policy = tl._unwrap_if_constexpr(eviction_policy)
    assert eviction_policy in ("", "evict_first", "evict_last"), \
        f"eviction_policy must be '', 'evict_first', or 'evict_last', got '{eviction_policy}'"
    store_reduce = tl._unwrap_if_constexpr(store_reduce)
    assert store_reduce in ("", "add", "min", "max", "and", "or", "xor"), \
        f"store_reduce must be one of '', 'add', 'min', 'max', 'and', 'or', 'xor', got '{store_reduce}'"

    ndim = len(desc.block_shape)
    assert len(offsets) == ndim
    source_handle = require_nv_mma_shared_layout(source, True,
                                                 _semantic.builder)
    offsets = _semantic._convert_to_ir_values(offsets, require_i64=False)

    if store_reduce == "":
        # Use gluon: create_async_tma_copy_local_to_global(desc, coord, src)
        _semantic.builder.create_async_tma_copy_local_to_global(
            desc.handle, offsets, source_handle)
    else:
        reduce_kind_map = {
            "add": ir.DESCRIPTOR_REDUCE_KIND.ADD,
            "min": ir.DESCRIPTOR_REDUCE_KIND.MIN,
            "max": ir.DESCRIPTOR_REDUCE_KIND.MAX,
            "and": ir.DESCRIPTOR_REDUCE_KIND.AND,
            "or": ir.DESCRIPTOR_REDUCE_KIND.OR,
            "xor": ir.DESCRIPTOR_REDUCE_KIND.XOR,
        }
        reduce_kind = reduce_kind_map[store_reduce]
        # Use gluon: create_async_tma_reduce(kind, desc, coord, src)
        _semantic.builder.create_async_tma_reduce(reduce_kind, desc.handle,
                                                  offsets, source_handle)


@tl.builtin
def async_descriptor_store_wait(
    pendings: tl.constexpr,
    _semantic=None,
) -> None:
    """Wait for completion of prior asynchronous TMA store operations."""
    pendings = tl._unwrap_if_constexpr(pendings)
    # Use gluon: create_async_tma_store_wait(pendings)
    _semantic.builder.create_async_tma_store_wait(pendings)


# Monkey-patch __getitem__ for indexing support
@tl.builtin
def _buffered_tensor_getitem(self, buffer_idx, _semantic=None):
    return local_view(self, buffer_idx, _semantic=_semantic)


@tl.builtin
def _tensor_descriptor_ptr_getitem(self, index, _semantic=None):
    descriptor_size = self.descriptor_size
    if isinstance(index, tl.tensor):
        index_handle = index.handle
    elif isinstance(index, int) or isinstance(index, tl.constexpr):
        index_val = tl._unwrap_if_constexpr(index)
        index_handle = _semantic.builder.get_int32(index_val)
    else:
        raise TypeError(
            f"Index must be int, constexpr, or tensor, got {type(index)}")

    size_handle = _semantic.builder.get_int32(descriptor_size)
    offset_handle = _semantic.builder.create_mul(index_handle, size_handle)
    indexed_handle = _semantic.builder.create_addptr(self.handle,
                                                     offset_handle)
    return tlx.tensor_descriptor_ptr(indexed_handle, self.num, descriptor_size)


tlx.buffered_tensor.__getitem__ = _buffered_tensor_getitem
tlx.mbarrier.__getitem__ = _buffered_tensor_getitem
tlx.clc_response.__getitem__ = _buffered_tensor_getitem
tlx.tensor_descriptor_ptr.__getitem__ = _tensor_descriptor_ptr_getitem
