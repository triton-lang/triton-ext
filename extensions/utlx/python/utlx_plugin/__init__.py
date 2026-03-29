"""uTLX Plugin — out-of-tree Python DSL for the full TLX dialect."""

# Define __all__ early to break circular import:
# triton.language.extra.tlx.__init__ does `from utlx_plugin import __all__`
# but our submodules import triton.language.core which triggers that import.
__all__ = [
    # async_tasks
    "async_tasks",
    "async_task",
    # types
    "layout_encoding",
    "shared_layout_encoding",
    "swizzled_shared_layout_encoding",
    "tensor_memory_layout_encoding",
    "tensor_memory_scales_layout_encoding",
    "nv_mma_shared_layout_encoding",
    "DummyRegisterLayoutEncoding",
    "DummyTMEMLayoutEncoding",
    "storage_kind",
    "buffered_tensor",
    "buffered_tensor_type",
    "storage_alias_spec",
    "storage_alias_spec_type",
    "storage_alias_spec_type_class",
    "reuse_group",
    "reuse_group_type",
    "reuse_group_ir_type",
    "mbarrier",
    "mbarrier_type",
    "clc_response",
    "clc_response_type",
    "CLCPipelineContext",
    "async_token",
    "tensor_descriptor_ptr",
    "tensor_descriptor_ptr_type",
    # mem_ops
    "async_store",
    "local_alloc",
    "local_view",
    "remote_view",
    "local_slice",
    "subslice",
    "async_load",
    "async_load_commit_group",
    "async_load_wait_group",
    "local_load",
    "local_store",
    "local_trans",
    "local_reinterpret",
    "allocate_tensor_descriptor",
    "async_descriptor_load",
    "async_descriptor_prefetch_tensor",
    "async_descriptor_store",
    "async_descriptor_store_wait",
    "fence",
    "fence_async_shared",
    "make_tensor_descriptor",
    "reinterpret_tensor_descriptor",
    "remote_shmem_store",
    "async_remote_shmem_store",
    "tmem_copy",
    # barriers
    "cluster_barrier",
    "alloc_barriers",
    "alloc_warp_barrier",
    "barrier_expect_bytes",
    "barrier_wait",
    "barrier_arrive",
    "named_barrier_wait",
    "named_barrier_arrive",
    # mma_ops
    "async_dot",
    "async_dot_scaled",
    "async_dot_wait",
    "tcgen05_commit",
    # utility
    "cluster_cta_rank",
    "cluster_size_1d",
    "thread_id",
    "async_task_replica_id",
    "dtype_of",
    "get_fp8_format_name",
    "is_hip",
    "size_of",
    "clock64",
    "stoch_round",
    # dynamic launcher ops
    "_alloc_clc_responses",
    "_clc_issue",
    "_clc_query",
    "clc_create_context",
    "clc_producer",
    "clc_consumer",
    # MXFP8
    "_to_mxfp8_block",
    # warp_ops
    "vote_ballot_sync",
]

from .async_task_utils import async_task, async_tasks
from .barrier import (
    alloc_barriers,
    alloc_warp_barrier,
    barrier_arrive,
    barrier_expect_bytes,
    barrier_wait,
    cluster_barrier,
    named_barrier_arrive,
    named_barrier_wait,
)
from .dynamic_launch import (
    _alloc_clc_responses,
    _clc_issue,
    _clc_query,
    clc_consumer,
    clc_create_context,
    clc_producer,
)
from .mem_ops import (
    allocate_tensor_descriptor,
    async_store,
    async_descriptor_load,
    async_descriptor_prefetch_tensor,
    async_descriptor_store,
    async_descriptor_store_wait,
    async_load,
    async_load_commit_group,
    async_load_wait_group,
    fence,
    fence_async_shared,
    local_alloc,
    local_load,
    local_reinterpret,
    local_slice,
    local_store,
    local_trans,
    local_view,
    make_tensor_descriptor,
    reinterpret_tensor_descriptor,
    remote_shmem_store,
    async_remote_shmem_store,
    remote_view,
    storage_alias_spec,
    subslice,
    tmem_copy,
)
from .mma_ops import async_dot, async_dot_scaled, async_dot_wait, tcgen05_commit
from .types import (
    async_token,
    buffered_tensor,
    buffered_tensor_type,
    clc_response,
    clc_response_type,
    CLCPipelineContext,
    DummyRegisterLayoutEncoding,
    DummyTMEMLayoutEncoding,
    layout_encoding,
    mbarrier,
    mbarrier_type,
    nv_mma_shared_layout_encoding,
    reuse_group,
    reuse_group_ir_type,
    reuse_group_type,
    storage_alias_spec as storage_alias_spec_type_class,
    storage_alias_spec_type,
    shared_layout_encoding,
    storage_kind,
    swizzled_shared_layout_encoding,
    tensor_descriptor_ptr,
    tensor_descriptor_ptr_type,
    tensor_memory_layout_encoding,
    tensor_memory_scales_layout_encoding,
)
from .utility import (
    async_task_replica_id,
    clock64,
    cluster_cta_rank,
    cluster_size_1d,
    dtype_of,
    get_fp8_format_name,
    is_hip,
    size_of,
    stoch_round,
    thread_id,
)
# Register this module as triton.language.extra.tlx so that
# `import triton.language.extra.tlx` works without a filesystem symlink.
# This must happen before importing mxfp8_utils which does that import.
import sys as _sys
import triton.language.extra as _extra

_sys.modules['triton.language.extra.tlx'] = _sys.modules[__name__]
_extra.tlx = _sys.modules[__name__]

from .mxfp8_utils import _to_mxfp8_block  # noqa: E402
from .warp_ops import vote_ballot_sync  # noqa: E402

from . import custom_stages  # noqa: E402

from triton import knobs  # noqa: E402

knobs.runtime.add_stages_inspection_hook = custom_stages.inspect_stages_hook


def _register_compiler_dispatch():
    """Register compiler dispatch for warp specialization (lazy)."""
    try:
        from triton.compiler.code_generator import WITH_DISPATCH
        from .compiler.dispatch import TLX_WITH_DISPATCH
        WITH_DISPATCH.update(TLX_WITH_DISPATCH)
    except (ImportError, AttributeError):
        pass


_register_compiler_dispatch()
