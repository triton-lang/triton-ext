"""uTLX Plugin — out-of-tree Python DSL for the full TLX dialect.

Provides the complete TLX API as plugin-based operations:
  - Memory ops: local_alloc, local_view, local_store, local_load
  - Barrier ops: alloc_barriers, barrier_wait, barrier_arrive, etc.
  - Storage alias: storage_alias_spec, reuse_group, set_buffer_overlap
  - Layout ops: require_layout, release_layout
  - MMA ops: async_dot, async_dot_wait

Usage:
    import utlx_plugin as tlx
    buf = tlx.local_alloc((M, K), tl.float16, 2)
    view = tlx.local_view(buf, 0)
    tlx.local_store(view, tensor)
    result = tlx.local_load(view)
"""

from .types import (
    buffered_tensor,
    buffered_tensor_type,
    layout_encoding,
    mbarrier,
    nv_mma_shared_layout_encoding,
    shared_layout_encoding,
    storage_kind,
    swizzled_shared_layout_encoding,
    tensor_memory_layout_encoding,
    storage_alias_spec as storage_alias_spec_type_class,
    storage_alias_spec_type,
    reuse_group,
    reuse_group_type,
    async_token,
)
from .mem_ops import (
    local_alloc,
    local_load,
    local_store,
    local_view,
    storage_alias_spec,
)
from .barrier import (
    alloc_barriers,
    alloc_warp_barrier,
    barrier_expect_bytes,
    barrier_wait,
    barrier_arrive,
    named_barrier_wait,
    named_barrier_arrive,
)
from .utility import dtype_of
from . import custom_stages

from triton import knobs
knobs.runtime.add_stages_inspection_hook = custom_stages.inspect_stages_hook

__all__ = [
    # types
    "buffered_tensor",
    "buffered_tensor_type",
    "layout_encoding",
    "mbarrier",
    "nv_mma_shared_layout_encoding",
    "shared_layout_encoding",
    "storage_kind",
    "swizzled_shared_layout_encoding",
    "tensor_memory_layout_encoding",
    "storage_alias_spec",
    "storage_alias_spec_type",
    "storage_alias_spec_type_class",
    "reuse_group",
    "reuse_group_type",
    "async_token",
    # mem_ops
    "local_alloc",
    "local_load",
    "local_store",
    "local_view",
    # barrier
    "alloc_barriers",
    "alloc_warp_barrier",
    "barrier_expect_bytes",
    "barrier_wait",
    "barrier_arrive",
    "named_barrier_wait",
    "named_barrier_arrive",
    # utility
    "dtype_of",
]
