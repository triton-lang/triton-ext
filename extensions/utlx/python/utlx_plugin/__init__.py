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
from pathlib import Path
import sys as _sys
import triton.language.extra as _extra
import triton._C.libtriton as _libtriton

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


def _make_tlx_op_builder():
    """Build a hybrid op-builder class for tlx (non-gluon) kernels.

    uTLX ops such as ``async_dot`` rely on native gluon builder methods
    (``create_warpgroup_mma``, ``create_tcgen05_mma``, ``create_async_tma_*``)
    that exist only on ``gluon_ir.GluonOpBuilder``. A plain ``@triton.jit``
    kernel compiles with ``ir.builder`` (``TritonOpBuilder``), which lacks
    those ops. ``GluonOpBuilder`` subclasses ``TritonOpBuilder`` and also
    inherits the ``create_utlx_*`` plugin ops, but it *overrides* a number of
    shared ops (``create_broadcast``, ``create_cat``, ``create_split``, ...)
    with gluon-specific signatures that are incompatible with the standard
    ``TritonSemantic`` used for regular kernels.

    We therefore derive a class from ``GluonOpBuilder`` that restores the base
    ``TritonOpBuilder`` implementation for every op the gluon builder overrides.
    The result speaks the regular Triton op ABI (so ``TritonSemantic`` and the
    ``create_utlx_*`` plugin ops work) while still exposing the gluon-exclusive
    ops that tlx needs.
    """
    from triton._C.libtriton import ir as _ir
    from triton._C.libtriton import gluon_ir as _gluon_ir

    base = _ir.builder
    gluon = _gluon_ir.GluonOpBuilder

    # Ops present on both classes but overridden by gluon -> restore the base
    # implementation so TritonSemantic keeps working. Gluon-exclusive ops (not
    # on the base) are left untouched and remain available.
    namespace = {}
    for name in dir(base):
        if name.startswith("__"):
            continue
        base_attr = getattr(base, name, None)
        gluon_attr = getattr(gluon, name, None)
        if base_attr is None or gluon_attr is None:
            continue
        if base_attr is gluon_attr:
            continue  # inherited unchanged; nothing to restore

        def _delegate(self, *args, _bm=base_attr, **kwargs):
            return _bm(self, *args, **kwargs)

        namespace[name] = _delegate

    return type("TLXOpBuilder", (gluon, ), namespace)


def _tag_module_num_warps(codegen):
    """Set ``ttg.num-warps``/``ttg.threads-per-warp`` on the module early.

    tlx kernels attach ttgpu distributed layouts (e.g. ``nvidia_mma``) to
    tensors during TTIR construction. Triton's ``VerifyTensorLayoutsTrait``
    validates those layouts against the module's warp counts, which normally are
    only set later by ``convert-triton-to-tritongpu``. Set them up front (from
    the compile options) so the initial ``module.verify()`` succeeds.
    """
    try:
        builder = codegen.builder
        module = codegen.module
        options = builder.options
        num_warps = int(options.num_warps)
        threads_per_warp = int(getattr(options, "warp_size", 32) or 32)
        if module.get_int_attr("ttg.num-warps") is None:
            module.set_attr("ttg.num-warps", builder.get_int32_attr(num_warps))
        if module.get_int_attr("ttg.threads-per-warp") is None:
            module.set_attr("ttg.threads-per-warp",
                            builder.get_int32_attr(threads_per_warp))
    except (AttributeError, TypeError):
        pass


def _patch_gluon_builder():
    """Route non-gluon kernel compilation through the hybrid tlx builder."""
    try:
        import triton.compiler.code_generator as _cg
        _tlx_builder = _make_tlx_op_builder()
    except (ImportError, AttributeError):
        return

    if getattr(_cg.CodeGenerator, "_utlx_gluon_builder", False):
        return

    _orig_init = _cg.CodeGenerator.__init__

    def _init(self, *args, **kwargs):
        # Only the non-gluon path constructs ``ir.builder(context)``; swap that
        # class for the hybrid builder for the duration of the original __init__.
        if not kwargs.get("is_gluon", False):
            _orig_builder = _cg.ir.builder
            _cg.ir.builder = _tlx_builder
            try:
                _orig_init(self, *args, **kwargs)
            finally:
                _cg.ir.builder = _orig_builder
            _tag_module_num_warps(self)
        else:
            _orig_init(self, *args, **kwargs)

    _cg.CodeGenerator.__init__ = _init
    _cg.CodeGenerator._utlx_gluon_builder = True


_patch_gluon_builder()

# Register the uTLX plugin library with Triton.
PLUGIN_DIR = Path(__file__).resolve().parent
PLUGIN_LIBRARY = PLUGIN_DIR / "libutlx.so"
_libtriton.passes.plugin.extend_with(str(PLUGIN_LIBRARY))  # adds passes
_libtriton.ir.extend_dialects_with(str(PLUGIN_LIBRARY))  # adds dialects
_libtriton.ir.builder.extend_with(str(PLUGIN_LIBRARY))  # adds ops
