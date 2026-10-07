"""uTLX Plugin — out-of-tree Python DSL for the full TLX dialect."""

# Define __all__ early to break circular import:
# triton.language.extra.tlx.__init__ does `from utlx_plugin import __all__`
# but our submodules import triton.language.core which triggers that import.
__all__ = [
    # async_tasks
    "async_tasks",
    "async_task",
    # warp_pipeline
    "warp_pipeline_stage",
    # amd_ops (Meta-fork AMD ops on upstream Triton)
    "buffer_load_to_local",
    "buffer_atomic_add",
    "assert_same_layout",
    "extract_slice",
    "rematerialized_range",
    "amd_register_resident",
    "amd_register_class_anchor",
    "assume_uniform",
    "amd_scheduled_mfma",
    "amd_mfma_commit",
    "amd_sched_barrier",
    "amd_iglp_opt",
    "num_warps",
    "warp_any",
    "warp_predicate",
    # types
    "layout",
    "layout_encoding",
    "shared_layout_encoding",
    "swizzled_shared_layout_encoding",
    "padded_shared_layout_encoding",
    "shared_linear_layout_encoding",
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
    "warp_redux",
    # layout_ops
    "amd_mfma_layout",
    "dot_operand_layout",
    "slice_layout",
    "swizzled_layout",
    "require_layout",
    "release_layout",
    "zeros",
    "buffer_load",
    "buffer_store",
]

# Imported first, ahead of anything that pulls in triton: importing it is the
# last chance to set TRITON_PLUGIN_PATHS for a pre-`extend_with` Triton, which
# reads the variable only while its own libtriton is being imported.
from . import _compat
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
from .layout_ops import (amd_mfma_layout, buffer_load, buffer_store,
                         dot_operand_layout, release_layout, require_layout,
                         slice_layout, swizzled_layout, zeros)
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
    layout,
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
    padded_shared_layout_encoding,
    shared_linear_layout_encoding,
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
import os as _os
import sys as _sys
import triton.language.extra as _extra

_sys.modules['triton.language.extra.tlx'] = _sys.modules[__name__]
_extra.tlx = _sys.modules[__name__]

# Multi-CTA (a 2-CTA cluster sharing one MMA) needs tlx.remote_view, which
# lowers to ttng.map_to_remote_buffer -- see the refusal in mem_ops. Meta's TLX
# does not publish this flag, so a consumer should read it as True when absent.
SUPPORTS_MULTI_CTA = False


def _install_tensor_descriptor_layout_patch():
    """Give tt.tensordesc types the NVMMA layout TLX's TMA copies require.

    Plain Triton creates descriptor types with no sharedLayout and fills it in
    much later, in the optimize-descriptor-encoding pass. That is fine for
    upstream, whose TMA copies are created in TTGIR. TLX builds its copies
    while generating TTIR, so ttng.async_tma_copy_global_to_local is verified
    long before that pass ever runs and rejects the descriptor with
    "TMA descriptor layout must match shared layout ... got <<NULL ATTRIBUTE>>".

    Emit the layout up front instead. The default NVMMA encoding for the block
    shape and element type is what TLX's own SMEM buffers get
    (nv_mma_shared_layout_encoding.make_default, via require_nv_mma_shared_layout),
    so descriptor and destination match by construction -- and it is the same
    encoding optimize-descriptor-encoding would have chosen anyway.
    """
    from triton.language import core as _core

    base = _core.tensor_descriptor_base_type
    if getattr(base, '_utlx_descriptor_layout_patched', False):
        return
    _orig_flatten_ir_types = base._flatten_ir_types

    def _flatten_ir_types(self, builder, out):
        # Only on the plugin's builder, and only for the 2-D+ tiles NVMMA
        # describes; anything else keeps the unencoded type.
        make_layout_ty = getattr(builder, 'get_tensor_descriptor_layout_type',
                                 None)
        block_ty = self.block_type
        shape = [int(d) for d in block_ty.shape]
        if make_layout_ty is None or len(shape) < 2:
            return _orig_flatten_ir_types(self, builder, out)

        from .types import nv_mma_shared_layout_encoding
        layout = nv_mma_shared_layout_encoding.make_default(
            shape, block_ty.element_ty)
        out.append(
            make_layout_ty(block_ty.to_ir(builder),
                           block_ty.element_ty.is_int_signed(),
                           layout.to_ir(builder)))

    base._flatten_ir_types = _flatten_ir_types
    base._utlx_descriptor_layout_patched = True


_install_tensor_descriptor_layout_patch()


def _install_make_tensor_descriptor_layout_patch():
    """Same fix as above, for descriptors built inside the kernel.

    `tl.make_tensor_descriptor` goes through TritonSemantic rather than a type's
    _flatten_ir_types, and tt.MakeTensorDescOp's own builder infers a result
    type with no sharedLayout. Gluon binds an overload of
    create_make_tensor_descriptor that takes the result type explicitly, so
    shadow the builder method for the duration of the call: upstream keeps
    doing all of its own validation, only the op construction changes.
    """
    from triton.language import core as _core
    from triton.language.semantic import TritonSemantic

    if getattr(TritonSemantic, '_utlx_descriptor_layout_patched', False):
        return
    _orig_make = TritonSemantic.make_tensor_descriptor

    def make_tensor_descriptor(self,
                               base,
                               shape,
                               strides,
                               block_shape,
                               padding_option="zero"):
        builder = self.builder
        make_layout_ty = getattr(builder, 'get_tensor_descriptor_layout_type',
                                 None)
        block = list(_core._unwrap_shape(block_shape))
        if make_layout_ty is None or len(block) < 2:
            return _orig_make(self, base, shape, strides, block_shape,
                              padding_option)

        from .types import nv_mma_shared_layout_encoding
        element_ty = base.type.element_ty
        layout = nv_mma_shared_layout_encoding.make_default(block, element_ty)
        result_ty = make_layout_ty(
            _core.block_type(element_ty, block).to_ir(builder),
            element_ty.is_int_signed(), layout.to_ir(builder))

        # _make_tlx_builder restores the ir.builder implementation of every
        # method gluon overrides, so builder.create_make_tensor_descriptor is
        # the base one and the result-type overload is not reachable through
        # the instance. Call gluon's unbound.
        from triton._C.libtriton.gluon_ir import GluonOpBuilder

        def _create_with_layout(base_handle, shape_handles, stride_handles,
                                tensor_shape, is_signed, padding):
            # Gluon's overload: (resultTy, base, shape, strides, padding).
            return GluonOpBuilder.create_make_tensor_descriptor(
                builder, result_ty, base_handle, shape_handles, stride_handles,
                padding)

        builder.create_make_tensor_descriptor = _create_with_layout
        try:
            return _orig_make(self, base, shape, strides, block_shape,
                              padding_option)
        finally:
            del builder.create_make_tensor_descriptor

    TritonSemantic.make_tensor_descriptor = make_tensor_descriptor
    TritonSemantic._utlx_descriptor_layout_patched = True


_install_make_tensor_descriptor_layout_patch()

from .mxfp8_utils import _to_mxfp8_block  # noqa: E402
from .warp_ops import vote_ballot_sync, warp_redux  # noqa: E402
from .warp_pipeline import warp_pipeline_stage  # noqa: E402
from .amd_ops import (  # noqa: E402
    buffer_load_to_local, buffer_atomic_add, assert_same_layout, extract_slice,
    rematerialized_range, amd_register_resident, amd_register_class_anchor,
    assume_uniform, amd_scheduled_mfma, amd_mfma_commit, amd_sched_barrier,
    amd_iglp_opt, num_warps, warp_any, warp_predicate,
)

from . import custom_stages  # noqa: E402
from .compiler.semantic import install_semantic  # noqa: E402

# Layout propagation, entirely inside a TritonSemantic subclass -- the ops as
# overrides and the result type via make_tensor. No monkeypatching. No-ops for
# values without an explicit layout. UTLX_NO_LAYOUT_PROPAGATION=1 opts out.
if _os.environ.get("UTLX_NO_LAYOUT_PROPAGATION") != "1":
    install_semantic()

from triton import knobs  # noqa: E402

knobs.runtime.add_stages_inspection_hook = custom_stages.inspect_stages_hook


def _register_compiler_dispatch():
    """Register compiler dispatch for warp specialization (lazy)."""
    try:
        from triton.compiler.code_generator import WITH_DISPATCH
        from .compiler.dispatch import TLX_WITH_DISPATCH
        WITH_DISPATCH.update(TLX_WITH_DISPATCH)
        return True
    except (ImportError, AttributeError):
        return False


def _patch_visit_with():
    """Dispatch `with tlx.async_task(s)(...)` to TLX codegen on upstream Triton.

    Meta's fork rewrites ``CodeGenerator.visit_With`` to look the context class
    up in a ``WITH_DISPATCH`` registry and hand the *AST node* to the handler.
    Upstream has no such registry: it instantiates every context manager as
    ``fn(*args, _semantic=..., **kws)`` and then runs ``__enter__`` / body /
    ``__exit__``. That protocol cannot express warp specialization, which has to
    split the body across the regions of a ``ttg.warp_specialize`` op, so
    ``visit_withAsyncTasks`` needs the unvisited statements.

    Without this the failure is two-layered: constructing ``async_tasks``
    raises on the unexpected ``_semantic`` keyword, and had it not, the body
    would be emitted inline with no warp specialization at all.

    Wrap ``visit_With`` so a TLX context manager reaches its AST-level handler
    and every other `with` keeps upstream behaviour.
    """
    import ast

    import triton.compiler.code_generator as _cg

    if getattr(_cg.CodeGenerator, "_utlx_visit_with", False):
        return

    from .compiler.dispatch import TLX_WITH_DISPATCH
    _orig_visit_With = _cg.CodeGenerator.visit_With

    def _visit_With(self, node):
        # Only a single-item `with` can be a TLX region; anything else (and any
        # non-call context expression) is upstream's to handle.
        if len(node.items) == 1:
            context = node.items[0].context_expr
            if isinstance(context, ast.Call):
                handler = TLX_WITH_DISPATCH.get(self.visit(context.func))
                if handler:
                    return handler(self, node)
        return _orig_visit_With(self, node)

    _cg.CodeGenerator.visit_With = _visit_With
    _cg.CodeGenerator._utlx_visit_with = True


# Meta's fork owns visit_With, so only patch a Triton that has no registry.
if not _register_compiler_dispatch():
    _patch_visit_with()


def _patch_verify_loop_carried_variable():
    """Reject loop-carried TLX values whose type changes inside the loop.

    Upstream's ``CodeGenerator._verify_loop_carried_variable`` only compares
    types for ``tl.tensor``, so a TLX value -- e.g. a 2-buffer ``local_alloc``
    reassigned to one ``local_view`` of it -- slips through and the mismatch
    surfaces later as a raw MLIR verifier error on ``scf.for``/``scf.while``.
    Meta's fork extends the check to TLX values; do the same here.

    Compare the Python-side ``.type``, as upstream does, not the IR handles:
    the check runs after the loop's dry-run block is erased, so ``loop_val``'s
    handle is already dangling.
    """
    import triton.compiler.code_generator as _cg

    from .types import tlx_value

    if getattr(_cg.CodeGenerator, "_utlx_verify_loop_carried", False):
        return

    _orig_verify = _cg.CodeGenerator._verify_loop_carried_variable

    def _verify_loop_carried_variable(self, name, loop_val, live_val):
        _orig_verify(self, name, loop_val, live_val)
        if not isinstance(loop_val, tlx_value):
            return
        try:
            same = loop_val.type == live_val.type
        except NotImplementedError:
            # tl.base_type's default __eq__; nothing structural to compare.
            return
        if not same:
            # Same wording as upstream's tl.tensor check.
            raise AssertionError(
                f'Loop-carried variable {name} has initial type '
                f'{live_val.type} but is re-assigned to {loop_val.type} in '
                f'loop! Please make sure that the type stays consistent.')

    _cg.CodeGenerator._verify_loop_carried_variable = (
        _verify_loop_carried_variable)
    _cg.CodeGenerator._utlx_verify_loop_carried = True


_patch_verify_loop_carried_variable()


def _patch_reconcile_region_types():
    """Reconcile scf yields that differ from their region only in encoding.

    ``require_layout`` makes a value's IR type encoded during code generation,
    so a variable can be plain on one control-flow edge and encoded on the
    other -- e.g. ``acc = tl.zeros(...)`` before a loop that reassigns
    ``acc`` from a ``require_layout``'d result. Meta's fork accepts this;
    upstream emits an ``scf.for``/``scf.if`` whose yield types mismatch and
    fails verification. ``tt.call`` and ``tt.return`` have the same problem:
    their signatures come from frontend types, which miss encodings a value
    inherits from its operands (``tl.dot`` with an encoded acc). Once a
    function's returns are finalised, the plugin casts each such operand to
    the type its region, callee or function expects.
    """
    import triton.compiler.code_generator as _cg

    if getattr(_cg.CodeGenerator, "_utlx_reconcile_region_types", False):
        return

    _orig_handle_returns = _cg.CodeGenerator.handle_returns

    def handle_returns(self):
        _orig_handle_returns(self)
        # Insertion point is back inside the function, after the final return.
        reconcile = getattr(self.builder, "utlx_reconcile_region_types", None)
        if reconcile is not None:
            reconcile([])

    _cg.CodeGenerator.handle_returns = handle_returns
    _cg.CodeGenerator._utlx_reconcile_region_types = True


_patch_reconcile_region_types()


def _patch_extern_elementwise():
    """Keep an explicit layout across ``tl.extern_elementwise``.

    The builtin types its result as ``broadcast_arg.type.with_element_ty``,
    which rebuilds a plain ``block_type``. With encoded operands that emits a
    ``tt.extern_elementwise`` whose result is unencoded, which fails
    ``SameOperandsAndResultEncoding`` during layout conversion (libdevice
    calls such as HIP's ``fast_dividef`` on a ``require_layout``'d value).
    Like ``cast``, round-trip: release the operands, call, re-apply the layout.
    """
    import triton.language.core as tl_core

    from .compiler.semantic import _has_layout, _promote_value
    from .layout_ops import _require

    orig = tl_core.extern_elementwise
    if getattr(orig, "_utlx", False):
        return

    @tl_core.builtin
    def extern_elementwise(lib_name,
                           lib_path,
                           args,
                           arg_type_symbol_dict,
                           is_pure,
                           _semantic=None):
        args = [_promote_value(a) for a in args]
        carrier = next((a for a in args if _has_layout(a)), None)
        drop = getattr(_semantic, "_drop_layout", None)
        if carrier is not None and drop is not None:
            args = [drop(a) for a in args]
        out = orig(lib_name,
                   lib_path,
                   args,
                   arg_type_symbol_dict,
                   is_pure,
                   _semantic=_semantic)
        if (carrier is not None and drop is not None and out.type.is_block()
                and list(out.shape) == list(carrier.shape)):
            out = _require(_semantic, out, carrier)
        return out

    extern_elementwise._utlx = True
    tl_core.extern_elementwise = extern_elementwise
    import triton.language as tl_lang
    if getattr(tl_lang, "extern_elementwise", None) is orig:
        tl_lang.extern_elementwise = extern_elementwise


_patch_extern_elementwise()


def _patch_inline_asm_elementwise():
    """Keep an explicit layout across ``tl.inline_asm_elementwise``.

    Same failure as ``extern_elementwise``: the results are typed with
    ``broadcast_arg.type.with_element_ty``, and an operand pinned with
    ``require_layout`` next to one that is not (fbtriton's gfx950 MXFP8 flash
    attention converts bf16 P with a pinned scale) gives a
    ``tt.elementwise_inline_asm`` that fails ``SameOperandsAndResultEncoding``.
    Round-trip the same way: release the operands, emit, re-apply the layout
    to every result.
    """
    import triton.language.core as tl_core

    from .compiler.semantic import _has_layout, _promote_value
    from .layout_ops import _require

    orig = tl_core.inline_asm_elementwise
    if getattr(orig, "_utlx", False):
        return

    @tl_core.builtin
    def inline_asm_elementwise(asm,
                               constraints,
                               args,
                               dtype,
                               is_pure,
                               pack,
                               _semantic=None):
        args = [_promote_value(a) for a in args]
        carrier = next((a for a in args if _has_layout(a)), None)
        drop = getattr(_semantic, "_drop_layout", None)
        if carrier is not None and drop is not None:
            args = [drop(a) for a in args]
        out = orig(asm,
                   constraints,
                   args,
                   dtype,
                   is_pure,
                   pack,
                   _semantic=_semantic)
        if carrier is None or drop is None:
            return out

        def pin(t):
            if t.type.is_block() and list(t.shape) == list(carrier.shape):
                return _require(_semantic, t, carrier)
            return t

        return tuple(pin(t)
                     for t in out) if isinstance(out, tuple) else pin(out)

    inline_asm_elementwise._utlx = True
    tl_core.inline_asm_elementwise = inline_asm_elementwise
    import triton.language as tl_lang
    if getattr(tl_lang, "inline_asm_elementwise", None) is orig:
        tl_lang.inline_asm_elementwise = inline_asm_elementwise


_patch_inline_asm_elementwise()

# Loop options Meta's fork adds to ``tl.range`` for NVIDIA automatic warp
# specialization and its modulo scheduler. They are scheduling hints with no
# meaning on upstream Triton, so they are accepted and dropped.
_FORK_ONLY_RANGE_OPTIONS = frozenset({
    "data_partition_factor",
    "multi_cta",
    "list_schedule_pick",
    "mem_plan_pick",
    "merge_epilogue",
    "merge_epilogue_to_computation",
    "merge_correction",
    "separate_epilogue_store",
    "tmem_alloc_algo",
    "smem_alloc_algo",
    "smem_budget",
    "smem_circular_reuse",
})


def _patch_range_options():
    """Let ``tl.range`` accept the fork's extra loop options.

    TLX op kernels (e.g. the HSTU reference) pass them, and upstream's
    ``range.__init__`` raises on the unknown keywords. Patch ``__init__`` in
    place: the code generator recognises the iterator by identity
    (``IteratorClass is language.range``), so a subclass would not work.
    """
    from triton.language import core as _core

    if getattr(_core.range, "_utlx_fork_options", False):
        return
    _orig_init = _core.range.__init__

    def __init__(self, *args, **kwargs):
        for name in _FORK_ONLY_RANGE_OPTIONS.intersection(kwargs):
            del kwargs[name]
        _orig_init(self, *args, **kwargs)

    _core.range.__init__ = __init__
    _core.range._utlx_fork_options = True


_patch_range_options()


def _make_tlx_op_builder():
    """Build a hybrid op-builder class for tlx (non-gluon) kernels.

    uTLX ops such as ``async_dot`` rely on native gluon builder methods
    (``create_warpgroup_mma``, ``create_tcgen05_mma``, ``create_async_tma_*``)
    that exist only on ``gluon_ir.GluonOpBuilder``. A plain ``@triton.jit``
    kernel compiles with ``ir.builder`` (``TritonOpBuilder``), which lacks
    those ops. ``GluonOpBuilder`` subclasses ``TritonOpBuilder`` and also
    inherits the ``utlx_*`` plugin ops, but it *overrides* a number of
    shared ops (``create_broadcast``, ``create_cat``, ``create_split``, ...)
    with gluon-specific signatures that are incompatible with the standard
    ``TritonSemantic`` used for regular kernels.

    We therefore derive a class from ``GluonOpBuilder`` that restores the base
    ``TritonOpBuilder`` implementation for every op the gluon builder overrides.
    The result speaks the regular Triton op ABI (so ``TritonSemantic`` and the
    ``utlx_*`` plugin ops work) while still exposing the gluon-exclusive
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

    # Stock Triton has none of the fork's make_*_encoding_attr factories, so
    # every tlx layout encoding's to_ir() would fail. Rebuild them on the
    # upstream get_*_layout getters.
    from . import layout_compat
    for name, fn in layout_compat.FACTORIES.items():
        if not hasattr(gluon, name):
            namespace[name] = fn

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

PLUGIN_DIR = _compat.PLUGIN_DIR
PLUGIN_LIBRARY = _compat.PLUGIN_LIBRARY
_compat.register_plugin(PLUGIN_LIBRARY)
_compat.install_semantic_helpers()

# Accept TLX's ctas_per_cga launch option, converting it to the num_ctas
# spelling upstream understands. Patches only Triton's Config and launch path,
# so it is inert on a fork that already supports the option -- see
# _ctas_per_cga for why a straight port is not possible.
from . import _ctas_per_cga as _utlx_ctas_per_cga  # noqa: E402

_utlx_ctas_per_cga.install()

# Accept the fork's extra AMD compile options (LLVM codegen flags and the
# sched-group-barrier scheduler knobs) -- see _hip_options.
from . import _hip_options as _utlx_hip_options  # noqa: E402

_utlx_hip_options.install()
