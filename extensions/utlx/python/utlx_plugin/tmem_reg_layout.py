"""Concrete register layout for a TMEM load.

TLX normally pins the destination layout of `tlx.local_load` on tensor memory
with a placeholder encoding (`#tlx.dummy_register_layout`) and lets
`tlx-resolve-placeholder-layouts` settle it once warp counts are known. A stock
Triton cannot carry that placeholder: its verifiers linearise every tensor
encoding they meet and `TritonGPUDialect::toLinearLayout` is a closed dispatch
over built-in layouts, so an out-of-tree encoding aborts the verifier long
before any tlx pass runs.

Compute the layout up front instead. Gluon already knows how -- it is the same
derivation its own `tensor_memory_descriptor.load()` uses -- and the one input
that made TLX defer the decision, the warp count of the enclosing region, is
recorded by the code generator while it builds each `async_task` body.
"""

def _cga_layout_bases(cta_split_m, cta_split_n):
    from .layout_compat import cga_bases
    splits = [int(cta_split_m), int(cta_split_n)]
    return cga_bases(splits, splits, [0, 1])


def _gluon_tmem_layout(layout):
    """tlx `tensor_memory_layout_encoding` -> gluon `TensorMemoryLayout`."""
    from triton.experimental.gluon.language.nvidia.blackwell import TensorMemoryLayout
    return TensorMemoryLayout(
        block=(int(layout.blockM), int(layout.blockN)),
        col_stride=int(layout.colStride),
        cga_layout=_cga_layout_bases(layout.CTASplitM, layout.CTASplitN),
    )


def num_warps_for_load(builder):
    """Warp count the TMEM load executes under.

    Inside a warp-specialized region that is the region's own warp count, not
    the kernel's; outside one the kernel's is right.
    """
    from .compiler.code_generator import current_task_num_warps
    default = int(builder.options.num_warps)
    num_warps = current_task_num_warps(default)
    return int(num_warps) if num_warps else default


def register_layout(builder, src):
    """Destination layout for loading the whole of TMEM buffer `src`.

    `src` is a tlx `buffered_tensor` in tensor memory. Returns a gluon
    distributed layout object; call `_to_ir(builder)` for the MLIR attribute.
    """
    from triton.experimental.gluon.language._semantic import _compute_tmem_reg_layout

    shape = [int(d) for d in src.type.shape]
    # `alloc_shape` describes the whole allocation the descriptor views into;
    # for a single-buffer view that is just the shape.
    alloc_shape = list(shape)
    return _compute_tmem_reg_layout(
        src.type.element_ty,
        shape,
        alloc_shape,
        _gluon_tmem_layout(src.type.layout),
        num_warps_for_load(builder),
        "32x32b",
    )


def result_tensor_type(builder, src):
    """The tensor type a TMEM load of `src` produces, as a gluon type."""
    from triton.experimental.gluon.language import _core as ttgl
    return ttgl.distributed_type(src.type.element_ty,
                                 [int(d) for d in src.type.shape],
                                 register_layout(builder, src))


def result_type_carrier(builder, src, scalar_zero):
    """A value whose type is what a TMEM load of `src` produces.

    `utlx_tmem_load` reads the result type off a carrier value, because the
    plugin op ABI passes only values. The carrier is a splat nothing consumes,
    so it is dead the moment the load is built.
    """
    ty = result_tensor_type(builder, src).to_ir(builder)
    return builder.create_splat(ty, scalar_zero)
