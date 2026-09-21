"""Give TMA descriptor types an explicit NVMMA shared layout.

`tt.tensordesc` types built by core Triton carry no shared-memory encoding:
upstream fills one in later, in the `optimize-descriptor-encoding` pass, which
runs on the high-level `tt.descriptor_load`/`store` ops. TLX does not use those
-- it emits `ttng.async_tma_copy_*` directly while building TTIR -- so the pass
never gets a say, and upstream's verifier rejects the unencoded descriptor:

    'ttng.async_tma_copy_global_to_local' op TMA descriptor layout must match
    shared layout, but got descriptor layout <<NULL ATTRIBUTE>> ...

Meta's build skips that check when the descriptor has no encoding yet. Upstream
does not, so decide the encoding at the point the descriptor type is built. The
choice has to agree with the shared buffer the copy targets, which TLX pins with
`require_nv_mma_shared_layout` -- the default swizzled NVMMA layout for the
tile's shape and element type. Deriving it from the same (shape, order, dtype)
here gives the same answer, and it is also what `optimize-descriptor-encoding`
would have picked.

This patches a core Triton type, so it applies to every descriptor compiled
while the plugin is loaded, not only those in TLX kernels. That is the same
scope as the plugin's existing code-generator patch.
"""


def _encoded_flatten_ir_types(self, builder, out):
    block = self.block_type
    shape = [int(d) for d in block.shape]
    element_ty = block.element_ty
    rank = len(shape)

    layout = builder.make_nv_mma_shared_encoding_attr(
        shape,
        list(reversed(range(rank))),
        element_ty.to_ir(builder),
        [1] * rank,
        [1] * rank,
        [1] * rank,
        False,  # fp4Padded
        True,  # swizzled
    )
    out.append(
        builder.get_tensor_descriptor_layout_type(block.to_ir(builder),
                                                  element_ty.is_int_signed(),
                                                  layout))


def install():
    """Make descriptor parameter/result types carry an NVMMA layout."""
    from .host_caps import host_is_meta_intree
    if host_is_meta_intree():
        return

    from triton.language import core as tl

    base = tl.tensor_descriptor_base_type
    if getattr(base, "_utlx_encoded_descriptor_types", False):
        return
    base._flatten_ir_types = _encoded_flatten_ir_types
    base._utlx_encoded_descriptor_types = True
