"""Core-Triton modules Meta's fork ships and upstream Triton does not.

Nothing here is TLX. These are pieces of ``triton.language.extra`` that TLX
kernels import by their core paths:

    from triton.language.extra.subtile_ops import _split_n_2D
    from triton.language.extra.cuda.inline_ptx_lib import _fma_f32x2

Upstream has no such modules, so those imports fail and the kernel never gets
as far as any TLX op. Both are small and written entirely against stock Triton
(``core.inline_asm_elementwise``, ``tl.join``/``permute``/``split``/
``reshape``), so uTLX can supply them rather than leaving the kernels
unimportable.

They are published under their core names by ``UtlxAliasFinder`` in
``_utlx_autoregister``, alongside the other triton-namespaced paths uTLX
stands in for. That finder is appended to ``sys.meta_path``, so a Triton that
really does ship these -- Meta's fork -- keeps resolving them itself.
"""
