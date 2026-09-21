"""Features a kernel may ask for that the host Triton build has to provide.

The plugin ships the tlx dialect and its ops, but some TLX kernels lean on
Triton features that live in *core*, outside any plugin's reach. Upstream
Triton has neither the ops nor the lowering for them, so say so plainly at the
point of use: emitting the IR anyway either yields a null value and segfaults
the compiler, or trips an assertion inside the MLIR verifier, neither of which
a caller can act on.
"""

class HostFeatureUnavailable(NotImplementedError):
    """A kernel asked for something this Triton build cannot lower.

    A distinct type so callers can tell "this target cannot run this kernel"
    apart from an ordinary bug, and decline the shape instead of failing.
    """


_FEATURES = {
    "remote_view": (
        "tlx.remote_view and a remote tlx.barrier_arrive need "
        "`ttng.map_to_remote_buffer` (mapa.shared::cluster), which upstream "
        "Triton has no op for.",
        "Cross-CTA (NUM_CTAS=2) kernels need Meta's in-tree Triton; "
        "single-CTA kernels work on a stock wheel.",
    ),
    "two_ctas_tma": (
        "a 2-CTA tlx.async_descriptor_load needs the `.cta_group::2` TMA "
        "modifier, which upstream's ttng.async_tma_copy_global_to_local does "
        "not expose.",
        "Cross-CTA (NUM_CTAS=2) kernels need Meta's in-tree Triton; "
        "single-CTA kernels work on a stock wheel.",
    ),
    "placeholder_layouts": (
        "a TMEM load pins its register layout with a tlx placeholder encoding "
        "(#tlx.dummy_register_layout) for tlx-resolve-placeholder-layouts to "
        "settle once warp counts are known.",
        "Core Triton verifiers linearise every tensor encoding they see, and "
        "TritonGPUDialect::toLinearLayout is a closed dispatch over built-in "
        "layouts, so an out-of-tree encoding aborts the verifier before that "
        "pass can run. Meta's build hides the placeholder behind a no_verify "
        "wrapper its verifiers skip. Supporting this on a stock wheel means "
        "computing the concrete TMEM register layout up front (see "
        "gluon's compute_tmem_reg_layout), which in turn needs the warp count "
        "of the enclosing async_task.",
    ),
}


def host_is_meta_intree():
    """True for Meta's in-tree Triton, which carries the extra core support.

    That build registers the `make_*_encoding_attr` layout factories on the op
    builder from the same source tree that provides these features, so their
    presence is a reliable proxy and needs no probing of the compiled pass
    pipeline.
    """
    try:
        from triton._C.libtriton import ir
    except ImportError:  # pragma: no cover - triton is always importable here
        return False
    return hasattr(ir.builder, "make_nv_mma_shared_encoding_attr")


def require_host_feature(name):
    """Raise unless the host Triton can lower the named feature."""
    if host_is_meta_intree():
        return
    what, why = _FEATURES.get(name, (name, ""))
    raise HostFeatureUnavailable(f"{what} This Triton build does not provide it. {why}".strip())
