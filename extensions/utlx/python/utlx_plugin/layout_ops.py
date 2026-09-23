"""uTLX explicit register-layout ops.

Layouts are represented as *carrier values*: a value whose RankedTensorType
carries the encoding. Plugin ops can only take and return ``mlir::Value``, so an
encoding is passed around as a value and the consumer reads the encoding back
off its type. A carrier's own shape and element type are immaterial.

``require_layout`` and ``dot_operand_layout`` lower to ``tlx.require_layout``,
which ``utlx_propagate_layout`` turns into ``ttg.convert_layout``. That pass must
be in the pipeline or these ops reach the backend unhandled -- see
``custom_stages.py``.
"""

import triton.language.core as tl


@tl.builtin
def amd_mfma_layout(version,
                    instr_shape,
                    transposed=False,
                    warps_per_cta=None,
                    _semantic=None):
    """Build an AMD MFMA (matrix-core) register layout.

    Args:
        version: MFMA instruction version (e.g. 4 for gfx950).
        instr_shape: [M, N, K] of the MFMA instruction, e.g. [16, 16, 32].
        transposed: whether the MFMA result is stored transposed.
        warps_per_cta: warp grid, e.g. [4, 1]. Defaults to [num_warps, 1].

    Returns a layout carrier; pass it to :func:`require_layout`,
    :func:`dot_operand_layout`, or ``zeros(..., layout=)``.
    """
    version = tl._unwrap_if_constexpr(version)
    transposed = tl._unwrap_if_constexpr(transposed)
    instr_shape = [tl._unwrap_if_constexpr(v) for v in instr_shape]
    if len(instr_shape) != 3:
        raise ValueError(
            f"instr_shape must be [M, N, K]; got {len(instr_shape)} entries")

    if warps_per_cta is None:
        warps_per_cta = [_semantic.builder.options.num_warps, 1]
    else:
        warps_per_cta = [tl._unwrap_if_constexpr(v) for v in warps_per_cta]
    if len(warps_per_cta) != 2:
        raise ValueError(
            f"warps_per_cta must have 2 entries; got {len(warps_per_cta)}")

    b = _semantic.builder
    args = [b.get_int32(int(version))]
    args += [b.get_int32(int(v)) for v in instr_shape]
    args.append(b.get_int32(1 if transposed else 0))
    args += [b.get_int32(int(v)) for v in warps_per_cta]
    return tl.tensor(b.utlx_make_amd_mfma_layout(args), tl.float32)


@tl.builtin
def require_layout(src, layout, pin=False, _semantic=None):
    """Require that ``src`` be materialised in ``layout``.

    ``layout`` is a carrier value from e.g. :func:`amd_mfma_layout`. ``pin`` is
    accepted for source compatibility and currently ignored: tlx.require_layout
    has no pin attribute, so there is nothing to forward it to.
    """
    handle = _semantic.builder.utlx_require_with_layout_carrier(
        [src.handle, layout.handle])
    return tl.tensor(handle, src.type)


@tl.builtin
def release_layout(src, _semantic=None):
    """Drop an explicit layout, returning ``src`` with a default encoding."""
    return tl.tensor(_semantic.builder.utlx_release_layout([src.handle]),
                     src.type)
