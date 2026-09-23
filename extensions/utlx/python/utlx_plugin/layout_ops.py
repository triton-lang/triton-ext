"""uTLX explicit register-layout ops.

Layouts are represented as *carrier values*: a value whose RankedTensorType
carries the encoding. Plugin ops can only take and return ``mlir::Value``, so an
encoding is passed around as a value and the consumer reads the encoding back
off its type. A carrier's own shape and element type are immaterial.

require_layout/release_layout lower to tlx.require_layout / tlx.release_layout,
which the plugin's ConvertTritonToTritonGPU turns into ttg.convert_layout.
"""

import triton.language as tlang
import triton.language.core as tl
from triton.runtime.jit import jit


def _uw(x):
    return tl._unwrap_if_constexpr(x)


def _require(semantic, src, layout):
    """Apply a layout carrier's encoding to ``src`` (a tl.tensor)."""
    handle = semantic.builder.utlx_require_with_layout_carrier(
        [src.handle, layout.handle])
    return tl.tensor(handle, src.type)


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
    """
    version = _uw(version)
    transposed = _uw(transposed)
    instr_shape = [_uw(v) for v in instr_shape]
    if len(instr_shape) != 3:
        raise ValueError(
            f"instr_shape must be [M, N, K]; got {len(instr_shape)} entries")

    if warps_per_cta is None:
        warps_per_cta = [_semantic.builder.options.num_warps, 1]
    else:
        warps_per_cta = [_uw(v) for v in warps_per_cta]
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
def dot_operand_layout(op_idx, parent, k_width=None, _semantic=None):
    """Layout for dot operand ``op_idx`` (0=lhs, 1=rhs) of an mma ``parent``.

    ``k_width`` must be given for AMD MFMA parents; the element-type-inferred
    form only covers NVIDIA MMA.
    """
    op_idx = _uw(op_idx)
    k_width = _uw(k_width)
    b = _semantic.builder
    args = [
        parent.handle,
        b.get_int32(int(op_idx)), parent.handle,
        b.get_int32(int(k_width) if k_width else 0)
    ]
    return tl.tensor(b.utlx_require_dot_operand_layout(args), parent.type)


@tl.builtin
def slice_layout(parent, dim, _semantic=None):
    """Layout of ``parent`` with dimension ``dim`` sliced away."""
    dim = _uw(dim)
    b = _semantic.builder
    return tl.tensor(
        b.utlx_make_slice_layout([parent.handle,
                                  b.get_int32(int(dim))]), parent.type)


@tl.builtin
def require_layout(src, layout, pin=False, _semantic=None):
    """Require that ``src`` be materialised in ``layout``.

    ``pin`` is accepted for source compatibility and currently ignored:
    tlx.require_layout has no pin attribute to forward it to.
    """
    return _require(_semantic, src, layout)


@tl.builtin
def release_layout(src, _semantic=None):
    """Drop an explicit layout, returning ``src`` with a default encoding."""
    return tl.tensor(_semantic.builder.utlx_release_layout([src.handle]),
                     src.type)


@tl.builtin
def zeros(shape, dtype, layout=None, _semantic=None):
    """``tl.zeros`` materialised directly in ``layout``."""
    shape = [_uw(s) for s in shape]
    dtype = _uw(dtype)
    z = _semantic.full(shape, 0, dtype)
    return z if layout is None else _require(_semantic, z, layout)


@tl.builtin
def swizzled_layout(vector_size,
                    per_phase,
                    max_phase,
                    order,
                    num_ctas=1,
                    _semantic=None):
    """Swizzled shared-memory layout.

    The spelling TLX kernels use for :class:`swizzled_shared_layout_encoding`,
    with the CTA/CGA parameters defaulted to the single-CTA case. A builtin
    rather than a plain function so it can be called from a @triton.jit kernel
    (the JIT's reference walker rejects ordinary Python callables), and returns
    a constexpr so it can be bound to a `tl.constexpr` name.
    """
    from .types import swizzled_shared_layout_encoding
    vector_size = _uw(vector_size)
    per_phase = _uw(per_phase)
    max_phase = _uw(max_phase)
    order = [_uw(o) for o in _uw(order)]
    num_ctas = _uw(num_ctas)
    rank = len(order)
    return tl.constexpr(swizzled_shared_layout_encoding(
        vector_size,
        per_phase,
        max_phase,
        order,
        num_ctas,
        [1] * rank,
        [1] * rank,
        list(reversed(range(rank))),
    ))


# --- AMD buffer ops --------------------------------------------------------
#
# Plain tl.load/tl.store against base+offsets. Triton's AMD backend already
# rewrites these into amdgpu.buffer_load/buffer_store in
# add_convert_to_buffer_ops (gated on knobs.amd.use_buffer_ops), so naming them
# explicitly buys nothing over letting the backend do it -- and this way they
# work unchanged on non-AMD targets.


@jit
def buffer_load(base, offsets, mask=None, other=None):
    """Load ``base[offsets]``; the AMD backend lowers this to a buffer load."""
    return tlang.load(base + offsets, mask=mask, other=other)


@jit
def buffer_store(value, base, offsets, mask=None):
    """Store ``value`` to ``base[offsets]`` as a buffer store."""
    tlang.store(base + offsets, value, mask=mask)
