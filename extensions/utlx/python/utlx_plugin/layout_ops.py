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

from .types import _value_layout, layout_encoding


def _uw(x):
    return tl._unwrap_if_constexpr(x)


class _carrier_type(tl.block_type):
    """A block_type that lowers back to the exact encoded IR type.

    tl.block_type.to_ir() rebuilds an *unencoded* tensor type, which silently
    drops the layout whenever a carrier crosses a @triton.jit boundary (Triton
    builds the callee signature from arg.type.to_ir()). Carriers exist only to
    hold an encoding, so remember the concrete IR type instead of rebuilding it.
    """

    def __init__(self, element_ty, shape, ir_type):
        super().__init__(element_ty, shape)
        self._ir_type = ir_type

    def to_ir(self, builder):
        return self._ir_type


def _carrier(handle, element_ty=None):
    """Wrap a layout-carrying handle so its encoding survives jit boundaries."""
    element_ty = element_ty or tl.float32
    try:
        return tl.tensor(
            handle,
            _carrier_type(element_ty, handle.get_shape(), handle.get_type()))
    except AttributeError:
        return tl.tensor(handle, element_ty)


def _require(semantic, src, layout):
    """Apply a layout carrier's encoding to ``src`` (a tl.tensor).

    The result keeps a carrier type so the encoding survives a @triton.jit
    return as well as a jit argument -- Triton builds the function result type
    from the frontend type of the returned value, which would otherwise be an
    unencoded block_type.
    """
    layout = _uw(layout)
    handle = semantic.builder.utlx_require_with_layout_carrier(
        [src.handle, _layout_carrier(semantic, src, layout)])
    return _carrier(handle, src.type.scalar)


def _layout_carrier(semantic, src, layout):
    """Return a carrier handle for ``layout``, as required for ``src``.

    Layouts built by the ops in this module are already carriers. Encoding
    objects such as ``tlx.layout`` (shape/stride) are attributes instead; give
    them a poison value of the encoded type, as only its type is ever read.
    """
    if not isinstance(layout, layout_encoding):
        return layout.handle
    b = semantic.builder
    shape = [int(d) for d in src.shape]
    try:
        enc = layout.to_ir(b, shape, src.dtype)
    except TypeError:
        enc = layout.to_ir(b)
    return b.create_poison(b.get_distributed_ty(src.dtype.to_ir(b), shape,
                                                enc))


class _register_layout(_value_layout, layout_encoding):
    """A register layout kept as a description until an op applies it.

    The fork's layouts are attributes. As carrier values instead, every layout
    a kernel defines lands in the IR whether used or not, and the module
    verifier rejects one whose warp count does not match the kernel's -- e.g.
    the 4-warp variant a kernel defines next to the 8-warp one a config picks.
    These lower through :func:`_layout_carrier` only when applied, and are
    constexprs, so they cross @triton.jit calls as the fork's do.
    """

    def to_ir(self, builder, *_):
        # The extra arguments are the shape and dtype _layout_carrier offers
        # layouts that depend on them; register layouts do not.
        raise NotImplementedError


class amd_mfma_layout_encoding(_register_layout):

    def __init__(self, version, instr_shape, transposed, warps_per_cta):
        self.version = version
        self.instr_shape = list(instr_shape)
        self.transposed = transposed
        self.warps_per_cta = list(warps_per_cta)

    def to_ir(self, builder, *_):
        rank = len(self.warps_per_cta)
        return builder.get_amd_mfma_layout(self.version, self.warps_per_cta,
                                           self.instr_shape, self.transposed,
                                           [], [1] * rank, 32)


class dot_operand_layout_encoding(_register_layout):

    def __init__(self, op_idx, parent, k_width):
        self.op_idx = op_idx
        self.parent = parent
        self.k_width = k_width

    def to_ir(self, builder, *_):
        return builder.get_dot_operand_layout(self.op_idx,
                                              self.parent.to_ir(builder),
                                              self.k_width)


class slice_layout_encoding(_register_layout):

    def __init__(self, parent, dim):
        self.parent = parent
        self.dim = dim

    def to_ir(self, builder, *_):
        return builder.get_slice_layout(self.dim, self.parent.to_ir(builder))


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
        warps_per_cta: warp grid, e.g. [4, 1], or [B, M, N] for a batched
            (rank-3) layout. Defaults to [num_warps, 1].
    """
    version = _uw(version)
    transposed = _uw(transposed)
    instr_shape = [_uw(v) for v in _uw(instr_shape)]
    if len(instr_shape) != 3:
        raise ValueError(
            f"instr_shape must be [M, N, K]; got {len(instr_shape)} entries")

    if warps_per_cta is None:
        warps_per_cta = [_semantic.builder.options.num_warps, 1]
    else:
        warps_per_cta = [_uw(v) for v in _uw(warps_per_cta)]
    if len(warps_per_cta) not in (2, 3):
        raise ValueError("warps_per_cta must have 2 entries (or 3 with a "
                         f"leading batch dim); got {len(warps_per_cta)}")

    return tl.constexpr(
        amd_mfma_layout_encoding(int(version), instr_shape, bool(transposed),
                                 warps_per_cta))


@tl.builtin
def dot_operand_layout(op_idx, parent, k_width=None, _semantic=None):
    """Layout for dot operand ``op_idx`` (0=lhs, 1=rhs) of an mma ``parent``.

    ``k_width`` must be given for AMD MFMA parents; the element-type-inferred
    form only covers NVIDIA MMA.
    """
    op_idx = _uw(op_idx)
    k_width = _uw(k_width)
    parent = _uw(parent)
    if isinstance(parent, _register_layout):
        if not k_width:
            raise ValueError("dot_operand_layout of an AMD MFMA layout needs "
                             "k_width")
        return tl.constexpr(
            dot_operand_layout_encoding(int(op_idx), parent, int(k_width)))
    b = _semantic.builder
    args = [
        parent.handle,
        b.get_int32(int(op_idx)), parent.handle,
        b.get_int32(int(k_width) if k_width else 0)
    ]
    return _carrier(b.utlx_require_dot_operand_layout(args))


@tl.builtin
def slice_layout(parent, dim, _semantic=None):
    """Layout of ``parent`` with dimension ``dim`` sliced away."""
    dim = _uw(dim)
    parent = _uw(parent)
    if isinstance(parent, _register_layout):
        return tl.constexpr(slice_layout_encoding(parent, int(dim)))
    b = _semantic.builder
    return _carrier(
        b.utlx_make_slice_layout([parent.handle,
                                  b.get_int32(int(dim))]))


@tl.builtin
def require_layout(src,
                   layout,
                   pin=False,
                   late_address_compute=False,
                   _semantic=None):
    """Require that ``src`` be materialised in ``layout``.

    ``pin`` and ``late_address_compute`` are accepted for source compatibility
    and currently ignored: tlx.require_layout has no attributes to forward
    them to. Neither changes the values produced.
    """
    return _require(_semantic, src, layout)


@tl.builtin
def release_layout(src, relaxed=False, _semantic=None):
    """Drop an explicit layout, returning ``src`` with a default encoding.

    Passes through anything that is not an IR tensor (constexprs, scalars) so it
    can be applied unconditionally in helpers that accept either.

    ``relaxed`` is accepted for fork compatibility. There it lets layout
    optimization remove the release; here every release lowers to a plain
    convert_layout that later passes may fold anyway, so it has no effect.
    """
    relaxed = tl._unwrap_if_constexpr(relaxed)
    assert isinstance(relaxed, bool), (
        f"relaxed must be a constexpr bool, got {type(relaxed).__name__}")
    if not isinstance(src, tl.tensor):
        return src
    handle = _semantic.builder.utlx_release_layout([src.handle])
    # Deliberately a plain block_type: if src came from require_layout its type
    # is a carrier that would lower straight back to the encoded type, which is
    # the opposite of releasing it.
    ty = src.type
    if isinstance(ty, _carrier_type):
        ty = tl.block_type(ty.scalar, ty.shape)
    return tl.tensor(handle, ty)


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
    return tl.constexpr(
        swizzled_shared_layout_encoding(
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


@tl.builtin
def _load_keeping_layout(ptr, mask, other, cache, _semantic=None):
    """tt.load whose result keeps ``ptr``'s register layout.

    The semantic's load shim strips pointer layouts. buffer_load offsets are
    laid out on purpose, though -- often in a dot operand layout so the values
    reach MFMA registers without a conversion through LDS -- and Meta's fork
    loads straight into that layout. Do the same, giving ``mask`` and
    ``other`` the pointer's layout since tt.load requires all three to agree.
    """
    from .compiler.semantic import UTLXSemantic, _has_layout, _promote_value
    cache = _uw(cache) or ""
    ptr = _promote_value(ptr)
    if not _has_layout(ptr):
        # As tl.load: mask and other may be constexprs or Python scalars.
        mask, other = _uw(mask), _uw(other)
        mask = None if mask is None else _semantic.to_tensor(mask)
        other = None if other is None else _semantic.to_tensor(other)
        return _semantic.load(ptr, mask, other, (), "", cache, "", False)

    def like_ptr(v):
        v = _uw(v)
        if v is None:
            return None
        v = _semantic.to_tensor(v)
        if not v.type.is_block():
            v = _semantic.splat(v, list(ptr.type.shape))
        return v if _has_layout(v) else _require(_semantic, v, ptr)

    out = super(UTLXSemantic, _semantic).load(ptr, like_ptr(mask),
                                              like_ptr(other), (), "", cache,
                                              "", False)
    # Coalesce would re-lay the tensor-of-pointer load out; the
    # utlx_keep_load_layout pass restores this layout afterwards.
    out.handle.set_attr("utlx.keep_layout", _semantic.builder.get_unit_attr())
    return out


@jit
def buffer_load(base,
                offsets,
                mask=None,
                other=None,
                cache: tl.constexpr = None,
                contiguity: tl.constexpr = 1):
    """Load ``base[offsets]``; the AMD backend lowers this to a buffer load.

    The result has the layout of ``offsets``, if they carry one.

    ``contiguity`` is the fork's trusted vector-width promise. It is accepted
    and checked but not forwarded: the backend's own axis analysis picks the
    width here, which can only be narrower, never wrong.
    """
    tlang.static_assert(
        contiguity > 0 and (contiguity & (contiguity - 1)) == 0,
        "contiguity must be a positive power of two")
    return _load_keeping_layout(base + offsets, mask, other, cache)


@jit
def buffer_store(value,
                 base,
                 offsets,
                 mask=None,
                 cache: tl.constexpr = None,
                 contiguity: tl.constexpr = 1):
    """Store ``value`` to ``base[offsets]`` as a buffer store.

    Layouts are stripped from the operands by the store shim, and
    ``contiguity`` is accepted but not forwarded, as for buffer_load.
    """
    tlang.static_assert(
        contiguity > 0 and (contiguity & (contiguity - 1)) == 0,
        "contiguity must be a positive power of two")
    tlang.store(base + offsets, value, mask=mask, cache_modifier=cache)
