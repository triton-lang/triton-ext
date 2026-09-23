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
            handle, _carrier_type(element_ty, handle.get_shape(),
                                  handle.get_type()))
    except AttributeError:
        return tl.tensor(handle, element_ty)


def _require(semantic, src, layout):
    """Apply a layout carrier's encoding to ``src`` (a tl.tensor).

    The result keeps a carrier type so the encoding survives a @triton.jit
    return as well as a jit argument -- Triton builds the function result type
    from the frontend type of the returned value, which would otherwise be an
    unencoded block_type.
    """
    handle = semantic.builder.utlx_require_with_layout_carrier(
        [src.handle, layout.handle])
    return _carrier(handle, src.type.scalar)


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
    return _carrier(b.utlx_make_amd_mfma_layout(args))


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
    return _carrier(b.utlx_require_dot_operand_layout(args))


@tl.builtin
def slice_layout(parent, dim, _semantic=None):
    """Layout of ``parent`` with dimension ``dim`` sliced away."""
    dim = _uw(dim)
    b = _semantic.builder
    return _carrier(
        b.utlx_make_slice_layout([parent.handle,
                                  b.get_int32(int(dim))]))


@tl.builtin
def require_layout(src, layout, pin=False, _semantic=None):
    """Require that ``src`` be materialised in ``layout``.

    ``pin`` is accepted for source compatibility and currently ignored:
    tlx.require_layout has no pin attribute to forward it to.
    """
    return _require(_semantic, src, layout)


@tl.builtin
def release_layout(src, _semantic=None):
    """Drop an explicit layout, returning ``src`` with a default encoding.

    Passes through anything that is not an IR tensor (constexprs, scalars) so it
    can be applied unconditionally in helpers that accept either.
    """
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
    """Load ``base[offsets]``; the AMD backend lowers this to a buffer load.

    Offsets and mask are released first. A pointer tensor cannot carry a
    register layout, and tt.load requires ptr/mask/other to agree, so an encoded
    index expression reaching here fails to verify. release_layout is a no-op on
    an unencoded value.
    """
    if mask is None:
        return tlang.load(base + release_layout(offsets))
    return tlang.load(base + release_layout(offsets),
                      mask=release_layout(mask), other=other)


@jit
def buffer_store(value, base, offsets, mask=None):
    """Store ``value`` to ``base[offsets]`` as a buffer store.

    Released for the same reason as buffer_load: tt.store requires the value and
    pointer types to agree and a pointer tensor carries no register layout.
    """
    if mask is None:
        tlang.store(base + release_layout(offsets), release_layout(value))
    else:
        tlang.store(base + release_layout(offsets), release_layout(value),
                    mask=release_layout(mask))


def install_encoding_preserving_tensor():
    """Make frontend tensor types lower back to their actual IR type.

    Triton's frontend has no notion of layout encodings: ``TritonSemantic``
    builds each result's ``tl.block_type`` from shape+dtype alone. That is
    invisible until a type is turned *back* into IR -- at a @triton.jit argument,
    a jit return, or a cast -- at which point the encoding an explicit
    require_layout established is silently dropped and the verifier rejects the
    mismatch.

    Rather than teach every semantic op about encodings, re-type any tensor whose
    IR value actually carries one, so to_ir() reproduces it. Idempotent.
    """
    if getattr(tl.tensor, "_utlx_encoding_shim", False):
        return
    orig_init = tl.tensor.__init__

    def __init__(self, handle, type):
        orig_init(self, handle, type)
        if not isinstance(type, tl.block_type) or isinstance(type, _carrier_type):
            return
        try:
            ir_ty = handle.get_type()
        except Exception:
            return
        # Encoded tensor types print as `tensor<...xT, #enc>`.
        if "#" in str(ir_ty):
            self.type = _carrier_type(type.scalar, type.shape, ir_ty)

    tl.tensor.__init__ = __init__
    tl.tensor._utlx_encoding_shim = True


def install_cast_shim():
    """Let ``.to(dtype)`` work on a value that carries an explicit layout.

    ``TritonSemantic.cast`` builds its result type from shape+dtype alone, so on
    an encoded operand it emits e.g.
    ``arith.extf : tensor<...xbf16, #dot_op> -> tensor<...xf32>`` and the
    verifier rejects it as cast-incompatible. Encodings are element-type
    agnostic, so round-trip instead: release the layout, cast, then re-apply the
    same encoding using the original value as the carrier. Only carrier-typed
    inputs take this path, so ordinary kernels are unaffected. Idempotent.
    """
    from triton.language.semantic import TritonSemantic
    if getattr(TritonSemantic, "_utlx_cast_shim", False):
        return
    orig_cast = TritonSemantic.cast

    def cast(self, input, dst_ty, fp_downcast_rounding=None):
        if not isinstance(getattr(input, "type", None), _carrier_type):
            return orig_cast(self, input, dst_ty, fp_downcast_rounding)
        if input.type.scalar == dst_ty.scalar:
            return input
        plain = tl.tensor(
            self.builder.utlx_release_layout([input.handle]),
            tl.block_type(input.type.scalar, input.type.shape))
        casted = orig_cast(self, plain, dst_ty, fp_downcast_rounding)
        return _require(self, casted, input)

    TritonSemantic.cast = cast
    TritonSemantic._utlx_cast_shim = True


def install_binop_shim():
    """Propagate an explicit layout across a mixed-encoding binary op.

    ``binary_op_type_checking_impl`` broadcasts/splats the other operand into a
    plain ``block_type``, so ``encoded_tensor != 0`` ends up comparing an encoded
    tensor against an unencoded splat and the verifier rejects it. When exactly
    one side carries a layout and both have the same shape, re-apply that layout
    to the other side. This is the frontend propagation the TLX fork does
    implicitly. Idempotent.
    """
    from triton.language.semantic import TritonSemantic
    if getattr(TritonSemantic, "_utlx_binop_shim", False):
        return
    orig = TritonSemantic.binary_op_type_checking_impl

    def binary_op_type_checking_impl(self, lhs, rhs, *args, **kwargs):
        lhs, rhs = orig(self, lhs, rhs, *args, **kwargs)
        lhs_c = isinstance(getattr(lhs, "type", None), _carrier_type)
        rhs_c = isinstance(getattr(rhs, "type", None), _carrier_type)
        if lhs_c == rhs_c:
            return lhs, rhs
        enc, plain = (lhs, rhs) if lhs_c else (rhs, lhs)
        if not isinstance(getattr(plain, "type", None), tl.block_type):
            return lhs, rhs
        if list(plain.type.shape) != list(enc.type.shape):
            return lhs, rhs
        fixed = _require(self, plain, enc)
        return (lhs, fixed) if lhs_c else (fixed, rhs)

    TritonSemantic.binary_op_type_checking_impl = binary_op_type_checking_impl
    TritonSemantic._utlx_binop_shim = True


def install_where_shim():
    """Propagate an explicit layout through ``tl.where``.

    arith.select requires condition, both arms and the result to agree, but the
    frontend types the arms independently, so mixing a layout-carrying arm with a
    plain one fails to verify. Re-apply the layout to whichever parts lack it.
    Idempotent.
    """
    from triton.language.semantic import TritonSemantic
    if getattr(TritonSemantic, "_utlx_where_shim", False):
        return
    orig = TritonSemantic.where

    def where(self, condition, x, y):
        carrier = next(
            (v for v in (x, y, condition)
             if isinstance(getattr(v, "type", None), _carrier_type)), None)
        if carrier is not None:
            shape = list(carrier.type.shape)

            def fix(v):
                ty = getattr(v, "type", None)
                if (isinstance(ty, tl.block_type)
                        and not isinstance(ty, _carrier_type)
                        and list(ty.shape) == shape):
                    return _require(self, v, carrier)
                return v

            condition, x, y = fix(condition), fix(x), fix(y)
        return orig(self, condition, x, y)

    TritonSemantic.where = where
    TritonSemantic._utlx_where_shim = True
