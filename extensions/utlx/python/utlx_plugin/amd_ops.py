"""Meta-fork AMD TLX ops, expressed with upstream Triton.

Meta's fork adds a family of AMD-specific TLX builtins that its gfx9 op
kernels (mm, flash attention, HSTU) and inductor templates call. Most of them
lower to fork-only TTGIR ops whose purpose is scheduling or register
allocation control -- they do not change the values a kernel computes. This
module provides each one on upstream Triton with the same results:

* Value-preserving markers (``amd_register_resident``,
  ``amd_register_class_anchor``, ``amd_mfma_commit``, ``assert_same_layout``,
  ``amd_sched_barrier``, ``amd_iglp_opt``, ``assume_uniform``) validate their
  arguments as the fork does and return their inputs unchanged.
* Ops with an upstream equivalent are rewritten onto it:
  ``amd_scheduled_mfma`` is ``tl.dot``, ``rematerialized_range`` is
  ``tl.arange``, ``buffer_load_to_local`` is ``async_load`` on ``ptr +
  offsets`` (the AMD backend's buffer-op conversion turns that back into a
  buffer load), and ``buffer_atomic_add`` is ``tl.atomic_add`` on
  ``ptr + offsets``.
* ``extract_slice`` selects the slice with reshape/permute/split rather than
  ``amdg.extract_slice``, which needs encoded types TTIR does not have yet.
* ``warp_predicate`` computes the body for every lane and selects the result
  per lane, and ``warp_any`` reduces over the whole CTA rather than one wave.
  Both are conservative: the fork's kernels only use them to skip work whose
  result is masked anyway, so doing the work everywhere gives the same values.

The scheduling these ops ask for is lost, so kernels written around them run
correctly but not at the fork's speed.
"""

import builtins

import triton.language as tlang
import triton.language.core as tl
from triton.runtime.jit import jit

from . import types as tlx
from .compiler.semantic import _promote
from .layout_ops import _carrier, _carrier_type, _require


def _uw(x):
    return tl._unwrap_if_constexpr(x)


def _is_pow2(n):
    return n > 0 and n & (n - 1) == 0


def _check_contiguity(contiguity):
    contiguity = _uw(contiguity)
    assert (isinstance(contiguity, int) and not isinstance(contiguity, bool)
            and _is_pow2(contiguity)), \
        f"contiguity must be a positive power of two, got {contiguity!r}"


def _verify_buffer_ops(ptr, offsets, mask=None, other=None):
    assert ptr.type.is_ptr(), "ptr must be a scalar pointer type"
    assert offsets.type.is_block(), "offsets must be a tensor type"
    assert offsets.dtype.is_int32() or offsets.dtype.is_uint32(), \
        "offsets element type must be int32 or uint32"
    if other is not None:
        assert mask is not None, "when other is set, mask must also be set"


def _keep_type(handle, like):
    """Wrap ``handle`` like ``like``, keeping a layout carrier a carrier."""
    if isinstance(like.type, _carrier_type):
        return _carrier(handle, like.type.scalar)
    return tl.tensor(handle, _promote(handle, like.type))


# --- memory ---------------------------------------------------------------


@tl.builtin
def buffer_load_to_local(dest,
                         ptr,
                         offsets,
                         mask=None,
                         other=None,
                         cache_modifier: str = "",
                         contiguity=1,
                         _semantic=None) -> tlx.async_token:
    """Async copy ``ptr[offsets]`` into ``dest``; returns an async token.

    ``contiguity`` is a trusted vectorization hint in the fork. Here the
    vector width comes from the backend's own axis analysis instead.
    """
    from .mem_ops import async_load

    mask = _uw(mask)
    other = _uw(other)
    _verify_buffer_ops(ptr, offsets, mask, other)
    _check_contiguity(contiguity)
    ptrs = _semantic.add(ptr, offsets, False)
    return async_load(ptrs,
                      dest,
                      mask=mask,
                      other=other,
                      cache_modifier=cache_modifier,
                      _semantic=_semantic)


@tl.builtin
def buffer_atomic_add(ptr,
                      offsets,
                      value,
                      mask=None,
                      sem=None,
                      scope=None,
                      contiguity=1,
                      _semantic=None):
    """``atomic_add(ptr + offsets, value)``; returns the old values."""
    mask = _uw(mask)
    _verify_buffer_ops(ptr, offsets, mask)
    _check_contiguity(contiguity)
    element_ty = ptr.type.scalar.element_ty
    assert (element_ty.is_standard_floating() or
            (element_ty.is_int() and element_ty.primitive_bitwidth in (32, 64))), \
        "buffer_atomic_add supports only f16, bf16, f32, f64, i32, and i64 values"
    value = _semantic.cast(_semantic.to_tensor(_uw(value)), element_ty)
    if mask is not None:
        mask = _semantic.cast(_semantic.to_tensor(mask), tl.int1)
    ptrs = _semantic.add(ptr, offsets, False)
    return _semantic.atomic_add(ptrs, value, mask, _uw(sem), _uw(scope))


@tl.builtin
def assert_same_layout(lhs, rhs, _semantic=None) -> None:
    """Accepted and not checked.

    The fork compares the final LinearLayouts after layout propagation, a
    pass uTLX does not have. It never changes the program, only rejects it.
    """
    rhs = _uw(rhs)
    if not isinstance(rhs,
                      (tl.tensor, tlx.buffered_tensor, tlx.layout_encoding)):
        raise TypeError(
            "`rhs` must be a TLX tensor/buffer value or layout encoding")


# --- registers / MFMA -----------------------------------------------------


@tl.builtin
def extract_slice(source, shape, offsets, _semantic=None):
    """Return the static ``shape`` slice of ``source`` starting at ``offsets``.

    Each sliced axis must split evenly into a power-of-two number of
    ``shape``-sized pieces with ``offsets`` on a piece boundary -- the aligned
    slices the fork accepts. The piece is picked by reshaping the axis into
    ``[pieces, extent]`` and splitting the ``pieces`` dim bit by bit.
    """
    assert isinstance(source, tl.tensor), \
        f"source must be a tensor, got {type(source).__name__}"
    shape = [_uw(d) for d in shape]
    offsets = [_uw(o) for o in offsets]
    src_shape = [_uw(d) for d in source.shape]
    rank = len(src_shape)
    assert len(shape) == rank, f"shape must have rank {rank}, got {len(shape)}"
    assert len(offsets) == rank, \
        f"offsets must have rank {rank}, got {len(offsets)}"
    b = _semantic.builder
    if (isinstance(source.type, _carrier_type)
            and getattr(b.options, "backend_name", None) == "hip"):
        # An encoded source, e.g. a dot operand after require_layout: slice it
        # in place so the result keeps that layout.
        handle = b.utlx_amd_extract_slice([source.handle] +
                                          [b.get_int32(d) for d in shape] +
                                          [b.get_int32(o) for o in offsets])
        return _carrier(handle, source.type.scalar)
    x = source
    for axis, (extent, offset,
               src_extent) in enumerate(zip(shape, offsets, src_shape)):
        assert isinstance(extent, int) and extent > 0, \
            "shape must contain positive constexpr integers"
        assert isinstance(offset, int) and offset >= 0, \
            "offsets must contain non-negative constexpr integers"
        assert offset + extent <= src_extent, (
            f"slice exceeds source extent at axis {axis}: "
            f"{offset} + {extent} > {src_extent}")
        if extent == src_extent:
            continue
        pieces = src_extent // extent
        assert src_extent % extent == 0 and _is_pow2(pieces) \
            and offset % extent == 0, (
                f"unaligned slice at axis {axis}: extent {extent} at offset "
                f"{offset} of {src_extent}")
        x = _select_piece(_semantic, x, axis, extent, pieces, offset // extent)
    return x


def _select_piece(semantic, x, axis, extent, pieces, index):
    cur = [_uw(d) for d in x.shape]
    rest = cur[:axis] + [extent] + cur[axis + 1:]
    # [.., pieces * extent, ..] -> [.., pieces, extent, ..] -> pieces last.
    x = semantic.reshape(x, cur[:axis] + [pieces, extent] + cur[axis + 1:],
                         False)
    perm = [d for d in range(len(cur) + 1) if d != axis] + [axis]
    x = semantic.permute(x, perm)
    bits = pieces.bit_length() - 1
    x = semantic.reshape(x, rest + [2] * bits, False)
    # Row-major: the last dim is the lowest bit of the piece index.
    for bit in range(bits):
        lo, hi = semantic.split(x)
        x = hi if (index >> bit) & 1 else lo
    return x


@tl.builtin
def rematerialized_range(start, end, identity, placement=None, _semantic=None):
    """``tl.arange(start, end)``.

    The fork keeps distinct-``identity`` copies apart under CSE so the range
    is recomputed near each use; the values are the same.
    """
    start = _uw(start)
    end = _uw(end)
    identity = _uw(identity)
    assert isinstance(start, int) and not isinstance(start, bool), \
        "start must be a constexpr integer"
    assert isinstance(end, int) and not isinstance(end, bool) \
        and end > start, "end must be a constexpr integer greater than start"
    assert isinstance(identity, int) and not isinstance(identity, bool), \
        "identity must be a constexpr integer"
    return _semantic.arange(start, end)


@tl.builtin
def amd_register_resident(value,
                          register_class: tl.constexpr = "agpr",
                          registers_per_group: tl.constexpr = 1,
                          _semantic=None):
    """Register-class placement hint; returns ``value`` unchanged."""
    register_class = _uw(register_class)
    assert register_class in ("agpr", "vgpr"), \
        f'register_class must be "agpr" or "vgpr", got {register_class!r}'
    assert isinstance(value, tl.tensor), "value must be a tensor"
    return value


@tl.builtin
def assume_uniform(value, _semantic=None):
    """Wave-uniformity hint; returns ``value`` unchanged.

    The fork emits ``amdg.assume_uniform`` (``v_readfirstlane``) so a scalar
    loaded from memory -- typically a buffer-op base pointer -- can live in an
    SGPR. Upstream has no such op; without it the backend keeps the value
    per-lane, which may mean a waterfall loop around buffer accesses, but the
    results are the same.
    """
    assert isinstance(value, tl.tensor), "value must be a tensor"
    ty = value.type
    assert ty.is_ptr() or ty.primitive_bitwidth >= 16, \
        f"assume_uniform expects a scalar pointer or a 16/32/64-bit value, got {ty}"
    return value


@tl.builtin
def amd_register_class_anchor(value,
                              register_class: tl.constexpr = "vgpr",
                              _semantic=None):
    """Register-class anchor hint; returns ``value`` unchanged."""
    register_class = _uw(register_class)
    assert register_class in ("agpr", "vgpr"), \
        f'register_class must be "agpr" or "vgpr", got {register_class!r}'
    assert isinstance(value, tl.tensor), "value must be a tensor"
    return value


@tl.builtin
def amd_scheduled_mfma(a,
                       b,
                       acc,
                       accumulator_role: tl.constexpr,
                       resident_operand: tl.constexpr = None,
                       accumulator_register_class: tl.constexpr = None,
                       initialize: tl.constexpr = False,
                       _semantic=None):
    """``acc + a @ b``, or ``a @ b`` with ``initialize=True``.

    The role / residency / register-class arguments only steer the fork's
    MFMA scheduling and register assignment; they are validated and dropped.
    """
    accumulator_role = _uw(accumulator_role)
    resident_operand = _uw(resident_operand)
    accumulator_register_class = _uw(accumulator_register_class)
    initialize = _uw(initialize)
    assert isinstance(a, tl.tensor) and isinstance(b, tl.tensor), \
        "a and b must be distributed tensors"
    assert isinstance(acc, tl.tensor), "acc must be a distributed tensor"
    assert resident_operand is None or (
        isinstance(resident_operand, int)
        and not isinstance(resident_operand, bool)
        and resident_operand in (0, 1)), \
        "resident_operand must be None, 0, or 1"
    assert accumulator_role in ("transient", "persistent"), \
        'accumulator_role must be either "transient" or "persistent"'
    assert accumulator_register_class in (None, "agpr", "vgpr"), \
        'accumulator_register_class must be None, "agpr", or "vgpr"'
    assert isinstance(initialize, bool), "initialize must be a constexpr bool"
    if initialize:
        zeros = _semantic.full([_uw(d) for d in acc.shape], 0, acc.dtype)
        acc = _require(_semantic, zeros, acc) if isinstance(
            acc.type, _carrier_type) else zeros
    out = _semantic.dot(a, b, acc, None, None, acc.dtype)
    return _carrier(out.handle, acc.dtype)


@tl.builtin
def amd_mfma_commit(value, preserve=None, _semantic=None):
    """MFMA completion boundary; returns every value unchanged.

    A single value is returned directly; a tuple keeps its arity; a
    ``preserve`` operand is appended to the result.
    """
    single_value = isinstance(value, tl.tensor)
    values = (value, ) if single_value else tuple(value)
    assert len(values) > 0 and all(isinstance(v, tl.tensor) for v in values), \
        "value must be a tensor or nonempty tensor tuple"
    assert preserve is None or isinstance(preserve, tl.tensor), \
        "preserve must be None or a tensor"
    if preserve is None:
        return values[0] if single_value else values
    return values + (preserve, )


@tl.builtin
def amd_sched_barrier(mask: tl.constexpr = 0, _semantic=None):
    """Instruction-scheduling fence hint; emits nothing.

    Not a workgroup barrier or memory fence in the fork either, so dropping
    it changes instruction order only.
    """
    mask = _uw(mask)
    assert isinstance(mask, int), \
        f"mask must be a constexpr integer, got {type(mask).__name__}"
    assert 0 <= mask <= 0xFFF, \
        f"mask must use only AMD scheduling-class bits 0..11, got {mask:#x}"


@tl.builtin
def amd_iglp_opt(variant: tl.constexpr, _semantic=None):
    """``llvm.amdgcn.iglp.opt`` scheduling hint; emits nothing."""
    variant = _uw(variant)
    assert isinstance(variant, int) and not isinstance(variant, bool), \
        "variant must be a constexpr integer"
    assert 0 <= variant <= 3, \
        f"variant must be one of 0, 1, 2, or 3, got {variant}"


# --- wave-level control ---------------------------------------------------


@tl.builtin
def num_warps(_semantic=None):
    """The number of warps executing the current kernel."""
    return tl.constexpr(_semantic.builder.options.num_warps)


@jit
def warp_any(pred):
    """Whether any lane's predicate is true -- across the CTA, not one wave.

    A CTA-wide vote is true whenever some wave's vote would be, so a guarded
    region runs at least where the fork would run it.
    """
    return tlang.max(pred.to(tlang.int32)) != 0


@tl.builtin
def warp_predicate(predicate,
                   inits,
                   body,
                   args=(),
                   wave_uniform=False,
                   _semantic=None,
                   _generator=None):
    """``body(*inits, *args)`` where ``predicate`` holds, ``inits`` elsewhere.

    The fork runs ``body`` under an EXEC mask and skips inactive waves. Here
    every lane computes ``body`` and the carried values are selected per
    element, which yields the same values: ``body`` must be straight-line and
    free of cross-wave synchronization in the fork too. A tensor predicate
    may cover a leading prefix of the carried shape (e.g. one per row).
    """
    if isinstance(inits, tl.tensor):
        inits = (inits, )
    elif not isinstance(inits, (builtins.tuple, builtins.list, tl.tuple)):
        raise TypeError(
            "warp_predicate inits must be a tensor, tuple, or list")
    if not isinstance(args, (builtins.tuple, builtins.list, tl.tuple)):
        args = (args, )
    inits = builtins.list(inits)
    args = builtins.list(args)
    if not inits:
        raise ValueError("warp_predicate requires at least one carried value")
    if not all(isinstance(v, tl.tensor) for v in inits):
        raise TypeError("warp_predicate carried values must be tensors")
    predicate = _semantic.to_tensor(predicate)
    if predicate.dtype != tl.int1:
        raise TypeError("warp_predicate predicate must have bool dtype, got "
                        f"{predicate.dtype}")
    if not isinstance(_uw(wave_uniform), builtins.bool):
        raise TypeError("warp_predicate wave_uniform must be a bool")

    body_result = _generator.call_JitFunction(body, inits + args, kwargs={})
    if isinstance(body_result, tl.tensor):
        results = [body_result]
    elif isinstance(body_result, (builtins.tuple, builtins.list, tl.tuple)):
        results = builtins.list(body_result)
    else:
        raise TypeError(
            "warp_predicate body must return a tensor, tuple, or list")
    if len(results) != len(inits):
        raise TypeError(f"warp_predicate body returned {len(results)} values "
                        f"for {len(inits)} carried values")

    merged = []
    for index, (init, result) in enumerate(zip(inits, results)):
        if not isinstance(result, tl.tensor):
            raise TypeError(f"warp_predicate result {index} is not a tensor")
        cond = predicate
        if cond.type.is_block():
            while len(cond.shape) < len(init.shape):
                cond = _semantic.expand_dims(cond, len(cond.shape))
            # Full shape up front, so where() can give it init's layout.
            cond = _semantic.broadcast_impl_shape(cond,
                                                  [_uw(d) for d in init.shape])
        out = _semantic.where(cond, result, init)
        merged.append(_keep_type(out.handle, init))
    return merged[0] if len(merged) == 1 else builtins.tuple(merged)
