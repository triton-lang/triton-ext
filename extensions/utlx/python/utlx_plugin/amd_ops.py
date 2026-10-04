"""Meta-fork AMD TLX ops, expressed with upstream Triton.

Meta's fork adds a family of AMD-specific TLX builtins that its gfx9 op
kernels (mm, flash attention, HSTU) and inductor templates call. Most of them
lower to fork-only TTGIR ops whose purpose is scheduling or register
allocation control -- they do not change the values a kernel computes. This
module provides each one on upstream Triton with the same results:

* Value-preserving markers (``amd_register_resident``,
  ``amd_register_class_anchor``, ``amd_mfma_commit``, ``assert_same_layout``,
  ``amd_sched_barrier``, ``amd_iglp_opt``) validate their arguments as the fork
  does and return their inputs unchanged.
* Ops with an upstream equivalent are rewritten onto it:
  ``amd_scheduled_mfma`` is ``tl.dot``, ``rematerialized_range`` is
  ``tl.arange``, ``buffer_load_to_local`` is ``async_load`` on ``ptr +
  offsets`` (the AMD backend's buffer-op conversion turns that back into a
  buffer load), and ``buffer_atomic_add`` is ``tl.atomic_add`` on
  ``ptr + offsets``.

The scheduling these ops ask for is lost, so kernels written around them run
correctly but not at the fork's speed.
"""

import triton.language.core as tl

from . import types as tlx
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
    if not isinstance(rhs, (tl.tensor, tlx.buffered_tensor, tlx.layout_encoding)):
        raise TypeError(
            "`rhs` must be a TLX tensor/buffer value or layout encoding")


# --- registers / MFMA -----------------------------------------------------


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
