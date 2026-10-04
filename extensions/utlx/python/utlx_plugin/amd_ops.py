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

The scheduling these ops ask for is lost, so kernels written around them run
correctly but not at the fork's speed.
"""

import triton.language.core as tl

from . import types as tlx


def _uw(x):
    return tl._unwrap_if_constexpr(x)


# --- memory ---------------------------------------------------------------


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
