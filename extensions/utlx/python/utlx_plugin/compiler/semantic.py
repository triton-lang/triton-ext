"""A ``TritonSemantic`` subclass that propagates explicit layouts.

Triton's frontend has no notion of layout encodings: ``TritonSemantic`` types
every result from shape and dtype alone. That stays invisible until a frontend
type is turned back into IR -- at a ``@triton.jit`` argument or return, or at a
cast -- where the encoding an explicit ``require_layout`` established is dropped
and the verifier rejects the mismatch. uTLX therefore has to supply the
propagation itself.

It lives in a subclass rather than in wrappers patched onto
``TritonSemantic``'s methods: the behaviour is then readable in one place,
composes through ``super()``, and needs a single injection point.
``triton.compiler.code_generator`` imports ``TritonSemantic`` from this module
at call time, so rebinding the module attribute is enough.

Every override is a no-op on values that carry no explicit layout, so ordinary
Triton kernels are unaffected.
"""

from typing import Generic, TypeVar

import triton.language.core as tl
from triton.language import semantic as _semantic_mod
# Imported directly, not as `_semantic_mod.TritonSemantic`: mypy rejects a
# module attribute as a base class ("not valid as a type"). This also pins the
# base to the original class, which is what we want -- `_semantic_mod` is only
# used for the rebind in install_semantic().
from triton.language.semantic import TritonSemantic as _BaseSemantic

from ..layout_ops import _carrier_type, _require

_TensorTy = TypeVar("_TensorTy")


def _has_layout(v):
    return isinstance(getattr(v, "type", None), _carrier_type)


class UTLXSemantic(_BaseSemantic[_TensorTy], Generic[_TensorTy]):
    """TritonSemantic that keeps explicit register layouts consistent.

    Generic like its base: ``GluonSemantic`` is declared as
    ``TritonSemantic[TensorTy]``, so whatever the name is bound to must stay
    subscriptable.
    """

    # -- casts --------------------------------------------------------------

    def cast(self, input, dst_ty, fp_downcast_rounding=None):
        """``.to(dtype)`` on a value that carries an explicit layout.

        The base implementation builds its result type from shape and dtype, so
        on an encoded operand it emits e.g.
        ``arith.extf : tensor<...xbf16, #dot_op> -> tensor<...xf32>``, which is
        rejected as cast-incompatible. Encodings are element-type agnostic, so
        round-trip: release, cast, re-apply the same encoding.
        """
        if not _has_layout(input):
            return super().cast(input, dst_ty, fp_downcast_rounding)
        if input.type.scalar == dst_ty.scalar:
            return input
        plain = self._drop_layout(input)
        return _require(self,
                        super().cast(plain, dst_ty, fp_downcast_rounding),
                        input)

    # -- mixed-encoding binary ops -----------------------------------------

    def binary_op_type_checking_impl(self, lhs, rhs, *args, **kwargs):
        """Give both operands the same layout when only one has it.

        The base implementation broadcasts or splats the other side into a plain
        ``block_type``, so ``encoded != 0`` compares an encoded tensor against an
        unencoded splat.
        """
        lhs, rhs = super().binary_op_type_checking_impl(
            lhs, rhs, *args, **kwargs)
        lhs_c, rhs_c = _has_layout(lhs), _has_layout(rhs)
        if lhs_c == rhs_c:
            return lhs, rhs
        enc, plain = (lhs, rhs) if lhs_c else (rhs, lhs)
        if not isinstance(getattr(plain, "type", None), tl.block_type):
            return lhs, rhs
        if list(plain.type.shape) != list(enc.type.shape):
            return lhs, rhs
        fixed = _require(self, plain, enc)
        return (lhs, fixed) if lhs_c else (fixed, rhs)

    # -- select -------------------------------------------------------------

    def where(self, condition, x, y):
        """``arith.select`` requires condition, both arms and result to agree."""
        carrier = next((v for v in (x, y, condition) if _has_layout(v)), None)
        if carrier is not None:
            shape = list(carrier.type.shape)

            def fix(v):
                ty = getattr(v, "type", None)
                if (isinstance(ty, tl.block_type) and not _has_layout(v)
                        and list(ty.shape) == shape):
                    return _require(self, v, carrier)
                return v

            condition, x, y = fix(condition), fix(x), fix(y)
        return super().where(condition, x, y)

    # -- pointer ops --------------------------------------------------------

    def _drop_layout(self, v):
        """Strip an explicit layout, returning a plainly-typed tensor."""
        if not _has_layout(v):
            return v
        return tl.tensor(self.builder.utlx_release_layout([v.handle]),
                         tl.block_type(v.type.scalar, v.type.shape))

    def load(self, ptr, mask, other, *args, **kwargs):
        """A pointer tensor carries no register layout, and tt.load requires
        ptr, mask and other to agree."""
        return super().load(self._drop_layout(ptr), self._drop_layout(mask),
                            self._drop_layout(other), *args, **kwargs)

    def store(self, ptr, val, mask, *args, **kwargs):
        """As :meth:`load`; tt.store requires value and pointer to agree."""
        return super().store(self._drop_layout(ptr), self._drop_layout(val),
                             self._drop_layout(mask), *args, **kwargs)


def install_semantic():
    """Make uTLX's semantic the one the code generator builds. Idempotent.

    Gluon is imported first on purpose: ``GluonSemantic`` subclasses
    ``TritonSemantic`` at module scope, and importing it before the rebind
    pins it to the original base. Gluon has its own layout model and should not
    inherit these overrides, and this keeps that true regardless of the order
    the user happens to import things in.
    """
    if getattr(_semantic_mod.TritonSemantic, "_utlx_semantic", False):
        return
    try:
        import triton.experimental.gluon.language._semantic  # noqa: F401
    except ImportError:
        pass  # Triton built without gluon; nothing to pin
    UTLXSemantic._utlx_semantic = True
    _semantic_mod.TritonSemantic = UTLXSemantic


def install_encoding_preserving_tensor():
    """Make frontend tensor types lower back to their actual IR type.

    Unlike the semantic overrides above this has to be a patch: ``tl.tensor`` is
    constructed directly in a hundred places inside ``TritonSemantic``, and
    Triton exposes no hook for the type a result is given. Without it, ops whose
    result type is inferred in C++ (``tl.dot``, comparisons, reductions) leave an
    encoded IR value behind a plain ``block_type``, and the layout is lost at the
    next boundary. Idempotent.
    """
    if getattr(tl.tensor, "_utlx_encoding_shim", False):
        return
    orig_init = tl.tensor.__init__

    def __init__(self, handle, type):
        orig_init(self, handle, type)
        if not isinstance(type, tl.block_type) or isinstance(
                type, _carrier_type):
            return
        try:
            ir_ty = handle.get_type()
        except AttributeError:
            return  # not an ir.value (constexpr placeholder, etc.)
        # Encoded tensor types print as `tensor<...xT, #enc>`.
        if "#" in str(ir_ty):
            self.type = _carrier_type(type.scalar, type.shape, ir_ty)

    tl.tensor.__init__ = __init__
    tl.tensor._utlx_encoding_shim = True


__all__ = [
    "UTLXSemantic", "install_encoding_preserving_tensor", "install_semantic"
]
