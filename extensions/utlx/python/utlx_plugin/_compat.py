"""Shims that let uTLX load on Tritons with a different plugin ABI.

uTLX is developed against the Triton pinned in ``ci/triton-hash.txt``, where a
plugin is registered explicitly with ``extend_with`` / ``extend_dialects_with``
and its ops and passes are bound as ``create_<op>`` and ``add_<pass>``. The
Triton 3.8.0 release -- the build PyTorch ships, and the one most users get --
predates both: it loads every library named in ``TRITON_PLUGIN_PATHS`` while
``libtriton`` is imported, and binds ops and passes under their bare names.

Neither Triton has ``TritonSemantic.dot_precheck`` or
``TritonSemantic._prepare_legacy_load``; those were only ever split out of
``dot`` and ``load`` in Meta's TLX fork, so uTLX has to supply them itself.

Every shim is feature-detected, so the package runs unchanged on either Triton.
"""

import os
import sys
from pathlib import Path

PLUGIN_PATHS_ENV = "TRITON_PLUGIN_PATHS"

PLUGIN_DIR = Path(__file__).resolve().parent
PLUGIN_LIBRARY = PLUGIN_DIR / "libutlx.so"


def _announce_plugin(library):
    """Second chance to set ``TRITON_PLUGIN_PATHS`` for a Triton that auto-loads.

    ``_utlx_autoregister`` normally does this from a .pth at interpreter
    startup. This covers installs where the .pth never ran -- an opted-out or
    vendored tree -- and only helps if triton has not been imported yet.
    """
    if "triton._C.libtriton" in sys.modules:
        # Too late to matter. If this Triton needed the variable,
        # register_plugin() is the one that says so.
        return

    paths = [
        p for p in os.environ.get(PLUGIN_PATHS_ENV, "").split(os.pathsep) if p
    ]
    if str(library) not in paths:
        paths.append(str(library))
    os.environ[PLUGIN_PATHS_ENV] = os.pathsep.join(paths)


_announce_plugin(PLUGIN_LIBRARY)


def register_plugin(library):
    """Make *library*'s dialects, passes and ops visible under uTLX's names."""
    from triton._C.libtriton import ir, passes

    if hasattr(passes.plugin, "extend_with"):
        passes.plugin.extend_with(str(library))  # adds passes
        ir.extend_dialects_with(str(library))  # adds dialects
        ir.builder.extend_with(str(library))  # adds ops
        _bind_unprefixed(ir.builder, "create_")
        _bind_unprefixed(passes.plugin, "add_")
        return

    if not any(name.startswith("utlx_") for name in dir(ir.builder)):
        raise RuntimeError(
            f"Triton {_triton_version()} loads plugins from {PLUGIN_PATHS_ENV} "
            f"while it is imported, and {library} was not among them. The "
            f"utlx_plugin.pth that normally sets it did not run: unset "
            f"UTLX_NO_AUTOREGISTER, or set {PLUGIN_PATHS_ENV}={library}, or "
            f"import utlx_plugin before triton.")


def install_semantic_helpers():
    """Add the ``TritonSemantic`` entry points uTLX calls but upstream lacks."""
    import triton.language as tl
    from triton.language import core
    from triton.language.semantic import TritonSemantic

    if not hasattr(tl, "_unwrap_if_constexpr"):
        tl._unwrap_if_constexpr = core._unwrap_if_constexpr
    if not hasattr(TritonSemantic, "dot_precheck"):
        TritonSemantic.dot_precheck = dot_precheck
    if not hasattr(TritonSemantic, "_prepare_legacy_load"):
        TritonSemantic._prepare_legacy_load = _prepare_legacy_load


def _bind_unprefixed(namespace, prefix):
    """Bind ``utlx_<x>`` for every ``<prefix>utlx_<x>`` symbol in *namespace*."""
    for name in dir(namespace):
        if name.startswith(prefix + "utlx_"):
            bare = name[len(prefix):]
            if not hasattr(namespace, bare):
                setattr(namespace, bare, getattr(namespace, name))


def _triton_version():
    import triton

    return getattr(triton, "__version__", "unknown")


def dot_precheck(self,
                 lhs,
                 rhs,
                 acc,
                 input_precision,
                 allow_tf32,
                 max_num_imprecise_acc,
                 out_dtype,
                 tlx_paired_ctas=False):
    """``TritonSemantic.dot``'s checks and type inference, without the op.

    Returns the operands plus everything ``async_dot`` needs to build the mma
    itself: ``(lhs, rhs, acc_handle, input_precision, max_num_imprecise_acc,
    ret_ty)``. Mirrors ``TritonSemantic.dot``; keep the two in step.

    ``tlx_paired_ctas`` types the result for a CTA-pair mma, where the two CTAs
    each hold half of N.
    """
    from triton import knobs
    from triton.language import core
    import triton.language as tl

    unwrap = core._unwrap_if_constexpr

    input_precision = unwrap(input_precision)
    allow_tf32 = unwrap(allow_tf32)
    assert input_precision is None or allow_tf32 is None, (
        "Only one of input_precision and allow_tf32 can be specified")
    if input_precision is None:
        supports_tf32 = "tf32" in self.builder.options.allowed_dot_input_precisions
        input_precision = knobs.language.fp32_default or (
            "tf32" if supports_tf32 and allow_tf32 is not False else "ieee")

    out_dtype = unwrap(out_dtype)
    max_num_imprecise_acc = unwrap(max_num_imprecise_acc)
    acc = unwrap(acc)

    assert lhs.type.is_block() and rhs.type.is_block(), (
        "dot operands must be block tensors (not scalars)")

    if lhs.dtype.is_fp8() and rhs.dtype.is_fp8():
        pass  # all combinations of supported fp8 x fp8 are permitted
    else:
        assert lhs.dtype in (tl.int8, tl.uint8, tl.float16, tl.bfloat16,
                             tl.float32,
                             tl.float64), f"Unsupported lhs dtype {lhs.dtype}"
        assert rhs.dtype in (tl.int8, tl.uint8, tl.float16, tl.bfloat16,
                             tl.float32,
                             tl.float64), f"Unsupported rhs dtype {rhs.dtype}"
        assert lhs.dtype == rhs.dtype, (
            f"Both operands must be same dtype. Got {lhs.dtype} and {rhs.dtype}"
        )

    if lhs.dtype.is_fp8e4b15() or rhs.dtype.is_fp8e4b15():
        if "fp8e4b15" in self.builder.options.deprecated_fp8_dot_operand_dtypes:
            import warnings
            warnings.warn(
                "the use of fp8e4b15 is deprecated on Hopper and later architectures "
                "and can cause significant slow down. It will be removed in a future "
                "triton release")
        # There is no fp8e4b15 type in MLIR, so upcast.
        lhs = self.cast(lhs, tl.float16)
        rhs = self.cast(rhs, tl.float16)

    uses_fp8e4b8 = lhs.dtype.is_fp8e4b8() or rhs.dtype.is_fp8e4b8()
    uses_fp8e5b16 = lhs.dtype.is_fp8e5b16() or rhs.dtype.is_fp8e5b16()
    if uses_fp8e4b8 or uses_fp8e5b16:
        type_name = "fp8e4b8" if uses_fp8e4b8 else "fp8e5b16"
        if type_name in self.builder.options.deprecated_fp8_dot_operand_dtypes:
            arch = self.builder.options.arch
            import warnings
            warnings.warn(
                f"{type_name} is AMD gfx942 specific and not supported on {arch} so "
                f"it's upcasted to fp16 and can cause significant slow down. Please "
                f"use OCP fp8 variants on {arch} for performance")
            lhs = self.cast(lhs, tl.float16)
            rhs = self.cast(rhs, tl.float16)

    if input_precision is None:
        input_precision = self.builder.options.default_dot_input_precision
    if out_dtype is None:
        out_dtype = tl.float32 if acc is None else acc.type.element_ty

    input_precision = self._str_to_dot_input_precision(input_precision)

    lhs_rank = len(lhs.shape)
    rhs_rank = len(rhs.shape)
    assert lhs_rank == rhs_rank == 2 or lhs_rank == rhs_rank == 3, (
        f"Both inputs must be either 2D or 3D; (lhs: {lhs.shape} vs rhs: {rhs.shape})"
    )
    assert unwrap(lhs.shape[-1]) == unwrap(rhs.shape[-2]), (
        f"First input shape ({lhs.shape}) and second input shape {rhs.shape} are not "
        f"compatible for matmul (second index of first shape ({unwrap(lhs.shape[-1])}) "
        f"must be equal to first index of second shape ({unwrap(rhs.shape[-2])}))"
    )

    assert self.builder.codegen_fns.get("min_dot_size") is not None, (
        "target doesn't provide lower shape bounds for dot.")
    min_dot_size = self.builder.codegen_fns["min_dot_size"](lhs.type, rhs.type)
    assert (unwrap(lhs.shape[-2]) >= min_dot_size[0]
            and unwrap(lhs.shape[-1]) >= min_dot_size[2]
            and unwrap(rhs.shape[-1]) >= min_dot_size[1]), (
                f"Input shapes should have M >= {min_dot_size[0]}, "
                f"N >= {min_dot_size[1]} and K >= {min_dot_size[2]}")

    if lhs.type.scalar.is_int():
        assert lhs.type.scalar == tl.int8, "only int8 supported!"
        _0 = self.builder.get_int32(0)
        ret_scalar_ty = tl.int32
    elif out_dtype.is_bf16():
        raise ValueError(
            "out_dtype=bfloat16 is unsupported. Please use out_dtype=float32/float16 "
            "and cast with `.to(tl.bfloat16)`")
    elif lhs.type.scalar.is_fp32() or lhs.type.scalar.is_bf16():
        _0 = self.builder.get_fp32(0)
        ret_scalar_ty = tl.float32
    elif lhs.type.scalar.is_fp64():
        _0 = self.builder.get_fp64(0)
        ret_scalar_ty = tl.float64
    else:
        _0 = self.builder.get_fp16(
            0) if out_dtype.is_fp16() else self.builder.get_fp32(0)
        ret_scalar_ty = out_dtype

    M = lhs.type.shape[-2]
    if tlx_paired_ctas:
        # rhs is [K, N/2] in two-CTA mode, so scale N back up. M per CTA is
        # whatever the tile shape says -- tcgen05 pair-CTA MMA handles 64 as
        # well as 128, and the in-tree semantic imposes no restriction here.
        N = 2 * rhs.type.shape[-1]
    else:
        N = rhs.type.shape[-1]
    K = lhs.type.shape[-1]
    B = lhs.type.shape[0] if lhs_rank == 3 else None
    ret_ty = tl.block_type(ret_scalar_ty, [B, M, N] if B else [M, N])

    if acc is None:
        acc_handle = self.builder.create_splat(ret_ty.to_ir(self.builder), _0)
    else:
        acc_handle = acc.handle
        assert acc.type.shape == ret_ty.shape, (
            f"expected accumulator shape {ret_ty.shape}, got {acc.type.shape}")
        assert acc.type.element_ty == out_dtype, (
            f"expected accumulator dtype {out_dtype}, got {acc.type.element_ty}; "
            f"pass out_dtype={acc.type.element_ty} to use this accumulator dtype"
        )

    # max_num_imprecise_acc only applies to fp8 -> fp32 dot on sm_90
    if max_num_imprecise_acc is None:
        if lhs.dtype.is_fp8() and rhs.dtype.is_fp8():
            max_num_imprecise_acc = self.builder.options.max_num_imprecise_acc_default
        else:
            max_num_imprecise_acc = 0
    elif lhs.dtype.is_fp8() and rhs.dtype.is_fp8(
    ) and max_num_imprecise_acc > K:
        raise ValueError(
            f"max_num_imprecise_acc ({max_num_imprecise_acc}) must be <= K ({K})"
        )

    return (lhs, rhs, acc_handle, input_precision, max_num_imprecise_acc,
            ret_ty)


def _prepare_legacy_load(self, ptr, mask, other, boundary_check, padding):
    """``TritonSemantic.load``'s operand coercion, without the load op.

    Returns ``(dst_ty, ptr, mask, other, is_bool)`` so an async load can build
    its own op from operands Triton has already broadcast and cast. Mirrors
    ``TritonSemantic.load``; keep the two in step.
    """
    import triton.language as tl

    if not ptr.type.scalar.is_ptr():
        raise ValueError(
            f"Unsupported ptr type {ptr.type.__repr__()} in `tl.load`")

    if mask is None and other is not None:
        raise ValueError("`other` cannot be provided without `mask`")
    if padding or boundary_check:
        raise ValueError(
            "`padding_option` or `boundary_check` argument is not supported for loading "
            "a tensor of pointers or loading a scalar. Because the compiler does not "
            "know the boundary; please use block pointers (defined by `make_block_ptr`) "
            "instead")

    if not ptr.type.is_block():
        if mask and mask.type.is_block():
            raise ValueError(
                "Mask argument cannot be block type if pointer argument is not a block"
            )
        if other and other.type.is_block():
            raise ValueError(
                "Other argument cannot be block type if pointer argument is not a block"
            )

    if ptr.type.is_block():
        if mask is not None:
            ptr, mask = self.broadcast_impl_value(ptr, mask)
        if other is not None:
            ptr, other = self.broadcast_impl_value(ptr, other)

    ptr_ty = ptr.type.scalar
    elt_ty = ptr_ty.element_ty

    # Treat `pointer_type<tl.int1>` as `pointer_type<tl.int8>`
    is_bool = elt_ty == tl.int1
    if is_bool:
        elt_ty = tl.int8
        ptr_ty = tl.pointer_type(elt_ty, ptr_ty.address_space)
        ptr = self.cast(ptr, ptr_ty)

    if other is not None:
        other = self.cast(other, elt_ty)

    if not ptr.type.is_block():
        dst_ty = elt_ty
    elif hasattr(ptr.type, "with_element_ty"):
        dst_ty = ptr.type.with_element_ty(elt_ty)
    else:
        dst_ty = tl.block_type(elt_ty, ptr.type.get_block_shapes())

    return dst_ty, ptr, mask, other, is_bool
