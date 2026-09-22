"""uTLX warp-level operations."""

import triton
import triton.language as tl_module
import triton.language.core as tl


@triton.jit
def _warp_redux_max(x):
    return tl_module.inline_asm_elementwise(
        "{ .reg .b32 t; mov.b32 t, $1; redux.sync.max.f32 t, t, 0xFFFFFFFF; mov.b32 $0, t; }",
        "=f,f",
        [x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _warp_redux_min(x):
    return tl_module.inline_asm_elementwise(
        "{ .reg .b32 t; mov.b32 t, $1; redux.sync.min.f32 t, t, 0xFFFFFFFF; mov.b32 $0, t; }",
        "=f,f",
        [x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _warp_redux_abs_max(x):
    return tl_module.inline_asm_elementwise(
        "{ .reg .b32 t; mov.b32 t, $1; and.b32 t, t, 0x7FFFFFFF; redux.sync.max.f32 t, t, 0xFFFFFFFF; mov.b32 $0, t; }",
        "=f,f",
        [x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _warp_redux_abs_min(x):
    return tl_module.inline_asm_elementwise(
        "{ .reg .b32 t; mov.b32 t, $1; and.b32 t, t, 0x7FFFFFFF; redux.sync.min.f32 t, t, 0xFFFFFFFF; mov.b32 $0, t; }",
        "=f,f",
        [x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def warp_redux(x, op: tl.constexpr):
    """Warp-level reduction with implicit broadcast.

    Each element is replaced with the reduction result across all 32 lanes in
    its warp, replicated back to every lane -- no explicit broadcast needed.

    ``redux.sync.{max,min}.f32`` requires sm_100 (Blackwell).

    Args:
        x: f32 tensor
        op: "max", "min", "abs_max", or "abs_min"
    """
    if op == "abs_max":
        return _warp_redux_abs_max(x)
    elif op == "abs_min":
        return _warp_redux_abs_min(x)
    elif op == "max":
        return _warp_redux_max(x)
    elif op == "min":
        return _warp_redux_min(x)


@tl.builtin
def vote_ballot_sync(
    mask: tl.constexpr,
    pred: tl.tensor,
    _semantic=None,
) -> tl.tensor:
    """Perform a warp-level vote ballot operation."""
    if pred.dtype != tl.int1:
        pred = pred != 0

    if isinstance(mask, tl.constexpr):
        mask_val = mask.value
    else:
        mask_val = mask

    mask_handle = _semantic.builder.get_int32(mask_val)
    result = _semantic.builder.utlx_vote_ballot_sync(
        [mask_handle, pred.handle])

    if pred.type.is_block():
        shape = [s.value if hasattr(s, "value") else s for s in pred.shape]
        ret_ty = tl.block_type(tl.int32, shape)
        return tl.tensor(result, ret_ty)
    else:
        return tl.tensor(result, tl.int32)
