"""uTLX warp-level operations."""

import triton.language.core as tl


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
