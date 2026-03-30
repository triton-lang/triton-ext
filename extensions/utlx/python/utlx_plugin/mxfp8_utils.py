"""MXFP8 conversion helpers."""

from __future__ import annotations

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx


@triton.jit
def _compute_scale_and_quantize(data_block, VEC_SIZE: tl.constexpr,
                                dtype: tl.constexpr):
    BLOCK_M: tl.constexpr = data_block.shape[0]
    BLOCK_K: tl.constexpr = data_block.shape[1]
    NUM_SCALES: tl.constexpr = BLOCK_K // VEC_SIZE

    if dtype == tl.float8e4nv:
        FLOAT_MAX: tl.constexpr = 448.0
    else:
        tl.static_assert(dtype == tl.float8e5)
        FLOAT_MAX = 57344.0

    data_reshaped = tl.reshape(data_block, [BLOCK_M, NUM_SCALES, VEC_SIZE])
    abs_data = tl.abs(data_reshaped)
    max_abs = tl.max(abs_data, axis=2)

    descale = max_abs / FLOAT_MAX
    descale_exponent = (descale.to(tl.uint32, bitcast=True) +
                        0x007FFFFF) & 0x7F800000
    descale_rounded = descale_exponent.to(tl.float32, bitcast=True)
    scale_e8m0 = (descale_exponent >> 23).to(tl.uint8)

    quant_scale = tl.where(descale_rounded == 0, 0.0, 1.0 / descale_rounded)
    quant_scale_expanded = tl.reshape(quant_scale, [BLOCK_M, NUM_SCALES, 1])
    scaled_data = data_reshaped * quant_scale_expanded
    scaled_data = tl.clamp(scaled_data, -FLOAT_MAX, FLOAT_MAX)
    data_scaled_flat = tl.reshape(scaled_data, [BLOCK_M, BLOCK_K])
    data_fp8 = data_scaled_flat.to(dtype)

    return scale_e8m0, data_fp8


@triton.jit
def _to_mxfp8_block(data_input, data_out_tile, scale_out_tile,
                    VEC_SIZE: tl.constexpr, dtype: tl.constexpr):
    """Convert float32 tensor to MXFP8 format and store results."""
    BLOCK_M: tl.constexpr = data_input.shape[0]
    BLOCK_K: tl.constexpr = data_input.shape[1]
    tl.static_assert(BLOCK_M == 128)
    tl.static_assert(BLOCK_K == 128)
    tl.static_assert(VEC_SIZE == 32)

    scale_e8m0, data_fp8 = _compute_scale_and_quantize(data_input, VEC_SIZE,
                                                       dtype)
    tlx.local_store(data_out_tile, data_fp8)
    tlx.local_store(scale_out_tile, scale_e8m0)
