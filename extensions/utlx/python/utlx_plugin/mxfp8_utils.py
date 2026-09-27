"""MXFP8 conversion helpers."""

from __future__ import annotations

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.language.extra.cuda.inline_ptx_lib import _mul_f32x2
from triton.language.extra.tlx.warp_ops import warp_redux


@triton.jit
def _fused_amax_to_e8m0(amax, max_norm_rcp):
    """Fused amax-to-E8M0 scale conversion in a single PTX asm block.

    Computes the E8M0 biased exponent (RCEIL of ``amax / max_norm``) and the
    power-of-two reciprocal quantization scale in one pass, replacing roughly
    eight separate Triton operations.

    Returns ``(e8m0_exp as uint32, inv_scale as float32)``; callers cast the
    exponent to uint8.
    """
    return tl.inline_asm_elementwise(
        """
        {
            .reg .f32 fae_scale, fae_zero;
            .reg .b16 fae_packed;
            .reg .u32 fae_exp, fae_inv_exp, fae_inv_bits;
            mul.f32 fae_scale, $2, $3;
            mov.b32 fae_zero, 0;
            cvt.rp.satfinite.ue8m0x2.f32 fae_packed, fae_zero, fae_scale;
            cvt.u32.u16 fae_exp, fae_packed;
            and.b32 fae_exp, fae_exp, 0xFF;
            mov.u32 $0, fae_exp;
            sub.u32 fae_inv_exp, 254, fae_exp;
            shl.b32 fae_inv_bits, fae_inv_exp, 23;
            mov.b32 $1, fae_inv_bits;
        }
        """,
        "=r,=f,f,f",
        [amax, max_norm_rcp],
        dtype=(tl.uint32, tl.float32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def _cvt_e4m3x4_f32(a):
    """Vectorized FP32 -> FP8 E4M3 conversion.

    Converts four float32 values to four packed FP8 values with two
    ``cvt.rn.satfinite.e4m3x2`` instructions, avoiding scalar conversions and
    PRMT byte permutes. The satfinite modifier saturates to +/-448 (the e4m3
    max), so no explicit clamp is needed.
    """
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b16 lo, hi;
            cvt.rn.satfinite.e4m3x2.f32 lo, $2, $1;
            cvt.rn.satfinite.e4m3x2.f32 hi, $4, $3;
            mov.b32 $0, {lo, hi};
        }
        """,
        "=r,f,f,f,f",
        [a],
        dtype=tl.float8e4nv,
        is_pure=True,
        pack=4,
    )


@triton.jit
def _cvt_e5m2x4_f32(a):
    """Vectorized FP32 -> FP8 E5M2 conversion. See :func:`_cvt_e4m3x4_f32`."""
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b16 lo, hi;
            cvt.rn.satfinite.e5m2x2.f32 lo, $2, $1;
            cvt.rn.satfinite.e5m2x2.f32 hi, $4, $3;
            mov.b32 $0, {lo, hi};
        }
        """,
        "=r,f,f,f,f",
        [a],
        dtype=tl.float8e5,
        is_pure=True,
        pack=4,
    )


@triton.jit
def _amax_to_e8m0_and_quantize(
    data_input,
    block_amax,
    VEC_SIZE: tl.constexpr,
    dtype: tl.constexpr,
):
    """Derive E8M0 scales from precomputed block amaxes and quantize to FP8.

    Skips the per-block ``max(abs(data))`` by taking amaxes computed from the
    raw QK values, which is valid because exp2 is monotonic:
    ``max(exp2(x)) == exp2(max(x))``.

    Returns ``(scale_e8m0 [BLOCK_M, NUM_SCALES], data_fp8 [BLOCK_M, BLOCK_K])``.
    """
    BLOCK_M: tl.constexpr = data_input.shape[0]
    BLOCK_K: tl.constexpr = data_input.shape[1]
    NUM_SCALES: tl.constexpr = BLOCK_K // VEC_SIZE

    if dtype == tl.float8e4nv:
        FLOAT_MAX: tl.constexpr = 448.0
    else:
        tl.static_assert(dtype == tl.float8e5)
        FLOAT_MAX: tl.constexpr = 57344.0

    scale_u32, quant_scale = _fused_amax_to_e8m0(block_amax, 1.0 / FLOAT_MAX)
    scale_e8m0 = scale_u32.to(tl.uint8)

    data_reshaped = tl.reshape(data_input, [BLOCK_M, NUM_SCALES, VEC_SIZE])
    quant_scale_expanded = tl.reshape(quant_scale, [BLOCK_M, NUM_SCALES, 1])
    scaled_data = _mul_f32x2(data_reshaped, quant_scale_expanded)

    if dtype == tl.float8e4nv:
        data_fp8 = _cvt_e4m3x4_f32(scaled_data)
    else:
        data_fp8 = _cvt_e5m2x4_f32(scaled_data)

    data_fp8_flat = tl.reshape(data_fp8, [BLOCK_M, BLOCK_K])
    return scale_e8m0, data_fp8_flat


@triton.jit
def _to_mxfp8_32x32_block(
    data_input,
    VEC_SIZE: tl.constexpr,
    dtype: tl.constexpr,
):
    """Convert float32 to MXFP8 with one scale factor per 32x32 block.

    Per-thread max over 32 k-values, then a warp-level redux across 32
    m-values, giving one scale per (32m x 32k) block implicitly replicated to
    every lane.
    """
    BLOCK_M: tl.constexpr = data_input.shape[0]
    BLOCK_K: tl.constexpr = data_input.shape[1]
    NUM_SCALES: tl.constexpr = BLOCK_K // VEC_SIZE
    tl.static_assert(VEC_SIZE == 32)
    tl.static_assert(BLOCK_M % VEC_SIZE == 0)

    if dtype == tl.float8e4nv:
        FLOAT_MAX: tl.constexpr = 448.0
    else:
        tl.static_assert(dtype == tl.float8e5)
        FLOAT_MAX: tl.constexpr = 57344.0

    data_reshaped = tl.reshape(data_input, [BLOCK_M, NUM_SCALES, VEC_SIZE])

    # Per-row amax: register-local max over 32 k-values.
    per_row_amax = tl.max(tl.abs(data_reshaped), axis=2)

    # Warp-level redux: max across 32 lanes (= 32 m-rows), result replicated.
    block_amax = warp_redux(per_row_amax, "max")

    scale_u32, quant_scale = _fused_amax_to_e8m0(block_amax, 1.0 / FLOAT_MAX)
    scale_e8m0 = scale_u32.to(tl.uint8)

    quant_scale_expanded = tl.reshape(quant_scale, [BLOCK_M, NUM_SCALES, 1])
    scaled_data = _mul_f32x2(data_reshaped, quant_scale_expanded)

    if dtype == tl.float8e4nv:
        data_fp8 = _cvt_e4m3x4_f32(scaled_data)
    else:
        data_fp8 = _cvt_e5m2x4_f32(scaled_data)

    data_fp8_flat = tl.reshape(data_fp8, [BLOCK_M, BLOCK_K])
    return data_fp8_flat, scale_e8m0


@triton.jit
def _to_mxfp8_block_with_block_amax(
    data_input,
    block_amax,
    VEC_SIZE: tl.constexpr,
    dtype: tl.constexpr,
):
    """Blockscaled variant of :func:`_to_mxfp8_block` taking precomputed amaxes.

    Returns the FP8 data and E8M0 scales; callers store them.
    """
    BLOCK_K: tl.constexpr = data_input.shape[1]
    tl.static_assert(BLOCK_K % VEC_SIZE == 0)
    tl.static_assert(VEC_SIZE == 32)

    scale_e8m0, data_fp8 = _amax_to_e8m0_and_quantize(data_input, block_amax,
                                                      VEC_SIZE, dtype)

    return data_fp8, scale_e8m0


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
