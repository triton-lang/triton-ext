"""uTLX's tcgen5 ``async_dot_scaled`` test for mxfp4 (e2m1) operands.

Kept from the former test_tlx.py, which otherwise duplicated fbtriton's
``python/test/unit/language/test_tlx_*.py`` tests: fbtriton has no mxfp4
counterpart for this one.
"""
import math

import pytest
import torch
import triton
import triton.language as tl
from conftest import tlx, is_blackwell
from triton.tools.tensor_descriptor import TensorDescriptor


def _swizzle_scale_to_5d(scale, outer_chunks, k_chunks):
    """Convert raw E8M0 scales to swizzled 5D format for TMA/async_dot_scaled.

    Applies the cuBLAS block scaling layout within each 128x4 block.
    dest[row%32 * 16 + row//32 * 4 + col] = src[row, col]

    Args:
        scale: Raw scale tensor of shape (batch, rows, K//32) in uint8.
        outer_chunks: Number of 128-row chunks (rows // 128).
        k_chunks: Number of 4-column chunks (K // 32 // 4).

    Returns:
        Swizzled 5D tensor of shape (batch, outer_chunks, k_chunks, 2, 256).
    """
    batch = scale.shape[0]
    cols = scale.shape[2]
    padded_cols = k_chunks * 4

    if cols < padded_cols:
        scale = torch.nn.functional.pad(scale, (0, padded_cols - cols))

    blocks = (scale.reshape(batch, outer_chunks, 128, k_chunks,
                            4).permute(0, 1, 3, 2,
                                       4).reshape(batch, outer_chunks,
                                                  k_chunks, 512))

    _r = torch.arange(128)
    _c = torch.arange(4)
    _rg, _cg = torch.meshgrid(_r, _c, indexing="ij")
    idx = ((_rg % 32) * 16 + (_rg // 32) * 4 + _cg).reshape(-1)
    idx = idx.to(scale.device).expand_as(blocks)
    output = torch.empty_like(blocks)
    output.scatter_(-1, idx, blocks)

    return output.reshape(batch, outer_chunks, k_chunks, 2, 256)


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
def test_async_dot_scaled_mxfp4(device):
    """
    Test D = (A * A_scale) * (B * B_scale) with mxfp4 (e2m1) format for both A and B.

    For mxfp4 format:
    - Two fp4 (e2m1) elements are packed into a single uint8
    - A has logical shape (M, K), packed along K to get physical shape (M, K//2)
    - B is stored in transposed layout (N, K), packed along K to get (N, K//2)
    - B is transposed in SMEM before being passed to MMA to get (K//2, N)

    Scale layout uses 5D TMA descriptor [1, rep_m, rep_k, 2, 256] with uint8 elements,
    matching cuBLAS block scaling layout.
    """
    from triton.tools.mxfp import MXFP4Tensor

    VEC_SIZE = 32  # mxfp4 uses 32 elements per scale factor

    @triton.jit
    def tcgen5_dot_scaled_mxfp4_kernel(
        a_desc,
        a_scale_desc,
        b_desc,
        b_scale_desc,
        c_desc,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        # Scale tile dimensions for 5D TMA (per cuBLAS block scaling layout)
        REP_M: tl.constexpr = triton.cdiv(BLOCK_M, 128)
        REP_N: tl.constexpr = triton.cdiv(BLOCK_N, 128)
        REP_K: tl.constexpr = triton.cdiv(BLOCK_K, 128)

        # Allocate SMEM buffers
        # A: (M, K//2) - packed along K
        # B: (N, K//2) - stored in transposed layout, packed along K
        a_tile = tlx.local_alloc((BLOCK_M, BLOCK_K // 2), tl.uint8,
                                 tl.constexpr(1))
        b_tile = tlx.local_alloc((BLOCK_N, BLOCK_K // 2), tl.uint8,
                                 tl.constexpr(1))
        # 5D scale buffers: [1, REP_M/N, REP_K, 2, 256] for cuBLAS block scaling layout
        a_scale_tile = tlx.local_alloc((1, REP_M, REP_K, 2, 256), tl.uint8,
                                       tl.constexpr(1))
        b_scale_tile = tlx.local_alloc((1, REP_N, REP_K, 2, 256), tl.uint8,
                                       tl.constexpr(1))

        load_bar = tlx.alloc_barriers(tl.constexpr(1))
        DATA_BYTES: tl.constexpr = BLOCK_M * BLOCK_K // 2 + BLOCK_N * BLOCK_K // 2
        SCALE_BYTES: tl.constexpr = (REP_M + REP_N) * REP_K * 2 * 256
        tlx.barrier_expect_bytes(load_bar[0], DATA_BYTES + SCALE_BYTES)
        tlx.async_descriptor_load(a_desc, a_tile[0], [0, 0], load_bar)
        tlx.async_descriptor_load(b_desc, b_tile[0], [0, 0], load_bar)
        # 5D offset with leading 0
        tlx.async_descriptor_load(a_scale_desc, a_scale_tile[0],
                                  [0, 0, 0, 0, 0], load_bar)
        tlx.async_descriptor_load(b_scale_desc, b_scale_tile[0],
                                  [0, 0, 0, 0, 0], load_bar)
        tlx.barrier_wait(load_bar[0], 0)

        # Transpose B from (N, K//2) to (K//2, N) for MMA
        b_tile_T = tlx.local_trans(b_tile[0])

        c_tile = tlx.local_alloc((BLOCK_M, BLOCK_N), tl.float32,
                                 tl.constexpr(1), tlx.storage_kind.tmem)
        tlx.async_dot_scaled(a_tile[0],
                             b_tile_T,
                             c_tile[0],
                             a_scale_tile[0],
                             "e2m1",
                             b_scale_tile[0],
                             "e2m1",
                             use_acc=False)

        result = tlx.local_load(c_tile[0])
        c = result.to(tlx.dtype_of(c_desc))
        c_desc.store([0, 0], c)

    torch.manual_seed(0)
    M, N, K = (128, 128, 128)
    BLOCK_M, BLOCK_N, BLOCK_K = (M, N, K)

    # Create mxfp4 tensors and pack them
    # A has logical shape (M, K), packed along K to get physical shape (M, K//2)

    A = torch.full((M, K), 2, dtype=torch.float32, device=device)
    B = torch.full((N, K), 2, dtype=torch.float32, device=device)
    AMXFP4 = MXFP4Tensor(data=A, device=device)
    BMXFP4 = MXFP4Tensor(data=B, device=device)
    APACKED = AMXFP4.to_packed_tensor(dim=1)
    BPACKED = BMXFP4.to_packed_tensor(dim=1)

    a_ref = AMXFP4.to(torch.float32)

    # B is stored in transposed layout (N, K), packed along K to get (N, K//2)
    # This matches the hardware expectation for mxfp4
    b_ref = BMXFP4.to(
        torch.float32).T  # Transpose for reference matmul -> (K, N)

    c = torch.zeros((M, N), device=device, dtype=torch.float16)

    # TMA descriptors for packed mxfp4 data
    a_desc = TensorDescriptor.from_tensor(APACKED, [BLOCK_M, BLOCK_K // 2])
    b_desc = TensorDescriptor.from_tensor(
        BPACKED, [BLOCK_N, BLOCK_K // 2])  # B stored as (N, K//2)
    c_desc = TensorDescriptor.from_tensor(c, block_shape=[BLOCK_M, BLOCK_N])

    # Create E8M0 scale tensors using 5D TMA layout: [1, rep_m, rep_k, 2, 256]
    # This matches cuBLAS block scaling layout used by tcgen5_mma_scaled
    a_scale = torch.randint(127,
                            128, (M, K // VEC_SIZE),
                            dtype=torch.uint8,
                            device=device)
    b_scale = torch.randint(127,
                            128, (N, K // VEC_SIZE),
                            dtype=torch.uint8,
                            device=device)

    # Swizzle to 5D cuBLAS block scaling layout for TMA: [1, rep_m, rep_k, 2, 256]
    a_scale_5d = _swizzle_scale_to_5d(a_scale.reshape(1, M, K // VEC_SIZE),
                                      M // 128, K // VEC_SIZE // 4)
    b_scale_5d = _swizzle_scale_to_5d(b_scale.reshape(1, N, K // VEC_SIZE),
                                      N // 128, K // VEC_SIZE // 4)

    a_scale_block_shape = [1, BLOCK_M // 128, BLOCK_K // 32 // 4, 2, 2 * 128]
    b_scale_block_shape = [1, BLOCK_N // 128, BLOCK_K // 32 // 4, 2, 2 * 128]
    a_scale_desc = TensorDescriptor.from_tensor(
        a_scale_5d, block_shape=a_scale_block_shape)
    b_scale_desc = TensorDescriptor.from_tensor(
        b_scale_5d, block_shape=b_scale_block_shape)

    kern_kwargs = {"BLOCK_M": BLOCK_M, "BLOCK_K": BLOCK_K, "BLOCK_N": BLOCK_N}
    kernel = tcgen5_dot_scaled_mxfp4_kernel[(1, 1)](
        a_desc,
        a_scale_desc,
        b_desc,
        b_scale_desc,
        c_desc,
        **kern_kwargs,
    )

    ttgir = kernel.asm["ttgir"]
    assert ttgir.count("ttng.async_tma_copy_global_to_local") == 4
    assert ttgir.count("ttng.tc_gen5_mma_scaled") == 1

    # Converts E8M0 format scale values to float32 by bit-shifting the exponent bits
    # into the correct position for IEEE 754 float32 representation
    def fp8e8m0_to_float32(scale):
        scale = scale.view(torch.uint8)
        scale = scale.to(torch.int32)
        scale = scale << 23
        scale = scale.view(torch.float32)
        return scale

    # Compute reference (use original 2D scales, not swizzled 5D)
    a_scale_f32 = fp8e8m0_to_float32(a_scale)
    b_scale_f32 = fp8e8m0_to_float32(b_scale)
    # Repeat each scale value VEC_SIZE times along dim 1
    a_scale_f32 = a_scale_f32.repeat_interleave(VEC_SIZE, dim=1)[:M, :K]
    b_scale_f32 = b_scale_f32.repeat_interleave(VEC_SIZE,
                                                dim=1).T.contiguous()[:K, :N]
    ref_out = torch.matmul(a_ref * a_scale_f32,
                           b_ref * b_scale_f32).to(torch.float16)
    atol = 1e-2 * math.sqrt(K / 32)
    torch.testing.assert_close(ref_out, c, atol=atol, rtol=0)
