"""TDM row-index operands must acquire GPU encodings during conversion."""

import pytest

import triton
import triton.language as tl
from conftest import tlx
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


@triton.jit
def _roundtrip(x, y, indices, PAD: tl.constexpr):
    layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for(
        [(64, PAD)], [32, 64], [1, 0])
    src = tlx.make_tensor_descriptor(None, x, [128, 64], [64, 1], [32, 64])
    dst = tlx.make_tensor_descriptor(None, y, [128, 64], [64, 1], [32, 64])
    buf = tlx.local_alloc((32, 64), x.dtype.element_ty, 1, layout=layout)
    rows = tl.load(indices + tl.arange(0, 32))
    tlx.tdm_async_gather(src, rows, tlx.local_view(buf, 0))
    tlx.tdm_async_wait(0)
    tlx.tdm_async_scatter(dst, rows, tlx.local_view(buf, 0))
    tlx.tdm_async_wait(0)


@pytest.mark.parametrize("dtype,pad", [("fp16", 8), ("fp8e4nv", 16)])
def test_tdm_row_indices_codegen(dtype, pad):
    compiled = triton.compile(
        ASTSource(_roundtrip,
                  signature={
                      "x": "*" + dtype,
                      "y": "*" + dtype,
                      "indices": "*i32",
                      "PAD": "constexpr"
                  },
                  constexprs={"PAD": pad}),
        target=GPUTarget("hip", "gfx1250", 32),
        options={"num_warps": 4},
    )
    assert "amdg.async_tdm_gather" in compiled.asm["ttgir"]
    assert "amdg.async_tdm_scatter" in compiled.asm["ttgir"]
    amdgcn = compiled.asm["amdgcn"]
    # 32 i32 rows over 4 warps: one 8-row TDM instruction per warp.
    assert amdgcn.count("tensor_load_to_lds") == 1
    assert amdgcn.count("tensor_store_from_lds") == 1
    assert compiled.asm["hsaco"]
