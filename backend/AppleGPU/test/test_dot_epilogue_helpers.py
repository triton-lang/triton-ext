"""A min/max folded into a dot's epilogue gets its prelude helper for the type
the epilogue evaluates in, including C's stored type past a rounding."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="the dispatch path is torch's")

if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
    pytest.skip("needs an Apple GPU", allow_module_level=True)

import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def mm_bias_relu(a, b, bias, c, M: tl.constexpr, N: tl.constexpr,
                 K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                 BK: tl.constexpr):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K // BK):
        x = tl.load(a + rm[:, None] * K + (k * BK + rk)[None, :])
        w = tl.load(b + (k * BK + rk)[:, None] * N + rn[None, :])
        acc += tl.dot(x, w)
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    shift = tl.load(bias + tl.broadcast_to(rn[None, :], [BM, BN]), mask)
    # Inductor rounds to the stored type after every op, as here.
    t = c.dtype.element_ty
    biased = (shift + acc).to(t).to(tl.float32)
    out = tl.maximum(tl.full([1], 0, tl.int32), biased, tl.PropagateNan.ALL)
    tl.store(c + rm[:, None] * N + rn[None, :], out.to(t).to(tl.float32), mask)


@pytest.mark.parametrize("dtype",
                         [torch.bfloat16, torch.float16, torch.float32])
def test_epilogue_relu_compiles_and_matches(dtype):
    M, N, K, BM, BN, BK = 64, 128, 64, 32, 64, 16
    torch.manual_seed(0)
    a = torch.randn(M, K, device="mps", dtype=dtype)
    b = torch.randn(K, N, device="mps", dtype=dtype)
    bias = torch.randn(N, device="mps", dtype=dtype)
    c = torch.empty(M, N, device="mps", dtype=dtype)
    mm_bias_relu[(M // BM, N // BN)](a, b, bias, c, M, N, K, BM, BN, BK)
    ref = torch.relu(a.float() @ b.float() + bias.float()).to(dtype)
    torch.testing.assert_close(c, ref, rtol=2e-2, atol=2e-2)
