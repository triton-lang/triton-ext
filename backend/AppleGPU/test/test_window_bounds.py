"""A fused dot's store window proven from its mask, including a column window
that starts at zero under a compile-time limit."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="the dispatch path is torch's")

if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
    pytest.skip("needs an Apple GPU", allow_module_level=True)

import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def masked_fused_store(a_ptr, b_ptr, c_ptr, M, NC: tl.constexpr,
                       K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                       BK: tl.constexpr):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        a = tl.load(a_ptr + rm[:, None] * K + k0 + rk[None, :],
                    mask=rm[:, None] < M,
                    other=0.0)
        b = tl.load(b_ptr + (k0 + rk[:, None]) * BN + rn[None, :])
        acc = tl.dot(a, b, acc, input_precision="ieee")
    tl.store(c_ptr + rm[:, None] * BN + rn[None, :],
             acc,
             mask=(rm[:, None] < M) & (rn[None, :] < NC))


@pytest.mark.parametrize("nc", [64, 40])
def test_column_window_at_zero_under_a_constant_limit(nc):
    M, K, BM, BN, BK = 72, 64, 32, 64, 32
    a = torch.randn(M, K, device="mps")
    b = torch.randn(K, BN, device="mps")
    out = torch.zeros(M, BN, device="mps")
    masked_fused_store[(triton.cdiv(M, BM), )](a,
                                               b,
                                               out,
                                               M,
                                               NC=nc,
                                               K=K,
                                               BM=BM,
                                               BN=BN,
                                               BK=BK)
    ref = a @ b
    ref[:, nc:] = 0
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
