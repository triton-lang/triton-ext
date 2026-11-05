"""Reductions to a scalar straight from the sliced layout a first reduction
leaves, where every lane of a warp holds the same values."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="the dispatch path is torch's")

if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
    pytest.skip("needs an Apple GPU", allow_module_level=True)

import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def causal_row_sums(x_ptr, out_ptr, count_ptr, N: tl.constexpr,
                    BM: tl.constexpr, BN: tl.constexpr):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    acc = tl.zeros((BM, ), tl.float32)
    count = 0
    for start in range(0, N, BN):
        rn = start + tl.arange(0, BN)
        live = (rm[:, None] >= rn[None, :]).to(tl.int32)
        count += tl.sum(tl.sum(live, 1), 0)
        if tl.max(tl.max(live, 1), 0) > 0:
            x = tl.load(x_ptr + rm[:, None] * N + rn[None, :])
            acc += tl.sum(tl.where(live > 0, x, 0.0), 1)
    tl.store(out_ptr + rm, acc)
    tl.store(count_ptr + tl.program_id(0), count)


@pytest.mark.parametrize("BM,BN,num_warps", [(64, 32, 4), (32, 64, 4),
                                             (64, 64, 8)])
def test_tile_skip_and_count(BM, BN, num_warps):
    N = 256
    x = torch.randn(N, N, device="mps")
    out = torch.empty(N, device="mps")
    count = torch.empty(N // BM, dtype=torch.int32, device="mps")
    causal_row_sums[(N // BM, )](x,
                                 out,
                                 count,
                                 N=N,
                                 BM=BM,
                                 BN=BN,
                                 num_warps=num_warps)
    torch.testing.assert_close(out, x.tril().sum(1), rtol=1e-4, atol=1e-4)
    rows = torch.arange(N, device="mps").view(-1, BM)
    torch.testing.assert_close(count, (rows + 1).sum(1).to(torch.int32))
