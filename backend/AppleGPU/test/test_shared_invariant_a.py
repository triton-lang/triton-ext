"""Two loops whose dots share one loop-invariant A, the shape of flex
attention's masked and full passes over the same query tile."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="the dispatch path is torch's")

if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
    pytest.skip("needs an Apple GPU", allow_module_level=True)

import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def two_passes(q_ptr, k_ptr, out_ptr, N: tl.constexpr, SPLIT: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rd = tl.arange(0, D)
    q = tl.load(q_ptr + rm[:, None] * D + rd[None, :])
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(0, SPLIT, BN):
        rn = start + tl.arange(0, BN)
        kt = tl.load(k_ptr + rn[None, :] * D + rd[:, None])
        acc += tl.dot(q, kt, input_precision="ieee")
    for start in range(SPLIT, N, BN):
        rn = start + tl.arange(0, BN)
        kt = tl.load(k_ptr + rn[None, :] * D + rd[:, None])
        acc += 2.0 * tl.dot(q, kt, input_precision="ieee")
    tl.store(out_ptr + rm[:, None] * BN + tl.arange(0, BN)[None, :], acc)


@pytest.mark.parametrize("BM,BN,num_warps", [(32, 32, 4), (64, 32, 8)])
def test_two_loops_share_a(BM, BN, num_warps):
    M, N, D, SPLIT = 128, 256, 64, 96
    q = torch.randn(M, D, device="mps")
    k = torch.randn(N, D, device="mps")
    out = torch.empty(M, BN, device="mps")
    two_passes[(M // BM, )](q,
                            k,
                            out,
                            N=N,
                            SPLIT=SPLIT,
                            BM=BM,
                            BN=BN,
                            D=D,
                            num_warps=num_warps)
    tiles = (q @ k.T).view(M, N // BN, BN)
    ref = tiles[:, :SPLIT // BN].sum(1) + 2 * tiles[:, SPLIT // BN:].sum(1)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-3)
