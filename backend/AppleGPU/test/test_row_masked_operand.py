"""A loop-invariant dot operand loaded under a mask on its rows alone is read
where it lies each trip instead of restaged: each lane reads a row clamped
into the tensor and takes the load's fill past the bound."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="the dispatch path is torch's")

if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
    pytest.skip("needs an Apple GPU", allow_module_level=True)

import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def row_masked_attn(q_ptr, k_ptr, v_ptr, o_ptr, M, N: tl.constexpr,
                    D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                    FILL: tl.constexpr):
    # Q is loaded once under a row bound and read by every trip's first dot.
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rd = tl.arange(0, D)
    q = tl.load(q_ptr + rm[:, None] * D + rd[None, :],
                mask=rm[:, None] < M,
                other=FILL)
    acc = tl.zeros((BM, D), tl.float32)
    for n0 in range(0, N, BN):
        rn = n0 + tl.arange(0, BN)
        kt = tl.load(k_ptr + rd[:, None] + rn[None, :] * D)
        s = tl.dot(q, kt, input_precision="ieee")
        v = tl.load(v_ptr + rn[:, None] * D + rd[None, :])
        acc = tl.dot(s, v, acc, input_precision="ieee")
    tl.store(o_ptr + rm[:, None] * D + rd[None, :], acc)


@pytest.mark.parametrize("M", [0, 1, 31, 50, 64])
@pytest.mark.parametrize("fill", [0.0, 2.0])
def test_row_masked_q(M, fill):
    # Two programs whatever M is: some tiles start past the bound, and with
    # M = 0 no row of Q is in it at all.
    BM, BN, N, D = 32, 32, 128, 64
    rows = 2 * BM
    q = torch.randn(rows, D, device="mps")
    k = torch.randn(N, D, device="mps")
    v = torch.randn(N, D, device="mps")
    o = torch.empty(rows, D, device="mps")
    compiled = row_masked_attn[(rows // BM, )](q,
                                               k,
                                               v,
                                               o,
                                               M,
                                               N,
                                               D,
                                               BM,
                                               BN,
                                               fill,
                                               num_warps=4)
    assert "rowsA" in compiled.asm["msl"], "Q was restaged, not read in place"
    padded = torch.full((rows, D), fill, device="mps")
    padded[:M] = q[:M]
    torch.testing.assert_close(o, (padded @ k.T) @ v, rtol=1e-4, atol=1e-3)
