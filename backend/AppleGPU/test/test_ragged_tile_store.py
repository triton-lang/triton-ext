"""The last tile of a ragged M stores its own rows. Shifting its store onto
the last full window is only right when nothing but the row start tells the
tiles apart; here the program id also sets how much of K a tile sums."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="the dispatch path is torch's")

if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
    pytest.skip("needs an Apple GPU", allow_module_level=True)

import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def ragged_tile_store(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr,
                      K: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    rm = pid * BM + tl.arange(0, BM)
    rn = tl.arange(0, N)
    acc = tl.zeros((BM, N), tl.float32)
    for k0 in range(0, (pid + 1) * BK, BK):
        rk = k0 + tl.arange(0, BK)
        a = tl.load(a_ptr + rm[:, None] * K + rk[None, :],
                    mask=rm[:, None] < M,
                    other=0.0)
        b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
        acc += tl.dot(a, b, input_precision="ieee")
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc, mask=rm[:, None] < M)


@pytest.mark.parametrize("M", [45, 50, 63])
def test_ragged_rows_keep_their_tile(M):
    BM, BK, N = 16, 16, 32
    tiles = triton.cdiv(M, BM)
    K = tiles * BK
    a = torch.randn(M, K, device="mps")
    b = torch.randn(K, N, device="mps")
    c = torch.zeros(M, N, device="mps")
    want = torch.empty(M, N)
    for t in range(tiles):
        rows = slice(t * BM, min((t + 1) * BM, M))
        want[rows] = a[rows, :(t + 1) * BK].cpu() @ b[:(t + 1) * BK].cpu()
    for _ in range(3):
        ragged_tile_store[(tiles, )](a, b, c, M, N, K, BM, BK, num_warps=4)
        torch.testing.assert_close(c.cpu(), want, rtol=1e-4, atol=1e-4)
