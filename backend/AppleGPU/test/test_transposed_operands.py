"""Dot operands whose registers run down their rows, as a transposed tensor
loads: each lane's stores scatter down a column of the staged tile, at the
pitch that spreads them over the banks."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="the dispatch path is torch's")

if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
    pytest.skip("needs an Apple GPU", allow_module_level=True)

import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def mm(a_ptr, b_ptr, c_ptr, K, sam, sak, sbk, sbn, BM: tl.constexpr,
       BN: tl.constexpr, BK: tl.constexpr):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K, BK):
        a = tl.load(a_ptr + rm[:, None] * sam + (k0 + rk)[None, :] * sak)
        b = tl.load(b_ptr + (k0 + rk)[:, None] * sbk + rn[None, :] * sbn)
        acc += tl.dot(a, b, input_precision="ieee")
    tl.store(c_ptr + rm[:, None] * (tl.num_programs(1) * BN) + rn[None, :],
             acc.to(c_ptr.dtype.element_ty))


@triton.jit
def invariant_a(a_ptr, b_ptr, c_ptr, sam, sak, NT, BM: tl.constexpr,
                BN: tl.constexpr, BK: tl.constexpr):
    rm = tl.arange(0, BM)
    rn = tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a = tl.load(a_ptr + rm[:, None] * sam + rk[None, :] * sak)
    acc = tl.zeros((BM, BN), tl.float32)
    for j in range(NT):
        b = tl.load(b_ptr + j * BK * BN + rk[:, None] * BN + rn[None, :])
        acc = tl.dot(a, b, acc, input_precision="ieee")
    tl.store(c_ptr + rm[:, None] * BN + rn[None, :], acc)


def _layout(x, col_major):
    return x.t().contiguous().t() if col_major else x


@pytest.mark.parametrize("a_col,b_col", [(True, False), (False, True),
                                         (True, True)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("BM,BN,BK,num_warps", [(128, 64, 32, 8),
                                                (64, 64, 32, 4),
                                                (16, 16, 16, 1)])
def test_mm(a_col, b_col, dtype, BM, BN, BK, num_warps):
    M, N, K = 2 * BM, 2 * BN, 4 * BK
    a = _layout(torch.randn(M, K, device="mps", dtype=dtype), a_col)
    b = _layout(torch.randn(K, N, device="mps", dtype=dtype), b_col)
    c = torch.empty(M, N, device="mps", dtype=dtype)
    mm[(M // BM, N // BN)](a,
                           b,
                           c,
                           K,
                           *a.stride(),
                           *b.stride(),
                           BM=BM,
                           BN=BN,
                           BK=BK,
                           num_warps=num_warps)
    tol = 1e-4 if dtype == torch.float32 else 1e-2
    torch.testing.assert_close(c.float(),
                               a.float() @ b.float(),
                               rtol=tol,
                               atol=tol * K**0.5)


@pytest.mark.parametrize("BM,BN,BK,num_warps", [(64, 64, 32, 4),
                                                (32, 32, 32, 4)])
def test_loop_invariant_a(BM, BN, BK, num_warps):
    NT = 5
    a = _layout(torch.randn(BM, BK, device="mps"), True)
    b = torch.randn(NT, BK, BN, device="mps")
    c = torch.empty(BM, BN, device="mps")
    invariant_a[(1, )](a,
                       b,
                       c,
                       *a.stride(),
                       NT,
                       BM=BM,
                       BN=BN,
                       BK=BK,
                       num_warps=num_warps)
    torch.testing.assert_close(c, a @ b.sum(0), rtol=1e-4, atol=1e-3)
