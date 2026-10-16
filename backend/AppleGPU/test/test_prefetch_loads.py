"""K loops pipelined by num_stages: operand loads run ahead into registers
the loop carries, the first ones before the loop, and the last iterations,
with nothing left to load, in a second loop."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="the dispatch path is torch's")

if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
    pytest.skip("needs an Apple GPU", allow_module_level=True)

import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def mm(a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr,
       BK: tl.constexpr):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a_ptrs = a_ptr + rm[:, None] * K + rk[None, :]
    b_ptrs = b_ptr + rk[:, None] * N + rn[None, :]
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K, BK):
        in_k = k0 + rk < K
        a = tl.load(a_ptrs, mask=in_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=in_k[:, None], other=0.0)
        acc += tl.dot(a, b, input_precision="ieee")
        a_ptrs += BK
        b_ptrs += BK * N
    tl.store(c_ptr + rm[:, None] * N + rn[None, :],
             acc.to(c_ptr.dtype.element_ty))


@pytest.mark.parametrize("num_stages", [1, 2, 3])
@pytest.mark.parametrize("K", [0, 16, 32, 48, 200, 1024])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_mm(num_stages, K, dtype):
    M, N, BM, BN, BK = 64, 64, 32, 32, 16
    a = torch.randn(M, K, dtype=dtype)
    b = torch.randn(K, N, dtype=dtype)
    c = torch.empty(M, N, device="mps", dtype=dtype)
    mm[(M // BM, N // BN)](a.to("mps"),
                           b.to("mps"),
                           c,
                           M,
                           N,
                           K,
                           BM=BM,
                           BN=BN,
                           BK=BK,
                           num_warps=4,
                           num_stages=num_stages)
    tol = 1e-4 if dtype == torch.float32 else 1e-2
    torch.testing.assert_close(c.cpu().float(),
                               a.float() @ b.float(),
                               rtol=tol,
                               atol=tol * max(K, 1)**0.5)


def test_stages_past_the_cap_share_one_compile():
    M = N = K = 64
    a = torch.randn(M, K, device="mps")
    b = torch.randn(K, N, device="mps")
    c = torch.empty(M, N, device="mps")
    hashes = {
        ns: mm[(2, 2)](a, b, c, M, N, K, BM=32, BN=32, BK=16,
                       num_stages=ns).hash
        for ns in (1, 2, 3, 5)
    }
    assert hashes[2] == hashes[3] == hashes[5]
    assert hashes[1] != hashes[2]


@pytest.mark.parametrize("num_warps", [4, 8])
def test_mm_tile_too_big_to_keep_c(num_warps):
    # The first loop's C does not fit beside A and B, so its dot walks panels
    # and keeps no fragments; the split-off last iteration must not expect them.
    M, N, K, BM, BN, BK = 128, 128, 256, 128, 128, 64
    a = torch.randn(M, K, dtype=torch.float16)
    b = torch.randn(K, N, dtype=torch.float16)
    c = torch.empty(M, N, device="mps", dtype=torch.float32)
    mm[(1, 1)](a.to("mps"),
               b.to("mps"),
               c,
               M,
               N,
               K,
               BM=BM,
               BN=BN,
               BK=BK,
               num_warps=num_warps,
               num_stages=3)
    torch.testing.assert_close(c.cpu(),
                               a.float() @ b.float(),
                               rtol=1e-2,
                               atol=1e-1)


@triton.jit
def mm_rowsum(a_ptr, b_ptr, c_ptr, s_ptr, M, N, K, BM: tl.constexpr,
              BN: tl.constexpr, BK: tl.constexpr):
    # A feeds the dot and a row sum, so the prefetched tile has two readers.
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a_ptrs = a_ptr + rm[:, None] * K + rk[None, :]
    b_ptrs = b_ptr + rk[:, None] * N + rn[None, :]
    acc = tl.zeros((BM, BN), tl.float32)
    rows = tl.zeros((BM, ), tl.float32)
    for _ in range(0, K, BK):
        a = tl.load(a_ptrs)
        acc += tl.dot(a, tl.load(b_ptrs), input_precision="ieee")
        rows += tl.sum(a, axis=1)
        a_ptrs += BK
        b_ptrs += BK * N
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc)
    tl.store(s_ptr + rm, rows)


@pytest.mark.parametrize("num_stages", [1, 2])
@pytest.mark.parametrize("K", [16, 64, 512])
def test_load_feeds_dot_and_reduction(num_stages, K):
    M, N, BM, BK = 64, 32, 32, 16
    a = torch.randn(M, K)
    b = torch.randn(K, N)
    c = torch.empty(M, N, device="mps")
    s = torch.empty(M, device="mps")
    mm_rowsum[(M // BM, )](a.to("mps"),
                           b.to("mps"),
                           c,
                           s,
                           M,
                           N,
                           K,
                           BM=BM,
                           BN=N,
                           BK=BK,
                           num_warps=4,
                           num_stages=num_stages)
    torch.testing.assert_close(c.cpu(), a @ b, rtol=1e-4, atol=1e-3)
    torch.testing.assert_close(s.cpu(), a.sum(1), rtol=1e-4, atol=1e-3)


@triton.jit
def persistent_mm(a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr,
                  BN: tl.constexpr, BK: tl.constexpr, PROGS: tl.constexpr):
    # Flattened into one loop whose scf.ifs start and finish each tile.
    tiles_n = tl.cdiv(N, BN)
    for tile in tl.range(tl.program_id(0),
                         tl.cdiv(M, BM) * tiles_n,
                         PROGS,
                         flatten=True):
        rm = (tile // tiles_n) * BM + tl.arange(0, BM)
        rn = (tile % tiles_n) * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        a_ptrs = a_ptr + rm[:, None] * K + rk[None, :]
        b_ptrs = b_ptr + rk[:, None] * N + rn[None, :]
        acc = tl.zeros((BM, BN), tl.float32)
        for k0 in range(0, K, BK):
            in_k = k0 + rk < K
            a = tl.load(a_ptrs, mask=in_k[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=in_k[:, None], other=0.0)
            acc += tl.dot(a, b, input_precision="ieee")
            a_ptrs += BK
            b_ptrs += BK * N
        tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc)


@pytest.mark.parametrize("num_stages", [1, 2, 3])
@pytest.mark.parametrize("K", [16, 48, 200])
@pytest.mark.parametrize("progs", [1, 4])
def test_persistent_mm(num_stages, K, progs):
    M, N, BM, BN, BK = 96, 64, 32, 32, 16
    a = torch.randn(M, K)
    b = torch.randn(K, N)
    c = torch.empty(M, N, device="mps")
    persistent_mm[(progs, )](a.to("mps"),
                             b.to("mps"),
                             c,
                             M,
                             N,
                             K,
                             BM=BM,
                             BN=BN,
                             BK=BK,
                             PROGS=progs,
                             num_warps=4,
                             num_stages=num_stages)
    torch.testing.assert_close(c.cpu(), a @ b, rtol=1e-4, atol=1e-3)


# A static first loop's last dot is peeled to straight-line code, so the
# second loop starts from that dot's result rather than a loop's fragments.
@triton.jit
def two_loops(a_ptr, b_ptr, c_ptr, n2, K, N1: tl.constexpr, BM: tl.constexpr,
              BN: tl.constexpr, BK: tl.constexpr):
    rm = tl.arange(0, BM)
    rn = tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for i in range(N1):
        a = tl.load(a_ptr + rm[:, None] * K + (i * BK + rk)[None, :])
        b = tl.load(b_ptr + (i * BK + rk)[:, None] * BN + rn[None, :])
        acc += tl.dot(a, b, input_precision="ieee")
    for j in range(n2):
        k0 = (N1 + j) * BK
        a = tl.load(a_ptr + rm[:, None] * K + (k0 + rk)[None, :])
        b = tl.load(b_ptr + (k0 + rk)[:, None] * BN + rn[None, :])
        acc += tl.dot(a, b, input_precision="ieee")
    tl.store(c_ptr + rm[:, None] * BN + rn[None, :], acc)


@pytest.mark.parametrize("num_stages", [1, 2, 3])
@pytest.mark.parametrize("n1", [1, 8])
@pytest.mark.parametrize("n2", [0, 2, 5])
def test_second_loop_continues_a_peeled_dot(num_stages, n1, n2):
    BM, BN, BK = 32, 32, 16
    K = (n1 + n2) * BK
    a = torch.randn(BM, K)
    b = torch.randn(K, BN)
    c = torch.empty(BM, BN, device="mps")
    two_loops[(1, )](a.to("mps"),
                     b.to("mps"),
                     c,
                     n2,
                     K,
                     N1=n1,
                     BM=BM,
                     BN=BN,
                     BK=BK,
                     num_warps=4,
                     num_stages=num_stages)
    torch.testing.assert_close(c.cpu(), a @ b, rtol=1e-4, atol=1e-3)
