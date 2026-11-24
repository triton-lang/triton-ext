"""Gluon shared buffers: multi-buffered allocations indexed at run time, and
dots reading their operands from them."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="the dispatch path is torch's")

if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
    pytest.skip("needs an Apple GPU", allow_module_level=True)

from triton.experimental import gluon  # noqa: E402
from triton.experimental.gluon import language as gl  # noqa: E402


@gluon.jit
def ring_copy(x_ptr, y_ptr, T, NB: gl.constexpr, R: gl.constexpr,
              C: gl.constexpr):
    # Tile t goes through slot t % NB of an NB-deep ring on its way out.
    ld: gl.constexpr = gl.BlockedLayout([1, 4], [8, 4], [4, 1], [1, 0])
    sl: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    ring = gl.allocate_shared_memory(gl.float32, [NB, R, C], sl)
    rows = gl.arange(0, R, layout=gl.SliceLayout(1, ld))
    cols = gl.arange(0, C, layout=gl.SliceLayout(0, ld))
    offs = rows[:, None] * C + cols[None, :]
    for t in range(T):
        slot = ring.index(t % NB)
        slot.store(gl.load(x_ptr + t * R * C + offs))
        gl.barrier()
        gl.store(y_ptr + t * R * C + offs, slot.load(ld))
        gl.barrier()


@pytest.mark.parametrize("NB", [2, 3])
def test_ring_copy(NB):
    T, R, C = 5, 32, 16
    x = torch.randn(T, R, C, device="mps")
    y = torch.empty_like(x)
    ring_copy[(1, )](x, y, T, NB, R, C, num_warps=4)
    torch.testing.assert_close(y, x)


@gluon.jit
def padded_half(x_ptr, y_ptr, R: gl.constexpr, C: gl.constexpr,
                INTERVAL: gl.constexpr, PAD: gl.constexpr):
    # The lower half of a padded buffer, read through a subslice: its rows
    # cross padding intervals the subslice does not start on.
    ld: gl.constexpr = gl.BlockedLayout([1, 4], [8, 4], [4, 1], [1, 0])
    sl: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[INTERVAL, PAD]], [R, C], [1, 0])
    buf = gl.allocate_shared_memory(gl.float32, [R, C], sl)
    rows = gl.arange(0, R, layout=gl.SliceLayout(1, ld))
    cols = gl.arange(0, C, layout=gl.SliceLayout(0, ld))
    buf.store(gl.load(x_ptr + rows[:, None] * C + cols[None, :]))
    gl.barrier()
    half = buf.slice(R // 2, R // 2, dim=0)
    hrows = gl.arange(0, R // 2, layout=gl.SliceLayout(1, ld))
    gl.store(y_ptr + hrows[:, None] * C + cols[None, :], half.load(ld))


@pytest.mark.parametrize("interval", [16, 32, 64])
def test_padded_subslice(interval):
    R, C = 16, 16
    x = torch.randn(R, C, device="mps")
    y = torch.empty(R // 2, C, device="mps")
    padded_half[(1, )](x, y, R, C, interval, 4, num_warps=4)
    torch.testing.assert_close(y, x[R // 2:])


@gluon.jit
def gemm_ring(a_ptr, b_ptr, c_ptr, M, N, K, BM: gl.constexpr, BN: gl.constexpr,
              BK: gl.constexpr, WARPS: gl.constexpr):
    # A and B stage through two-deep rings: step k reads slot k % 2 while the
    # next K block is loaded into registers, then stored into the other.
    ld_a: gl.constexpr = gl.BlockedLayout([1, 4], [8, 4], [WARPS, 1], [1, 0])
    ld_b: gl.constexpr = gl.BlockedLayout([1, 4], [2, 16], [WARPS, 1], [1, 0])
    acc_l: gl.constexpr = gl.BlockedLayout([2, 4], [8, 4], [WARPS, 1], [1, 0])
    a_op: gl.constexpr = gl.DotOperandLayout(0, acc_l, 0)
    b_op: gl.constexpr = gl.DotOperandLayout(1, acc_l, 0)
    sl: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])

    rm = gl.program_id(0) * BM + gl.arange(
        0, BM, layout=gl.SliceLayout(1, ld_a))
    rka = gl.arange(0, BK, layout=gl.SliceLayout(0, ld_a))
    rkb = gl.arange(0, BK, layout=gl.SliceLayout(1, ld_b))
    rn = gl.program_id(1) * BN + gl.arange(
        0, BN, layout=gl.SliceLayout(0, ld_b))
    a_ptrs = a_ptr + rm[:, None] * K + rka[None, :]
    b_ptrs = b_ptr + rkb[:, None] * N + rn[None, :]

    a_ring = gl.allocate_shared_memory(gl.float32, [2, BM, BK], sl)
    b_ring = gl.allocate_shared_memory(gl.float32, [2, BK, BN], sl)
    a_ring.index(0).store(gl.load(a_ptrs))
    b_ring.index(0).store(gl.load(b_ptrs))
    acc = gl.zeros([BM, BN], gl.float32, layout=acc_l)
    steps = K // BK
    for k in range(steps):
        a_ptrs += BK
        b_ptrs += BK * N
        more = k + 1 < steps
        a = gl.load(a_ptrs, mask=more, other=0.0)
        b = gl.load(b_ptrs, mask=more, other=0.0)
        gl.barrier()
        acc = gl.dot_fma(
            a_ring.index(k % 2).load(a_op),
            b_ring.index(k % 2).load(b_op), acc)
        a_ring.index((k + 1) % 2).store(a)
        b_ring.index((k + 1) % 2).store(b)

    rmo = gl.program_id(0) * BM + gl.arange(
        0, BM, layout=gl.SliceLayout(1, acc_l))
    rno = gl.program_id(1) * BN + gl.arange(
        0, BN, layout=gl.SliceLayout(0, acc_l))
    gl.store(c_ptr + rmo[:, None] * N + rno[None, :], acc)


@pytest.mark.parametrize("tile", [(32, 32, 16, 4), (64, 64, 16, 4),
                                  (64, 64, 32, 8)])
def test_gemm_ring(tile):
    BM, BN, BK, W = tile
    M, N, K = 128, 128, 256
    a = torch.randn(M, K, device="mps")
    b = torch.randn(K, N, device="mps")
    c = torch.empty(M, N, device="mps")
    gemm_ring[(M // BM, N // BN)](a, b, c, M, N, K, BM, BN, BK, W, num_warps=W)
    torch.testing.assert_close(c, a @ b, rtol=1e-4, atol=1e-3)
