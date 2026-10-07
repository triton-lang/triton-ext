"""2D tiles: indexing one with `tl.arange` and `None` moves no data.

`expand_dims` and `broadcast` only say which register holds which coordinate,
so these run without threadgroup memory.
"""

import pytest

torch = pytest.importorskip("torch", reason="needs torch")

if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
    pytest.skip("needs an Apple GPU", allow_module_level=True)

import triton  # noqa: E402
import triton.language as tl  # noqa: E402

DEVICE = torch.device("mps")


@triton.jit
def add_2d(x_ptr, y_ptr, o_ptr, M, N, BLOCK_M: tl.constexpr,
           BLOCK_N: tl.constexpr):
    offs_m = tl.program_id(axis=0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.program_id(axis=1) * BLOCK_N + tl.arange(0, BLOCK_N)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(o_ptr + offs,
             tl.load(x_ptr + offs, mask=mask) +
             tl.load(y_ptr + offs, mask=mask),
             mask=mask)


@triton.jit
def row_times_col(a_ptr, b_ptr, o_ptr, N: tl.constexpr):
    r = tl.arange(0, N)
    a = tl.load(a_ptr + r)[:, None]
    b = tl.load(b_ptr + r)[None, :]
    tl.store(o_ptr + r[:, None] * N + r[None, :], a * b)


@pytest.mark.parametrize("shape,block", [((64, 64), (32, 32)),
                                         ((128, 32), (32, 32)),
                                         ((40, 72), (16, 32))])
def test_elementwise_2d(shape, block):
    M, N = shape
    torch.manual_seed(0)
    x = torch.randn(shape, device=DEVICE)
    y = torch.randn(shape, device=DEVICE)
    out = torch.zeros(shape, device=DEVICE)

    grid = (triton.cdiv(M, block[0]), triton.cdiv(N, block[1]))
    add_2d[grid](x, y, out, M, N, BLOCK_M=block[0], BLOCK_N=block[1])
    torch.testing.assert_close(out, x + y)


def test_outer_product_broadcasts_both_axes():
    N = 32
    torch.manual_seed(0)
    a = torch.randn((N, ), device=DEVICE)
    b = torch.randn((N, ), device=DEVICE)
    out = torch.zeros((N, N), device=DEVICE)

    row_times_col[(1, )](a, b, out, N=N)
    torch.testing.assert_close(out, a[:, None] * b[None, :])
