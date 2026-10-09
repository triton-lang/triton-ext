"""List comprehensions in kernels, as Meta's fork's code generator accepts them.

Upstream Triton iterates only a ``tl.tuple`` in a comprehension; the fork also
takes a constexpr ``range(...)`` and ``if`` filters, which TLX kernels use to
build tuples of buffers. ``_compat.install_codegen_helpers`` adds both.
"""
import torch
import triton
import triton.language as tl
from conftest import DEVICE

BLOCK = tl.constexpr(16)


@triton.jit
def _sum_chunks(X, Y, N: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    chunks = tl.tuple([tl.load(X + i * BLOCK + offs) for i in range(N)])
    acc = tl.zeros((BLOCK, ), tl.float32)
    for i in tl.static_range(N):
        acc += chunks[i]
    tl.store(Y + offs, acc)


@triton.jit
def _sum_even_chunks(X, Y, N: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    chunks = tl.tuple(
        [tl.load(X + i * BLOCK + offs) for i in range(N) if i % 2 == 0])
    acc = tl.zeros((BLOCK, ), tl.float32)
    for i in tl.static_range(len(chunks)):
        acc += chunks[i]
    tl.store(Y + offs, acc)


@triton.jit
def _scale_tuple(X, Y):
    offs = tl.arange(0, BLOCK)
    pair = (tl.load(X + offs), tl.load(X + BLOCK + offs))
    scaled = [v * 2.0 for v in pair]  # upstream's own tuple comprehension
    tl.store(Y + offs, scaled[0] + scaled[1])


def _chunks(n):
    return torch.randn(n, 16, device=DEVICE, dtype=torch.float32)


def test_comprehension_over_constexpr_range():
    x = _chunks(4)
    y = torch.empty(16, device=DEVICE, dtype=torch.float32)
    _sum_chunks[(1, )](x, y, 4)
    torch.testing.assert_close(y, x.sum(0))


def test_comprehension_with_filter():
    x = _chunks(5)
    y = torch.empty(16, device=DEVICE, dtype=torch.float32)
    _sum_even_chunks[(1, )](x, y, 5)
    torch.testing.assert_close(y, x[0::2].sum(0))


def test_tuple_comprehension_still_upstream():
    x = _chunks(2)
    y = torch.empty(16, device=DEVICE, dtype=torch.float32)
    _scale_tuple[(1, )](x, y)
    torch.testing.assert_close(y, 2.0 * x.sum(0))
