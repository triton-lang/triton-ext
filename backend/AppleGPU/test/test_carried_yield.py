"""Loops whose yield moves carried values between one another's places: every
value must be read before any of them is overwritten."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="the dispatch path is torch's")

if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
    pytest.skip("needs an Apple GPU", allow_module_level=True)

import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def rotate(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    off = tl.arange(0, BLOCK)
    a = tl.load(x_ptr + off)
    b = tl.load(x_ptr + BLOCK + off)
    c = tl.load(x_ptr + 2 * BLOCK + off)
    i = 0
    j = 1
    for _ in range(n):
        a, b, c = b, c, a
        i, j = j, i
    tl.store(out_ptr + off, a + i)
    tl.store(out_ptr + BLOCK + off, b + j)
    tl.store(out_ptr + 2 * BLOCK + off, c)


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 7])
def test_rotate(n):
    BLOCK = 64
    x = torch.randn(3, BLOCK, device="mps")
    out = torch.empty_like(x)
    rotate[(1, )](x, out, n, BLOCK=BLOCK)
    want = x.roll(-n, 0).clone()
    want[0] += n % 2
    want[1] += 1 - n % 2
    torch.testing.assert_close(out, want)
