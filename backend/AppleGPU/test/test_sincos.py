"""A sin and a cos of the same value share one sincos."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="the dispatch path is torch's")

if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
    pytest.skip("needs an Apple GPU", allow_module_level=True)

import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def rotate_kernel(x_ptr, a_ptr, o_ptr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + i)
    a = tl.load(a_ptr + i)
    tl.store(o_ptr + i, x * tl.cos(a) - x * x * tl.sin(a))


def test_a_sin_cos_pair_is_one_sincos():
    x = torch.randn(4096, device="mps")
    a = torch.rand(4096, device="mps") * 300
    out = torch.empty_like(x)
    k = rotate_kernel[(4, )](x, a, out, BLOCK=1024)
    body = k.asm["msl"].split("kernel void")[-1]
    assert "metal::precise::sincos(" in body
    assert "metal::precise::sin(" not in body
    assert "metal::precise::cos(" not in body
    ref = (x.cpu().double() * a.cpu().double().cos() -
           x.cpu().double()**2 * a.cpu().double().sin()).float()
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-5, atol=1e-5)
