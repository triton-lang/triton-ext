"""`enable_reflect_ftz` reaches a libdevice stub's fma and nothing else."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="the dispatch path is torch's")

if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
    pytest.skip("needs an Apple GPU", allow_module_level=True)

import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from triton.language.extra import libdevice  # noqa: E402


@triton.jit
def pow_kernel(x_ptr, o_ptr, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(o_ptr + off, libdevice.pow(10000.0, tl.load(x_ptr + off)))


@triton.jit
def fma_kernel(x_ptr, o_ptr, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + off)
    tl.store(o_ptr + off, tl.math.fma(x, 1.5, -0.25))


def _body(kernel, **options):
    x = torch.rand(64, device="mps")
    out = torch.empty_like(x)
    return kernel[(1, )](x, out, BLOCK=64,
                         **options).asm["msl"].split("kernel void")[-1]


def test_a_stub_fma_is_bare_by_default():
    body = _body(pow_kernel)
    assert "metal::fma(" in body
    assert "__agpu_fma(" not in body


def test_a_stub_fma_is_guarded_without_ftz():
    assert "__agpu_fma(" in _body(pow_kernel, enable_reflect_ftz=False)


def test_a_user_fma_is_guarded_either_way():
    assert "__agpu_fma(" in _body(fma_kernel)
    assert "__agpu_fma(" in _body(fma_kernel, enable_reflect_ftz=False)


def test_the_knob_changes_no_bits_over_the_rope_range():
    y = torch.linspace(-2, 2, 1 << 16, device="mps")
    a, b = torch.empty_like(y), torch.empty_like(y)
    pow_kernel[(y.numel() // 1024, )](y, a, BLOCK=1024)
    pow_kernel[(y.numel() // 1024, )](y,
                                      b,
                                      BLOCK=1024,
                                      enable_reflect_ftz=False)
    assert torch.equal(a.view(torch.int32), b.view(torch.int32))
