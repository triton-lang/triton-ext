"""A row masked by `cols < N` moves in vector runs, each under one guard, when
the mask is constant over every run: a run past N is skipped whole, and the
runs before it stay vectors."""

from __future__ import annotations

import re

import pytest

torch = pytest.importorskip("torch", reason="the dispatch path is torch's")

if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
    pytest.skip("needs an Apple GPU", allow_module_level=True)

import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def center_rows(x_ptr, y_ptr, N, stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(x_ptr + row * stride + cols, mask=mask, other=0.0)
    x = x.to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    tl.store(y_ptr + row * stride + cols,
             (x - mean).to(y_ptr.dtype.element_ty),
             mask=mask)


@pytest.mark.parametrize("N", [1040, 4100, 4608])
def test_masked_row_runs(N):
    M, pad = 64, 16
    stride = N + pad
    x = torch.randn(M, stride, device="mps", dtype=torch.float16)
    y = torch.full_like(x, float("nan"))
    k = center_rows[(M, )](x,
                           y,
                           N,
                           stride,
                           BLOCK=triton.next_power_of_2(N),
                           num_warps=8)
    xs = x[:, :N].float()
    torch.testing.assert_close(y[:, :N].float(),
                               xs - xs.mean(1, keepdim=True),
                               rtol=1e-2,
                               atol=1e-2)
    assert torch.isnan(y[:, N:]).all()
    if N % 16 == 0:
        body = k.asm["msl"].split("kernel void")[-1]
        assert not re.search(r"if \(c\d+_\d+\) l\d+_\d+ = ", body)
