"""A scalar pointer whose offset it computes itself, defined before an early
return and read after it: the offset has to outlive the block that set it."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="the dispatch path is torch's")

if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
    pytest.skip("needs an Apple GPU", allow_module_level=True)

import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def offset_across_return(out, src, shifts, stride, n: tl.constexpr):
    pid = tl.program_id(0)
    row = src + pid * stride + tl.load(shifts + pid)
    if tl.load(row) < 0:
        return
    acc = 0
    for i in range(n):
        if i % 2 == 0:
            acc += tl.load(row + i)
    tl.store(out + pid, acc)


def test_offset_across_return():
    rows, stride, n = 4, 16, 6
    src = torch.arange(rows * stride, dtype=torch.int32, device="mps")
    src[stride + 1] = -1
    shifts = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device="mps")
    out = torch.full((rows, ), 7, dtype=torch.int32, device="mps")
    offset_across_return[(rows, )](out, src, shifts, stride, n=n)

    s, sh = src.cpu(), shifts.cpu()
    want = []
    for pid in range(rows):
        base = pid * stride + int(sh[pid])
        want.append(7 if s[base] < 0 else int(s[base:base + n:2].sum()))
    assert out.cpu().tolist() == want
