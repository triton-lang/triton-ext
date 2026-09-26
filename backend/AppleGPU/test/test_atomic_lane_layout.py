"""Tensor atomics give each lane one element of the fastest axis."""

from __future__ import annotations

import re

import pytest

torch = pytest.importorskip("torch", reason="the dispatch path is torch's")

if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
    pytest.skip("needs an Apple GPU", allow_module_level=True)

import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def scatter_add_kernel(v_ptr, idx_ptr, o_ptr, R: tl.constexpr):
    x = tl.program_id(0)
    r = tl.arange(0, R)
    row = tl.load(idx_ptr + x)
    tl.atomic_add(o_ptr + r + R * row,
                  tl.load(v_ptr + r + R * x),
                  sem="relaxed")


def _atomic_size_per_thread(ttgir):
    enc = re.search(r"tt\.atomic_rmw .*?-> tensor<\d+xf32, (#\w+)>",
                    ttgir).group(1)
    decl = re.search(
        re.escape(enc) + r" = #ttg\.blocked<\{sizePerThread = \[(\d+)\]",
        ttgir)
    return int(decl.group(1))


@pytest.mark.parametrize("rows", [7, 512])
def test_scatter_add_matches_index_add(rows):
    n, r = 512, 256
    v = torch.randn(n, r, device="mps")
    idx = torch.randint(0, rows, (n, ), device="mps")
    out = torch.zeros(rows, r, device="mps")
    k = scatter_add_kernel[(n, )](v, idx, out, R=r, num_warps=2)
    torch.testing.assert_close(out,
                               torch.zeros_like(out).index_add_(0, idx, v))
    assert _atomic_size_per_thread(k.asm["ttgir"]) == 1
