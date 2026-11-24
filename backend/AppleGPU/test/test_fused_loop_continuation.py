"""A loop continuing another loop's fused accumulator must own the same C tiles."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="the dispatch path is torch's")

if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
    pytest.skip("needs an Apple GPU", allow_module_level=True)

import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def two_loops_kernel(a_ptr, b_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr,
                     BK: tl.constexpr, STEPS: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, BK)
    acc = tl.zeros((M, N), dtype=tl.float32)
    for s in range(STEPS):
        a = tl.load(a_ptr + rm[:, None] * (2 * STEPS * BK) + s * BK +
                    rk[None, :])
        b = tl.load(b_ptr + (s * BK + rk[:, None]) * N + rn[None, :])
        acc = tl.dot(a, b, acc, input_precision="ieee")
    for s in range(STEPS, 2 * STEPS):
        a = tl.load(a_ptr + rm[:, None] *
                    (2 * STEPS * BK) + s * BK + rk[None, :]) * 2.0
        b = tl.load(b_ptr + (s * BK + rk[:, None]) * N + rn[None, :])
        acc = tl.dot(a, b, acc, input_precision="ieee")
    tl.store(o_ptr + rm[:, None] * N + rn[None, :], acc)


def test_differently_read_operands_continue_one_accumulator():
    M, N, BK, STEPS = 32, 64, 32, 2
    a = torch.randn(M, 2 * STEPS * BK, device="mps")
    b = torch.randn(2 * STEPS * BK, N, device="mps")
    out = torch.empty(M, N, device="mps")
    two_loops_kernel[(1, )](a,
                            b,
                            out,
                            M=M,
                            N=N,
                            BK=BK,
                            STEPS=STEPS,
                            num_warps=4)
    scale = torch.ones(2 * STEPS * BK, device="mps")
    scale[STEPS * BK:] = 2.0
    torch.testing.assert_close(out, (a * scale) @ b, rtol=1e-4, atol=1e-4)
