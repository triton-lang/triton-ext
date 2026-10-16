"""A whole-tensor reduction reshapes its operand with `can_reorder=True`. When
both layouts hold every element once in the same number of registers, each
thread keeps its own values: no round trip through threadgroup memory."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="the dispatch path is torch's")

if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
    pytest.skip("needs an Apple GPU", allow_module_level=True)

import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def row_stats(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + row * N + cols, mask=cols < N, other=0.0)
    tl.store(out_ptr + row * 2, tl.sum(x))
    tl.store(out_ptr + row * 2 + 1, tl.max(x))


@pytest.mark.parametrize("num_warps", [4, 8, 32])
def test_whole_reduction_keeps_registers(num_warps):
    M, N = 16, 4000
    x = torch.rand(M, N, device="mps") + 0.5
    out = torch.empty(M, 2, device="mps")
    k = row_stats[(M, )](x, out, N, BLOCK=4096, num_warps=num_warps)
    torch.testing.assert_close(out[:, 0].cpu(),
                               x.cpu().sum(1),
                               rtol=1e-4,
                               atol=1e-3)
    torch.testing.assert_close(out[:, 1].cpu(), x.cpu().amax(1))
    assert "tt.reshape" in k.asm["ttgir"]
    assert "sc[" not in k.asm["msl"]
