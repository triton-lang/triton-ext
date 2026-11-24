"""An i64 tt.addptr offset stays 64-bit through the chain, so an element index
past 2^31 does not wrap. Checked on the emitted MSL, without a 2 GB buffer."""

from __future__ import annotations

import re

import pytest

torch = pytest.importorskip("torch", reason="the dispatch path is torch's")

if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
    pytest.skip("needs an Apple GPU", allow_module_level=True)

import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def rows(x, y, stride, WIDE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    if WIDE:
        row = row.to(tl.int64)
    col = tl.arange(0, BLOCK)
    tl.store(y + row * stride + col, tl.load(x + row * stride + col))


@triton.jit
def masked_tile(x, y, M, N, stride, BM: tl.constexpr, BN: tl.constexpr):
    rm = (tl.program_id(0) * BM + tl.arange(0, BM)).to(tl.int64)
    rn = tl.arange(0, BN)
    m = (rm[:, None] < M) & (rn[None, :] < N)
    off = rm[:, None] * stride + rn[None, :]
    tl.store(y + off, tl.load(x + off, mask=m, other=0.0) + 1.0, mask=m)


@triton.jit
def carried(x, y, stride, STEPS: tl.constexpr, BLOCK: tl.constexpr):
    p = x + tl.program_id(0).to(tl.int64) * stride + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK, ), tl.float32)
    for _ in range(STEPS):
        acc += tl.load(p)
        p += BLOCK
    tl.store(y + tl.program_id(0) * BLOCK + tl.arange(0, BLOCK), acc)


@triton.jit
def early_return(x, y, stride, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row = x + pid.to(tl.int64) * stride
    if pid >= n:
        return
    tl.store(y + pid * BLOCK + tl.arange(0, BLOCK),
             tl.load(row + tl.arange(0, BLOCK)))


def offset_types(kernel):
    body = kernel.asm["msl"].split("kernel void")[-1]
    return set(re.findall(r"\b(int|long) off\w+ =", body))


@pytest.mark.parametrize("wide", [False, True])
def test_chain_keeps_its_offset_width(wide):
    x = torch.arange(4 * 64, dtype=torch.float32, device="mps")
    y = torch.empty_like(x)
    k = rows[(4, )](x, y, 64, WIDE=wide, BLOCK=64)
    assert torch.equal(y, x)
    assert offset_types(k) == ({"long"} if wide else {"int"})


def test_masked_tile():
    x = torch.randn(70, 50, device="mps")
    y = torch.empty_like(x)
    masked_tile[(triton.cdiv(70, 32), )](x, y, 70, 50, 50, BM=32, BN=64)
    assert torch.equal(y, x + 1)


def test_loop_carried_pointer():
    x = torch.randn(4 * 256, device="mps")
    y = torch.empty(4 * 64, device="mps")
    k = carried[(4, )](x, y, 256, STEPS=4, BLOCK=64)
    torch.testing.assert_close(y, x.view(4, 4, 64).sum(1).flatten())
    assert "long" in offset_types(k)


def test_offset_across_early_return():
    x = torch.randn(4 * 64, device="mps")
    y = torch.zeros(4 * 64, device="mps")
    k = early_return[(6, )](x, y, 64, 4, BLOCK=64)
    assert torch.equal(y, x)
    assert "long" in offset_types(k)
