"""A load that only feeds one arm of a select is masked by that arm's condition."""

from __future__ import annotations

import re

import pytest

torch = pytest.importorskip("torch", reason="the dispatch path is torch's")

if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
    pytest.skip("needs an Apple GPU", allow_module_level=True)

import triton  # noqa: E402
import triton.language as tl  # noqa: E402

N = 1024


@triton.jit
def stack_kernel(a_ptr, b_ptr, o_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    j = i % N
    s = i // N
    a = tl.where(s == 0, tl.load(a_ptr + j), 0.0)
    b = tl.where(s == 1, tl.load(b_ptr + j) * 2.0, 0.0)
    tl.store(o_ptr + i, a + b)


@triton.jit
def false_arm_kernel(a_ptr, o_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(o_ptr + i, tl.where(i < N, 0.0, tl.load(a_ptr + i % N) + 1.0))


@triton.jit
def shared_kernel(a_ptr, o_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(a_ptr + i % N)
    tl.store(o_ptr + i, tl.where(i < N, x, x * 2.0))


@triton.jit
def dot_rows_kernel(a_ptr, o_ptr, N: tl.constexpr):
    r = tl.arange(0, N)
    a = tl.load(a_ptr + r[:, None] * N + r[None, :])
    d = tl.dot(a, a, input_precision="ieee")
    tl.store(o_ptr + r[:, None] * N + r[None, :],
             tl.where(r[:, None] < N // 2, d, 0.0))


def _loads(ttir):
    return re.findall(r"tt\.load (%[\w.]+)((?:, %[\w.]+)*)", ttir)


def test_each_arm_load_gets_its_condition():
    a = torch.randn(N, device="mps")
    b = torch.randn(N, device="mps")
    out = torch.empty(2 * N, device="mps")
    k = stack_kernel[(2 * N // 256, )](a, b, out, N=N, BLOCK=256)
    loads = _loads(k.asm["ttir"])
    assert len(loads) == 2
    assert all(extra.count(",") == 2 for _, extra in loads)
    assert torch.equal(out, torch.cat([a, 2 * b]))


def test_a_false_arm_load_gets_the_negated_condition():
    a = torch.randn(N, device="mps")
    out = torch.empty(2 * N, device="mps")
    k = false_arm_kernel[(2 * N // 256, )](a, out, N=N, BLOCK=256)
    assert all(extra.count(",") == 2 for _, extra in _loads(k.asm["ttir"]))
    assert torch.equal(out, torch.cat([torch.zeros_like(a), a + 1]))


def test_a_load_reached_through_a_dot_stays_unmasked():
    a = torch.randn(32, 32, device="mps")
    out = torch.empty_like(a)
    k = dot_rows_kernel[(1, )](a, out, N=32)
    rows = torch.arange(32, device="mps")[:, None] < 16
    ref = torch.where(rows,
                      (a.cpu().double() @ a.cpu().double()).float().to("mps"),
                      0.0)
    torch.testing.assert_close(out, ref)
    assert [extra for _, extra in _loads(k.asm["ttir"])] == [""]


def test_a_load_both_arms_read_stays_unmasked():
    a = torch.randn(N, device="mps")
    out = torch.empty(2 * N, device="mps")
    k = shared_kernel[(2 * N // 256, )](a, out, N=N, BLOCK=256)
    assert [extra for _, extra in _loads(k.asm["ttir"])] == [""]
    assert torch.equal(out, torch.cat([a, 2 * a]))
