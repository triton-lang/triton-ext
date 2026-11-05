"""Work only one arm of a select reads runs under that arm's condition."""

from __future__ import annotations

import re

import pytest

torch = pytest.importorskip("torch", reason="the dispatch path is torch's")

if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
    pytest.skip("needs an Apple GPU", allow_module_level=True)

import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def alternate_kernel(x_ptr, o_ptr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + i)
    tl.store(o_ptr + i, tl.where(i % 2 == 0, tl.sin(x), tl.cos(x)))


@triton.jit
def nested_kernel(x_ptr, o_ptr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + i)
    inner = tl.where(i % 64 < 48, tl.cos(x), tl.exp(x))
    tl.store(o_ptr + i, tl.where(i % 64 < 32, tl.sin(x), inner))


@triton.jit
def halves_2d_kernel(x_ptr, o_ptr, R: tl.constexpr, C: tl.constexpr):
    r = tl.arange(0, R)[:, None]
    c = tl.arange(0, C)[None, :]
    x = tl.load(x_ptr + r * C + c)
    tl.store(o_ptr + r * C + c, tl.where(c < C // 2, tl.sin(x), tl.exp(x)))


@triton.jit
def cheap_kernel(x_ptr, o_ptr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + i)
    tl.store(o_ptr + i, tl.where(i % 64 < 32, x * 2.0 + 1.0, x - 3.0))


@triton.jit
def dot_arm_kernel(a_ptr, o_ptr, N: tl.constexpr):
    r = tl.arange(0, N)
    a = tl.load(a_ptr + r[:, None] * N + r[None, :])
    d = tl.dot(a, a, input_precision="ieee")
    out = tl.where(r[:, None] < N // 2, tl.exp(d), 0.0)
    tl.store(o_ptr + r[:, None] * N + r[None, :], out)


@triton.jit
def scalar_cond_kernel(x_ptr, s_ptr, o_ptr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + i)
    a = tl.load(s_ptr + tl.program_id(0))
    scalar = tl.where(a > 1.0, tl.exp(a), tl.sin(a))
    tl.store(o_ptr + i, x + scalar)


@triton.jit
def load_then_store_kernel(p_ptr, o_ptr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    x = tl.load(p_ptr + i)
    tl.store(p_ptr + i, tl.zeros([BLOCK], tl.float32))
    tl.store(o_ptr + i, tl.where(i % 64 < 32, tl.sin(x), 1.0))


def _body(k):
    return k.asm["msl"].split("kernel void")[-1]


def _predicated(k):
    return re.search(r"if \(!?c\d+_\d+( \|\||\) \{)", _body(k)) is not None


# 512 elements over 16 warps is one per lane, so `i % 64 < 32` is the same in
# every lane of a simdgroup.
def test_a_simdgroup_uniform_condition_predicates():
    x = torch.randn(2048, device="mps")
    out = torch.empty_like(x)
    k = nested_kernel[(4, )](x, out, BLOCK=512, num_warps=16)
    j = torch.arange(2048, device="mps") % 64
    ref = torch.where(j < 32, x.sin(), torch.where(j < 48, x.cos(), x.exp()))
    torch.testing.assert_close(out, ref)
    assert _predicated(k)


@pytest.mark.parametrize("num_warps", [1, 4])
def test_a_lane_varying_condition_stays_branch_free(num_warps):
    x = torch.randn(4096, device="mps")
    out = torch.empty_like(x)
    k = alternate_kernel[(4, )](x, out, BLOCK=1024, num_warps=num_warps)
    even = torch.arange(4096, device="mps") % 2 == 0
    torch.testing.assert_close(out, torch.where(even, x.sin(), x.cos()))
    assert not _predicated(k)


def test_a_scalar_condition_stays_branch_free():
    x = torch.randn(2048, device="mps")
    s = torch.tensor([2.0, 0.5, 3.0, 0.25], device="mps")
    out = torch.empty_like(x)
    k = scalar_cond_kernel[(4, )](x, s, out, BLOCK=512, num_warps=16)
    scalar = torch.where(s > 1.0, s.exp(), s.sin()).repeat_interleave(512)
    torch.testing.assert_close(out, x + scalar)
    assert not _predicated(k)


def test_a_2d_select_predicates_per_thread():
    x = torch.randn(32, 64, device="mps")
    out = torch.empty_like(x)
    halves_2d_kernel[(1, )](x, out, R=32, C=64)
    ref = torch.cat([x[:, :32].sin(), x[:, 32:].exp()], dim=1)
    torch.testing.assert_close(out, ref)


def test_an_arm_load_stays_ahead_of_a_later_store():
    x = torch.randn(1024, device="mps")
    x0 = x.clone()
    out = torch.empty_like(x)
    load_then_store_kernel[(1, )](x, out, BLOCK=1024)
    j = torch.arange(1024, device="mps") % 64
    torch.testing.assert_close(out, torch.where(j < 32, x0.sin(), 1.0))


def test_a_select_on_a_dot_result_stays_branch_free():
    a = torch.randn(32, 32, device="mps") * 0.1
    out = torch.empty_like(a)
    k = dot_arm_kernel[(1, )](a, out, N=32)
    ref = torch.where(
        torch.arange(32, device="mps")[:, None] < 16, (a @ a).exp(), 0.0)
    torch.testing.assert_close(out, ref)
    assert not _predicated(k)


def test_cheap_arms_stay_branch_free():
    x = torch.randn(2048, device="mps")
    out = torch.empty_like(x)
    k = cheap_kernel[(4, )](x, out, BLOCK=512, num_warps=16)
    j = torch.arange(2048, device="mps") % 64
    torch.testing.assert_close(out, torch.where(j < 32, x * 2 + 1, x - 3))
    assert not _predicated(k)
