"""A program polling with volatile loads or value-preserving atomics sees what
an earlier program of the same launch stored. Only earlier: Apple GPUs promise
no forward progress to a program waiting on a later one."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="the dispatch path is torch's")

if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
    pytest.skip("needs an Apple GPU", allow_module_level=True)

import triton  # noqa: E402
import triton.language as tl  # noqa: E402

LIMIT = 100_000
DELAY = 1_000


@triton.jit
def handoff(flag, seen, polls, scratch, DELAY: tl.constexpr,
            LIMIT: tl.constexpr):
    if tl.program_id(0) == 0:
        for _ in range(DELAY):
            tl.atomic_add(scratch, 1)
        tl.store(flag, 5)
    else:
        v = tl.load(flag, volatile=True)
        n = 1
        while (v == 0) & (n < LIMIT):
            v = tl.load(flag, volatile=True)
            n += 1
        tl.store(seen, v)
        tl.store(polls, n)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.float16])
def test_scalar_poll(dtype):
    flag = torch.zeros(1, dtype=dtype, device="mps")
    seen = torch.zeros(1, dtype=dtype, device="mps")
    polls = torch.zeros(1, dtype=torch.int32, device="mps")
    scratch = torch.zeros(1, dtype=torch.int32, device="mps")
    handoff[(2, )](flag, seen, polls, scratch, DELAY=DELAY, LIMIT=LIMIT)
    assert seen.item() == 5
    assert 1 < polls.item() < LIMIT


@triton.jit
def handoff_rmw(flag, seen, polls, scratch, DELAY: tl.constexpr,
                LIMIT: tl.constexpr):
    if tl.program_id(0) == 0:
        for _ in range(DELAY):
            tl.atomic_add(scratch, 1)
        tl.store(flag, 5)
    else:
        v = tl.atomic_add(flag, 0)
        n = 1
        while (v == 0) & (n < LIMIT):
            v = tl.atomic_add(flag, 0)
            n += 1
        tl.store(seen, v)
        tl.store(polls, n)


def test_scalar_rmw_poll():
    flag = torch.zeros(1, dtype=torch.int32, device="mps")
    seen = torch.zeros(1, dtype=torch.int32, device="mps")
    polls = torch.zeros(1, dtype=torch.int32, device="mps")
    scratch = torch.zeros(1, dtype=torch.int32, device="mps")
    handoff_rmw[(2, )](flag, seen, polls, scratch, DELAY=DELAY, LIMIT=LIMIT)
    assert seen.item() == 5
    assert 1 < polls.item() < LIMIT


@triton.jit
def handoff_block(flags, seen, polls, scratch, N: tl.constexpr,
                  BLOCK: tl.constexpr, DELAY: tl.constexpr,
                  LIMIT: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    if tl.program_id(0) == 0:
        for _ in range(DELAY):
            tl.atomic_add(scratch, 1)
        tl.store(flags + offs, offs + 1, mask=mask)
    else:
        v = tl.load(flags + offs, mask=mask, other=1, volatile=True)
        n = 1
        while (tl.min(v, axis=0) == 0) & (n < LIMIT):
            v = tl.load(flags + offs, mask=mask, other=1, volatile=True)
            n += 1
        tl.store(seen + offs, v, mask=mask)
        tl.store(polls, n)


def test_masked_block_poll():
    N, BLOCK = 100, 128
    flags = torch.zeros(N, dtype=torch.int32, device="mps")
    seen = torch.zeros(N, dtype=torch.int32, device="mps")
    polls = torch.zeros(1, dtype=torch.int32, device="mps")
    scratch = torch.zeros(1, dtype=torch.int32, device="mps")
    handoff_block[(2, )](flags,
                         seen,
                         polls,
                         scratch,
                         N=N,
                         BLOCK=BLOCK,
                         DELAY=DELAY,
                         LIMIT=LIMIT)
    assert torch.equal(seen.cpu(), torch.arange(1, N + 1, dtype=torch.int32))
    assert 1 < polls.item() < LIMIT


@triton.jit
def handoff_atomic_poll(flag, seen, polls, scratch, DELAY: tl.constexpr,
                        LIMIT: tl.constexpr):
    if tl.program_id(0) == 0:
        for _ in range(DELAY):
            tl.atomic_add(scratch, 1)
        tl.store(flag, 5)
    else:
        ok = tl.atomic_poll(flag, 5, timeout_ns=0).to(tl.int32)
        n = 1
        while (ok == 0) & (n < LIMIT):
            ok = tl.atomic_poll(flag, 5, timeout_ns=0).to(tl.int32)
            n += 1
        tl.store(seen, ok)
        tl.store(polls, n)


def test_atomic_poll():
    flag = torch.zeros(1, dtype=torch.int32, device="mps")
    seen = torch.zeros(1, dtype=torch.int32, device="mps")
    polls = torch.zeros(1, dtype=torch.int32, device="mps")
    scratch = torch.zeros(1, dtype=torch.int32, device="mps")
    handoff_atomic_poll[(2, )](flag,
                               seen,
                               polls,
                               scratch,
                               DELAY=DELAY,
                               LIMIT=LIMIT)
    assert seen.item() == 1
    assert 1 < polls.item() < LIMIT


@triton.jit
def publish(data, flag, before, after, polls, scratch, BLOCK: tl.constexpr,
            DELAY: tl.constexpr, LIMIT: tl.constexpr, ACQUIRE: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    if tl.program_id(0) == 0:
        for _ in range(DELAY):
            tl.atomic_add(scratch, 1)
        tl.store(data + offs, offs + 1)
        tl.atomic_xchg(flag, 1, sem="release")
    else:
        tl.store(before + offs, tl.load(data + offs))
        n = 1
        if ACQUIRE:
            ok = tl.atomic_poll(flag, 1, sem="acquire",
                                timeout_ns=0).to(tl.int32)
            while (ok == 0) & (n < LIMIT):
                ok = tl.atomic_poll(flag, 1, sem="acquire",
                                    timeout_ns=0).to(tl.int32)
                n += 1
        else:
            v = tl.load(flag, volatile=True)
            while (v == 0) & (n < LIMIT):
                v = tl.load(flag, volatile=True)
                n += 1
        tl.store(after + offs, tl.load(data + offs))
        tl.store(polls, n)


@pytest.mark.parametrize("acquire", [False, True])
def test_release_publishes_data(acquire):
    BLOCK = 128
    data = torch.zeros(BLOCK, dtype=torch.int32, device="mps")
    flag = torch.zeros(1, dtype=torch.int32, device="mps")
    before = torch.zeros(BLOCK, dtype=torch.int32, device="mps")
    after = torch.zeros(BLOCK, dtype=torch.int32, device="mps")
    polls = torch.zeros(1, dtype=torch.int32, device="mps")
    scratch = torch.zeros(1, dtype=torch.int32, device="mps")
    publish[(2, )](data,
                   flag,
                   before,
                   after,
                   polls,
                   scratch,
                   BLOCK=BLOCK,
                   DELAY=DELAY,
                   LIMIT=LIMIT,
                   ACQUIRE=acquire)
    assert 1 < polls.item() < LIMIT
    assert torch.equal(after.cpu(),
                       torch.arange(1, BLOCK + 1, dtype=torch.int32))
