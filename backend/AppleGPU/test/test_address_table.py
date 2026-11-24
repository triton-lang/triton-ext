"""Address tables: a kernel reads through an address loaded from a tensor.

The grouped-GEMM and mixture-of-experts idiom. Upstream covers it only in
test_warp_specialization.py::test_grouped_gemm, which is gated to Blackwell
and wraps it in TMA descriptors; these isolate the address table itself.
"""

import pytest

torch = pytest.importorskip("torch", reason="the dispatch path is torch's")

if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
    pytest.skip("needs an Apple GPU", allow_module_level=True)

import triton  # noqa: E402
import triton.language as tl  # noqa: E402

DEVICE = torch.device("mps")


@triton.jit
def capture_address(src_ptr, tab_ptr, i):
    tl.store(tab_ptr + i, tl.cast(src_ptr, tl.int64, bitcast=True))


@triton.jit
def read_through_table(tab_ptr, out_ptr, n, BLOCK: tl.constexpr):
    e = tl.program_id(0)
    off = tl.arange(0, BLOCK)
    mask = off < n
    addr = tl.load(tab_ptr + e)
    src = addr.to(tl.pointer_type(tl.float32))
    tl.store(out_ptr + e * n + off,
             tl.load(src + off, mask=mask, other=0.0),
             mask=mask)


@triton.jit
def read_through_table_bitcast(tab_ptr, out_ptr, n, BLOCK: tl.constexpr):
    e = tl.program_id(0)
    off = tl.arange(0, BLOCK)
    mask = off < n
    addr = tl.load(tab_ptr + e)
    src = tl.cast(addr, tl.pointer_type(tl.float32), bitcast=True)
    tl.store(out_ptr + e * n + off,
             tl.load(src + off, mask=mask, other=0.0),
             mask=mask)


def device_table(tensors):
    tab = torch.zeros(len(tensors), dtype=torch.int64, device=DEVICE)
    for i, t in enumerate(tensors):
        capture_address[(1, )](t, tab, i)
    return tab


def host_table(tensors):
    from triton_apple_backend.address import address_table
    return address_table(tensors)


def gather(tab, experts, n, kernel=read_through_table):
    out = torch.zeros(len(experts), n, dtype=torch.float32, device=DEVICE)
    kernel[(len(experts), )](tab, out, n, BLOCK=n)
    return out


def separate_experts(count, n):
    return [
        torch.arange(i * n, (i + 1) * n, dtype=torch.float32, device=DEVICE)
        for i in range(count)
    ]


@pytest.mark.parametrize("kernel",
                         [read_through_table, read_through_table_bitcast])
def test_device_built_table(kernel):
    n, count = 256, 8
    experts = separate_experts(count, n)
    out = gather(device_table(experts), experts, n, kernel)
    torch.testing.assert_close(out.cpu(),
                               torch.stack([e.cpu() for e in experts]))


def test_table_of_views():
    n, count = 256, 8
    stacked = torch.arange(count * n, dtype=torch.float32,
                           device=DEVICE).reshape(count, n)
    experts = [stacked[i] for i in range(count)]
    assert experts[1].storage_offset() == n

    out = gather(device_table(experts), experts, n)
    torch.testing.assert_close(out.cpu(), stacked.cpu())


def test_table_survives_allocator_churn():
    n, count = 256, 8
    experts = separate_experts(count, n)
    tab = device_table(experts)

    churn = [torch.rand(1 << 16, device=DEVICE) for _ in range(64)]
    del churn

    want = torch.stack([e.cpu() for e in experts])
    for _ in range(8):
        torch.testing.assert_close(gather(tab, experts, n).cpu(), want)


def test_host_built_table():
    n, count = 256, 8
    experts = separate_experts(count, n)
    out = gather(host_table(experts), experts, n)
    torch.testing.assert_close(out.cpu(),
                               torch.stack([e.cpu() for e in experts]))


def test_data_ptr_is_not_a_gpu_address():
    src = torch.arange(256, dtype=torch.float32, device=DEVICE)
    from triton_apple_backend.address import gpu_address
    assert gpu_address(src) != src.data_ptr()


@triton.jit
def copy_through_tables(src_tab, dst_tab, n, BLOCK: tl.constexpr):
    e = tl.program_id(0)
    src = tl.load(src_tab + e).to(tl.pointer_type(tl.float16))
    dst = tl.load(dst_tab + e).to(tl.pointer_type(tl.float16))
    for i in range(0, n, BLOCK):
        off = i + tl.arange(0, BLOCK)
        tl.store(dst + off, tl.load(src + off))


# Buffers reached only through a table are bound to no argument, so they are
# resident only because the launch names them. Large, separate allocations sit
# in heaps no argument shares; small ones next to the arguments pass by luck.
@pytest.mark.parametrize("build", [host_table, device_table])
def test_separate_buffers_reached_through_tables(build):
    count, n = 4, 8 << 19
    for rep in range(3):
        src = [
            torch.full((n, ),
                       float(i + 1 + 10 * rep),
                       dtype=torch.float16,
                       device=DEVICE) for i in range(count)
        ]
        dst = [
            torch.zeros(n, dtype=torch.float16, device=DEVICE)
            for _ in range(count)
        ]
        tables = build(src), build(dst)
        # A capture kernel binds its buffers; the copy must not ride on that.
        torch.mps.synchronize()
        copy_through_tables[(count, )](*tables, n, BLOCK=1024)
        for s, d in zip(src, dst):
            assert torch.equal(d.cpu(), s.cpu())


def test_launch_flags_name_address_use():
    src = torch.ones(256, device=DEVICE)
    tab = torch.zeros(1, dtype=torch.int64, device=DEVICE)
    out = torch.zeros(256, device=DEVICE)
    capture = capture_address[(1, )](src, tab, 0)
    read = read_through_table[(1, )](tab, out, 256, BLOCK=256)
    assert capture.metadata.exposes_addresses
    assert not capture.metadata.reads_addresses
    assert read.metadata.reads_addresses
    assert not read.metadata.exposes_addresses


def test_host_table_of_views():
    n, count = 256, 8
    stacked = torch.arange(count * n, dtype=torch.float32,
                           device=DEVICE).reshape(count, n)
    experts = [stacked[i] for i in range(count)]
    out = gather(host_table(experts), experts, n)
    torch.testing.assert_close(out.cpu(), stacked.cpu())
