"""
Two-dimensional Elementwise
===========================

In this tutorial, you will write a kernel over a 2D tile and see how a tile's
shape is built out of one-dimensional ranges.

A 2D index is two 1D ranges combined: one down the rows, one across the
columns. `tl.arange` gives each range, `[:, None]` and `[None, :]` say which
axis each one runs along, and the arithmetic between them broadcasts to the
full tile.

None of that moves data. Every program already holds the values it needs; the
indexing only decides which register means which coordinate.

In doing so, you will learn about:

* Indexing a 2D tile with `tl.arange` and `None`.

* How a 2D kernel is launched over a grid of tiles.

"""

# %%
# Compute Kernel
# --------------

import torch

import triton
import triton.language as tl

DEVICE = torch.device("mps")


@triton.jit
def add_2d_kernel(x_ptr, y_ptr, out_ptr, M, N, BLOCK_M: tl.constexpr,
                  BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # The tile's addresses: a column of row offsets plus a row of column
    # offsets, broadcast against each other.
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


def add_2d(x: torch.Tensor, y: torch.Tensor):
    M, N = x.shape
    out = torch.empty_like(x)
    BLOCK_M, BLOCK_N = 64, 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    add_2d_kernel[grid](x, y, out, M, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    return out


# %%
# Let's check it against torch.

torch.manual_seed(0)
x = torch.randn((512, 512), device=DEVICE, dtype=torch.float32)
y = torch.randn((512, 512), device=DEVICE, dtype=torch.float32)
out_torch = x + y
out_triton = add_2d(x, y)
print(out_torch)
print(out_triton)
print(f"The maximum difference between torch and triton is "
      f"{torch.max(torch.abs(out_torch - out_triton))}")

# %%
# Benchmark
# ---------
#
# Two loads and one store per element, so this is bandwidth-bound and the
# tile shape is the whole story.


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],
        x_vals=[2**i for i in range(7, 13, 1)],
        x_log=True,
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'Torch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',
        plot_name='2d-elementwise-performance',
        args={},
    ))
def benchmark(size, provider):
    x = torch.randn((size, size), device=DEVICE, dtype=torch.float32)
    y = torch.randn((size, size), device=DEVICE, dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y,
                                                     quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add_2d(x, y),
                                                     quantiles=quantiles)

    def gbps(ms):
        return 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)

    return gbps(ms), gbps(max_ms), gbps(min_ms)


# %%
# Pass `print_data=True` to see the numbers, `show_plots=True` to plot them,
# and/or `save_path='/path/to/results/'` to save them along with raw CSV data:
benchmark.run(print_data=True, show_plots=True)
