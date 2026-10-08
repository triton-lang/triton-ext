"""
Matrix Multiplication: intensity across the tuning space
========================================================

This is the kernel of Triton's ``python/tutorials/03-matrix-multiplication.py``
(blocked FP16 matmul with grouped program ordering) used to compare the
*whole* autotuning space with ``triton-intensity``, instead of
only benchmarking the configuration the autotuner ends up picking:

* Importing :mod:`triton_intensity` installs the pass at the end
  of the ``ttir`` stage, so every kernel compiled below carries per-argument
  ``tint.load_bytes`` / ``tint.store_bytes`` / ``tint.op_count`` equations in its
  metadata.
* An :class:`triton_intensity.IntensityPruner` is installed as
  the autotuner's ``early_config_prune``: *before* the autotuner benchmarks
  anything, the hook compiles every ``triton.Config`` of the tuning space for
  the shape at hand (a :class:`~triton_intensity.ConfigLaunch`),
  evaluates its equations and removes the configurations whose intensity
  is below ``--min-intensity`` (40 FLOP/byte by default) from the
  tuning space, so the autotuner never times them. The ``BLOCK_SIZE_*`` and
  ``GROUP_SIZE_M`` meta-parameters are folded into each compiled kernel and
  therefore into its equations, so evaluating them gives the bytes each
  configuration moves and the FLOPs it performs for a given shape -- also
  for configurations that compile but do not fit the device.
* An :class:`triton_intensity.AutotuneRecorder` is installed as
  the autotuner's ``post_hook``: while the autotuner benchmarks the remaining
  configurations, the hook captures the compiled kernel of every one it
  tries and the error of those that do not fit the device.
* Each remaining configuration is then re-launched and timed on its own,
  the autotuner's own pick is marked, and the tutorial's cuBLAS/rocBLAS
  reference is measured for comparison. Pruned configurations appear in the
  table and plot with their intensity but without a time.

The interesting part is that the bytes the kernel *requests* are dominated by
the tile shape: every program streams a ``BLOCK_SIZE_M x K`` slab of ``A`` and
a ``K x BLOCK_SIZE_N`` slab of ``B``, so the arithmetic intensity is
approximately ``1 / (1 / BLOCK_SIZE_M + 1 / BLOCK_SIZE_N)`` FLOP per byte and
does not depend on ``BLOCK_SIZE_K``, ``GROUP_SIZE_M``, ``num_stages`` or
``num_warps``. The plot shows how much of the achieved performance is
explained by that intensity and how much by the remaining knobs.

Running the script sweeps a few square shapes, prints a table (per shape and
configuration: FLOPs, bytes, intensity, time, achieved throughput, whether the
autotuner selected it) and saves a plot::

    python matrix-multiplication.py                  # fp16, 1024..8192
    python matrix-multiplication.py --quick          # fewer shapes
    python matrix-multiplication.py --fp8            # float8_e5m2 inputs
    python matrix-multiplication.py --shapes 4096 2048,4096,1024
    python matrix-multiplication.py --min-intensity 0  # benchmark everything
    python matrix-multiplication.py --save-path out/
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Tuple

import torch

import triton
import triton.language as tl
import triton_intensity as tint

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# Gather the symbolic equations of every kernel function compiled from here on.
LISTENER = tint.enable()
#: Configurations below this intensity (FLOP/byte) are removed
#: from the tuning space before the autotuner benchmarks it (`--min-intensity`).
MIN_INTENSITY = 40.0
# Compile and evaluate every configuration for the shape being tuned and drop
# the low-intensity ones (installed as the autotuner's `early_config_prune`
# below).
PRUNER = tint.IntensityPruner(min_intensity=MIN_INTENSITY)
# Capture the compiled kernel of every configuration the autotuner benchmarks
# (installed as the autotuner's `post_hook` below).
RECORDER = tint.AutotuneRecorder()


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


# ---------------------------------------------------------------------------
# Tuning space and kernel (as in the tutorial)
# ---------------------------------------------------------------------------


def get_cuda_autotune_config():
    return [
        triton.Config(
            {
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_N': 256,
                'BLOCK_SIZE_K': 64,
                'GROUP_SIZE_M': 8
            },
            num_stages=3,
            num_warps=8),
        triton.Config(
            {
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 256,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 8
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_N': 128,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 8
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_N': 64,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 8
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 128,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 8
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_N': 32,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 8
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 32,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 8
            },
            num_stages=5,
            num_warps=2),
        triton.Config(
            {
                'BLOCK_SIZE_M': 32,
                'BLOCK_SIZE_N': 64,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 8
            },
            num_stages=5,
            num_warps=2),
        # Good config for fp8 inputs.
        triton.Config(
            {
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_N': 256,
                'BLOCK_SIZE_K': 128,
                'GROUP_SIZE_M': 8
            },
            num_stages=3,
            num_warps=8),
        triton.Config(
            {
                'BLOCK_SIZE_M': 256,
                'BLOCK_SIZE_N': 128,
                'BLOCK_SIZE_K': 128,
                'GROUP_SIZE_M': 8
            },
            num_stages=3,
            num_warps=8),
        triton.Config(
            {
                'BLOCK_SIZE_M': 256,
                'BLOCK_SIZE_N': 64,
                'BLOCK_SIZE_K': 128,
                'GROUP_SIZE_M': 8
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 256,
                'BLOCK_SIZE_K': 128,
                'GROUP_SIZE_M': 8
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_N': 128,
                'BLOCK_SIZE_K': 128,
                'GROUP_SIZE_M': 8
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_N': 64,
                'BLOCK_SIZE_K': 64,
                'GROUP_SIZE_M': 8
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 128,
                'BLOCK_SIZE_K': 64,
                'GROUP_SIZE_M': 8
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_N': 32,
                'BLOCK_SIZE_K': 64,
                'GROUP_SIZE_M': 8
            },
            num_stages=4,
            num_warps=4)
    ]


def get_hip_autotune_config():
    sizes = [
        {
            'BLOCK_SIZE_M': 32,
            'BLOCK_SIZE_N': 32,
            'BLOCK_SIZE_K': 64,
            'GROUP_SIZE_M': 6
        },
        {
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 32,
            'BLOCK_SIZE_K': 64,
            'GROUP_SIZE_M': 4
        },
        {
            'BLOCK_SIZE_M': 32,
            'BLOCK_SIZE_N': 64,
            'BLOCK_SIZE_K': 64,
            'GROUP_SIZE_M': 6
        },
        {
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 64,
            'BLOCK_SIZE_K': 64,
            'GROUP_SIZE_M': 6
        },
        {
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 64,
            'BLOCK_SIZE_K': 64,
            'GROUP_SIZE_M': 4
        },
        {
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 64,
            'GROUP_SIZE_M': 4
        },
        {
            'BLOCK_SIZE_M': 256,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 64,
            'GROUP_SIZE_M': 4
        },
        {
            'BLOCK_SIZE_M': 256,
            'BLOCK_SIZE_N': 256,
            'BLOCK_SIZE_K': 64,
            'GROUP_SIZE_M': 6
        },
    ]
    return [
        triton.Config(s | {'matrix_instr_nonkdim': 16},
                      num_warps=8,
                      num_stages=2) for s in sizes
    ]


def get_autotune_config():
    if is_cuda():
        return get_cuda_autotune_config()
    else:
        return get_hip_autotune_config()


@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
    # Called once per shape, before benchmarking, with the whole tuning space.
    prune_configs_by={'early_config_prune': PRUNER},
    post_hook=RECORDER,  # called after every benchmark launch of a config
)
@triton.jit
def matmul_kernel(
        # Pointers to matrices
        a_ptr,
        b_ptr,
        c_ptr,
        # Matrix dimensions
        M,
        N,
        K,
        # The stride variables represent how much to increase the ptr by when
        # moving by 1 element in a particular dimension. E.g. `stride_am` is
        # how much to increase `a_ptr` by to get the element one row down (A
        # has M rows).
        stride_am,
        stride_ak,  #
        stride_bk,
        stride_bn,  #
        stride_cm,
        stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
        ACTIVATION: tl.constexpr  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See the tutorial's `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # -----------------------------------------------------------
    # Add some integer bound assumptions.
    # This helps to guide integer analysis in the backend to optimize
    # load/store offset address calculation
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See the tutorial's `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am +
                      offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk +
                      offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K
        # dimension. If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs,
                    mask=offs_k[None, :] < K - k * BLOCK_SIZE_K,
                    other=0.0)
        b = tl.load(b_ptrs,
                    mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                    other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:,
                                         None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# The hooks compile the autotuner's configurations, resp. identify the
# benchmarked `triton.Config` among them.
PRUNER.attach(matmul_kernel)
RECORDER.attach(matmul_kernel)


# We can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter
# in `matmul_kernel`.
@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)


# ---------------------------------------------------------------------------
# Launch helpers
# ---------------------------------------------------------------------------


def matmul_grid(M: int, N: int):
    """1D launch grid where each block of C gets its own program."""
    return lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(
        N, META['BLOCK_SIZE_N']), )


def matmul_args(a: torch.Tensor, b: torch.Tensor,
                c: torch.Tensor) -> Tuple[Any, ...]:
    """Positional launch arguments of :func:`matmul_kernel`."""
    M, K = a.shape
    K, N = b.shape
    return (
        a,
        b,
        c,  #
        M,
        N,
        K,  #
        a.stride(0),
        a.stride(1),  #
        b.stride(0),
        b.stride(1),  #
        c.stride(0),
        c.stride(1),
    )


def matmul(a: torch.Tensor, b: torch.Tensor, activation="") -> torch.Tensor:
    """The tutorial's autotuned matmul."""
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, _ = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    matmul_kernel[matmul_grid(M, N)](*matmul_args(a, b, c),
                                     ACTIVATION=activation)
    return c


def autotune_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    activation=""
) -> Tuple[torch.Tensor, Dict[triton.Config, tint.ConfigLaunch], Dict[
        triton.Config, BaseException], List[triton.Config]]:
    """Autotune ``matmul`` for this shape, recording every configuration.

    The autotuner tunes this shape (the tuning cache is cleared first so it
    does so even for a shape seen before). Its ``early_config_prune``,
    :data:`PRUNER`, first compiles the whole tuning space and evaluates the
    equations of every configuration for this shape, removing those below
    :attr:`~triton_intensity.IntensityPruner.min_intensity`; the
    autotuner then benchmarks the rest and its ``post_hook``,
    :data:`RECORDER`, captures the compiled kernel of every configuration it
    runs -- whether or not it fits the device: a configuration can compile
    fine and still raise ``OutOfResources`` when launched, while the pass
    equations are already in the compiled kernel's metadata.

    Returns ``C`` (as computed by the autotuner's pick), the compiled launch
    of every configuration -- pruned or benchmarked (each writes into this
    same ``C`` when re-run) --, the error of every configuration that did not
    compile or run, and the configurations pruned for their intensity.
    """
    matmul_kernel.cache.clear()
    RECORDER.clear()
    PRUNER.clear()
    c = matmul(a, b, activation)
    # With pruning disabled only the recorder sees the configurations.
    launches = {**PRUNER.launches, **RECORDER.launches}
    errors = {**PRUNER.errors, **RECORDER.errors}
    return c, launches, errors, list(PRUNER.pruned)


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def config_label(config: triton.Config) -> str:
    kw = config.kwargs
    return (f"{kw['BLOCK_SIZE_M']}x{kw['BLOCK_SIZE_N']}x{kw['BLOCK_SIZE_K']}"
            f" g{kw['GROUP_SIZE_M']} s{config.num_stages} w{config.num_warps}")


def tile_label(config: triton.Config) -> str:
    kw = config.kwargs
    return f"{kw['BLOCK_SIZE_M']}x{kw['BLOCK_SIZE_N']}"


def make_inputs(M: int,
                N: int,
                K: int,
                fp8: bool,
                device=DEVICE) -> Tuple[torch.Tensor, torch.Tensor]:
    a = torch.randn((M, K), device=device, dtype=torch.float16)
    if not fp8:
        b = torch.randn((K, N), device=device, dtype=torch.float16)
        return a, b
    # Column-major b (pre-transposed for efficiency, as in the tutorial).
    b = torch.randn((N, K), device=device, dtype=torch.float16).T
    return a.to(torch.float8_e5m2), b.to(torch.float8_e5m2)


def reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.matmul(a.to(torch.float16), b.to(torch.float16))


def sweep(M: int, N: int, K: int, fp8: bool,
          configs: List[triton.Config]) -> List[Dict[str, Any]]:
    """Evaluate and time every ``config`` the autotuner tries for one shape."""
    a, b = make_inputs(M, N, K, fp8)
    c_ref = reference(a, b)
    shape = f"{M}x{N}x{K}"

    # Autotune: the hooks record the compiled kernel (and thus the equations)
    # of every configuration -- those pruned for their intensity before the
    # autotuner ran and those it benchmarked, including ones that compile but
    # do not fit the device; `best` is the autotuner's own pick.
    c, launches, errors, pruned = autotune_matmul(a, b)
    best = matmul_kernel.best_config

    # cuBLAS / rocBLAS reference; torch.matmul has no fp8 kernel.
    ref_ms = None
    if not fp8:
        ref_ms = triton.testing.do_bench(lambda: torch.matmul(a, b))

    rows: List[Dict[str, Any]] = []
    for config in configs:
        row: Dict[str, Any] = {
            "shape": shape,
            "M": M,
            "N": N,
            "K": K,
            "config": config_label(config),
            "tile": tile_label(config),
            **config.kwargs,
            "num_stages": config.num_stages,
            "num_warps": config.num_warps,
            "autotuned": config == best,
            "ref_ms": ref_ms,
        }
        rows.append(row)
        launch = launches.get(config)
        error = errors.get(config)
        if launch is None:
            # Did not compile (e.g. a compile-time assertion) or was not
            # benchmarked at all (pruned, or tuning results from disk).
            row["error"] = type(error).__name__ if error else "not benchmarked"
            print(f"{shape:>16} {row['config']:<22} skipped: "
                  f"{error or 'not benchmarked by the autotuner'}")
            continue
        # The equations are known once compiled, whether or not the
        # configuration can be launched on this device.
        work = launch.work()
        row.update({
            "flops": work.flops,
            "bytes": work.bytes,
            "intensity": work.intensity,
            "num_programs": work.num_cores,
            "launch": launch,
        })
        if config in pruned:  # too little intensity: not benchmarked
            row["error"] = "pruned"
            print(f"{shape:>16} {row['config']:<22}  intensity="
                  f"{work.intensity:6.1f} FLOP/B  pruned "
                  f"(< {PRUNER.min_intensity:g} FLOP/B)")
            continue
        if error is not None:  # compiled, but failed when the autotuner ran it
            row["error"] = type(error).__name__
            print(f"{shape:>16} {row['config']:<22}  intensity="
                  f"{work.intensity:6.1f} FLOP/B  not run: {error}")
            continue
        launch.run()  # a cache hit: overwrites `c` with this config's result
        ok = torch.allclose(c, c_ref, atol=0.125 if fp8 else 1e-2, rtol=1e-2)
        ms = triton.testing.do_bench(launch.run)
        row.update({
            "error": "" if ok else "mismatch",
            "ms": ms,
            "tflops": work.tflops(ms),
            "gbps": work.gbps(ms),
            "ref_tflops": work.tflops(ref_ms) if ref_ms else None,
        })
        mark = "*" if row["autotuned"] else " "
        ref = f"  (ref {row['ref_tflops']:6.1f} TFLOP/s)" if ref_ms else ""
        print(f"{shape:>16} {row['config']:<22}{mark} {ms:8.3f} ms  "
              f"{row['tflops']:6.1f} TFLOP/s  {row['gbps']:7.1f} GB/s  "
              f"intensity={work.intensity:6.1f} FLOP/B{ref}"
              f"{'  ' + row['error'] if row['error'] else ''}")
    return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_equations(launch: tint.ConfigLaunch) -> None:
    """Print the pass equations of one compiled configuration.

    ``args[i]`` symbols are resolved to parameter names and the folded
    ``constexpr`` / specialized parameters are listed, so the dependence of
    the per-program bytes on the tile shape is visible in the equations
    themselves.
    """
    # The listener gathered the same kernel function as it was compiled.
    assert launch.name in LISTENER, f"listener did not see {launch.name}"
    print()
    tint.print_equations(launch, title="Intensity equations (per program):")


def plot(rows: List[Dict[str, Any]], path: str, fp8: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = [r for r in rows if "tflops" in r]
    shapes = list(dict.fromkeys(r["shape"] for r in rows))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, (ax_scatter,
          ax_bars) = plt.subplots(1,
                                  2,
                                  figsize=(15, 5.5),
                                  gridspec_kw={"width_ratios": [1, 1.3]})

    # Left: achieved TFLOP/s against intensity, one marker per
    # configuration and shape; the autotuner's pick is outlined.
    for i, shape in enumerate(shapes):
        color = colors[i % len(colors)]
        group = [r for r in valid if r["shape"] == shape]
        if not group:
            continue
        ax_scatter.scatter([r["intensity"] for r in group],
                           [r["tflops"] for r in group],
                           color=color,
                           alpha=0.75,
                           label=shape)
        picks = [r for r in group if r["autotuned"]]
        ax_scatter.scatter([r["intensity"] for r in picks],
                           [r["tflops"] for r in picks],
                           s=160,
                           facecolors="none",
                           edgecolors=color,
                           linewidths=2)
        if group[0]["ref_ms"]:
            ax_scatter.axhline(group[0]["ref_tflops"],
                               color=color,
                               linestyle="--",
                               alpha=0.6)
    # Label each intensity cluster with its tile shape(s) (largest shape
    # only); tiles like 128x64 and 64x128 share the same intensity.
    largest = shapes[-1]
    clusters: Dict[float, Tuple[List[str], float]] = {}
    for r in valid:
        if r["shape"] == largest:
            tiles, y = clusters.setdefault(round(r["intensity"], 1), ([], 0.0))
            if r["tile"] not in tiles:
                tiles.append(r["tile"])
            clusters[round(r["intensity"], 1)] = (tiles, max(y, r["tflops"]))
    for x, (tiles, y) in clusters.items():
        ax_scatter.annotate("\n".join(tiles), (x, y),
                            textcoords="offset points",
                            xytext=(0, 6),
                            ha="center",
                            va="bottom",
                            fontsize=8)
    if valid:
        ax_scatter.set_ylim(top=1.15 * max(r["tflops"] for r in valid))
    if PRUNER.min_intensity:
        ax_scatter.axvline(PRUNER.min_intensity,
                           color="gray",
                           linestyle=":",
                           label=f"pruned below {PRUNER.min_intensity:g}")
    ax_scatter.set_xlabel("Intensity (FLOP/byte, from the pass)")
    ax_scatter.set_ylabel("Achieved TFLOP/s")
    ax_scatter.set_title("Tuning space: performance vs. intensity\n"
                         "(circled: autotuner's pick, dashed: torch.matmul)")
    ax_scatter.grid(True, alpha=0.3)
    ax_scatter.legend(title="M x N x K", fontsize=8)

    # Right: every configuration for the largest shape, ordered by intensity;
    # bars are achieved TFLOP/s, the line is the intensity (known for every
    # compiled configuration), missing bars were pruned for their intensity
    # or did not fit the device.
    group = sorted((r for r in rows if r["shape"] == largest),
                   key=lambda r:
                   (r.get("intensity", float("inf")), r["config"]))
    labels = [r["config"] for r in group]
    xs = range(len(group))
    ax_bars.bar(xs, [r.get("tflops", 0.0) for r in group],
                color=[
                    "tab:red" if r["autotuned"] else
                    ("lightgray" if "tflops" not in r else "tab:blue")
                    for r in group
                ])
    for x, r in zip(xs, group):
        if "tflops" not in r:
            ax_bars.text(x,
                         0,
                         r.get("error", "n/a"),
                         rotation=90,
                         ha="center",
                         va="bottom",
                         fontsize=7,
                         color="gray")
    ax_bars.set_xticks(list(xs))
    ax_bars.set_xticklabels(labels, rotation=75, ha="right", fontsize=8)
    ax_bars.set_ylabel("Achieved TFLOP/s (red: autotuner's pick)")
    ax_bars.set_title(f"All configurations for {largest}, sorted by intensity")
    ax_bars.grid(True, axis="y", alpha=0.3)
    ax_line = ax_bars.twinx()
    with_work = [(x, r) for x, r in zip(xs, group) if "intensity" in r]
    ax_line.plot([x for x, _ in with_work],
                 [r["intensity"] for _, r in with_work],
                 color="tab:green",
                 marker="o",
                 label="intensity")
    if PRUNER.min_intensity:
        ax_line.axhline(PRUNER.min_intensity,
                        color="tab:green",
                        linestyle=":",
                        alpha=0.7)
    ax_line.set_ylabel("Intensity (FLOP/byte)", color="tab:green")
    ax_line.tick_params(axis="y", colors="tab:green")

    device = torch.cuda.get_device_name(DEVICE) if is_cuda() else str(DEVICE)
    dtype = "float8_e5m2" if fp8 else "fp16"
    fig.suptitle(f"Matmul ({dtype}) tuning space, work from "
                 f"triton-intensity -- {device}")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"\nSaved plot to {path}")


def parse_shape(text: str) -> Tuple[int, int, int]:
    """``N`` for a square matmul or ``M,N,K``."""
    parts = [int(p) for p in text.replace("x", ",").split(",")]
    if len(parts) == 1:
        return parts[0], parts[0], parts[0]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"expected N or M,N,K, got {text!r}")
    return parts[0], parts[1], parts[2]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the intensity and performance of "
        "every matmul autotune configuration.")
    parser.add_argument("--shapes",
                        type=parse_shape,
                        nargs="+",
                        default=None,
                        help="N (square) or M,N,K; default 1024..8192 "
                        "(1024..2048 with --quick)")
    parser.add_argument("--fp8",
                        action="store_true",
                        help="use float8_e5m2 inputs")
    parser.add_argument("--quick", action="store_true", help="fewer shapes")
    parser.add_argument("--min-intensity",
                        type=float,
                        default=MIN_INTENSITY,
                        metavar="FLOP/B",
                        help="prune configurations below this intensity "
                        "before benchmarking; 0 benchmarks all "
                        f"(default {MIN_INTENSITY:g})")
    parser.add_argument("--save-path",
                        default=".",
                        help="directory for the plot and CSV")
    args = parser.parse_args()
    if args.shapes is None:
        args.shapes = [(2**i, ) * 3
                       for i in range(10, 12 if args.quick else 14)]
    if args.fp8 and not (hasattr(torch, "float8_e5m2") and is_cuda()):
        parser.error("--fp8 requires CUDA and a torch with float8_e5m2")
    # `None` lets every configuration through to the autotuner.
    PRUNER.min_intensity = args.min_intensity if args.min_intensity > 0 else None

    torch.manual_seed(0)
    os.makedirs(args.save_path, exist_ok=True)
    configs = matmul_kernel.configs
    rows: List[Dict[str, Any]] = []
    for M, N, K in args.shapes:
        rows.extend(sweep(M, N, K, args.fp8, configs))

    # The equations of the autotuner's pick for the last shape.
    picks = [r for r in rows if r["autotuned"] and "launch" in r]
    if picks:
        print_equations(picks[-1]["launch"])

    import pandas as pd
    df = pd.DataFrame([{
        k: v
        for k, v in row.items() if k not in ("launch", "M", "N", "K", "tile")
    } for row in rows])
    print("\n" + df.drop(columns=["ref_ms"]).to_string(
        index=False, float_format=lambda f: f"{f:.3f}"))
    stem = "matrix-multiplication-intensity" + ("-fp8" if args.fp8 else "")
    csv_path = os.path.join(args.save_path, stem + ".csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved data to {csv_path}")
    plot(rows, os.path.join(args.save_path, stem + ".png"), args.fp8)


if __name__ == "__main__":
    main()
