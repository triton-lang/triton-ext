"""
Persistent Matmul: one problem, many kernels, measured by the pass
===================================================================

This is the kernel zoo of Triton's ``python/tutorials/09-persistent-matmul.py``
-- naive, persistent, TMA (host tensor descriptors), TMA persistent, CLC TMA
(compute capability >= 10.0) and device-descriptor persistent matmuls, each
with and without warp specialization -- all solving the *same* ``C = A @ B``
problem, with the tutorial's hand-written ``flops``/``bytes`` launch metadata
replaced by what ``triton-arithmetic-intensity`` derives from the kernels:

* Importing :mod:`triton_arithmetic_intensity` installs the pass at the end
  of the ``ttir`` stage; every kernel below then carries per-argument
  ``tai.load_bytes`` / ``tai.store_bytes`` / ``tai.op_count`` equations in its
  metadata.
* Each launch goes through :func:`triton_arithmetic_intensity.launch` and is
  recorded (:func:`triton_arithmetic_intensity.record`) so the equations can
  be evaluated for the exact grid, arguments and autotuned configuration. The
  persistent kernels are the interesting case: their per-program work is a
  ``program_id``-dependent trip count of ``range(pid, num_tiles, NUM_SMS)``,
  which the evaluator sums over the ``min(NUM_SMS, num_tiles)`` programs.
* Host tensor descriptors (``a_desc.load``) and device descriptors
  (``tl.make_tensor_descriptor``) are attributed to the descriptor / pointer
  argument, so TMA variants are reported exactly like the pointer variants.

Every variant is validated, timed with ``triton.testing.do_bench`` and listed
next to cuBLAS/hipBLAS and ``torch.matmul``. The table also shows the
tutorial's analytic byte count ``elem * (M*K + N*K + M*N)`` next to the bytes
the kernels actually request, i.e. how many times the operands are
re-streamed through the memory hierarchy for the chosen tile shape::

    python persistent-matmul.py                       # fp16, M=N=8192, K=128..1024
    python persistent-matmul.py -K 512                # a single K
    python persistent-matmul.py --prec fp8 --K_range 128 1024 --K_step 128
    python persistent-matmul.py --quick               # smaller problem, one K
    python persistent-matmul.py --save-path out/
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

import triton
import triton.language as tl
import triton_arithmetic_intensity as tai
from triton.tools.tensor_descriptor import TensorDescriptor
from triton_arithmetic_intensity import launch

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# Gather the symbolic equations of every kernel function compiled from here on.
LISTENER = tai.enable()


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


if is_cuda():
    from triton._C.libtriton import nvidia
    device_workspace = torch.empty(32 * 1024 * 1024,
                                   device="cuda",
                                   dtype=torch.uint8)
    device_blas = nvidia.cublas.CublasLt(device_workspace)
elif is_hip():
    from triton._C.libtriton import amd
    device_workspace = torch.empty(32 * 1024 * 1024,
                                   device="cuda",
                                   dtype=torch.uint8)
    device_blas = amd.hipblas.HipblasLt(device_workspace)
else:
    device_blas = None


def device_blas_name():
    return 'cuBLAS' if is_cuda() else 'hipBLAS'


def supports_tma():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


def is_hopper():
    return torch.cuda.get_device_capability()[0] == 9


def supports_ws():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


def supports_clc():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 10


HAS_TENSOR_DESC = supports_tma() and hasattr(tl, "make_tensor_descriptor")
HAS_HOST_TENSOR_DESC = supports_tma() and hasattr(
    triton.tools.tensor_descriptor, "TensorDescriptor")
HAS_WARP_SPECIALIZE = supports_ws() and HAS_TENSOR_DESC
HAS_TMA_CLC = supports_clc() and HAS_HOST_TENSOR_DESC

# ---------------------------------------------------------------------------
# Kernels (as in the tutorial, launching through `tai.launch`)
# ---------------------------------------------------------------------------


def matmul_get_configs(pre_hook=None):
    return [
        triton.Config(
            {
                'BLOCK_SIZE_M': BM,
                'BLOCK_SIZE_N': BN,
                "BLOCK_SIZE_K": BK,
                "GROUP_SIZE_M": 8
            },
            num_stages=s,
            num_warps=w,
            pre_hook=pre_hook) for BM in [128] for BN in [128, 256]
        for BK in [64, 128] for s in ([2, 3, 4]) for w in [4, 8]
    ]


@triton.autotune(
    configs=matmul_get_configs(),
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
        a_ptr,
        b_ptr,
        c_ptr,  #
        M,
        N,
        K,  #
        stride_am,
        stride_ak,  #
        stride_bk,
        stride_bn,  #
        stride_cm,
        stride_cn,  #
        BLOCK_SIZE_M: tl.constexpr,  #
        BLOCK_SIZE_N: tl.constexpr,  #
        BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N

    offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_am = tl.where(offs_am < M, offs_am, 0)
    offs_bn = tl.where(offs_bn < N, offs_bn, 0)

    offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M),
                                BLOCK_SIZE_M)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N),
                                BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am +
                      offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk +
                      offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs,
                    mask=offs_k[None, :] < K - k * BLOCK_SIZE_K,
                    other=0.0)
        b = tl.load(b_ptrs,
                    mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                    other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if (c_ptr.dtype.element_ty == tl.float8e4nv):
        c = accumulator.to(tl.float8e4nv)
    else:
        c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:,
                                         None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype

    c = torch.empty((M, N), device=a.device, dtype=dtype)

    # 1D launch kernel where each block gets its own program.
    def grid(META):
        return (triton.cdiv(M, META["BLOCK_SIZE_M"]) *
                triton.cdiv(N, META["BLOCK_SIZE_N"]), )

    launch(
        matmul_kernel,
        grid,
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
        c.stride(1),  #
    )
    return c


def matmul_tma_set_block_size_hook(nargs):
    EPILOGUE_SUBTILE = nargs.get("EPILOGUE_SUBTILE", False)
    BLOCK_M = nargs["BLOCK_SIZE_M"]
    BLOCK_N = nargs["BLOCK_SIZE_N"]
    BLOCK_K = nargs["BLOCK_SIZE_K"]
    nargs["a_desc"].block_shape = [BLOCK_M, BLOCK_K]
    nargs["b_desc"].block_shape = [BLOCK_N, BLOCK_K]
    if EPILOGUE_SUBTILE:
        nargs["c_desc"].block_shape = [BLOCK_M, BLOCK_N // 2]
    else:
        nargs["c_desc"].block_shape = [BLOCK_M, BLOCK_N]


@triton.autotune(
    configs=matmul_get_configs(pre_hook=matmul_tma_set_block_size_hook),
    key=["M", "N", "K", "WARP_SPECIALIZE"],
)
@triton.jit
def matmul_kernel_tma(
        a_desc,
        b_desc,
        c_desc,  #
        M,
        N,
        K,  #
        BLOCK_SIZE_M: tl.constexpr,  #
        BLOCK_SIZE_N: tl.constexpr,  #
        BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
        FP8_OUTPUT: tl.constexpr,  #
        WARP_SPECIALIZE: tl.constexpr,  #
):
    dtype = tl.float8e4nv if FP8_OUTPUT else tl.float16

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)

    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in tl.range(k_tiles, warp_specialize=WARP_SPECIALIZE):
        offs_k = k * BLOCK_SIZE_K
        a = a_desc.load([offs_am, offs_k])
        b = b_desc.load([offs_bn, offs_k])
        accumulator = tl.dot(a, b.T, accumulator)

    c = accumulator.to(dtype)

    offs_cm = pid_m * BLOCK_SIZE_M
    offs_cn = pid_n * BLOCK_SIZE_N
    c_desc.store([offs_cm, offs_cn], c)


def matmul_tma(a, b, warp_specialize: bool):
    # Check constraints.
    assert a.shape[1] == b.shape[
        1], "Incompatible dimensions"  # b is transposed
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype

    c = torch.empty((M, N), device=a.device, dtype=dtype)

    # A dummy block value that will be overwritten when we have the real
    # block size
    dummy_block = [1, 1]
    a_desc = TensorDescriptor.from_tensor(a, dummy_block)
    b_desc = TensorDescriptor.from_tensor(b, dummy_block)
    c_desc = TensorDescriptor.from_tensor(c, dummy_block)

    def grid(META):
        BLOCK_M = META["BLOCK_SIZE_M"]
        BLOCK_N = META["BLOCK_SIZE_N"]
        return (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), )

    launch(
        matmul_kernel_tma,
        grid,
        a_desc,
        b_desc,
        c_desc,  #
        M,
        N,
        K,  #
        FP8_OUTPUT=dtype == torch.float8_e4m3fn,  #
        WARP_SPECIALIZE=warp_specialize,  #
    )
    return c


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.autotune(
    configs=matmul_get_configs(),
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel_persistent(
        a_ptr,
        b_ptr,
        c_ptr,  #
        M,
        N,
        K,  #
        stride_am,
        stride_ak,  #
        stride_bk,
        stride_bn,  #
        stride_cm,
        stride_cn,  #
        BLOCK_SIZE_M: tl.constexpr,  #
        BLOCK_SIZE_N: tl.constexpr,  #
        BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
        NUM_SMS: tl.constexpr,  #
):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    # NOTE: There is currently a bug in blackwell pipelining that means it
    # can't handle a value being used in both the prologue and epilogue, so
    # we duplicate the counters as a work-around.
    tile_id_c = start_pid - NUM_SMS

    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m,
                                    GROUP_SIZE_M, NUM_SMS)
        start_m = pid_m * BLOCK_SIZE_M
        start_n = pid_n * BLOCK_SIZE_N
        offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
        offs_am = tl.where(offs_am < M, offs_am, 0)
        offs_bn = tl.where(offs_bn < N, offs_bn, 0)
        offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M),
                                    BLOCK_SIZE_M)
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N),
                                    BLOCK_SIZE_N)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + (offs_am[:, None] * stride_am +
                              offs_k[None, :] * stride_ak)
            b_ptrs = b_ptr + (offs_k[:, None] * stride_bk +
                              offs_bn[None, :] * stride_bn)

            a = tl.load(a_ptrs,
                        mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K,
                        other=0.0)
            b = tl.load(b_ptrs,
                        mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K,
                        other=0.0)
            accumulator = tl.dot(a, b, accumulator)

        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(tile_id_c, num_pid_in_group, num_pid_m,
                                    GROUP_SIZE_M, NUM_SMS)
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[
            None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        if (c_ptr.dtype.element_ty == tl.float8e4nv):
            c = accumulator.to(tl.float8e4nv)
        else:
            c = accumulator.to(tl.float16)
        tl.store(c_ptrs, c, mask=c_mask)


def num_sms() -> int:
    return torch.cuda.get_device_properties(DEVICE).multi_processor_count


def matmul_persistent(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    NUM_SMS = num_sms()
    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=dtype)

    # 1D launch kernel where each block gets its own program.
    def grid(META):
        return (min(
            NUM_SMS,
            triton.cdiv(M, META["BLOCK_SIZE_M"]) *
            triton.cdiv(N, META["BLOCK_SIZE_N"])), )

    launch(
        matmul_kernel_persistent,
        grid,
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
        c.stride(1),  #
        NUM_SMS=NUM_SMS,  #
    )
    return c


def matmul_tma_persistent_get_configs(pre_hook=None):
    return [
        triton.Config(
            {
                'BLOCK_SIZE_M': BM,
                'BLOCK_SIZE_N': BN,
                "BLOCK_SIZE_K": BK,
                "GROUP_SIZE_M": 8,
                "EPILOGUE_SUBTILE": SUBTILE
            },
            num_stages=s,
            num_warps=w,
            pre_hook=pre_hook)  #
        for BM in [128]  #
        for BN in [128, 256]  #
        for BK in [64, 128]  #
        for s in ([2, 3, 4])  #
        for w in [4, 8]  #
        for SUBTILE in [True, False]  #
    ]


@triton.autotune(
    configs=matmul_tma_persistent_get_configs(
        pre_hook=matmul_tma_set_block_size_hook),
    key=["M", "N", "K", "WARP_SPECIALIZE"],
)
@triton.jit
def matmul_kernel_tma_persistent(
        a_desc,
        b_desc,
        c_desc,  #
        M,
        N,
        K,  #
        BLOCK_SIZE_M: tl.constexpr,  #
        BLOCK_SIZE_N: tl.constexpr,  #
        BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
        FP8_OUTPUT: tl.constexpr,  #
        EPILOGUE_SUBTILE: tl.constexpr,  #
        NUM_SMS: tl.constexpr,  #
        WARP_SPECIALIZE: tl.constexpr,  #
):
    dtype = tl.float8e4nv if FP8_OUTPUT else tl.float16
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    # Enable warp specialization to leverage async warp scheduling in the
    # GPU.
    # FIXME: This only works on Blackwell right now. On older GPUs, this will
    # use software pipelining.
    for tile_id in tl.range(start_pid,
                            num_tiles,
                            NUM_SMS,
                            flatten=True,
                            warp_specialize=WARP_SPECIALIZE):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m,
                                    GROUP_SIZE_M, NUM_SMS)
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)

        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(tile_id_c, num_pid_in_group, num_pid_m,
                                    GROUP_SIZE_M, NUM_SMS)
        offs_am_c = pid_m * BLOCK_SIZE_M
        offs_bn_c = pid_n * BLOCK_SIZE_N

        # Epilogue subtiling is a technique to break our computation and
        # stores into multiple pieces. By subtiling we can reduce shared
        # memory consumption by the epilogue and instead use that memory to
        # increase our stage count. In this case we partition the
        # accumulator into 2 BLOCK_SIZE_M x BLOCK_SIZE_N // 2 tensors
        if EPILOGUE_SUBTILE:
            acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
            acc = tl.permute(acc, (0, 2, 1))
            acc0, acc1 = tl.split(acc)
            c0 = acc0.to(dtype)
            c_desc.store([offs_am_c, offs_bn_c], c0)
            c1 = acc1.to(dtype)
            c_desc.store([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2], c1)
        else:
            accumulator = accumulator.to(dtype)
            c_desc.store([offs_am_c, offs_bn_c], accumulator)


# Blackwell-only CLC scheduling keeps the full logical grid. The clc=True
# launch option makes the compiler wrap this one-tile kernel in the persistent
# CLC scheduling loop.
@triton.autotune(
    configs=matmul_tma_persistent_get_configs(
        pre_hook=matmul_tma_set_block_size_hook),
    key=["M", "N", "K", "WARP_SPECIALIZE"],
)
@triton.jit
def matmul_kernel_tma_clc(
        a_desc,
        b_desc,
        c_desc,  #
        M,
        N,
        K,  #
        BLOCK_SIZE_M: tl.constexpr,  #
        BLOCK_SIZE_N: tl.constexpr,  #
        BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
        FP8_OUTPUT: tl.constexpr,  #
        EPILOGUE_SUBTILE: tl.constexpr,  #
        WARP_SPECIALIZE: tl.constexpr,  #
):
    dtype = tl.float8e4nv if FP8_OUTPUT else tl.float16

    tile_id = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)

    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for ki in tl.range(k_tiles, warp_specialize=WARP_SPECIALIZE):
        offs_k = ki * BLOCK_SIZE_K
        a = a_desc.load([offs_am, offs_k])
        b = b_desc.load([offs_bn, offs_k])
        accumulator = tl.dot(a, b.T, accumulator)

    # Epilogue subtiling is a technique to break our computation and stores
    # into multiple pieces. By subtiling we can reduce shared memory
    # consumption by the epilogue and instead use that memory to increase our
    # stage count. In this case we partition the accumulator into 2
    # BLOCK_SIZE_M x BLOCK_SIZE_N // 2 tensors
    if EPILOGUE_SUBTILE:
        acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
        acc = tl.permute(acc, (0, 2, 1))
        acc0, acc1 = tl.split(acc)
        c0 = acc0.to(dtype)
        c_desc.store([offs_am, offs_bn], c0)
        c1 = acc1.to(dtype)
        c_desc.store([offs_am, offs_bn + BLOCK_SIZE_N // 2], c1)
    else:
        accumulator = accumulator.to(dtype)
        c_desc.store([offs_am, offs_bn], accumulator)


def matmul_tma_clc(a, b, warp_specialize: bool):
    assert HAS_TMA_CLC, "CLC TMA requires an NVIDIA SM100+ GPU with tensor descriptor support"
    assert a.shape[1] == b.shape[
        1], "Incompatible dimensions"  # b is transposed
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype
    c = torch.empty((M, N), device=a.device, dtype=dtype)

    dummy_block = [1, 1]
    a_desc = TensorDescriptor.from_tensor(a, dummy_block)
    b_desc = TensorDescriptor.from_tensor(b, dummy_block)
    c_desc = TensorDescriptor.from_tensor(c, dummy_block)

    def grid(META):
        return (triton.cdiv(M, META["BLOCK_SIZE_M"]) *
                triton.cdiv(N, META["BLOCK_SIZE_N"]), )

    launch(
        matmul_kernel_tma_clc,
        grid,
        a_desc,
        b_desc,
        c_desc,  #
        M,
        N,
        K,  #
        FP8_OUTPUT=dtype == torch.float8_e4m3fn,  #
        WARP_SPECIALIZE=warp_specialize,  #
        clc=True,  #
    )
    return c


def matmul_tma_persistent(a, b, warp_specialize: bool):
    # Check constraints.
    assert a.shape[1] == b.shape[
        1], "Incompatible dimensions"  # b is transposed
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype

    c = torch.empty((M, N), device=a.device, dtype=dtype)

    NUM_SMS = num_sms()

    # A dummy block value that will be overwritten when we have the real
    # block size
    dummy_block = [1, 1]
    a_desc = TensorDescriptor.from_tensor(a, dummy_block)
    b_desc = TensorDescriptor.from_tensor(b, dummy_block)
    c_desc = TensorDescriptor.from_tensor(c, dummy_block)

    def grid(META):
        BLOCK_M = META["BLOCK_SIZE_M"]
        BLOCK_N = META["BLOCK_SIZE_N"]
        return (min(
            NUM_SMS,
            triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),
        ), )

    launch(
        matmul_kernel_tma_persistent,
        grid,
        a_desc,
        b_desc,
        c_desc,  #
        M,
        N,
        K,  #
        FP8_OUTPUT=dtype == torch.float8_e4m3fn,  #
        NUM_SMS=NUM_SMS,  #
        WARP_SPECIALIZE=warp_specialize,  #
    )
    return c


def prune_invalid_configs(configs, named_args, **kwargs):
    FLATTEN = kwargs["FLATTEN"]
    # Filter out configs where EPILOGUE_SUBTILE is true and HOPPER is true
    return [
        conf for conf in configs
        if not (conf.kwargs.get("EPILOGUE_SUBTILE", True) and FLATTEN is False)
    ]


@triton.autotune(
    configs=matmul_tma_persistent_get_configs(),
    key=["M", "N", "K", "WARP_SPECIALIZE", "FLATTEN"],
    prune_configs_by={'early_config_prune': prune_invalid_configs})
@triton.jit
def matmul_kernel_descriptor_persistent(
    a_ptr,
    b_ptr,
    c_ptr,  #
    M,
    N,
    K,  #
    BLOCK_SIZE_M: tl.constexpr,  #
    BLOCK_SIZE_N: tl.constexpr,  #
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    EPILOGUE_SUBTILE: tl.constexpr,  #
    NUM_SMS: tl.constexpr,  #
    WARP_SPECIALIZE: tl.constexpr,  #
    FLATTEN: tl.constexpr,
):
    # Matmul using TMA and device-side descriptor creation
    dtype = c_ptr.dtype.element_ty
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[N, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[
            BLOCK_SIZE_M,
            BLOCK_SIZE_N if not EPILOGUE_SUBTILE else BLOCK_SIZE_N // 2
        ],
    )

    # tile_id_c is used in the epilogue to break the dependency between
    # the prologue and the epilogue
    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid,
                            num_tiles,
                            NUM_SMS,
                            flatten=FLATTEN,
                            warp_specialize=WARP_SPECIALIZE):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m,
                                    GROUP_SIZE_M, NUM_SMS)
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)

        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(tile_id_c, num_pid_in_group, num_pid_m,
                                    GROUP_SIZE_M, NUM_SMS)
        offs_cm = pid_m * BLOCK_SIZE_M
        offs_cn = pid_n * BLOCK_SIZE_N

        if EPILOGUE_SUBTILE:
            acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
            acc = tl.permute(acc, (0, 2, 1))
            acc0, acc1 = tl.split(acc)
            c0 = acc0.to(dtype)
            c_desc.store([offs_cm, offs_cn], c0)
            c1 = acc1.to(dtype)
            c_desc.store([offs_cm, offs_cn + BLOCK_SIZE_N // 2], c1)
        else:
            c = accumulator.to(dtype)
            c_desc.store([offs_cm, offs_cn], c)


def matmul_descriptor_persistent(a, b, warp_specialize: bool):
    # Check constraints.
    assert a.shape[1] == b.shape[
        1], "Incompatible dimensions"  # b is transposed
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype

    c = torch.empty((M, N), device=a.device, dtype=dtype)
    NUM_SMS = num_sms()

    # TMA descriptors require a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    # Hopper warpspec doesn't work with flatten
    flatten = False if (warp_specialize and is_hopper()) else True

    def grid(META):
        return (min(
            NUM_SMS,
            triton.cdiv(M, META["BLOCK_SIZE_M"]) *
            triton.cdiv(N, META["BLOCK_SIZE_N"])), )

    launch(
        matmul_kernel_descriptor_persistent,
        grid,
        a,
        b,
        c,  #
        M,
        N,
        K,  #
        NUM_SMS=NUM_SMS,  #
        WARP_SPECIALIZE=warp_specialize,  #
        FLATTEN=flatten,
    )
    return c


def device_blas_matmul(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[
        1], "Incompatible dimensions"  # b is transposed
    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype
    c = torch.empty((M, N), device=a.device, dtype=dtype)
    device_blas.matmul(a, b, c)
    return c


def torch_matmul(a, b):
    return torch.matmul(a, b.T)


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------


@dataclass
class Variant:
    """A way of computing ``C = A @ B``; ``fn(a, b)`` with ``b`` of shape ``(N, K)``."""

    name: str
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    kernel: Any = None  # the autotuner behind it, if a Triton kernel
    enabled: bool = True


def variants(dtype) -> List[Variant]:
    """The tutorial's benchmark line-up, in its order."""
    result = [
        Variant(device_blas_name(),
                device_blas_matmul,
                enabled=device_blas is not None),
        Variant("torch", torch_matmul, enabled=dtype == torch.float16),
        Variant("naive", lambda a, b: matmul(a, b.T), matmul_kernel),
        Variant("persistent", lambda a, b: matmul_persistent(a, b.T),
                matmul_kernel_persistent),
    ]
    warp_specialize = [False, True] if HAS_WARP_SPECIALIZE else [False]
    for ws in warp_specialize:
        ws_str = "_ws" if ws else ""
        # disable on-host warpspec on Hopper
        host_ok = HAS_HOST_TENSOR_DESC and not (is_hopper() and ws)
        result += [
            Variant(f"tma{ws_str}",
                    partial(matmul_tma, warp_specialize=ws),
                    matmul_kernel_tma,
                    enabled=host_ok),
            Variant(f"tma_persistent{ws_str}",
                    partial(matmul_tma_persistent, warp_specialize=ws),
                    matmul_kernel_tma_persistent,
                    enabled=host_ok),
            Variant(f"clc_tma{ws_str}",
                    partial(matmul_tma_clc, warp_specialize=ws),
                    matmul_kernel_tma_clc,
                    enabled=host_ok and HAS_TMA_CLC),
            Variant(f"descriptor_persistent{ws_str}",
                    partial(matmul_descriptor_persistent, warp_specialize=ws),
                    matmul_kernel_descriptor_persistent,
                    enabled=HAS_TENSOR_DESC),
        ]
    return result


def config_label(config: Optional[triton.Config]) -> str:
    if config is None:
        return ""
    kw = config.kwargs
    label = (f"{kw['BLOCK_SIZE_M']}x{kw['BLOCK_SIZE_N']}x{kw['BLOCK_SIZE_K']}"
             f" s{config.num_stages} w{config.num_warps}")
    if kw.get("EPILOGUE_SUBTILE"):
        label += " subtile"
    return label


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def make_inputs(M: int, N: int, K: int,
                dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16).to(dtype)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16).to(dtype)
    b = b.T.contiguous()  # (N, K): b is pre-transposed, as in the tutorial
    return a, b


def bench(M: int, N: int, K: int, dtype) -> List[Dict[str, Any]]:
    """Run, validate, evaluate and time every variant on one problem."""
    a, b = make_inputs(M, N, K, dtype)
    elem = a.element_size()
    analytic_flops = 2 * M * N * K
    analytic_bytes = elem * (M * K + N * K + M * N)  # the tutorial's count
    expect: Optional[torch.Tensor] = None

    rows: List[Dict[str, Any]] = []
    for variant in variants(dtype):
        row: Dict[str, Any] = {
            "M":
            M,
            "N":
            N,
            "K":
            K,
            "variant":
            variant.name,
            "kernel":
            getattr(getattr(variant.kernel, "base_fn", None), "__name__", ""),
            "analytic_flops":
            analytic_flops,
            "analytic_bytes":
            analytic_bytes,
            "analytic_intensity":
            analytic_flops / analytic_bytes,
        }
        rows.append(row)
        if not variant.enabled:
            row["error"] = "unsupported"
            print(f"K={K:<6} {variant.name:<26} unsupported here")
            continue
        try:
            # The first call autotunes/compiles; record a steady-state call.
            variant.fn(a, b)
            c, launches = tai.record(lambda: variant.fn(a, b))
        except Exception as exc:  # noqa: BLE001 - report and move on
            row["error"] = f"{type(exc).__name__}: {exc}"[:200]
            print(f"K={K:<6} {variant.name:<26} failed: {row['error']}")
            continue
        if expect is None:
            expect = c.to(torch.float16)
            row["error"] = ""
        else:
            ok = torch.allclose(expect, c.to(expect.dtype), atol=1.0)
            row["error"] = "" if ok else "mismatch"
        ms = triton.testing.do_bench(lambda: variant.fn(a, b))
        row["ms"] = ms
        row["analytic_tflops"] = analytic_flops * 1e-12 / (ms * 1e-3)
        if variant.kernel is not None:
            row["config"] = config_label(variant.kernel.best_config)
            if launches:
                work = tai.Work.of(launches)
                row.update({
                    "launch": launches[-1],
                    "num_programs": work.num_programs,
                    "flops": work.flops,
                    "bytes": work.bytes,
                    "intensity": work.intensity,
                    "bytes_ratio": work.bytes / analytic_bytes,
                    "tflops": work.tflops(ms),
                    "gbps": work.gbps(ms),
                })
        if "tflops" in row:
            print(f"K={K:<6} {variant.name:<26} {ms:8.3f} ms  "
                  f"{row['tflops']:6.1f} TFLOP/s  {row['gbps']:7.1f} GB/s  "
                  f"intensity={row['intensity']:6.1f} FLOP/B  "
                  f"({row['bytes_ratio']:4.1f}x analytic bytes)  "
                  f"[{row['config']}, {row['num_programs']} programs]"
                  f"{'  ' + row['error'] if row['error'] else ''}")
        else:
            print(f"K={K:<6} {variant.name:<26} {ms:8.3f} ms  "
                  f"{row['analytic_tflops']:6.1f} TFLOP/s (analytic)"
                  f"{'  ' + row['error'] if row['error'] else ''}")
    return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_equations(rows: List[Dict[str, Any]]) -> None:
    """Print the pass equations of each kernel variant (last problem size).

    ``args[i]`` symbols are resolved to parameter names and the folded
    ``constexpr`` / specialized parameters are listed. The persistent
    kernels show the ``program_id``-dependent trip count of their tile loop.
    """
    launches = [row["launch"] for row in rows if row.get("launch") is not None]
    for kernel_launch in launches:
        # The listener gathered the same kernel function as it was compiled.
        assert kernel_launch.name in LISTENER, (
            f"listener did not see {kernel_launch.name}")
    print()
    tai.print_equations(launches)


def plot(rows: List[Dict[str, Any]], path: str, dtype) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = sorted({r["K"] for r in rows})
    names = list(dict.fromkeys(r["variant"] for r in rows))
    by_variant = {
        name:
        sorted((r for r in rows if r["variant"] == name), key=lambda r: r["K"])
        for name in names
    }
    M, N = rows[0]["M"], rows[0]["N"]
    fig, (ax_int, ax_perf) = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax in (ax_int, ax_perf):  # more variants than default colors
        ax.set_prop_cycle(color=plt.get_cmap("tab20").colors)

    if len(ks) > 1:
        for name in names:
            group = [r for r in by_variant[name] if "ms" in r]
            if not group:
                continue
            style = "--" if not any("tflops" in r for r in group) else "-"
            ax_perf.plot(
                [r["K"] for r in group],
                [r.get("tflops", r["analytic_tflops"]) for r in group],
                style,
                marker="o",
                label=name)
            group = [r for r in group if "intensity" in r]
            if group:
                ax_int.plot([r["K"] for r in group],
                            [r["intensity"] for r in group],
                            marker="o",
                            label=name)
        analytic = [
            next(r for r in rows if r["K"] == k)["analytic_intensity"]
            for k in ks
        ]
        ax_int.plot(ks, analytic, "k--", label="analytic elem*(MK+NK+MN)")
        for ax in (ax_int, ax_perf):
            ax.set_xscale("log", base=2)
            ax.set_xlabel("K")
        ax_int.set_yscale("log", base=2)
        ax_perf.legend(fontsize=8)
    else:
        xs = range(len(names))
        perf = [
            by_variant[n][0].get("tflops",
                                 by_variant[n][0].get("analytic_tflops", 0.0))
            for n in names
        ]
        colors = [
            "tab:gray" if "tflops" not in by_variant[n][0] else "tab:blue"
            for n in names
        ]
        ax_perf.bar(xs, perf, color=colors)
        ax_int.bar(xs, [by_variant[n][0].get("intensity", 0.0) for n in names],
                   color="tab:green")
        ax_int.axhline(rows[0]["analytic_intensity"],
                       color="k",
                       linestyle="--",
                       label="analytic elem*(MK+NK+MN)")
        for ax in (ax_int, ax_perf):
            ax.set_xticks(list(xs))
            ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
        ax_perf.set_xlabel("gray: library reference, analytic 2MNK FLOPs")
    ax_int.set_ylabel("Arithmetic intensity (FLOP/byte)")
    ax_int.set_title("Requested work per variant (from the pass)")
    ax_int.legend(fontsize=8)
    ax_perf.set_ylabel("Achieved TFLOP/s")
    ax_perf.set_title("Performance (dashed: library reference)")
    for ax in (ax_int, ax_perf):
        ax.grid(True, which="both", alpha=0.3)

    device = torch.cuda.get_device_name(DEVICE) if is_cuda() else str(DEVICE)
    fig.suptitle(f"Persistent matmul variants ({str(dtype).split('.')[-1]}, "
                 f"M=N={M if M == N else f'{M}/{N}'}), work from "
                 f"triton-arithmetic-intensity -- {device}")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"\nSaved plot to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the tutorial's matmul kernel variants on one "
        "problem, with work measured by triton-arithmetic-intensity.")
    parser.add_argument("-K", type=int, required=False, default=None)
    parser.add_argument("--K_range", type=int, nargs=2)
    parser.add_argument("--K_step", type=int, default=128)
    parser.add_argument("-M", type=int, default=8192)
    parser.add_argument("-N", type=int, default=8192)
    parser.add_argument("--prec",
                        type=str,
                        choices=["fp8", "fp16"],
                        default="fp16")
    parser.add_argument("--quick",
                        action="store_true",
                        help="M=N=4096 and a single K")
    parser.add_argument("--save-path",
                        default=".",
                        help="directory for the plot and CSV")
    args = parser.parse_args()

    if args.prec == 'fp8' and (not hasattr(torch, "float8_e4m3fn")
                               or not is_cuda()):
        parser.error("--prec fp8 requires CUDA with fp8 support")
    dtype = torch.float8_e4m3fn if args.prec == 'fp8' else torch.float16

    if args.quick:
        args.M = args.N = 4096
        if args.K is None and args.K_range is None:
            args.K = 512
    if args.K is not None and args.K_range is None:
        args.K_range = [args.K, args.K]
    if args.K_range is None:
        args.K_range = [128, 1024]
    ks = list(range(args.K_range[0], args.K_range[1] + 1, args.K_step))

    torch.manual_seed(0)
    os.makedirs(args.save_path, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for K in ks:
        rows.extend(bench(args.M, args.N, K, dtype))
    print_equations(rows)

    import pandas as pd
    df = pd.DataFrame([{
        k: v
        for k, v in row.items() if k not in ("launch", "M", "N")
    } for row in rows])
    print("\n" + df.to_string(index=False, float_format=lambda f: f"{f:.3f}"))
    stem = "persistent-matmul-arithmetic-intensity" + ("-fp8" if args.prec
                                                       == "fp8" else "")
    csv_path = os.path.join(args.save_path, stem + ".csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved data to {csv_path}")
    plot(rows, os.path.join(args.save_path, stem + ".png"), dtype)


if __name__ == "__main__":
    main()
