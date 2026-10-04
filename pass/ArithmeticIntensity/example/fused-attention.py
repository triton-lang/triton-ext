"""
Fused Attention, instrumented with ``triton_arithmetic_intensity``
==================================================================

This is Triton's ``python/tutorials/06-fused-attention.py`` (the Flash
Attention v2 forward and backward kernels, credits: OpenAI kernel team) with
the hand-written FLOP formula of its benchmark replaced by the work computed by
the ``triton-arithmetic-intensity`` compiler pass:

* Importing :mod:`triton_arithmetic_intensity` installs the pass at the end of
  the ``ttir`` stage, so every kernel compiled below carries per-argument
  ``tai.load_bytes`` / ``tai.store_bytes`` / ``tai.op_count`` equations in its
  metadata.
* Each kernel launch goes through :func:`triton_arithmetic_intensity.launch`
  and is recorded (:func:`triton_arithmetic_intensity.record`) so that the
  equations can be evaluated for the exact grid and arguments that were used
  -- including the autotuned ``BLOCK_M`` / ``BLOCK_N``, which are folded into
  the compiled kernel and therefore into its equations. The forward and
  backward passes launch several kernels each; their work is summed with
  :class:`triton_arithmetic_intensity.Work`.
* An :class:`~triton_arithmetic_intensity.ArithmeticIntensityListener`
  gathers the symbolic equations per kernel function, which are printed
  alongside the evaluated numbers.

Running the script benchmarks the forward and backward passes over a range of
sequence lengths and head dimensions, prints a table with the pass-derived
FLOPs, bytes, arithmetic intensity and achieved throughput (next to the
tutorial's analytic FLOP count), and saves a plot::

    python fused-attention.py                # full autotune, like the tutorial
    python fused-attention.py --quick        # single config, fewer sizes
    python fused-attention.py --save-path out/

Credits: OpenAI kernel team (kernels), Tri Dao (Flash Attention v2,
https://tridao.me/publications/flash2/flash2.pdf).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional

import torch

import triton
import triton.language as tl
import triton_arithmetic_intensity as tai
from triton.tools.tensor_descriptor import TensorDescriptor
from triton_arithmetic_intensity import launch

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# Gather the symbolic equations of every kernel function compiled from here on.
LISTENER = tai.enable()

# `--quick` must be known before the autotune configs below are built.
QUICK = "--quick" in sys.argv or "PYTEST_VERSION" in os.environ


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def supports_host_descriptor():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


def is_blackwell():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 10


def is_hopper():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 9


# ---------------------------------------------------------------------------
# Kernels (unchanged from the tutorial)
# ---------------------------------------------------------------------------


@triton.jit
def _attn_fwd_inner(
        acc,
        l_i,
        m_i,
        q,  #
        desc_k,
        desc_v,  #
        off_hz,
        dtype: tl.constexpr,
        start_m,
        qk_scale,  #
        BLOCK_M: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_N: tl.constexpr,  #
        STAGE: tl.constexpr,
        offs_m: tl.constexpr,
        offs_n: tl.constexpr,  #
        N_CTX: tl.constexpr,
        warp_specialize: tl.constexpr,
        IS_HOPPER: tl.constexpr):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    offset_y = off_hz * N_CTX
    offsetk_y = offset_y + lo
    offsetv_y = offset_y + lo
    # loop over k, v and update accumulator
    for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=warp_specialize):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = desc_k.load([offsetk_y, 0]).T
        qk = tl.dot(q, k)
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        # -- compute correction factor
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        # -- update output accumulator --
        if not IS_HOPPER and warp_specialize and BLOCK_M == 128 and HEAD_DIM == 128:
            BM: tl.constexpr = acc.shape[0]
            BN: tl.constexpr = acc.shape[1]
            acc0, acc1 = acc.reshape([BM, 2, BN // 2]).permute(0, 2, 1).split()
            acc0 = acc0 * alpha[:, None]
            acc1 = acc1 * alpha[:, None]
            acc = tl.join(acc0, acc1).permute(0, 2, 1).reshape([BM, BN])
        else:
            acc = acc * alpha[:, None]
        # prepare p and v for the dot
        if dtype == tl.float8e5:
            v = desc_v.load([off_hz * HEAD_DIM, start_n]).T
        else:
            v = desc_v.load([offsetv_y, 0])
        p = p.to(dtype)
        # note that this non transposed v for FP8 is only supported on Blackwell
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        # place this at the end of the loop to reduce register pressure
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        offsetk_y += BLOCK_N
        offsetv_y += BLOCK_N
    return acc, l_i, m_i


def _host_descriptor_pre_hook(nargs):
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    HEAD_DIM = nargs["HEAD_DIM"]
    if not isinstance(nargs["desc_q"], TensorDescriptor):
        return
    nargs["desc_q"].block_shape = [BLOCK_M, HEAD_DIM]
    if nargs["FP8_OUTPUT"]:
        nargs["desc_v"].block_shape = [HEAD_DIM, BLOCK_N]
    else:
        nargs["desc_v"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_k"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_o"].block_shape = [BLOCK_M, HEAD_DIM]


if is_hip():
    NUM_STAGES_OPTIONS = [1]
elif supports_host_descriptor():
    NUM_STAGES_OPTIONS = [2, 3, 4]
else:
    NUM_STAGES_OPTIONS = [2, 3, 4]

configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w, pre_hook=_host_descriptor_pre_hook) \
    for BM in [64, 128]\
    for BN in [32, 64, 128]\
    for s in NUM_STAGES_OPTIONS \
    for w in [4, 8]\
]
if QUICK:
    # Use a single config in testing for reproducibility
    configs = [
        triton.Config(dict(BLOCK_M=128, BLOCK_N=64),
                      num_stages=2,
                      num_warps=4,
                      pre_hook=_host_descriptor_pre_hook),
    ]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    return not (is_hopper() and BLOCK_M * BLOCK_N < 128 * 128
                and conf.num_warps == 8)


def prune_invalid_configs(configs, named_args, **kwargs):
    N_CTX = kwargs["N_CTX"]
    STAGE = kwargs["STAGE"]

    # Filter out configs where BLOCK_M > N_CTX
    # Filter out configs where BLOCK_M < BLOCK_N when causal is True
    return [
        conf for conf in configs
        if conf.kwargs.get("BLOCK_M", 0) <= N_CTX and (conf.kwargs.get(
            "BLOCK_M", 0) >= conf.kwargs.get("BLOCK_N", 0) or STAGE == 1)
    ]


@triton.jit
def _maybe_make_tensor_desc(desc_or_ptr, shape, strides, block_shape):
    if isinstance(desc_or_ptr, tl.tensor_descriptor):
        return desc_or_ptr
    else:
        return tl.make_tensor_descriptor(desc_or_ptr, shape, strides,
                                         block_shape)


@triton.autotune(
    configs=list(filter(keep, configs)),
    key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT", "warp_specialize"],
    prune_configs_by={'early_config_prune': prune_invalid_configs})
@triton.jit
def _attn_fwd(
        sm_scale,
        M,  #
        Z,
        H,
        desc_q,
        desc_k,
        desc_v,
        desc_o,
        N_CTX,  #
        HEAD_DIM: tl.constexpr,  #
        BLOCK_M: tl.constexpr,  #
        BLOCK_N: tl.constexpr,  #
        FP8_OUTPUT: tl.constexpr,  #
        STAGE: tl.constexpr,  #
        warp_specialize: tl.constexpr,  #
        IS_HOPPER: tl.constexpr,  #
):
    dtype = tl.float8e5 if FP8_OUTPUT else tl.float16
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    y_dim = Z * H * N_CTX
    desc_q = _maybe_make_tensor_desc(desc_q,
                                     shape=[y_dim, HEAD_DIM],
                                     strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_M, HEAD_DIM])
    if FP8_OUTPUT:
        desc_v = _maybe_make_tensor_desc(desc_v,
                                         shape=[Z * H * HEAD_DIM, N_CTX],
                                         strides=[N_CTX, 1],
                                         block_shape=[HEAD_DIM, BLOCK_N])
    else:
        desc_v = _maybe_make_tensor_desc(desc_v,
                                         shape=[y_dim, HEAD_DIM],
                                         strides=[HEAD_DIM, 1],
                                         block_shape=[BLOCK_N, HEAD_DIM])
    desc_k = _maybe_make_tensor_desc(desc_k,
                                     shape=[y_dim, HEAD_DIM],
                                     strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_N, HEAD_DIM])
    desc_o = _maybe_make_tensor_desc(desc_o,
                                     shape=[y_dim, HEAD_DIM],
                                     strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_M, HEAD_DIM])

    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = desc_q.load([qo_offset_y, 0])
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,  #
            desc_k,
            desc_v,  #
            off_hz,
            dtype,
            start_m,
            qk_scale,  #
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,  #
            4 - STAGE,
            offs_m,
            offs_n,
            N_CTX,  #
            warp_specialize,
            IS_HOPPER)
    # stage 2: on-band
    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,  #
            desc_k,
            desc_v,  #
            off_hz,
            dtype,
            start_m,
            qk_scale,  #
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,  #
            2,
            offs_m,
            offs_n,
            N_CTX,  #
            warp_specialize,
            IS_HOPPER)
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    desc_o.store([qo_offset_y, 0], acc.to(dtype))


@triton.jit
def _attn_bwd_preprocess(
        O,  # noqa: E741
        DO,  #
        Delta,  #
        Z,
        H,
        N_CTX,  #
        BLOCK_M: tl.constexpr,
        HEAD_DIM: tl.constexpr  #
):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    off_n = tl.arange(0, HEAD_DIM)
    # load
    o = tl.load(O + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM +
                off_n[None, :])
    do = tl.load(DO + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM +
                 off_n[None, :]).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_hz * N_CTX + off_m, delta)


# The main inner-loop logic for computing dK and dV.
@triton.jit
def _attn_bwd_dkdv(
        dk,
        dv,  #
        Q,
        k,
        v,
        sm_scale,  #
        DO,  #
        M,
        D,  #
        # shared by Q/K/V/DO.
    stride_tok,
        stride_d,  #
        H,
        N_CTX,
        BLOCK_M1: tl.constexpr,  #
        BLOCK_N1: tl.constexpr,  #
        HEAD_DIM: tl.constexpr,  #
        # Filled in by the wrapper.
    start_n,
        start_m,
        num_steps,  #
        MASK: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_k = tl.arange(0, HEAD_DIM)
    qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1
    for blk_idx in range(num_steps):
        qT = tl.load(qT_ptrs)
        # Load m before computing qk to reduce pipeline stall.
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        m = tl.load(M + offs_m)
        qkT = tl.dot(k, qT)
        pT = tl.math.exp2(qkT - m[None, :])
        # Autoregressive masking.
        if MASK:
            mask = (offs_m[None, :] >= offs_n[:, None])
            pT = tl.where(mask, pT, 0.0)
        do = tl.load(do_ptrs)
        # Compute dV.
        ppT = pT
        ppT = ppT.to(tl.float16)
        dv += tl.dot(ppT, do)
        # D (= delta) is pre-divided by ds_scale.
        Di = tl.load(D + offs_m)
        # Compute dP and dS.
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
        dsT = pT * (dpT - Di[None, :])
        dsT = dsT.to(tl.float16)
        dk += tl.dot(dsT, tl.trans(qT))
        # Increment pointers.
        curr_m += step_m
        qT_ptrs += step_m * stride_tok
        do_ptrs += step_m * stride_tok
    return dk, dv


# the main inner-loop logic for computing dQ
@triton.jit
def _attn_bwd_dq(
        dq,
        q,
        K,
        V,  #
        do,
        m,
        D,
        # shared by Q/K/V/DO.
        stride_tok,
        stride_d,  #
        H,
        N_CTX,  #
        BLOCK_M2: tl.constexpr,  #
        BLOCK_N2: tl.constexpr,  #
        HEAD_DIM: tl.constexpr,
        # Filled in by the wrapper.
        start_m,
        start_n,
        num_steps,  #
        MASK: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_k = tl.arange(0, HEAD_DIM)
    kT_ptrs = K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    vT_ptrs = V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    # D (= delta) is pre-divided by ds_scale.
    Di = tl.load(D + offs_m)
    # BLOCK_M2 must be a multiple of BLOCK_N2, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n
    step_n = BLOCK_N2
    for blk_idx in range(num_steps):
        kT = tl.load(kT_ptrs)
        vT = tl.load(vT_ptrs)
        qk = tl.dot(q, kT)
        p = tl.math.exp2(qk - m)
        # Autoregressive masking.
        if MASK:
            offs_n = curr_n + tl.arange(0, BLOCK_N2)
            mask = (offs_m[:, None] >= offs_n[None, :])
            p = tl.where(mask, p, 0.0)
        # Compute dP and dS.
        dp = tl.dot(do, vT).to(tl.float32)
        ds = p * (dp - Di[:, None])
        ds = ds.to(tl.float16)
        # Compute dQ.
        # NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
        dq += tl.dot(ds, tl.trans(kT))
        # Increment pointers.
        curr_n += step_n
        kT_ptrs += step_n * stride_tok
        vT_ptrs += step_n * stride_tok
    return dq


@triton.jit
def _attn_bwd(
        Q,
        K,
        V,
        sm_scale,  #
        DO,  #
        DQ,
        DK,
        DV,  #
        M,
        D,
        # shared by Q/K/V/DO.
        stride_z,
        stride_h,
        stride_tok,
        stride_d,  #
        H,
        N_CTX,  #
        BLOCK_M1: tl.constexpr,  #
        BLOCK_N1: tl.constexpr,  #
        BLOCK_M2: tl.constexpr,  #
        BLOCK_N2: tl.constexpr,  #
        BLK_SLICE_FACTOR: tl.constexpr,  #
        HEAD_DIM: tl.constexpr,  #
        CAUSAL: tl.constexpr):
    LN2: tl.constexpr = 0.6931471824645996  # = ln(2)

    bhid = tl.program_id(2)
    off_chz = (bhid * N_CTX).to(tl.int64)
    adj = (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
    pid = tl.program_id(0)

    # offset pointers for batch/head
    Q += adj
    K += adj
    V += adj
    DO += adj
    DQ += adj
    DK += adj
    DV += adj
    M += off_chz
    D += off_chz

    # load scales
    offs_k = tl.arange(0, HEAD_DIM)

    start_n = pid * BLOCK_N1
    start_m = 0

    MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR
    offs_n = start_n + tl.arange(0, BLOCK_N1)

    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

    # load K and V: they stay in SRAM throughout the inner loop.
    k = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
    v = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)

    if CAUSAL:
        start_m = start_n
        num_steps = BLOCK_N1 // MASK_BLOCK_M1
        dk, dv = _attn_bwd_dkdv(
            dk,
            dv,  #
            Q,
            k,
            v,
            sm_scale,  #
            DO,  #
            M,
            D,  #
            stride_tok,
            stride_d,  #
            H,
            N_CTX,  #
            MASK_BLOCK_M1,
            BLOCK_N1,
            HEAD_DIM,  #
            start_n,
            start_m,
            num_steps,  #
            MASK=True,  #
        )

        start_m += num_steps * MASK_BLOCK_M1

    # Compute dK and dV for non-masked blocks.
    num_steps = (N_CTX - start_m) // BLOCK_M1
    dk, dv = _attn_bwd_dkdv(  #
        dk,
        dv,  #
        Q,
        k,
        v,
        sm_scale,  #
        DO,  #
        M,
        D,  #
        stride_tok,
        stride_d,  #
        H,
        N_CTX,  #
        BLOCK_M1,
        BLOCK_N1,
        HEAD_DIM,  #
        start_n,
        start_m,
        num_steps,  #
        MASK=False,  #
    )

    dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dv_ptrs, dv)

    # Write back dK.
    dk *= sm_scale
    dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dk_ptrs, dk)

    # THIS BLOCK DOES DQ:
    start_m = pid * BLOCK_M2
    start_n = 0
    num_steps = N_CTX // BLOCK_N2

    MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR
    offs_m = start_m + tl.arange(0, BLOCK_M2)

    q = tl.load(Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
    dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
    do = tl.load(DO + offs_m[:, None] * stride_tok +
                 offs_k[None, :] * stride_d)

    m = tl.load(M + offs_m)
    m = m[:, None]

    if CAUSAL:
        # Compute dQ for masked (diagonal) blocks.
        # NOTE: This code scans each row of QK^T backward (from right to left,
        # but inside each call to _attn_bwd_dq, from left to right), but that's
        # not due to anything important.  I just wanted to reuse the loop
        # structure for dK & dV above as much as possible.
        end_n = start_m + BLOCK_M2
        num_steps = BLOCK_M2 // MASK_BLOCK_N2
        dq = _attn_bwd_dq(
            dq,
            q,
            K,
            V,  #
            do,
            m,
            D,  #
            stride_tok,
            stride_d,  #
            H,
            N_CTX,  #
            BLOCK_M2,
            MASK_BLOCK_N2,
            HEAD_DIM,  #
            start_m,
            end_n - num_steps * MASK_BLOCK_N2,
            num_steps,  #
            MASK=True,  #
        )
        end_n -= num_steps * MASK_BLOCK_N2
        # stage 2
        num_steps = end_n // BLOCK_N2
        start_n = end_n - num_steps * BLOCK_N2

    dq = _attn_bwd_dq(
        dq,
        q,
        K,
        V,  #
        do,
        m,
        D,  #
        stride_tok,
        stride_d,  #
        H,
        N_CTX,  #
        BLOCK_M2,
        BLOCK_N2,
        HEAD_DIM,  #
        start_m,
        start_n,
        num_steps,  #
        MASK=False,  #
    )
    # Write back dQ.
    dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    dq *= LN2
    tl.store(dq_ptrs, dq)


# ---------------------------------------------------------------------------
# Attention (as in the tutorial, launching through `tai.launch`)
# ---------------------------------------------------------------------------


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, warp_specialize=True):
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        o = torch.empty_like(q)
        stage = 3 if causal else 1
        extra_kern_args = {}
        # Tuning for AMD target
        if is_hip():
            waves_per_eu = 3 if HEAD_DIM_K <= 64 else 2
            extra_kern_args = {
                "waves_per_eu": waves_per_eu,
                "allow_flush_denorm": True
            }

        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]),
                        device=q.device,
                        dtype=torch.float32)
        # Use device_descriptor for Hopper + warpspec.
        if supports_host_descriptor() and not (is_hopper()
                                               and warp_specialize):
            # Note that on Hopper we cannot perform a FP8 dot with a non-transposed second tensor
            y_dim = q.shape[0] * q.shape[1] * q.shape[2]

            dummy_block = [1, 1]
            desc_q = TensorDescriptor(q,
                                      shape=[y_dim, HEAD_DIM_K],
                                      strides=[HEAD_DIM_K, 1],
                                      block_shape=dummy_block)
            if q.dtype == torch.float8_e5m2:
                desc_v = TensorDescriptor(
                    v,
                    shape=[q.shape[0] * q.shape[1] * HEAD_DIM_K, q.shape[2]],
                    strides=[q.shape[2], 1],
                    block_shape=dummy_block)
            else:
                desc_v = TensorDescriptor(v,
                                          shape=[y_dim, HEAD_DIM_K],
                                          strides=[HEAD_DIM_K, 1],
                                          block_shape=dummy_block)
            desc_k = TensorDescriptor(k,
                                      shape=[y_dim, HEAD_DIM_K],
                                      strides=[HEAD_DIM_K, 1],
                                      block_shape=dummy_block)
            desc_o = TensorDescriptor(o,
                                      shape=[y_dim, HEAD_DIM_K],
                                      strides=[HEAD_DIM_K, 1],
                                      block_shape=dummy_block)
        else:
            desc_q = q
            desc_v = v
            desc_k = k
            desc_o = o

        def alloc_fn(size: int, align: int, _):
            return torch.empty(size, dtype=torch.int8, device="cuda")

        triton.set_allocator(alloc_fn)

        def grid(META):
            return (triton.cdiv(q.shape[2],
                                META["BLOCK_M"]), q.shape[0] * q.shape[1], 1)

        ctx.grid = grid
        if is_blackwell() and warp_specialize:
            if HEAD_DIM_K == 128 and q.dtype == torch.float16:
                extra_kern_args["maxnreg"] = 168
            else:
                extra_kern_args["maxnreg"] = 80
        launch(
            _attn_fwd,
            grid,  #
            sm_scale,
            M,  #
            q.shape[0],
            q.shape[1],  #
            desc_q,
            desc_k,
            desc_v,
            desc_o,  #
            N_CTX=q.shape[2],  #
            HEAD_DIM=HEAD_DIM_K,  #
            FP8_OUTPUT=q.dtype == torch.float8_e5m2,  #
            STAGE=stage,  #
            warp_specialize=warp_specialize,  #
            IS_HOPPER=is_hopper(),  #
            **extra_kern_args)

        ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, M = ctx.saved_tensors
        assert do.is_contiguous()
        assert q.stride() == k.stride() == v.stride() == o.stride(
        ) == do.stride()
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        BATCH, N_HEAD, N_CTX = q.shape[:3]
        PRE_BLOCK = 128
        NUM_WARPS, NUM_STAGES = 4, 5
        BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 128, 128, 32
        BLK_SLICE_FACTOR = 2
        RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
        arg_k = k
        arg_k = arg_k * (ctx.sm_scale * RCP_LN2)
        PRE_BLOCK = 128
        assert N_CTX % PRE_BLOCK == 0
        pre_grid = (N_CTX // PRE_BLOCK, BATCH * N_HEAD)
        delta = torch.empty_like(M)
        launch(
            _attn_bwd_preprocess,
            pre_grid,  #
            o,
            do,  #
            delta,  #
            BATCH,
            N_HEAD,
            N_CTX,  #
            BLOCK_M=PRE_BLOCK,
            HEAD_DIM=ctx.HEAD_DIM  #
        )
        grid = (N_CTX // BLOCK_N1, 1, BATCH * N_HEAD)
        launch(
            _attn_bwd,
            grid,  #
            q,
            arg_k,
            v,
            ctx.sm_scale,
            do,
            dq,
            dk,
            dv,  #
            M,
            delta,  #
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),  #
            N_HEAD,
            N_CTX,  #
            BLOCK_M1=BLOCK_M1,
            BLOCK_N1=BLOCK_N1,  #
            BLOCK_M2=BLOCK_M2,
            BLOCK_N2=BLOCK_N2,  #
            BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,  #
            HEAD_DIM=ctx.HEAD_DIM,  #
            num_warps=NUM_WARPS,  #
            num_stages=NUM_STAGES,  #
            CAUSAL=ctx.causal,  #
        )

        return dq, dk, dv, None, None, None, None


attention = _attention.apply

# ---------------------------------------------------------------------------
# Benchmark: pass-derived work vs. the tutorial's analytic FLOP count
# ---------------------------------------------------------------------------


def analytic_flops(BATCH, H, N_CTX, HEAD_DIM, causal, mode) -> float:
    """The hand-written FLOP count from the tutorial's benchmark."""
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    if mode == "bwd":
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    return total_flops


def check_correctness(q, k, v, causal, sm_scale, warp_specialize):
    ref = torch.nn.functional.scaled_dot_product_attention(q,
                                                           k,
                                                           v,
                                                           is_causal=causal,
                                                           scale=sm_scale)
    out = attention(q, k, v, causal, sm_scale, warp_specialize)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=0)


def bench(BATCH,
          H,
          N_CTX,
          HEAD_DIM,
          causal,
          mode,
          warp_specialize=False,
          device=DEVICE) -> Optional[Dict[str, Any]]:
    """Time one configuration and evaluate the work of the kernels it launched.

    Returns ``None`` when the device cannot run the configuration.
    """
    dtype = torch.float16
    q = torch.randn((BATCH, H, N_CTX, HEAD_DIM),
                    dtype=dtype,
                    device=device,
                    requires_grad=True)
    k = torch.randn((BATCH, H, N_CTX, HEAD_DIM),
                    dtype=dtype,
                    device=device,
                    requires_grad=True)
    v = torch.randn((BATCH, H, N_CTX, HEAD_DIM),
                    dtype=dtype,
                    device=device,
                    requires_grad=True)
    sm_scale = 1.3

    def forward():
        return attention(q, k, v, causal, sm_scale, warp_specialize)

    fn = forward
    if mode == "bwd":
        o = forward()
        do = torch.randn_like(o)

        def backward():
            o.backward(do, retain_graph=True)

        fn = backward
    # The first call autotunes/compiles; evaluate the launches of a
    # steady-state call.
    try:
        fn()
    except triton.runtime.errors.OutOfResources as exc:
        # e.g. the tutorial's fixed backward config needs more shared memory
        # than some devices offer for HEAD_DIM=128.
        print(
            f"{mode} d={HEAD_DIM:<3} causal={causal!s:<5} N_CTX={N_CTX:<6} skipped: {exc}"
        )
        return None
    _, launches = tai.record(fn)
    work = tai.Work.of(launches)
    ms = triton.testing.do_bench(fn)
    row = {
        "mode": mode,
        "causal": causal,
        "HEAD_DIM": HEAD_DIM,
        "N_CTX": N_CTX,
        "ms": ms,
        "flops": work.flops,
        "bytes": work.bytes,
        "intensity": work.intensity,
        "tflops": work.tflops(ms),
        "gbps": work.gbps(ms),
        "analytic_flops": analytic_flops(BATCH, H, N_CTX, HEAD_DIM, causal,
                                         mode),
    }
    row["analytic_tflops"] = row["analytic_flops"] * 1e-12 / (ms * 1e-3)
    row["kernels"] = {
        name: (w.flops, w.bytes)
        for name, w in work.per_kernel.items()
    }
    return row


def print_equations(launches: List[tai.KernelLaunch]) -> None:
    """Print the symbolic equations the pass produced for the launched kernels.

    Shows the last launch of each kernel function: the equations (per program,
    in bytes and FLOPs) with ``args[i]`` resolved to parameter names, and the
    ``constexpr`` / specialized parameters they were compiled with.
    """
    for kernel_launch in launches:
        # The listener gathered the same kernel function as it was compiled.
        assert kernel_launch.name in LISTENER, (
            f"listener did not see {kernel_launch.name}")
    print()
    tai.print_equations(launches)


def plot(rows: List[Dict[str, Any]], path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["mode"], row["HEAD_DIM"], row["causal"]),
                          []).append(row)
    for (mode, head_dim, causal), group in sorted(groups.items()):
        group = sorted(group, key=lambda r: r["N_CTX"])
        x = [r["N_CTX"] for r in group]
        label = f"{mode} d={head_dim} causal={causal}"
        style = "-" if mode == "fwd" else "--"
        axes[0].plot(x, [r["intensity"] for r in group],
                     style,
                     marker="o",
                     label=label)
        axes[1].plot(x, [r["tflops"] for r in group],
                     style,
                     marker="o",
                     label=label)
        axes[2].plot(x, [r["gbps"] for r in group],
                     style,
                     marker="o",
                     label=label)
    titles = [
        "Arithmetic intensity (FLOP/byte)", "Achieved TFLOP/s", "Achieved GB/s"
    ]
    for ax, title in zip(axes, titles):
        ax.set_xscale("log", base=2)
        ax.set_xlabel("N_CTX")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_yscale("log", base=2)
    axes[2].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    device = torch.cuda.get_device_name(DEVICE) if is_cuda() else str(DEVICE)
    fig.suptitle(
        f"Fused attention (fp16), work from triton-arithmetic-intensity -- {device}"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"\nSaved plot to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=
        "Benchmark fused attention and report its work as computed by triton-arithmetic-intensity."
    )
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--head-dims", type=int, nargs="+", default=[64, 128])
    parser.add_argument(
        "--n-ctx",
        type=int,
        nargs="+",
        default=None,
        help=
        "sequence lengths (default: 1024..16384, or 1024..4096 with --quick)")
    parser.add_argument("--modes",
                        nargs="+",
                        default=["fwd", "bwd"],
                        choices=["fwd", "bwd"])
    parser.add_argument("--quick",
                        action="store_true",
                        help="single autotune config and fewer sizes")
    parser.add_argument("--save-path",
                        default=".",
                        help="directory for the plot and CSV")
    args = parser.parse_args()
    if args.n_ctx is None:
        args.n_ctx = [2**i for i in range(10, 18 if args.quick else 15)]

    os.makedirs(args.save_path, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for head_dim in args.head_dims:
        for causal in (False, True):
            q = torch.randn((2, 2, 1024, head_dim),
                            dtype=torch.float16,
                            device=DEVICE)
            check_correctness(q, q.clone(), q.clone(), causal, 0.5, False)
            for mode in args.modes:
                for n_ctx in args.n_ctx:
                    row = bench(args.batch, args.heads, n_ctx, head_dim,
                                causal, mode)
                    if row is None:
                        break  # larger sizes need at least as many resources
                    rows.append(row)
                    print(
                        f"{mode} d={head_dim:<3} causal={causal!s:<5} N_CTX={n_ctx:<6} "
                        f"{row['ms']:8.3f} ms  {row['tflops']:7.1f} TFLOP/s  {row['gbps']:7.1f} GB/s  "
                        f"intensity={row['intensity']:8.1f} FLOP/B  "
                        f"(analytic: {row['analytic_tflops']:7.1f} TFLOP/s)")

    print_equations(list(tai.last_launches.values()))

    import pandas as pd
    df = pd.DataFrame([{
        k: v
        for k, v in row.items() if k != "kernels"
    } for row in rows])
    print("\n" + df.to_string(index=False, float_format=lambda f: f"{f:.3f}"))
    csv_path = os.path.join(args.save_path,
                            "fused-attention-arithmetic-intensity.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved data to {csv_path}")
    plot(
        rows,
        os.path.join(args.save_path,
                     "fused-attention-arithmetic-intensity.png"))


if __name__ == "__main__":
    main()
