"""Fuse the preparation the TLX KDA kernels expect their caller to do.

The TLX kernels take l2-normalized Q/K, a materialized per-K log decay, and a
sigmoided beta. Doing that with torch ops costs ten-odd elementwise launches,
each streaming the whole ``[T, H, D]`` tensor through HBM -- at T=16384 that
measured 0.33ms against a 1.20ms kernel, eating the kernel's entire advantage.

This does all of it in one pass. Plain Triton, no TLX: the work is a per-head
reduction plus elementwise math, which needs no async copies or MFMA control.

The l2-norm epsilon matches ``fused_recurrent_kda_packed_decode_kernel``
(``sqrt(sum + 1e-6)``, not ``max(sqrt(sum), eps)``) so the two paths stay
numerically comparable.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _kda_prepare_kernel(
    q,
    k,
    a,
    b,
    A_log,
    dt_bias,
    qn,
    kn,
    g,
    beta_out,
    lower_bound,
    stride_tok: tl.constexpr,
    stride_head: tl.constexpr,
    stride_b_tok: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    SIGMOID_BETA: tl.constexpr,
):
    i_t, i_h = tl.program_id(0), tl.program_id(1)
    o_d = tl.arange(0, BD)
    mask = o_d < D
    base = i_t * stride_tok + i_h * stride_head + o_d

    b_q = tl.load(q + base, mask=mask, other=0.0).to(tl.float32)
    b_k = tl.load(k + base, mask=mask, other=0.0).to(tl.float32)
    b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
    b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
    tl.store(qn + base, b_q.to(qn.dtype.element_ty), mask=mask)
    tl.store(kn + base, b_k.to(kn.dtype.element_ty), mask=mask)

    b_a = tl.load(a + base, mask=mask, other=0.0).to(tl.float32)
    b_dt = tl.load(dt_bias + i_h * D + o_d, mask=mask, other=0.0).to(tl.float32)
    A = tl.exp(tl.load(A_log + i_h).to(tl.float32))
    x = b_a + b_dt
    if USE_LOWER_BOUND:
        b_g = lower_bound * tl.sigmoid(A * x)
    else:
        softplus_x = tl.where(x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x)
        b_g = -A * softplus_x
    tl.store(g + base, b_g.to(g.dtype.element_ty), mask=mask)

    # One scalar per (token, head); every program owns exactly one.
    p_b = i_t * stride_b_tok + i_h
    b_val = tl.load(b + p_b).to(tl.float32)
    if SIGMOID_BETA:
        b_val = tl.sigmoid(b_val)
    tl.store(beta_out + p_b, b_val.to(beta_out.dtype.element_ty))


def prepare_kda_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    num_heads: int,
    head_dim: int,
    lower_bound: Optional[float] = None,
    sigmoid_beta: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(q_norm, k_norm, g, beta)`` shaped for the TLX kernels.

    ``q``/``k`` come back bf16 (the TLX contract), ``g`` and ``beta`` fp32.
    ``a`` may be any shape with ``T * num_heads * head_dim`` elements; ``b``
    any shape with ``T * num_heads``.

    Set ``sigmoid_beta=False`` when the caller already activated beta (the
    extend path does, decode does not); beta is then passed through as fp32.
    """
    T = a.numel() // (num_heads * head_dim)
    q = q.reshape(T, num_heads, head_dim)
    k = k.reshape(T, num_heads, head_dim)
    a = a.reshape(T, num_heads, head_dim)
    b = b.reshape(T, num_heads)
    if not (q.is_contiguous() and k.is_contiguous() and a.is_contiguous()):
        q, k, a = q.contiguous(), k.contiguous(), a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()

    qn = torch.empty_like(q)
    kn = torch.empty_like(k)
    g = torch.empty(T, num_heads, head_dim, dtype=torch.float32, device=q.device)
    beta = torch.empty(T, num_heads, dtype=torch.float32, device=q.device)

    _kda_prepare_kernel[(T, num_heads)](
        q,
        k,
        a,
        b,
        A_log.reshape(-1),
        dt_bias.reshape(-1),
        qn,
        kn,
        g,
        beta,
        lower_bound,
        stride_tok=num_heads * head_dim,
        stride_head=head_dim,
        stride_b_tok=num_heads,
        D=head_dim,
        BD=triton.next_power_of_2(head_dim),
        SOFTPLUS_THRESHOLD=20.0,
        USE_LOWER_BOUND=lower_bound is not None,
        SIGMOID_BETA=sigmoid_beta,
        num_warps=4,
    )
    return (
        qn.unsqueeze(0),
        kn.unsqueeze(0),
        g.unsqueeze(0),
        beta.unsqueeze(0),
    )
