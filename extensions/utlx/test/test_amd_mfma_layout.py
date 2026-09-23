"""AMD MFMA register-layout ops.

Covers the explicit-layout surface uTLX exposes for AMD -- amd_mfma_layout,
dot_operand_layout, slice_layout, require_layout/release_layout, zeros(layout=)
and the buffer helpers -- plus regressions for two bugs that were only
observable at the boundaries between the Triton frontend and the IR.
"""

import pytest
import torch
import triton
import triton.language as tl

from conftest import DEVICE, is_hip_cdna4, tlx

pytestmark = pytest.mark.skipif(not is_hip_cdna4(),
                                reason="AMD MFMA layouts require gfx950")

# The layout the AMD kernels in the wild ask for (MI350, bf16 x bf16 -> fp32).
MFMA_VERSION = tl.constexpr(4)
INSTR_SHAPE = tl.constexpr([16, 16, 32])
WARPS_PER_CTA = tl.constexpr([4, 1])
K_WIDTH = tl.constexpr(8)

# ---------------------------------------------------------------------------
# require_layout / release_layout
# ---------------------------------------------------------------------------


@triton.jit
def _require_release_roundtrip(X, Y, M: tl.constexpr, N: tl.constexpr):
    offs_m = tl.arange(0, M)[:, None]
    offs_n = tl.arange(0, N)[None, :]
    x = tl.load(X + offs_m * N + offs_n)
    mfma: tl.constexpr = tlx.amd_mfma_layout(MFMA_VERSION, INSTR_SHAPE, True,
                                             WARPS_PER_CTA)
    y = tlx.release_layout(tlx.require_layout(x, mfma))
    tl.store(Y + offs_m * N + offs_n, y)


def test_require_release_roundtrip():
    """Requiring then releasing an MFMA layout must not perturb the data."""
    M = N = 64
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    y = torch.empty_like(x)
    _require_release_roundtrip[(1, )](x, y, M, N, num_warps=4)
    torch.testing.assert_close(y, x, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# Layout carriers across a @triton.jit boundary
#
# Regression: a carrier's frontend type is a tl.block_type, which lowers back to
# an *unencoded* tensor type. Triton builds a callee's signature (and its return
# type) from those frontend types, so passing a layout into a jit helper -- or
# returning a layout-carrying value from one -- silently dropped the encoding and
# the verifier rejected the mismatch downstream.
# ---------------------------------------------------------------------------


@triton.jit
def _apply_layout(x, layout):
    return tlx.require_layout(x, layout)


@triton.jit
def _carrier_across_jit(X, Y, M: tl.constexpr, N: tl.constexpr):
    offs_m = tl.arange(0, M)[:, None]
    offs_n = tl.arange(0, N)[None, :]
    x = tl.load(X + offs_m * N + offs_n)
    mfma: tl.constexpr = tlx.amd_mfma_layout(MFMA_VERSION, INSTR_SHAPE, True,
                                             WARPS_PER_CTA)
    # layout passed *into* a jit helper, encoded value returned *out* of it
    y = tlx.release_layout(_apply_layout(x, mfma))
    tl.store(Y + offs_m * N + offs_n, y)


def test_layout_carrier_survives_jit_boundary():
    M = N = 64
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    y = torch.empty_like(x)
    _carrier_across_jit[(1, )](x, y, M, N, num_warps=4)
    torch.testing.assert_close(y, x, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# dot_operand_layout: a GEMM driven entirely by explicit layouts
# ---------------------------------------------------------------------------


@triton.jit
def _mfma_dot(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    offs_k = tl.arange(0, K)
    a = tl.load(A + offs_m[:, None] * K + offs_k[None, :])
    b = tl.load(B + offs_k[:, None] * N + offs_n[None, :])

    mfma: tl.constexpr = tlx.amd_mfma_layout(MFMA_VERSION, INSTR_SHAPE, True,
                                             WARPS_PER_CTA)
    a_layout: tl.constexpr = tlx.dot_operand_layout(0, mfma, k_width=K_WIDTH)
    b_layout: tl.constexpr = tlx.dot_operand_layout(1, mfma, k_width=K_WIDTH)

    lhs = tlx.require_layout(a, a_layout)
    rhs = tlx.require_layout(b, b_layout)
    acc = tlx.zeros((M, N), tl.float32, layout=mfma)
    acc = tl.dot(lhs, rhs, acc=acc, out_dtype=tl.float32)
    tl.store(C + offs_m[:, None] * N + offs_n[None, :],
             tlx.release_layout(acc))


def test_dot_operand_layout_gemm():
    """tl.dot with explicitly laid-out operands matches torch."""
    M = N = 64
    K = 64
    a = torch.randn(M, K, device=DEVICE, dtype=torch.bfloat16)
    b = torch.randn(K, N, device=DEVICE, dtype=torch.bfloat16)
    c = torch.empty(M, N, device=DEVICE, dtype=torch.float32)
    _mfma_dot[(1, )](a, b, c, M, N, K, num_warps=4)
    torch.testing.assert_close(c, (a.float() @ b.float()),
                               atol=1e-1,
                               rtol=1e-2)


def test_dot_operand_layout_reaches_ttgir():
    M = N = K = 64
    a = torch.randn(M, K, device=DEVICE, dtype=torch.bfloat16)
    b = torch.randn(K, N, device=DEVICE, dtype=torch.bfloat16)
    c = torch.empty(M, N, device=DEVICE, dtype=torch.float32)
    compiled = _mfma_dot.warmup(a, b, c, M, N, K, num_warps=4, grid=(1, ))
    ttgir = compiled.asm["ttgir"]
    # The requested MFMA parameters must survive into TTGIR verbatim. Note a
    # bare require->release round trip would NOT show up here: with nothing
    # consuming the layout it is a no-op and remove-layout-conversions drops it,
    # which is why this assertion lives on a kernel that actually dots.
    assert "ttg.amd_mfma" in ttgir, ttgir[:2000]
    assert "version = 4" in ttgir
    assert "instrShape = [16, 16, 32]" in ttgir
    assert "isTransposed = true" in ttgir
    assert "ttg.dot_op" in ttgir
    assert f"kWidth = {K_WIDTH.value}" in ttgir


# ---------------------------------------------------------------------------
# slice_layout
# ---------------------------------------------------------------------------


@triton.jit
def _slice_layout_kernel(X, Y, M: tl.constexpr, N: tl.constexpr):
    offs_m = tl.arange(0, M)[:, None]
    offs_n = tl.arange(0, N)[None, :]
    x = tl.load(X + offs_m * N + offs_n)
    mfma: tl.constexpr = tlx.amd_mfma_layout(MFMA_VERSION, INSTR_SHAPE, True,
                                             WARPS_PER_CTA)
    a_layout: tl.constexpr = tlx.dot_operand_layout(0, mfma, k_width=K_WIDTH)
    row_layout: tl.constexpr = tlx.slice_layout(a_layout, 0)
    # a slice-laid-out row, broadcast back up: expand_dims must recover the
    # parent (dot-operand) layout for the subtract to verify
    row = tlx.require_layout(tl.sum(x, axis=0), row_layout)
    y = tlx.require_layout(x, a_layout) - row[None, :]
    tl.store(Y + offs_m * N + offs_n, tlx.release_layout(y))


def test_slice_layout_expand_dims():
    """slice_layout's parent must be recovered by expand_dims."""
    M = N = 64
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    y = torch.empty_like(x)
    _slice_layout_kernel[(1, )](x, y, M, N, num_warps=4)
    torch.testing.assert_close(y,
                               x - x.sum(dim=0, keepdim=True),
                               atol=1e-3,
                               rtol=1e-3)


# ---------------------------------------------------------------------------
# zeros(layout=) and swizzled_layout
# ---------------------------------------------------------------------------


@triton.jit
def _zeros_with_layout(Y, M: tl.constexpr, N: tl.constexpr):
    mfma: tl.constexpr = tlx.amd_mfma_layout(MFMA_VERSION, INSTR_SHAPE, True,
                                             WARPS_PER_CTA)
    acc = tlx.zeros((M, N), tl.float32, layout=mfma)
    offs_m = tl.arange(0, M)[:, None]
    offs_n = tl.arange(0, N)[None, :]
    tl.store(Y + offs_m * N + offs_n, tlx.release_layout(acc))


def test_zeros_with_layout():
    M = N = 64
    y = torch.full((M, N), 7.0, device=DEVICE, dtype=torch.float32)
    _zeros_with_layout[(1, )](y, M, N, num_warps=4)
    torch.testing.assert_close(y, torch.zeros_like(y), atol=0, rtol=0)


def test_swizzled_layout_is_a_shared_encoding():
    """swizzled_layout is the TLX spelling of swizzled_shared_layout_encoding."""

    @triton.jit
    def _k(Y, M: tl.constexpr, N: tl.constexpr):
        layout: tl.constexpr = tlx.swizzled_layout(1, 1, 1, order=[1, 0])
        buf = tlx.local_alloc((M, N), tl.float32, 1, layout=layout)
        v = tl.zeros((M, N), tl.float32)
        tlx.local_store(tlx.local_view(buf, 0), v)
        offs = tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]
        tl.store(Y + offs, tlx.local_load(tlx.local_view(buf, 0)))

    M = N = 32
    y = torch.full((M, N), 3.0, device=DEVICE, dtype=torch.float32)
    _k[(1, )](y, M, N, num_warps=4)
    torch.testing.assert_close(y, torch.zeros_like(y), atol=0, rtol=0)


# ---------------------------------------------------------------------------
# async_load token
#
# Regression: ttg.AsyncCopyGlobalToLocalOp has an *operand* named `result` (the
# destination memdesc), so the generated getResult() accessor returns that
# operand and shadows Operation::getResult(). async_load handed back the memdesc
# instead of its async token, and async_commit_group then rejected it.
# ---------------------------------------------------------------------------


@triton.jit
def _async_load_token(X, Y, M: tl.constexpr, N: tl.constexpr):
    layout: tl.constexpr = tlx.swizzled_layout(1, 1, 1, order=[1, 0])
    buf = tlx.local_alloc((M, N), tl.float32, 1, layout=layout)
    offs = tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]
    token = tlx.async_load(X + offs, tlx.local_view(buf, 0))
    tlx.async_load_commit_group([token])
    wait = tlx.async_load_wait_group(0)
    tl.store(Y + offs, tlx.local_load(tlx.local_view(buf, 0), token=wait))


def test_async_load_returns_a_token():
    """async_load must hand back its token, not the destination buffer.

    This is the regression: AsyncCopyGlobalToLocalOp has an operand named
    `result` (the destination memdesc), so the generated getResult() accessor
    returns that operand and shadows Operation::getResult(). async_load returned
    the memdesc, and async_commit_group -- which takes async tokens -- then
    failed to verify. Compiling at all is the assertion; the data check confirms
    the copy still lands.
    """
    M = N = 32
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    y = torch.empty_like(x)
    _async_load_token[(1, )](x, y, M, N, num_warps=4)
    torch.testing.assert_close(y, x, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# buffer_load / buffer_store
# ---------------------------------------------------------------------------


@triton.jit
def _buffer_roundtrip(X, Y, N: tl.constexpr):
    offs = tl.arange(0, N)
    mask = offs < N
    v = tlx.buffer_load(X, offs, mask=mask, other=0.0)
    tlx.buffer_store(v * 2.0, Y, offs, mask=mask)


def test_buffer_load_store():
    N = 128
    x = torch.randn(N, device=DEVICE, dtype=torch.float32)
    y = torch.empty_like(x)
    _buffer_roundtrip[(1, )](x, y, N, num_warps=4)
    torch.testing.assert_close(y, x * 2.0, atol=0, rtol=0)
