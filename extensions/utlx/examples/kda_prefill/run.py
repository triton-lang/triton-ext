#!/usr/bin/env python3
"""Run the KDA prefill kernel and check it against a reference.

    python run.py                # correctness
    python run.py --sweep        # GFX950_PREFILL_FOCUS shapes, with timings

The reference is a sequential transcription of FLA's
``fused_recurrent_kda_packed_decode_kernel``: state is V-major ``h[v, k]``,
decay multiplies along ``k``, and the query is scaled. Agreement is expected at
bf16 precision (~5e-3) since the kernel accumulates per chunk rather than per
token.
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import utlx_plugin  # noqa: F401  registers the plugin and the tlx DSL

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import kda_prefill as kda
from kda_prefill._shapes import GFX950_PREFILL_FOCUS

DEV = "cuda"
D = V = 128


def make_inputs(T, H, nseq, seed=0, ragged=False):
    torch.manual_seed(seed)
    q = torch.randn(T, H, D, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(T, H, D, device=DEV, dtype=torch.bfloat16)
    a = torch.randn(T, H, D, device=DEV, dtype=torch.bfloat16)
    b = torch.randn(T, H, device=DEV, dtype=torch.float32)
    # Realistic decay. A_log ~ N(0, 1) gives per-token g ~ -11, and the kernel
    # exponentiates differences of *cumulative* decay, which overflows to NaN.
    A_log = -4.0 + 0.1 * torch.randn(H, device=DEV, dtype=torch.float32)
    dt = torch.randn(H * D, device=DEV, dtype=torch.float32)
    qn, kn, g, beta = kda.prepare_kda_inputs(q,
                                             k,
                                             a,
                                             b,
                                             A_log,
                                             dt,
                                             num_heads=H,
                                             head_dim=D)
    v = torch.randn(1, T, H, V, device=DEV, dtype=torch.bfloat16)
    # NB: V-major [N, H, V, K]. With D == V a transposed tensor still passes
    # validation and silently computes nonsense.
    S0 = (torch.randn(nseq, H, V, D, device=DEV, dtype=torch.float32) *
          0.1).contiguous()
    if ragged and nseq > 1:
        chunk = T // nseq
        bounds = [0]
        for i in range(nseq - 1):
            bounds.append(bounds[-1] + chunk + (64 if i % 2 == 0 else -64))
        bounds.append(T)
    else:
        bounds = [i * (T // nseq) for i in range(nseq)] + [T]
    cu = torch.tensor(bounds, device=DEV, dtype=torch.int32)
    return qn, kn, v, g, beta, S0, cu


def reference(qn, kn, v, g, beta, S0, cu, scale):
    """Sequential FLA recurrence, per sequence."""
    T, H = qn.shape[1], qn.shape[2]
    out = torch.empty(T, H, v.shape[3], device=qn.device, dtype=torch.float32)
    finals = []
    for s in range(len(cu) - 1):
        lo, hi = int(cu[s]), int(cu[s + 1])
        h = S0[s].float().clone()
        for t in range(lo, hi):
            q_t = qn[0, t].float() * scale
            k_t = kn[0, t].float()
            h = h * torch.exp(g[0, t].float()).unsqueeze(1)
            vv = (v[0, t].float() - (h * k_t.unsqueeze(1)).sum(-1)) \
                * beta[0, t].float().unsqueeze(-1)
            h = h + vv.unsqueeze(-1) * k_t.unsqueeze(1)
            out[t] = (h * q_t.unsqueeze(1)).sum(-1)
        finals.append(h)
    return out.unsqueeze(0), torch.stack(finals)


def rel(x, y):
    return float((x.float() - y.float()).abs().max() /
                 max(float(y.float().abs().max()), 1e-9))


def check():
    print(f"prefill available: {kda.is_prefill_available()}")
    if not kda.is_prefill_available():
        print(f"missing ops: {kda.missing_prefill_ops()}")
        return 1
    scale = D**-0.5
    worst = 0.0
    for T, H, nseq, ragged in [(256, 4, 1, False), (256, 4, 2, False),
                               (512, 4, 4, False), (512, 12, 4, True)]:
        args = make_inputs(T, H, nseq, ragged=ragged)
        out, fs = kda.kda_paged_prefill(*args[:5],
                                        scale=scale,
                                        initial_state=args[5],
                                        cu_seqlens=args[6])
        ref_out, ref_fs = reference(*args, scale)
        r_o, r_s = rel(out, ref_out), rel(fs, ref_fs)
        worst = max(worst, r_o, r_s)
        print(
            f"T={T:4d} H={H:3d} seqs={nseq} {'ragged' if ragged else 'even  '}"
            f"  out_rel={r_o:.2e}  state_rel={r_s:.2e}")
    ok = worst < 2e-2
    print(f"\n{'PASS' if ok else 'FAIL'} (worst {worst:.2e}, bf16 tolerance)")
    return 0 if ok else 1


def sweep():
    print(f"{'tokens':>8} {'seqs':>5} {'heads':>6} {'finite':>7} {'ms':>9}")
    print("-" * 40)
    for shape in GFX950_PREFILL_FOCUS:
        T, nseq, H = shape.total_tokens, shape.sequences, shape.heads
        qn, kn, v, g, beta, S0, cu = make_inputs(T, H, nseq)
        kw = {"scale": D**-0.5, "initial_state": S0, "cu_seqlens": cu}
        out, fs = kda.kda_paged_prefill(qn, kn, v, g, beta, **kw)  # warm up
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            out, fs = kda.kda_paged_prefill(qn, kn, v, g, beta, **kw)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 5 * 1e3
        ok = bool(torch.isfinite(out).all() and torch.isfinite(fs).all())
        print(f"{T:8d} {nseq:5d} {H:6d} {ok!s:>7} {ms:9.3f}")
        del qn, kn, v, g, beta, S0, cu, out, fs
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep",
                   action="store_true",
                   help="run the GFX950_PREFILL_FOCUS shapes with timings")
    raise SystemExit(sweep() if p.parse_args().sweep else check())
