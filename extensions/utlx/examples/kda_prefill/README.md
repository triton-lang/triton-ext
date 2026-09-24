# KDA prefill on gfx950

A real kernel that exercises uTLX's AMD register-layout surface end to end:
chunk-parallel Kimi Delta Attention prefill, using `amd_mfma_layout`,
`dot_operand_layout`, `slice_layout`, `require_layout`/`release_layout`,
`local_alloc`/`local_view`/`local_slice`, and the async-copy ops.

Where the unit tests in `../../test/test_amd_mfma_layout.py` cover each op in
isolation, this covers them under load — a kernel written against the layout API
without any regard for what uTLX happens to support.

## Provenance

The six `.py` files below are vendored **byte-for-byte** from sglang, Apache
License 2.0:

```text
repo    https://github.com/RolaoDenthu/sglang
branch  tlx/kimi-k3-kda
commit  66e3aa977d757aba9e5263d57e5aa98dfe9b43c7
path    python/sglang/kernels/ops/kimi_k3/tlx/
```

| file                     |                                                    |
| ------------------------ | -------------------------------------------------- |
| `kimi_k3_kda_prefill.py` | the kernel under test                              |
| `kimi_k3_kda_prepare.py` | builds normalized Q/K, log decays, beta            |
| `kimi_k3_kda_decode.py`  | recurrent decode, imported by `__init__`           |
| `_shapes.py`             | the shape sets upstream benchmarks sweep           |
| `check.py`               | upstream's own availability probe                  |
| `__init__.py`            | `is_prefill_available()` / `missing_prefill_ops()` |

They are **not** modified — that is the point. `run.py` and this README are the
only files added here. If you update them, re-vendor rather than patch, so this
keeps testing upstream's code and not ours.

## Running

Needs gfx950 (MI350), a ROCm torch, and the uTLX plugin installed.

```bash
python run.py            # correctness against a reference
python run.py --sweep    # GFX950_PREFILL_FOCUS shapes, with timings
```

Expected, on one MI350 die (the 4096-token shapes are launch-bound and vary run
to run; the long-context ones are stable):

```text
T= 256 H=  4 seqs=1 even    out_rel=5.11e-03  state_rel=3.11e-03
T= 256 H=  4 seqs=2 even    out_rel=4.62e-03  state_rel=5.08e-03
T= 512 H=  4 seqs=4 even    out_rel=4.79e-03  state_rel=4.62e-03
T= 512 H= 12 seqs=4 ragged  out_rel=6.93e-03  state_rel=4.27e-03

  tokens  seqs  heads  finite        ms
    4096     1      4    True     0.199
    4096     4      4    True     0.187
  131072     1      4    True     5.088
  131072     8      4    True     1.633
    4096     1     12    True     0.235
    4096     4     12    True     0.178
  131072     1     12    True     7.382
  131072     8     12    True     4.052
```

The reference in `run.py` is a sequential transcription of FLA's
`fused_recurrent_kda_packed_decode_kernel`. Errors around 5e-3 are expected: the
inputs and output are bf16 (eps ~7.8e-3) and the kernel accumulates per 64-token
chunk rather than per token, so it will not match bit-for-bit.

The timings are wall clock with no baseline. They show the kernel scales — note
batching at long context, 131072 tokens going 5.09 → 1.63 ms at H=4 — but say
nothing about whether it is *fast*.

## Two things that will waste your afternoon

**`initial_state` is V-major `[N, H, V, K]`.** The validator checks the shape,
so with `D == V` (the usual case) a transposed tensor sails through and the
kernel quietly computes nonsense. If your output disagrees with a reference by
`rel ~ 1.0`, check this first.

**Decay rates have to be physical.** The kernel exponentiates differences of
*cumulative* log decay. `A_log ~ N(0, 1)` gives per-token `g ~ -11`, which
overflows over a 64-token chunk and turns an entire head to NaN. `run.py` uses
`A_log = -4 + 0.1 * randn`. The NaNs this produces are easy to misread: they
land on tokens `≡ 15 mod 16`, exactly the MFMA tile height, which looks
convincingly like a layout bug and is not.
