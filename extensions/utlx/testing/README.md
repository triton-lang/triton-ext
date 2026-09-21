# Running the TLX op-library tests against a stock Triton

`extensions/utlx` builds the uTLX plugin, which supplies the TLX dialect and
language to an unmodified Triton. The TLX *op library* (`tlx.ops.mm`,
`tlx.ops.flash_attn`, ...) and its tests live in Meta's Triton fork,
[facebookexperimental/triton], not in any wheel. `install_tlx_ops_overlay.py`
copies that library into an environment that has stock Triton plus this
plugin, so the fork's own tests can run against it.

Nothing from the fork is vendored here; it is copied from the checkout you
pass in, at whatever version you check out.

## What the overlay adds

`overlay/triton/tlx/__init__.py` is the only adaptation layer:

1. imports `utlx_plugin` first -- that import is what registers the DSL as
   `triton.language.extra.tlx`, which every kernel under `ops/` imports;
2. accepts `triton.Config(ctas_per_cga=(n, 1, 1))`, a Meta extension, and maps
   it to upstream's `num_ctas=n` (the kernels only ever pass that form);
3. reports a configuration this Triton build cannot lower as the catalog's
   `UnsupportedOp`, so callers decline the shape instead of failing. Today that
   means `NUM_CTAS=2` kernels, which want `mapa.shared::cluster` and the
   `.cta_group::2` TMA modifier -- upstream has neither op, and its own 2-CTA
   scheme is different in kind (broadcast the tile through the CGA layout and
   let each CTA signal its own barrier), so bridging it is a kernel change, not
   a plugin one.

## Reproducing

See the "Testing against the TLX op library" section of
[`../README.md`](../README.md) for the full command sequence.

[facebookexperimental/triton]: https://github.com/facebookexperimental/triton
