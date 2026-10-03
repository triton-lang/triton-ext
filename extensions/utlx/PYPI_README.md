# uTLX — Triton Language Extensions (Plugin), Triton 3.8 build

`triton-utlx` ships a Triton plugin (`libutlx.so`) plus the `utlx` Python DSL
exposing the TLX dialect (async loads, warp-group MMA, shared/tensor-memory
buffers, etc.), built from
[triton-lang/triton-ext](https://github.com/triton-lang/triton-ext)
([`extensions/utlx`](https://github.com/triton-lang/triton-ext/tree/main/extensions/utlx))
for Triton 3.8.

## Install

The stock `triton` wheel that PyTorch depends on works; no Triton build of your
own is needed. Plugin support is compiled into Triton releases from 3.7 onward.

```bash
pip install torch            # torch 2.14 pulls triton ~=3.8.0
pip install triton-utlx
```

Then import it like any other package, in any order:

```python
import triton
import utlx_plugin as tlx
```

## Compatibility

`libutlx.so` is self-contained: it depends only on `libtriton.so`, and MLIR
symbols resolve from libtriton at load.

The 3.8.0 release and Triton `main` differ on the plugin ABI, so
`utlx_plugin._compat` bridges them at import, feature-detected in both
directions:

- **Plugin discovery.** 3.8.0 loads the libraries named in `TRITON_PLUGIN_PATHS`
  while libtriton is imported; `main` registers them explicitly with
  `extend_with`. Because 3.8.0 needs the variable set before `import triton`,
  this wheel installs a `utlx_plugin.pth` that sets it at interpreter startup,
  so import order does not matter. Set `UTLX_NO_AUTOREGISTER=1` to disable that
  and manage `TRITON_PLUGIN_PATHS` yourself.
- **Op and pass names.** 3.8.0 binds plugin ops as `utlx_<op>` and passes as
  `utlx_<pass>`; `main` prefixes them `create_` and `add_`.
- **Semantic helpers.** uTLX calls `TritonSemantic.dot_precheck` and
  `TritonSemantic._prepare_legacy_load`, which exist only in Meta's TLX fork of
  Triton, and `tl._unwrap_if_constexpr`, which upstream keeps in
  `triton.language.core`.

Pair the plugin with the Triton release line it was built for. A 3.7.x
`triton-utlx` against Triton 3.8 is an ABI mismatch; forcing it past Triton's
version check corrupts the heap.

## License

MIT. Copyright (c) 2026 Meta Platforms, Inc. and affiliates.
