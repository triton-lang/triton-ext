# uTLX — Triton Language Extensions (Plugin), Triton 3.9 build

`triton-utlx` ships a Triton plugin (`libutlx.so`) plus the `utlx` Python DSL
exposing the TLX dialect (async loads, warp-group MMA, shared/tensor-memory
buffers, etc.), built from
[triton-lang/triton-ext](https://github.com/triton-lang/triton-ext)
([`extensions/utlx`](https://github.com/triton-lang/triton-ext/tree/main/extensions/utlx))
for Triton 3.9.

## Install

The stock `triton` wheel that PyTorch depends on works; no Triton build of your
own is needed. Plugin support is compiled into Triton releases from 3.7 onward.

```bash
pip install torch            # torch 2.15 pulls triton ~=3.9.0
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

Triton 3.9 registers plugins explicitly: `import utlx_plugin` calls Triton's
`extend_with`, which binds the plugin's ops as `create_<op>` and its passes as
`add_<pass>`. Import order does not matter, and `TRITON_PLUGIN_PATHS` is not
needed.

uTLX also calls `TritonSemantic.dot_precheck` and
`TritonSemantic._prepare_legacy_load`, which exist only in Meta's TLX fork of
Triton, and `tl._unwrap_if_constexpr`, which upstream keeps in
`triton.language.core`. `utlx_plugin._compat` supplies them at import.

Pair the plugin with the Triton release line it was built for: this wheel
requires `triton~=3.9.0`. A `triton-utlx` built for another release line, forced
onto Triton 3.9 (e.g. with `pip install --no-deps`), is an ABI mismatch that
crashes or corrupts the heap rather than failing cleanly.

## License

MIT. Copyright (c) 2026 Meta Platforms, Inc. and affiliates.
