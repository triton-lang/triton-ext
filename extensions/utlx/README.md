# µTLX: Triton Language Extensions distributed as a Plugin

This package provides most of the function that Meta's [TLX] does, but without
any changes to a fork of Triton.

## Install against the Triton that ships with PyTorch

µTLX runs on the stock `triton` wheel PyTorch depends on, with no Triton build
of your own:

```bash
pip install torch            # torch 2.14 pulls triton ~=3.8.0
pip install triton-utlx      # matching the Triton release line
```

Then import it like any other package, in any order:

```python
import triton
import utlx_plugin as tlx
```

That Triton registers plugins from `TRITON_PLUGIN_PATHS` while `libtriton` is
imported, rather than from `extend_with`, so the wheel installs a `.pth` that
sets the variable at interpreter startup. Set `UTLX_NO_AUTOREGISTER=1` to
disable that and set `TRITON_PLUGIN_PATHS` yourself. It also predates two
`TritonSemantic` helpers µTLX calls, which `utlx_plugin/_compat.py` supplies.

Build from source, as below, if you need an unreleased Triton or are changing
the plugin itself.

## Create a Project Root Directory

```bash
mkdir TRITON-uTLX
export PROJECT_ROOT=`pwd`/TRITON-uTLX
```

## Build a plugable Triton

```bash
cd $PROJECT_ROOT
git clone https://github.com/triton-lang/triton.git

python -m venv ./triton/.venv --prompt triton
source ./triton/.venv/bin/activate
TRITON_EXT_ENABLED=1 make -C triton dev-install-llvm
```

## Build and install the µTLX wheel

µTLX is packaged as a self-contained wheel (via `scikit-build-core`). The native
plugin (`libutlx.so`) is compiled by CMake and bundled inside the `utlx_plugin`
package, so importing it registers the plugin with Triton automatically — no
`TRITON_PLUGIN_PATHS` needed.

Build inputs:

- `LLVM_INSTALL_DIR` — an LLVM/MLIR install (headers, `mlir-tblgen`, CMake
  modules). Pass it as an environment variable.
- `TRITON_WHEEL_DIR` — an installed Triton wheel built with
  `TRITON_EXT_ENABLED=1`. Discovered automatically from the active Python
  environment; override with `TRITON_WHEEL_DIR=...` if needed.

```bash
cd $PROJECT_ROOT
git clone -b tlx https://github.com/triton-lang/triton-ext

LLVM_INSTALL_DIR=$(realpath $PROJECT_ROOT/triton-ext/llvm-*) \
    pip install ./triton-ext/extensions/utlx --no-build-isolation
```

`--no-build-isolation` lets CMake discover the Triton wheel installed in the
active environment. To build a distributable wheel instead of installing:

```bash
LLVM_INSTALL_DIR=$(realpath $PROJECT_ROOT/triton-ext/llvm-*) \
    pip wheel ./triton-ext/extensions/utlx --no-build-isolation --no-deps -w dist
```

## Run AMD Group GEMM

```bash
python $PROJECT_ROOT/triton-ext/extensions/utlx/tlx/tutorials/amd-gemm-pipelined_test.py
```

## Run tests

```bash
cd $PROJECT_ROOT/triton-ext/extensions/utlx/test
python -m pytest -v
```

## Building against a Triton release

`ci/download_triton_wheel.py` fetches a nightly, which carries the C++ headers
an extension build needs. A wheel built from a Triton *release* tag does not:
the `wheel_headers` cmake install component postdates v3.8.0. Build the wheel
with `TRITON_EXT_ENABLED=1`, install it, then stage the headers:

```bash
(cd triton && TRITON_EXT_ENABLED=1 python setup.py bdist_wheel)
pip install --force-reinstall --no-deps triton/dist/triton-*.whl
python triton-ext/ci/stage_triton_headers.py ./triton
```

Pin the release when the plugin has to load into the stock PyPI wheel: the
plugin resolves Triton and MLIR symbols from `libtriton` at `dlopen`, so it has
to be built against the same commit. `triton==3.8.0` is `c01b6774`.

## Testing against the TLX op library

The TLX op library (`tlx.ops.mm`, `tlx.ops.flash_attn`, ...) and its tests live
in [TLX], not in any wheel. `testing/install_tlx_ops_overlay.py` copies that
library into an environment running stock Triton plus this plugin, so those
tests can be run against it. See [`testing/README.md`](./testing/README.md).

Blackwell (sm100) matmul, end to end, from nothing:

```bash
# 1. the environment: stock Triton, straight from PyTorch
python3.12 -m venv .venv && . .venv/bin/activate
pip install torch numpy pytest              # torch 2.14 -> triton 3.8.0

# 2. the plugin. The PyPI wheel is x86-64 only and predates the work below,
#    so build one from this tree (see "Build a plugable Triton" above) and:
pip install path/to/dist/triton_utlx-*.whl

# 3. the op library and its tests
git clone --filter=blob:none https://github.com/facebookexperimental/triton fb-triton
git clone --filter=blob:none https://github.com/triton-lang/triton-ext
python triton-ext/extensions/utlx/testing/install_tlx_ops_overlay.py \
    --fb-triton ./fb-triton

# 4. run
cd fb-triton/python/test/unit/tlx_ops
python -m pytest test_mm_sm100.py -q -rs
```

On a GB200 (sm100) this reports **130 passed, 6 skipped**. The skips are the
`NUM_CTAS=2` shapes, which the op declines with a reason -- see
[`testing/README.md`](./testing/README.md) for why they are out of reach
without changes to Triton itself.

The plugin caches compiled kernels under `~/.triton/cache`, keyed on
`custom_stages.py` only: if you edit anything else in the plugin, clear that
cache or you will be re-running an old cubin.

[tlx]: https://github.com/facebookexperimental/triton
