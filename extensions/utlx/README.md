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
cd $PROJECT_ROOT/triton-ext/extensions/utlx/test
python -m pytest test_amd_gemm_pipelined.py
```

## Run tests

```bash
cd $PROJECT_ROOT/triton-ext/extensions/utlx/test
python -m pytest -v
```

[tlx]: https://github.com/facebookexperimental/triton
