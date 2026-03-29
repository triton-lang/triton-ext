# µTLX: Triton Language Extensions distributed as a Plugin

This package provides most of the function that Meta's TLX (https://github.com/facebookexperimental/triton) does, but without any changes to a fork of Triton.


## Create a Project Root Directory
```
mkdir TRITON-uTLX
export PROJECT_ROOT=`pwd`/TRITON-uTLX
```

## Build a plugable Triton
```
cd $PROJECT_ROOT
git clone https://github.com/triton-lang/triton.git

python -m venv ./triton/.venv --prompt triton
source ./triton/.venv/bin/activate
TRITON_EXT_ENABLED=1 make -C triton dev-install-llvm
```

## Configure and Build the µTLX plugin
```
cd $PROJECT_ROOT
git clone -b tlx https://github.com/triton-lang/triton-ext

cmake -GNinja \
  -B./triton-ext/extensions/utlx/build \
  -S./triton-ext \
  -DTRITON_SOURCE_DIR=$PROJECT_ROOT/triton \
  -DTRITON_BUILD_DIR=$PROJECT_ROOT/triton/build/cmake.linux-x86_64-cpython-3.11 \
  -DLLVM_BUILD_DIR=$PROJECT_ROOT/triton/llvm-project/build \
  -DTRITON_EXT_NAMES="utlx"

ninja -C ./triton-ext/extensions/utlx/build
```

## Output: build/lib/libutlx.so

To use it:

## Set the plugin path
```
export TRITON_PLUGIN_PATHS=$PROJECT_ROOT/triton-ext/extensions/utlx/build/lib/libutlx.so
```

## Run AMD Group GEMM:

```
TRITON_PLUGIN_PATHS=$PROJECT_ROOT/triton-ext/extensions/utlx/build/lib/libutlx.so \
python $PROJECT_ROOT/triton-ext/extensions/utlx/tlx/tutorials/amd-gemm-pipelined_test.py
```

## Run tests
```
cd $PROJECT_ROOT/triton-ext/extensions/utlx/test
TRITON_PLUGIN_PATHS=$PROJECT_ROOT/triton-ext/extensions/utlx/build/lib/libutlx.so python -m pytest -v
```

The three required paths:
- TRITON_SOURCE_DIR — Triton source tree (for headers)
- TRITON_BUILD_DIR — Triton build directory (for libtriton.so and generated headers)
- LLVM_BUILD_DIR — LLVM/MLIR build directory (for mlir-tblgen, MLIR libs, and headers)
