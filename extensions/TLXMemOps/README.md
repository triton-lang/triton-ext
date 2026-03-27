# TLXMemOps — Triton plugin for TLX memory operations

This extension builds `libtlx_mem_ops.so`, a Triton plugin providing local memory
operations (local_alloc, local_view, local_store, local_load, alloc_barriers).

## Building

Built as part of `triton-ext`. From the triton-ext root:

```
  export TRITON_INSTALL_DIR=/path/to/triton/install
  export LLVM_INSTALL_DIR=/path/to/llvm/install
  cmake -B build
  cmake --build build
```

The output is `build/lib/libtlx_mem_ops.so`.

## Running

```
  TRITON_PLUGIN_PATHS=build/lib/libtlx_mem_ops.so python extensions/TLXMemOps/python/benchmarks/amd-gemm-pipelined.py
```
