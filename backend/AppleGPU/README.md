# Apple GPU Backend for Triton

Out-of-tree Apple GPU (Metal) backend for the Triton compiler, built as a triton-ext plugin.

## Architecture

```
triton (vanilla, pip install)
  │── TRITON_PASS_PLUGIN_PATH ──→ libTritonAppleGPUBackend.dylib (this plugin)
  │                                 MLIR dialect + passes for Apple GPU
  │── entry_points ──→ triton-apple (pip package)
  │                      compiler.py, driver.py, metal_utils.m
  │                        │
  │                        └── METALIR_DYLIB_PATH ──→ metal-ir-pipeline
  │                                                    LLVM IR → Metal AIR → metallib
```

## Prerequisites

- macOS 26+ with Xcode (Metal framework + clang)
- Python 3.10+
- CMake + Ninja

## Setup

### 1. Install Triton from source

```bash
git clone https://github.com/triton-lang/triton.git
cd triton

# Required for macOS plugin support: remove -fvisibility=hidden
# (TypeID symbols must be exported for dlopen'd plugins to resolve them)
sed -i '' 's/-fvisibility=hidden//' CMakeLists.txt

pip install -e . --no-build-isolation
cmake --install build/cmake.* --prefix build/install
```

### 2. Build metal-ir-pipeline

```bash
git clone https://github.com/imperatormk/metal-ir-pipeline.git
cd metal-ir-pipeline
# Point LLVM_DIR to the LLVM that Triton downloaded (under ~/.triton/llvm/)
cmake -B build -DLLVM_DIR=$(ls -d ~/.triton/llvm/llvm-*/lib/cmake/llvm | head -1)
cmake --build build
```

### 3. Build this plugin (triton-ext)

```bash
# From the triton-ext root
LLVM_INSTALL_DIR=/path/to/llvm/install \
TRITON_INSTALL_DIR=/path/to/triton/build/install \
cmake -S . -B build -G Ninja
cmake --build build -- TritonAppleGPUBackend

# Fix rpath for macOS
install_name_tool -add_rpath /path/to/triton/python/triton/_C \
  build/lib/libTritonAppleGPUBackend.dylib
```

### 4. Install triton-apple Python package

```bash
git clone https://github.com/imperatormk/triton-apple.git
cd triton-apple
pip install -e . --no-build-isolation
```

### 5. Run

```bash
export TRITON_PASS_PLUGIN_PATH=/path/to/triton-ext/build/lib/libTritonAppleGPUBackend.dylib
export METALIR_DYLIB_PATH=/path/to/metal-ir-pipeline/build/lib/Bridge/libMetalIRBridge.dylib

python your_triton_script.py
```

## What's included

### C++ MLIR Passes (loaded via TRITON_PASS_PLUGIN_PATH)

| Pass | Purpose |
|------|---------|
| `add_accelerate_matmul` | Rewrite tt.dot → AppleMmaEncoding (simdgroup MMA) |
| `add_simplify_gather` | Strip efficient_layout from large gathers (Metal JIT limit) |
| `add_to_llvmir` | Lower AppleMmaEncoding → LLVM IR with simdgroup intrinsics |
| `add_lower_gpu_to_air` | Lower gpu.thread_id/block_dim → AIR intrinsics |
| `add_reconcile_unrealized_casts` | Clean up leftover conversion casts |

### TritonAppleGPU Dialect

- `AppleMmaEncodingAttr` — 8x8 simdgroup matrix multiply encoding

## Test Status

- 72/72 backend-specific tests passing (elementwise, dot, GEMM, reduce, atomics, softmax, layernorm, attention, fla delta rule)
- 5735/9337 upstream test_core.py passing (remaining failures are float64, FP8, int64 atomics, NVIDIA-specific precision modes — all known MPS/Metal limitations)
- Qwen 3.5-2B inference working via fla (flash-linear-attention) Triton kernels

## Known Limitations

- `float64` — MPS doesn't support double precision
- `float8` (e4m3, e5m2) — NVIDIA-specific hardware types
- `int64` atomics — Metal doesn't support 64-bit atomic operations
- `num_warps >= 16` — exceeds Apple GPU max threads per threadgroup (384)
- Cross-threadgroup spinlocks — Apple GPU has no forward progress guarantee
