# Apple GPU Backend for Triton

Out-of-tree Apple GPU backend for the Triton compiler, built as a triton-ext
plugin. Codegen lowers TTGIR straight to Metal Shading Language (MSL) text,
which the Metal toolchain compiles to a `.metallib`.

## Architecture

```text
triton-ext/backend/AppleGPU/
  ├── ExportAppleGPU.cpp         Plugin registration (tritonGetPluginInfo API)
  ├── lib/TritonAppleGPUTransforms/   The emit-msl pass
  ├── lib/TritonAppleGPUToMSL/   TTGIR in, agpu facts out
  ├── agpu/                      The emitter; see agpu/README.md
  └── python/
        └── triton_apple_backend/
              ├── compiler.py    TTIR → TTGIR → MSL → metallib
              ├── driver.py      MPS dispatch, buffer binding, scalar packing
              └── metal_utils.m  ObjC++ Metal bridge, built beside its source
```

## Scope

`tt.get_program_id`, `tt.make_range`, `tt.splat`, `tt.addptr`, masked `tt.load`
and `tt.store`, `arith.constant` and the integer and float elementwise and
comparison operators. An op with no handler declines by name.

## Prerequisites

macOS 14+ with Xcode. Everything else (LLVM, Triton, cmake, ninja) is the
repo-wide setup in the [top-level README](../../README.md).

The emitter targets Metal 3. Per-device limits such as max threads per
threadgroup are always read back from the compiled pipeline.

## Build

From the repo root, `make build` builds every extension through pip. To build
this one alone, configure it directly. Triton is found by importing it with
`Python_EXECUTABLE`; `LLVM_INSTALL_DIR` names the LLVM that Triton pinned in its
`cmake/llvm-info.json`, and is read from the environment:

```bash
export LLVM_INSTALL_DIR=~/.triton/llvm/llvm-<hash>-<platform>-<build>

cmake -S . -B build -G Ninja -DPython_EXECUTABLE=$(which python)
ninja -C build libapplegpu_backend.dylib
```

The plugin target also builds `metal_utils`, the ObjC++ bridge that links the
torch cmake found. Point `-DTorch_DIR` at a different torch to link that one.

## Run

Install the package. Triton discovers the backend through its `triton.backends`
entry point, and importing it loads the bundled plugin library, so no
`TRITON_PLUGIN_PATHS` is needed:

```bash
export LLVM_INSTALL_DIR=~/.triton/llvm/llvm-<hash>-<platform>-<build>
pip install -e backend/AppleGPU --no-build-isolation --no-deps
```

```python
import torch, triton, triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

n = 4096
x = torch.randn(n, device="mps")
y = torch.randn(n, device="mps")
out = torch.empty_like(x)
add_kernel[(triton.cdiv(n, 1024),)](x, y, out, n, BLOCK=1024)
torch.mps.synchronize()
assert (out - (x + y)).abs().max().item() == 0.0
print("vecadd ok")
```

## Debug env vars

- `TRITON_MSL_DUMP=<path>` - write the emitted MSL for each kernel to `<path>`.
- `TRITON_MSL_DEBUG=1` - print the emitted MSL to stdout.

## Tests

`ctest` in `agpu/build` runs `metal_compiles`, which feeds
`agpu/test/emit_probe.cpp`'s output to `xcrun metal` and fails if the toolchain
rejects it. Numerical correctness is proved by running a kernel, as above.

## Known limitations

- `float64` - Metal has no double, so an f64 kernel silently computes in f32.
  See `narrowsSilently` in `agpu/include/agpu/plan/ElemType.h`.
- large `num_warps` - capped per kernel by the pipeline state's
  `maxTotalThreadsPerThreadgroup`, which the backend queries and reports so
  Triton drops over-large configs. Register pressure lowers it, so the ceiling
  is kernel-dependent.
