# Apple GPU Backend for Triton

Out-of-tree Apple GPU backend for the Triton compiler, built as a triton-ext
plugin. Codegen lowers TTGIR straight to Metal Shading Language (MSL) text,
which is compiled to a `.metallib` in-process via the Metal framework.

## Architecture

```text
triton-ext/backend/AppleGPU/
  ├── ExportAppleGPU.cpp         Plugin registration (tritonGetPluginInfo API)
  ├── lib/Dialect/               The TritonAppleGPU dialect (AppleMmaEncoding)
  ├── lib/TritonAppleGPUTransforms/   The passes, incl. the emit-msl shell
  ├── lib/TritonAppleGPUToMSL/   TTGIR in, agpu facts out: the one place MLIR
  │                              and agpu meet
  ├── agpu/                      The emitter. Holds no MLIR, so its suites run
  │                              in seconds; see agpu/README.md
  └── python/                    Python backend (pip installable)
        └── triton_apple_backend/
              ├── compiler.py    TTIR → TTGIR → MSL → metallib
              ├── driver.py      MPS GPU dispatch, buffer binding, scalar packing
              └── metal_utils.m  ObjC++ Metal bridge, built beside its source
```

## Prerequisites

macOS 14+ with Xcode, for the Metal framework and clang. Everything else (LLVM,
Triton, cmake, ninja) is the repo-wide setup in the
[top-level README](../../README.md).

### Metal version and hardware

The emitter targets Metal 3 (simdgroup_matrix MMA, bfloat, relaxed device
atomics) and uses nothing Metal 4 only, so the MSL runs on M3+ without change.
It does not yet use Metal 4's cooperative_tensor MMA or packed fp8/MX storage,
so GEMM and conv run at Metal 3 fragment speed there.

Per-device limits such as max threads per threadgroup are always read back from
the compiled pipeline.

## Setup

With LLVM and Triton built per the [top-level README](../../README.md):

### Build the plugin

From the repo root, `make build` builds every extension through pip. To build
this one alone, configure it directly. Triton is found by importing it with
`Python_EXECUTABLE`; `LLVM_INSTALL_DIR` names the LLVM that Triton pinned in its
`cmake/llvm-info.json`, and is read from the environment:

```bash
export LLVM_INSTALL_DIR=~/.triton/llvm/llvm-<hash>-<platform>-<build>

cmake -S . -B build -G Ninja -DPython_EXECUTABLE=$(which python)
ninja -C build libapplegpu_backend.dylib
```

This builds `libapplegpu_backend.dylib` (or `.so`) under `build/lib/`.

The plugin target also builds `metal_utils`, the ObjC++ bridge that links the
torch cmake found. Point `-DTorch_DIR` at a different torch and it links that
one instead.

### Run

Install the package. Triton discovers the backend through its `triton.backends`
entry point, and importing it loads the bundled plugin library, so no
`TRITON_PLUGIN_PATHS` is needed:

```bash
export LLVM_INSTALL_DIR=~/.triton/llvm/llvm-<hash>-<platform>-<build>
pip install -e backend/AppleGPU --no-build-isolation --no-deps
```

Then run:

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

## What's included

### C++ MLIR Passes

| Pass                       | Purpose                                                 |
| -------------------------- | ------------------------------------------------------- |
| `add_accelerate_matmul`    | Rewrite tt.dot → AppleMmaEncoding (simdgroup MMA)       |
| `add_store_shuffle_layout` | Re-lay MMA epilogue stores as within-simdgroup shuffles |
| `add_emit_msl`             | Emit MSL text from TTGIR (terminal codegen)             |

### TritonAppleGPU Dialect

- `AppleMmaEncodingAttr` - 8x8 simdgroup matrix multiply encoding

## Debug env vars

Every one of these is a diagnostic: it changes what is printed and never what is
emitted, so none of them keys the compilation cache. `CODEGEN_ENV` in
`compiler.py` is that key and is empty; `agpu/test/test_gates.cpp` reads it as
text and fails if any gate appears there.

That matters because the alternative bites quietly: a gate that did change
emitted code and was missing from the key meant a warm cache served a kernel
built the other way.

- `TRITON_MSL_DUMP=<path>` - write the emitted MSL for each kernel to `<path>`.
- `TRITON_MSL_DEBUG=1` - print the emitted MSL to stdout, plus per-kernel
  `.ll`/`.metallib` dumps.
- `MSL_LOG_REJECT=1` - log every fast-path rejection with its reason and site.
- `TRITON_MSL_TRACE_FAIL=1` - stack trace at the point emission first fails.
- `MSL_FUNC_BUDGET_DEBUG=1` - report per-function size budget accounting.
- `TRITON_MSL_NO_COMPILE=1` - stop after emitting MSL, skip the Metal compiler.

## Test Status

The backend's own suites are the ones this repo can speak for: `ctest` in
`agpu/build` covers the emitter and its planning, and `test/` holds the Python
ABI and emitted-MSL checks. Both run without a GPU except `metal_compiles`,
which needs the Metal toolchain.

Upstream's `test_core.py` runs against the MPS device but its pass rate is not
recorded here, because a figure without the date, the machine and the Triton
revision it was taken on goes stale silently. Measure it when you need it.

A test that inspects LLVM IR does not apply to this path, which emits MSL. Note
that `llvm.intr.assume` and `ub.poison` are both handled: the first is consumed
as vestigial, the second has a real lowering.

## Known Limitations

- `float64` - Metal has no double, so an f64 kernel silently computes in f32.
  The narrowing carries a decline note; see `narrowsSilently` in
  `agpu/include/agpu/plan/ElemType.h`.
- `float8` (e4m3, e5m2) - kernels compile and run when fp8 crosses the boundary
  as uint8 storage plus `triton.reinterpret`, which is what Triton's own fp8
  tests do. A native `torch.float8_*` device tensor is not possible: torch MPS
  has no fp8 dtype.
- `int64`/`uint64` atomics - Metal has no 64-bit device atomic, so these
  decline; see `kAtomicRules` in `agpu/include/agpu/plan/AtomicPlan.h`.
- acquire/release atomics - Metal device atomics are relaxed-only, so an ordered
  variant becomes a relaxed operation with device-scope fences around it rather
  than losing its ordering; see `fencesFor` in
  `agpu/include/agpu/plan/AtomicPlan.h`.
- `tf32` / `bf16xN` dot input precision - split-precision emulation modes with
  no Metal equivalent. The attribute is carried through the layout pass and
  otherwise ignored, so a dot asking for one gets the ordinary lowering rather
  than an error.
- large `num_warps` - capped per kernel by the pipeline state's
  `maxTotalThreadsPerThreadgroup`, which the backend queries and reports so
  Triton drops over-large configs. Register pressure lowers it, so the ceiling
  is kernel-dependent.
- Waiting on another threadgroup - Apple GPU guarantees no forward progress
  between threadgroups, so a kernel that spins until one it cannot schedule
  makes progress needs the whole grid co-resident. The backend classifies this
  and marks such kernels for the host; a mutex that merely spins on a device
  atomic is fine. See `residencyFor` in `agpu/include/agpu/plan/LaunchPlan.h`.
- libdevice patching is process-global. Importing this backend patches
  `triton.language.extra.libdevice`, so another backend loaded in the same
  interpreter sees Apple's stubs. Existing `libdevice` names are replaced;
  `tl.math` names are only added when missing.
