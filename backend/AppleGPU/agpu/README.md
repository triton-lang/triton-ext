# agpu

An MSL emitter for Apple GPUs.

Facts in, Metal Shading Language out. No MLIR, no LLVM, no Triton: what arrives
is a small facts struct and everything after that is integer arithmetic and a
syntax tree.

## Build

```bash
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build --output-on-failure
```

## Using it

One header, one object:

```cpp
#include "agpu/Emitter.h"

agpu::Emitter e;

e.addKernel(facts, [&](agpu::msl::Context &c) {
  agpu::msl::Block body;
  // ... build the body
  return body;
});

e.print(std::cout);
```

## Layers

```text
core/   arithmetic and names
plan/   decides, in integers. Never builds an AST.
emit/   consumes a plan, builds AST. Never re-decides.
msl/    the syntax tree, the printer
bind/   the dispatch table and the symbol table
```

Within `emit/`: an `EmitX.h` holds the `emitX()` for one op family.
`primitives/` holds the small types those emitters take and hold.

## The check that matters

```bash
ctest --test-dir build -R metal_compiles
```

It is the only test here, because a text assertion cannot catch a module that
does not compile. `test/emit_probe.cpp` emits a masked move, an elementwise
kernel and a whole `Emitter` module, and feeds the lot to `xcrun metal`.

It skips when there is no Metal toolchain and fails when one is present and
rejects the output. If `xcrun metal` reports a missing toolchain that is in fact
installed, set `TOOLCHAINS` to the Metal toolchain's bundle id.

## Design rules

1. **One owner per fact.** `kWarpSize` is declared once. A second implementation
   is the defect this exists to remove.
1. **Query, don't check.** A caller asks the plan what to do. It does not test a
   flag and decide again.
1. **Decline, don't fail.** A shape this emitter cannot express returns a
   `Decision` carrying a reason, so the output stays free of stray markers like
   `/*bad loop header*/`.
1. **Refuse only what the toolchain cannot diagnose.** Register pressure,
   alignment and buffer counts are Metal's job.
1. **Headers define, `src/` holds the recursive walks.** Everything is
   header-only except a definition that recurses over the tree, which every
   translation unit would otherwise re-instantiate. That is why the printer is
   the only thing in `src/`.
