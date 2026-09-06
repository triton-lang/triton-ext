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

Or, for a single file with no CMake:

```bash
c++ -std=c++17 -Iinclude -Itest test/test_tileview.cpp -o /tmp/t && /tmp/t
```

## Using it

One header, one object:

```cpp
#include "agpu/Emitter.h"

agpu::Emitter e;

e.addKernel(facts, [&](agpu::msl::Context &c, bool rollK) {
  agpu::msl::Block body;
  // ... build the body; emitters record what they need in `e.helpers`
  return body;
});

e.print(std::cout);   // includes, helpers, structs, prototypes, bodies, kernels
```

`Emitter` holds the two module-wide facts: the helper set and the pool
requirement. A helper named by the third kernel still reaches the prelude
printed before the first and the pool is the max over every function, since one
buffer serves them all.

A dot goes through its plan:

```cpp
agpu::msl::Block body;
e.dot(body, dotFacts, inputs);      // planDot chooses; emitDot spells it
```

`e.planFor(facts)` answers what a shape will cost (which strategy, how much
pool) without emitting anything.

## Layers

```text
core/   arithmetic. TileView, CoordGuard, Units, Swizzle, Names.
plan/   decides, in integers. Never builds an AST.
emit/   consumes a plan, builds AST. Never re-decides.
msl/    the syntax tree, its generated walk, the printer, the analyses.
```

Within `emit/`: an `EmitX.h` holds the `emitX()` for one op family.
`primitives/` holds the small types those emitters take and hold, each answering
one question about the hardware or a coordinate: `FragLane`, `Stride`,
`OperandSource`, `CoordHoist`.

`test_layering.cpp` reads the headers as text and fails if a planner includes
`msl/Context.h`.

## The check that matters

```bash
ctest --test-dir build -R metal_compiles
```

The other suites compare emitted text against expectations, which cannot catch a
module that does not compile. `test/emit_probe.cpp` emits every helper, a kernel
calling each, a ragged panel, an unstructured region, an integer reduce, both
loop forms, a device-function module, a planned dot and a whole `Emitter` module
and feeds the lot to `xcrun metal`.

It skips when there is no Metal toolchain and fails when one is present and
rejects the output. If `xcrun metal` reports a missing toolchain that is in fact
installed, set `TOOLCHAINS` to the Metal toolchain's bundle id.

## Design rules

1. **One owner per fact.** `TileView::offsetOf` is the only place a coordinate
   is multiplied by a stride; `kWarpSize` is declared once; a helper's name and
   its body key on one enum.
1. **Query, don't check.** A caller asks the plan what to do. It does not test a
   flag and decide again.
1. **Sizing and emission share the object.** Whoever reserves space calls
   `cosizeElems()` on the same view the emitter addresses through.
1. **Decline, don't fail.** A shape this emitter cannot express returns a
   `Decision` carrying a reason, so the output stays free of stray markers like
   `/*bad loop header*/`.
1. **Refuse only what the toolchain cannot diagnose.** An over-limit threadgroup
   declaration crashes MTLCompilerService, so the pool gate catches it. Register
   pressure, alignment and buffer counts are Metal's job.
1. **Verify numerically where structure is not enough.** A scan's emitted text
   can have the right shape and combine the wrong elements, so the suite
   simulates the ladder against the definition of a prefix sum.
1. **Headers define, `src/` holds the recursive walks.** Everything is
   header-only except a definition that recurses over the tree, which every
   translation unit would otherwise re-instantiate. That is why `eraseStmts` and
   the printer are the only two things in `src/`.
1. **A layer's files divide into deciders and vocabulary.** In `plan/`, a file
   that decides ends in `Plan` or `Schedule`; the rest are the types and tables
   those decisions are written in (`ElemType`, `MathFn`, `WarpSlots`). `emit/*`
   build AST and are named for the op family; `emit/primitives/` holds the small
   types an emitter holds, kept apart from the functions that emit.
