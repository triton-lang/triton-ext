// DeviceStoreTarget - where a fused dot's C drains straight to device and
// what happens to each element on the way.
#ifndef AGPU_EMIT_DEVICE_STORE_TARGET_H
#define AGPU_EMIT_DEVICE_STORE_TARGET_H

#include "agpu/emit/primitives/Stride.h"
#include "agpu/msl/Ast.h"
#include "agpu/plan/Elementwise.h"

#include <vector>

namespace agpu {

struct DeviceStoreTarget {
  msl::Str base; // the tensor's base pointer, empty when absent
  msl::Expr *baseOffset = nullptr; // elements past `base`, null for zero
  Stride leadingDim;               // elements between consecutive rows
  msl::Expr *rowStart = nullptr;   // window origin, null for zero
  msl::Expr *colStart = nullptr;

  // The mask, as per-axis bounds on the window's coordinates. Null: unmasked.
  msl::Expr *rowBound = nullptr;
  msl::Expr *colBound = nullptr;

  // Uniform across the tile (an inductor template's `xmask`): whether the
  // drain happens at all.
  msl::Expr *uniformGuard = nullptr;

  // Threadgroup scratch the drain may borrow for edge tiles. Empty means the
  // edge falls back to per-element stores through the lane.
  msl::Str edgeScratch;

  // Threadgroup tile extents in elements, when known; `start + extent <= bound`
  // proves every fragment inside at once. Zero means unknown.
  int64_t tileRows = 0;
  int64_t tileCols = 0;

  ElemType elem = f32();

  bool ok() const { return !base.empty(); }
  bool bounded() const { return rowBound || colBound; }
  bool narrows() const { return elem.bits < 32; }
};

// Where one folded step's second operand is read. A window operand carries no
// starts of its own; it is indexed with the store's own row and column.
struct DrainOperand {
  enum class Kind {
    None,     // unary step
    Splat,    // one value for every element
    Row,      // a device row at the store's columns, reused by every row
    Col,      // a device column at the store's rows, reused by every column
    Tile,     // a device window at the store's own coordinates
    AccChain, // `DrainStep::branch`, rendered from the accumulator element
  };
  Kind kind = Kind::None;
  msl::Expr *splat = nullptr; // Splat: the uniform value
  msl::Str base;              // Row/Tile: the device base pointer
  msl::Expr *baseOffset = nullptr;
  Stride leadingDim; // Tile: its own row stride

  // What `base` points at, for the memoised read. Must match the un-memoised
  // arm's type or `DrainStep::roundBefore` rounds the two differently.
  ElemType elem = f32();
};

inline msl::Expr *basePtr(msl::Context &c, const msl::Str &base,
                          msl::Expr *baseOffset) {
  return baseOffset ? c.binary(msl::BinOp::Add, c.var(base), baseOffset)
                    : c.var(base);
}

// One link of a step's accumulator-rooted operand chain: the op (EpilogueOps.h
// names the set) and its own operand, which is never another chain.
struct DrainBranchLink {
  msl::Str op;
  DrainOperand operand;
};

struct DrainStep {
  msl::Str op;
  DrainOperand operand;
  // `Kind::AccChain`: the step's right operand, folded from the running
  // value after `branchBase` steps (0: the accumulator element). Empty means
  // the operand is that value itself.
  std::vector<DrainBranchLink> branch;
  int branchBase = 0;
  // Whether the kernel rounded the running value to the tensor's element
  // before this op consumed it. The drain replays that rounding.
  bool roundBefore = false;
};

} // namespace agpu

#endif // AGPU_EMIT_DEVICE_STORE_TARGET_H
