// ScanPlan.h - a prefix scan's topology, decided before emission.
//
// The combine runs in order along the axis, so the lane phase is a
// shift-and-combine ladder and each warp needs the total of every warp
// before it. A symmetric XOR butterfly would not hold the order.
#ifndef AGPU_SCAN_PLAN_H
#define AGPU_SCAN_PLAN_H

#include "agpu/core/Decline.h"
#include "agpu/msl/Ast.h"
#include "agpu/msl/Builtins.h"
#include "agpu/plan/Combiner.h"
#include "agpu/plan/Elementwise.h"
#include "agpu/plan/ReductionPlan.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace agpu {

// One (bit, stride) pair: input bit `bit` of an index moves the scanned axis
// by `stride` elements.
struct AxisBit {
  int bit = 0;
  int32_t stride = 0;
};

// What the layout says about the scanned axis.
struct ScanFacts {
  std::vector<AxisBit> laneBits; // lane bits that move the axis
  std::vector<AxisBit> warpBits;
  // Register bits that move the axis. Empty means a thread's registers hold
  // different elements at the same axis position (separate columns of a
  // column scan), each its own independent scan.
  std::vector<AxisBit> regBits;
  int64_t numWarps = 1;
  int64_t regCount = 1;
  bool reverse = false;

  // What the scan accumulates, per operand. An integer scan accumulated
  // through `float` is exact only to 2^24. Empty means one f32 operand.
  std::vector<ElemType> elems;

  // Registers each operand occupies. The plan refuses a multi-operand scan
  // whose operands disagree, since the register grouping comes from the
  // scanned tensor's layout alone. Empty means one operand, or unstated.
  std::vector<int64_t> regsPerOperand;

  Combiner combiner = Combiner::Generic;

  ElemType elemAt(int k) const {
    return k < (int)elems.size() ? elems[(std::size_t)k] : f32();
  }
};

inline unsigned maskOf(const std::vector<AxisBit> &bits) {
  unsigned m = 0;
  for (const AxisBit &b : bits)
    m |= 1u << b.bit;
  return m;
}

// The largest stride any lane or warp bit contributes.
inline int32_t reachOf(const ScanFacts &f) {
  int32_t reach = 0;
  for (const AxisBit &b : f.laneBits)
    reach = std::max(reach, b.stride);
  for (const AxisBit &b : f.warpBits)
    reach = std::max(reach, b.stride);
  return reach;
}

// Shifted down by the lowest set bit, the mask has to be 2^k - 1, or a
// shuffle by a power of two does not connect two lanes adjacent on the axis.
inline bool laneBitsContiguous(unsigned laneMask) {
  if (laneMask == 0)
    return true;
  unsigned shifted = laneMask;
  while ((shifted & 1u) == 0)
    shifted >>= 1;
  return (shifted & (shifted + 1)) == 0; // 2^k - 1
}

// The axis strides must form an unbroken halving ladder down from the reach.
// A gap groups registers no shuffle or carry step joins.
inline bool stridesFormLadder(const ScanFacts &f) {
  std::vector<int32_t> strides;
  for (const AxisBit &b : f.laneBits)
    strides.push_back(b.stride);
  for (const AxisBit &b : f.warpBits)
    strides.push_back(b.stride);
  if (strides.empty())
    return true;

  std::sort(strides.begin(), strides.end());
  strides.erase(std::unique(strides.begin(), strides.end()), strides.end());
  for (std::size_t i = 1; i < strides.size(); ++i)
    if (strides[i] != strides[i - 1] * 2)
      return false;
  return true;
}

// A scan shuffles up by a fixed delta and combines only where the source lane
// is in range: a lane must not absorb a partial sum from ahead of it.
struct ScanStep {
  int64_t delta = 0;   // lanes to shuffle up by
  bool guarded = true; // lanes below `delta` take no contribution
};

// Deltas scaled by the axis stride in lanes, the lowest set bit of the mask,
// which need not be 1: a column scan of an 8x8 tile puts the axis on lane
// bits 3 and 4, giving a ladder of 8, 16 where a row scan would give 1, 2.
inline std::vector<ScanStep> laneLadder(unsigned laneMask) {
  std::vector<ScanStep> steps;
  if (laneMask == 0)
    return steps;
  const unsigned stride = laneMask & (~laneMask + 1u);
  int lanes = 0;
  for (unsigned m = laneMask; m; m >>= 1)
    lanes += (m & 1u);
  for (int i = 0; i < lanes; ++i)
    steps.push_back(ScanStep{(int64_t)stride << i, true});
  return steps;
}

struct ScanPlan {
  std::vector<ScanStep> laneSteps;
  unsigned laneMask = 0;

  // Bits of the warp id the scanned axis traverses. An axis
  // spanning only some warp bits splits the threadgroup into independent
  // scans; a warp outside the mask holds a different output element.
  unsigned warpMask = 0;

  int32_t reach = 0;

  // Registers per window the machinery can connect.
  int64_t windowRegs = 1;

  bool crossWarp = false;

  // How many consecutive windows are segments of one scan. A register bit
  // above the spread selects a block of the axis the window machinery cannot
  // connect, so each block is its own window but takes the grand totals of
  // the blocks before it. 1 when every window is a whole scan.
  int64_t chainedWindows = 1;

  bool windowsChain() const { return chainedWindows > 1; }

  // The warp position holding the axis's final element.
  int64_t finalWarp() const { return reverse ? 0 : (int64_t)warpMask; }

  // A chained scan's running total outlives the next window's publish, so it
  // gets its own rows above the warp totals, two parities alternating per
  // window: readers take the previous window's row while the writer fills the
  // other, so no slot is read and written in one barrier epoch.

  int64_t chainParityStride(int64_t numWarps) const {
    return ((int64_t)anchorMask(numWarps) + 1) * scratch.warpSize;
  }

  int64_t chainBase(int64_t numWarps, int64_t parity) const {
    return numWarps * scratch.warpSize + parity * chainParityStride(numWarps);
  }

  // Not just register order: reversing the order the caller walks registers
  // does not change which lane `simd_shuffle_up` reads from, so a reverse
  // scan needs the opposite shuffle, guard, publishing lane and carry
  // comparison.
  bool reverse = false;

  ScratchLayout scratch;

  std::vector<ElemType> elems;

  bool usable = false;

  Combiner combiner = Combiner::Generic;

  ElemType elemAt(int k) const {
    return k < (int)elems.size() ? elems[(std::size_t)k] : f32();
  }

  // The whole-simdgroup prefix that replaces the lane ladder. Only a forward
  // scan over every lane bit: the intrinsic fixes both the direction and the
  // span. Reverse and strided ladders keep the shuffle steps.
  const char *laneIntrinsic(int64_t warpSize) const {
    if (reverse || elems.size() > 1)
      return nullptr;
    if (laneMask != (unsigned)(warpSize - 1))
      return nullptr;
    return simdPrefixInclusiveFn(combiner, elemAt(0));
  }

  // The exclusive prefix carries the identity, so the fold it feeds needs no
  // guard: the first lane along the axis reads the identity.
  const char *prefixIntrinsic(int64_t warpSize) const {
    if (!laneIntrinsic(warpSize))
      return nullptr;
    return simdPrefixExclusiveFn(combiner, elemAt(0));
  }

  // Whether a lane's own registers fold before any shuffle.
  bool needsLocalPass() const { return windowRegs > 1; }

  // Registers group into windows of 2*reach, each an independent scan, so a
  // register at a boundary seeds.
  bool startsWindow(int64_t r) const {
    return r == 0 || (windowRegs > 0 && r % windowRegs == 0);
  }

  // The warp positions the carry walks: the mask's set positions in
  // increasing order. Unrolling `w = 0 .. numWarps-1` and guarding on
  // `warp > w` would combine a warp holding a different output element.
  std::vector<int64_t> carryWarps(int64_t numWarps) const {
    std::vector<int64_t> out;
    if (warpMask == 0)
      return out;
    for (int64_t w = 0; w < numWarps; ++w)
      if (((unsigned)w & ~warpMask) == 0)
        out.push_back(w);
    return out;
  }

  // In scan order: the earliest seeds the fold and each later one joins as the
  // second combine argument, which also lets the carry guard nest. The last
  // position is dropped, its block being dead.
  std::vector<int64_t> carryFoldOrder(int64_t numWarps) const {
    std::vector<int64_t> out = carryWarps(numWarps);
    if (reverse)
      std::reverse(out.begin(), out.end());
    if (!out.empty())
      out.pop_back();
    return out;
  }

  // The bits the axis does not traverse. `warpId & anchorMask` identifies
  // which independent scan a warp belongs to.
  unsigned anchorMask(int64_t numWarps) const {
    return ~warpMask & (unsigned)(numWarps - 1);
  }

  // ── the four direction answers ─────────────────────────────────────────

  // A forward scan takes from the lane behind it, a reverse one from ahead.
  const char *shuffleName() const {
    return reverse ? msl::builtin::simd::ShuffleDown
                   : msl::builtin::simd::ShuffleUp;
  }

  // Compared against the lane id masked to the axis bits. The mask is what
  // makes a replicated tensor correct: an 8-element tensor over a 32-lane warp
  // is loaded four times at `lane & 7`, and testing the raw lane id would
  // continue one running sum across all four copies.
  msl::BinOp guardOp() const {
    return reverse ? msl::BinOp::Le : msl::BinOp::Ge;
  }
  // Measured against the axis extent: the top position is `laneMask`.
  int64_t guardBound(int64_t delta, int64_t warpSize) const {
    const int64_t top = laneMask ? (int64_t)laneMask : warpSize - 1;
    return reverse ? top - delta : delta;
  }

  // Only when the axis leaves lane bits free, i.e. the tensor is replicated
  // across the warp.
  bool guardNeedsMask(int64_t warpSize) const {
    return laneMask != 0 && laneMask != (unsigned)(warpSize - 1);
  }

  // The lane holding the warp's total once the ladder has run: `laneMask`,
  // since only an axis filling the warp puts the total at lane 31.
  int64_t totalLane(int64_t warpSize) const {
    if (reverse)
      return 0;
    return laneMask ? (int64_t)laneMask : warpSize - 1;
  }

  // The bits the axis does not traverse: a lane reading a preceding warp's
  // total wants that warp's last position on the axis, but its own column.
  unsigned carryLaneKeepMask(int64_t warpSize) const {
    return (unsigned)(warpSize - 1) & ~laneMask;
  }

  // One published value serves the whole warp only when the axis covers every
  // lane bit. A free lane bit means those lanes hold different elements, so
  // each publishes its own value.
  bool publishesPerLane(int64_t warpSize) const {
    return laneMask != (unsigned)(warpSize - 1);
  }

  // Forward: a warp takes from every warp before it. Reverse: from every
  // warp after.
  msl::BinOp carryOp() const {
    return reverse ? msl::BinOp::Lt : msl::BinOp::Gt;
  }
};

// A multi-operand scan derives its register grouping from one layout and
// indexes every operand's name array with it. Operands that disagree on
// layout would read the wrong registers.
inline bool operandsShareLayout(const ScanFacts &f) {
  for (std::size_t k = 1; k < f.regsPerOperand.size(); ++k)
    if (f.regsPerOperand[k] != f.regsPerOperand[0])
      return false;
  return true;
}

// ── which registers fold locally ──────────────────────────────────────────
//
// The emitted order is fixed: registers, then lanes, then warps. That is a
// prefix only if a register step is the smallest step along the axis, so a
// layout needing the phases inverted declines.

// The largest stride the lane and warp phases reach.
inline int32_t spreadReach(const ScanFacts &f) {
  int32_t high = 0;
  for (const AxisBit &b : f.laneBits)
    high = std::max(high, b.stride);
  for (const AxisBit &b : f.warpBits)
    high = std::max(high, b.stride);
  return high;
}

// Register bits below the spread: the ones this thread folds locally, before
// anything crosses a lane. A register bit above the spread selects a whole
// block of the axis; counting it as a local fold once made a 32x32 column
// scan add row r+16 before the fifteen rows between. Each block is emitted as
// a segment, see `ScanPlan::chainedWindows`.
inline std::size_t localRegBits(const ScanFacts &f) {
  const int32_t reach = spreadReach(f);
  if (reach == 0)
    return f.regBits.size(); // no spread: every register folds locally
  std::size_t n = 0;
  for (const AxisBit &b : f.regBits)
    if (b.stride <= reach)
      ++n;
  return n;
}

inline Decision scanDecline(const ScanFacts &f) {
  if (!laneBitsContiguous(maskOf(f.laneBits)))
    return Decision::declined("emitScan", "axis lane bits are not contiguous");
  if (!stridesFormLadder(f))
    return Decision::declined("emitScan", "axis strides skip a halving step");
  if (!operandsShareLayout(f))
    return Decision::declined("emitScan",
                              "operands disagree on register layout");
  return Decision::emitted();
}

inline ScanPlan planScan(const ScanFacts &f) {
  ScanPlan p;
  p.reverse = f.reverse;
  p.elems = f.elems;
  p.combiner = f.combiner;
  p.laneMask = maskOf(f.laneBits);
  p.warpMask = maskOf(f.warpBits);

  if (!scanDecline(f).ok())
    return p;

  p.usable = true;
  p.reach = reachOf(f);
  // 2^|regBits|, the register bits the axis traverses. A 32x32 tile gives a
  // thread 8 registers
  // covering two rows of four columns and the axis moves only 2 register
  // bits, so registers 0-3 chain and register 4 starts the second row.
  p.windowRegs = std::min<int64_t>(f.regCount, (int64_t)1 << localRegBits(f));
  // One chain per off-axis register group: the axis's register bits above
  // the spread each double the number of segments.
  p.chainedWindows =
      std::min<int64_t>(std::max<int64_t>(f.regCount / p.windowRegs, 1),
                        (int64_t)1 << (f.regBits.size() - localRegBits(f)));
  p.laneSteps = laneLadder(p.laneMask);
  p.crossWarp = p.warpMask != 0 && f.numWarps > 1;
  if (p.crossWarp) {
    p.scratch = ScratchLayout{threadsFor(f.numWarps), kWarpSize};
    if (p.windowsChain())
      p.scratch.slotsPerOperand += 2 * p.chainParityStride(f.numWarps);
  }
  return p;
}

} // namespace agpu

#endif // AGPU_SCAN_PLAN_H
