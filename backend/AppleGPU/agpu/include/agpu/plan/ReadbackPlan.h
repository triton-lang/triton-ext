// ReadbackPlan.h - how C leaves the accumulator fragments: through the pool,
// or by naming `thread_elements()` when the consumer's registers are already
// the fragment's own lanes.
#ifndef AGPU_READBACK_PLAN_H
#define AGPU_READBACK_PLAN_H

#include "agpu/plan/LayoutBasis.h"
#include "agpu/plan/WarpSlots.h"

#include <cstdint>
#include <vector>

namespace agpu {

// The `thread_elements()` lane map as bases; `FragLane.h` spells it as
// expressions and `test_readbackplan` asserts the two agree.
inline constexpr int32_t kFragLaneRowBasis[5] = {0, 1, 2, 0, 4};
inline constexpr int32_t kFragLaneColBasis[5] = {2, 0, 0, 4, 0};
inline constexpr int64_t kFragElemBit = 1;

struct ReadbackPlan {
  enum class Kind { Pool, Rename };

  // Which accumulator and element a register names; Rename only.
  struct Elem {
    int64_t acc = 0;
    int64_t elem = 0;
  };

  Kind kind = Kind::Pool;
  std::vector<Elem> regs;

  bool rename() const { return kind == Kind::Rename; }
};

struct ReadbackWindow {
  int64_t rowLo = 0, rowHi = 0, colLo = 0, colHi = 0, batch = -1;
  bool set() const { return rowHi > rowLo; }
};

namespace detail {

inline bool basisRowIs(const BasisRow &row, const int32_t *want, int n) {
  if ((int)row.size() != n)
    return false;
  for (int i = 0; i < n; ++i)
    if (row[i] != want[i])
      return false;
  return true;
}

inline bool everyBasisIsFragAligned(const BasisRow &row, int32_t allowed) {
  for (int32_t b : row)
    if (b != allowed && (b % kSgFragDim) != 0)
      return false;
  return true;
}

inline bool allZero(const BasisRow &row) {
  for (int32_t b : row)
    if (b != 0)
      return false;
  return true;
}

} // namespace detail

// `dims` is the C layout's bases, row then column. A rename holds only when
// every register of every warp lands on the fragment its own cover assigns,
// which is the write set equalling the read set, checked per lane.
inline ReadbackPlan planReadback(const std::vector<LayoutBasis> &dims,
                                 const std::vector<WarpSlot> &slots,
                                 int64_t regCount, int64_t numWarps,
                                 const ReadbackWindow &window = {}) {
  ReadbackPlan p;
  if (dims.size() < 2 || slots.empty() || regCount <= 0)
    return p;
  const LayoutBasis &rowB = dims[dims.size() - 2];
  const LayoutBasis &colB = dims[dims.size() - 1];
  const LayoutBasis *batchB = dims.size() > 2 ? &dims[0] : nullptr;
  if (batchB &&
      (!window.set() || window.batch < 0 || !detail::allZero(batchB->lane) ||
       !detail::allZero(batchB->warp) || !detail::allZero(batchB->block)))
    return p;

  auto orOf = [](const BasisRow &row) {
    int32_t m = 0;
    for (int32_t b : row)
      m |= b;
    return m;
  };
  if (window.set()) {
    const int64_t rowLen = window.rowHi - window.rowLo;
    const int64_t colLen = window.colHi - window.colLo;
    if (colLen <= 0 || ((window.rowLo | rowLen) & orOf(rowB.warp)) ||
        ((window.colLo | colLen) & orOf(colB.warp)))
      return p;
  }

  if (!detail::basisRowIs(rowB.lane, kFragLaneRowBasis, 5) ||
      !detail::basisRowIs(colB.lane, kFragLaneColBasis, 5))
    return p;
  if (!detail::allZero(rowB.block) || !detail::allZero(colB.block))
    return p;
  if (!detail::everyBasisIsFragAligned(rowB.reg, 0) ||
      !detail::everyBasisIsFragAligned(colB.reg, kFragElemBit))
    return p;
  if (!detail::everyBasisIsFragAligned(rowB.warp, 0) ||
      !detail::everyBasisIsFragAligned(colB.warp, 0))
    return p;

  auto warpConst = [](const BasisRow &row, int64_t w) {
    int32_t v = 0;
    for (int b = 0; b < (int)row.size(); ++b)
      if (w & (int64_t(1) << b))
        v ^= row[b];
    return v;
  };

  std::vector<ReadbackPlan::Elem> out((std::size_t)regCount);
  for (int64_t r = 0; r < regCount; ++r) {
    const int32_t rc = rowB.registerConstant((int)r);
    const int32_t cc = colB.registerConstant((int)r);
    if (rc % kSgFragDim != 0 || (cc & ~kFragElemBit) % kSgFragDim != 0)
      return ReadbackPlan{};
    if (window.set() &&
        (rc < window.rowLo || rc >= window.rowHi ||
         (cc & ~kFragElemBit) < window.colLo ||
         (cc & ~kFragElemBit) >= window.colHi ||
         (batchB && batchB->registerConstant((int)r) != window.batch))) {
      out[(std::size_t)r] = {-1, 0};
      continue;
    }

    int64_t chosen = -1;
    for (int64_t w = 0; w < numWarps; ++w) {
      const int64_t mi =
          ((rc ^ warpConst(rowB.warp, w)) - window.rowLo) / kSgFragDim;
      const int64_t ni =
          (((cc & ~kFragElemBit) ^ warpConst(colB.warp, w)) - window.colLo) /
          kSgFragDim;
      int64_t found = -1;
      for (std::size_t s = 0; s < slots.size(); ++s)
        if (slots[s].mi.at(w) == mi && slots[s].ni.at(w) == ni) {
          found = (int64_t)s;
          break;
        }
      if (found < 0 || (chosen >= 0 && found != chosen))
        return ReadbackPlan{};
      chosen = found;
    }
    out[(std::size_t)r] = {slots[(std::size_t)chosen].acc, cc & kFragElemBit};
  }

  p.kind = ReadbackPlan::Kind::Rename;
  p.regs = std::move(out);
  return p;
}

} // namespace agpu

#endif // AGPU_READBACK_PLAN_H
