// StagePlan - which registers stage where and under what condition.
#ifndef AGPU_STAGE_PLAN_H
#define AGPU_STAGE_PLAN_H

#include "agpu/core/CoordGuard.h"
#include "agpu/core/TileView.h"
#include "agpu/core/Units.h"
#include "agpu/plan/LayoutBasis.h"

#include <numeric>
#include <optional>
#include <vector>

namespace agpu {

// `width` registers starting at `reg` form one access, merged by
// `planStageRuns`. Every consumer of an action list must honour the width.
struct StageAction {
  int reg = 0;           // first source register of the access
  int width = 1;         // registers this access covers
  bool packed = false;   // a wide access needs a packed_* type
  TileView::Coord coord; // its coordinate in the destination tile
  CoordGuard guard;      // dead actions are never in the list
};

// None when the register cannot reach the window at all.
inline std::optional<StageAction>
planStage(int reg, const std::vector<CoordRange> &ranges,
          const std::vector<CoordWindow> &windows,
          const TileView::Coord &coord) {
  CoordGuard g = planGuard(ranges, windows);
  if (g.isDead())
    return std::nullopt;
  return StageAction{reg, 1, false, coord, g};
}

// The alignment gcd omits the region's own offset in the pool, which is a
// multiple of 16 only for 2- or 4-byte elements. Over-claiming makes
// `limitFor` emit an align-16 `float4` against a misaligned address, so this
// does not hold for a 1-byte element with an odd stride.
inline PtrDims tilePtrDims(const TileView &v, int64_t elemBytes) {
  PtrDims out;
  for (int d = 0; d < v.rank(); ++d) {
    PtrInfo p;
    if (v.strideAt(d) == 1) {
      p.contiguity = v.extentAt(d);
      int64_t align = elemBytes > 0 ? kTGPoolAlignBytes / elemBytes : 1;
      for (int o = 0; o < v.rank(); ++o)
        if (o != d)
          align = std::gcd(align, v.strideAt(o));
      p.alignment = std::max<int64_t>(1, align);
    }
    out.push_back(p);
  }
  return out;
}

// A run merges only when every register of the group is present, in order,
// starting at a width-aligned register and all carrying the same guard.
// `GuardTerm::dim` is the dimension, so guards from a ragged staging edge
// compare equal even though the emitter spells each register's coordinate as
// a different `coordN`.
template <class Actions>
inline AccessPlan planStageRuns(Actions &actions,
                                const std::vector<LayoutBasis> &dims,
                                const TileView &dst, unsigned elemBits) {
  const AccessPlan w =
      planAccess(regBasesOf(dims), runtimeSpanOf(dims),
                 tilePtrDims(dst, (int64_t)elemBits / 8), vecElemOf(elemBits));
  if (!w.vectorised())
    return w;

  Actions out;
  std::size_t i = 0;
  while (i < actions.size()) {
    const StageAction &a = actions[i];
    bool merge =
        a.reg % w.width == 0 && i + (std::size_t)w.width <= actions.size();
    for (int64_t k = 0; merge && k < w.width; ++k) {
      const StageAction &n = actions[i + (std::size_t)k];
      merge = n.reg == a.reg + (int)k && n.guard == a.guard;
    }
    if (merge) {
      StageAction m = a;
      m.width = (int)w.width;
      m.packed = w.packed;
      out.push_back(m);
      i += (std::size_t)w.width;
    } else {
      out.push_back(a);
      ++i;
    }
  }
  actions = std::move(out);
  return w;
}

} // namespace agpu

#endif // AGPU_STAGE_PLAN_H
