// StagePlan - which registers stage where and under what condition.
#ifndef AGPU_STAGE_PLAN_H
#define AGPU_STAGE_PLAN_H

#include "agpu/core/CoordGuard.h"
#include "agpu/core/TileView.h"
#include "agpu/core/Units.h"
#include "agpu/cost/Banks.h"
#include "agpu/plan/LayoutBasis.h"
#include "agpu/plan/ReadbackPlan.h"

#include <algorithm>
#include <array>
#include <iterator>
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
      // A swizzle keeps only `vec` elements together, so a wider access would
      // straddle the XOR boundary and reach a permuted neighbour.
      p.contiguity = v.swizzle().permutes()
                         ? std::min(v.extentAt(d), v.swizzle().vec)
                         : v.extentAt(d);
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

// ── the staged pitch ──────────────────────────────────────────────────────
//
// A staged tile is written by its layout's register runs and read by the MMA
// in fragments, and its pitch decides which bank each lane's words land in.
// The pad is chosen by counting both at every candidate pitch:
//   1. list where each lane of each access starts, as (batch, row, col),
//      which no pitch changes, grouping accesses whose lanes form the same
//      shape;
//   2. at each even pitch within one bank cycle, turn those into byte
//      addresses and count the bank passes;
//   3. keep the cheapest pitch, the smallest on a tie.

// Accesses whose lanes stand the same way relative to lane 0: each lane's
// (batch, row, col) offset from it, the elements each lane covers, and where
// each access's lane 0 starts.
struct AccessShape {
  std::vector<std::array<int64_t, 3>> lanes;
  int64_t width = 1;
  std::vector<std::array<int64_t, 3>> origins;
};

// The accesses registers laid out as `dims` (rows, cols last) make into a
// `rows` x `cols` tile, one per run `planStageRuns` merges.
inline std::vector<AccessShape>
accessShapes(const std::vector<LayoutBasis> &dims, int64_t rows, int64_t cols,
             int64_t elemBytes) {
  std::vector<AccessShape> out;
  if (dims.size() < 2 || dims.size() > 3)
    return out;
  const TileView tile = dims.size() == 3 ? TileView::rowMajor({1, rows, cols})
                                         : TileView::rowMajor({rows, cols});
  const int64_t width =
      planAccess(regBasesOf(dims), runtimeSpanOf(dims),
                 tilePtrDims(tile, elemBytes), vecElemOf(elemBytes * 8))
          .width;
  const std::size_t lead = 3 - dims.size();
  std::vector<std::array<int64_t, 3>> laneBits(kWarpSize);
  for (int64_t lane = 0; lane < kWarpSize; ++lane)
    for (std::size_t d = 0; d < dims.size(); ++d)
      for (std::size_t b = 0; b < dims[d].lane.size(); ++b)
        if (lane >> b & 1)
          laneBits[lane][lead + d] ^= dims[d].lane[b];
  std::size_t regBits = 0;
  for (const LayoutBasis &lb : dims)
    regBits = std::max(regBits, lb.reg.size());
  for (int reg = 0; reg < (1 << regBits); reg += (int)width) {
    std::vector<std::array<int64_t, 3>> lanes = laneBits;
    for (std::size_t d = 0; d < dims.size(); ++d) {
      const int64_t k = dims[d].registerConstant(reg);
      for (std::array<int64_t, 3> &at : lanes)
        at[lead + d] ^= k;
    }
    const std::array<int64_t, 3> origin = lanes.front();
    for (std::array<int64_t, 3> &at : lanes)
      for (int d = 0; d < 3; ++d)
        at[d] -= origin[d];
    auto same = std::find_if(out.begin(), out.end(), [&](const AccessShape &s) {
      return s.lanes == lanes;
    });
    if (same == out.end())
      same = out.insert(out.end(), AccessShape{std::move(lanes), width, {}});
    same->origins.push_back(origin);
  }
  return out;
}

// Bank passes `shapes` take into a row-major tile of `pitch` elements a row,
// its slices `slice` elements apart. Moving an access by whole bank words
// turns every lane's bank alike, so accesses of one shape cost the same unless
// they start at different offsets within a word.
inline int64_t passesAtPitch(const std::vector<AccessShape> &shapes,
                             int64_t pitch, int64_t slice, int64_t elemBytes) {
  const auto bytesAt = [&](const std::array<int64_t, 3> &at) {
    return (at[0] * slice + at[1] * pitch + at[2]) * elemBytes;
  };
  int64_t passes = 0;
  std::vector<int64_t> starts;
  for (const AccessShape &s : shapes) {
    // Per offset within a word: the first such origin's byte, and how many.
    std::array<int64_t, cost::kTGBankBytes> first{}, count{};
    for (const std::array<int64_t, 3> &o : s.origins) {
      const int64_t at = bytesAt(o);
      if (count[at % cost::kTGBankBytes]++ == 0)
        first[at % cost::kTGBankBytes] = at;
    }
    for (int64_t within = 0; within < cost::kTGBankBytes; ++within) {
      if (count[within] == 0)
        continue;
      starts.clear();
      for (const std::array<int64_t, 3> &at : s.lanes)
        starts.push_back(first[within] + bytesAt(at));
      passes += count[within] * cost::bankPasses(starts, s.width * elemBytes);
    }
  }
  return passes;
}

// One fragment read as `loadFrag` issues it: each lane's element pair.
inline std::vector<LayoutBasis> fragmentReadDims() {
  LayoutBasis row, col;
  row.reg = {0};
  col.reg = {(int32_t)kFragElemBit};
  row.lane.assign(std::begin(kFragLaneRowBasis), std::end(kFragLaneRowBasis));
  col.lane.assign(std::begin(kFragLaneColBasis), std::end(kFragLaneColBasis));
  return {row, col};
}

// What one staging of a tile costs in accesses: the layout its registers
// stage from, stored by `warps` simdgroups, and the fragment reads it serves.
// With no layout only the reads place the pad; with no count each fragment is
// read once.
struct StageTraffic {
  std::vector<LayoutBasis> dims;
  int64_t warps = 0;
  int64_t reads = 0;
};

// Row pad, in elements, of a row-major `rows` x `cols` tile. The pitch stays
// even, since a lane reads its fragment pair as one 2-vector, and a pad past
// one bank cycle only repeats a shorter one.
inline int64_t stagedPadFor(int64_t rows, int64_t cols, int64_t elemBytes,
                            const StageTraffic &traffic = {}) {
  if (rows <= 0 || cols <= 0 || elemBytes <= 0)
    return 0;
  const std::vector<AccessShape> stores =
      accessShapes(traffic.dims, rows, cols, elemBytes);
  const std::vector<AccessShape> frag =
      accessShapes(fragmentReadDims(), rows, cols, elemBytes);
  const int64_t reads =
      traffic.reads > 0 ? traffic.reads : fragsFor(rows) * fragsFor(cols);
  const int64_t cycle =
      std::max<int64_t>(1, cost::kTGBanks * cost::kTGBankBytes / elemBytes);
  int64_t best = 0, bestPasses = -1;
  for (int64_t pad = 0; pad < cycle; ++pad) {
    const int64_t pitch = cols + pad;
    if (pitch % 2 != 0)
      continue;
    const int64_t slice = (rows - 1) * pitch + cols;
    const int64_t passes =
        traffic.warps * passesAtPitch(stores, pitch, slice, elemBytes) +
        reads * passesAtPitch(frag, pitch, slice, elemBytes);
    if (bestPasses < 0 || passes < bestPasses) {
      best = pad;
      bestPasses = passes;
    }
  }
  return best;
}

} // namespace agpu

#endif // AGPU_STAGE_PLAN_H
