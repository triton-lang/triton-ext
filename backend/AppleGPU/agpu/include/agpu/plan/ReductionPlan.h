// ReductionPlan - the topology of a reduction, decided before any AST exists.
//
// The result key comes from `dropAxis` on the source key.
#ifndef AGPU_REDUCTION_PLAN_H
#define AGPU_REDUCTION_PLAN_H

#include "agpu/core/Units.h"
#include "agpu/plan/Combiner.h"
#include "agpu/plan/Elementwise.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <vector>

namespace agpu {

// A register's coordinate vector, used as group identity.
class CoordKey {
public:
  using Storage = std::vector<int32_t>;

  CoordKey() = default;
  explicit CoordKey(Storage c) : coords_(std::move(c)) {}
  CoordKey(std::initializer_list<int32_t> c) : coords_(c) {}

  CoordKey dropAxis(int axis) const {
    assert(axis >= 0 && axis < (int)coords_.size());
    Storage out;
    out.reserve(coords_.size() - 1);
    for (int d = 0; d < (int)coords_.size(); ++d)
      if (d != axis)
        out.push_back(coords_[d]);
    return CoordKey(std::move(out));
  }

  int rank() const { return (int)coords_.size(); }
  int32_t at(int d) const { return coords_[d]; }
  const Storage &coords() const { return coords_; }

  bool operator==(const CoordKey &o) const { return coords_ == o.coords_; }
  bool operator!=(const CoordKey &o) const { return !(*this == o); }
  bool operator<(const CoordKey &o) const { return coords_ < o.coords_; }

private:
  Storage coords_;
};

// The registers whose coordinates agree on every dimension but the reduced
// one, folded into a single value.
struct ReductionGroup {
  CoordKey key;                // survivor coordinate, axis dropped
  std::vector<int> sourceRegs; // registers folded together, in order
};

// A cross-lane or cross-warp reduction step. `xorOffset` is the shuffle
// distance; the emitter turns it into simd_shuffle_xor.
struct ReduceStep {
  int64_t xorOffset = 0;
};

// Threadgroup scratch for the cross-warp phase, when there is one.
struct ScratchLayout {
  // numWarps * warpSize: every warp publishes, including ones outside this
  // reduction's subset.
  int64_t slotsPerOperand = 0;
  int64_t warpSize = kWarpSize;

  int64_t slotFor(int64_t warp, int64_t lane) const {
    return warp * warpSize + lane;
  }

  // The same address with the warp as a runtime value: the caller adds its
  // lane.
  int64_t warpStride() const { return warpSize; }

  int64_t anchorSlots(int64_t warpIndex) const { return warpIndex * warpSize; }
};

struct ReductionPlan {
  int reducedAxis = 0;

  std::vector<ReductionGroup> groups;

  // Lane bits whose basis moves the reduced axis: each contributes one XOR
  // shuffle step, emitted high bit first.
  std::vector<ReduceStep> laneSteps;

  // Warp ids in the cross-warp combine, as xor offsets from the executing
  // warp's anchor. Empty when lane-local. Every subset
  // shares these offsets; the anchor makes them absolute.
  std::vector<int> warpSubset;

  // Bits of the warp id the reduction spans. The anchor of the executing
  // warp's subset is `warpId & ~warpMask`.
  unsigned warpMask = 0;

  ScratchLayout scratch;

  // What each operand accumulates. Empty means one f32 operand.
  std::vector<ElemType> elems;

  // Registers each operand occupies. Groups address registers by index from
  // the reduced tensor's layout, so operands that disagree read the wrong
  // element. Empty means one operand.
  std::vector<int64_t> regsPerOperand;

  Combiner combiner = Combiner::Generic;

  // The whole-simdgroup fold that replaces the lane ladder, or null when the
  // ladder stands. The steps must cover every lane bit: a partial fold would
  // cross groups that the reduction keeps apart.
  const char *laneIntrinsic(int64_t warpSize) const {
    if (elems.size() > 1)
      return nullptr;
    unsigned covered = 0;
    for (const ReduceStep &s : laneSteps)
      covered |= (unsigned)s.xorOffset;
    if (covered != (unsigned)(warpSize - 1))
      return nullptr;
    return simdReduceFn(combiner, elemAt(0));
  }

  ElemType elemAt(int k) const {
    return k < (int)elems.size() ? elems[(std::size_t)k] : f32();
  }

  // Whether every operand is addressed by the same register indices.
  bool operandsShareLayout() const {
    for (std::size_t k = 1; k < regsPerOperand.size(); ++k)
      if (regsPerOperand[k] != regsPerOperand[0])
        return false;
    return true;
  }

  bool crossWarp() const { return warpSubset.size() > 1; }

  // The mask that clears the reduced bits, as the emitted expression needs it.
  unsigned anchorMask(int64_t numWarps) const {
    return ~warpMask & (unsigned)(numWarps - 1);
  }
  int groupCount() const { return (int)groups.size(); }

  // The group a result register belongs to, by its own coordinates.
  int groupFor(const CoordKey &resultKey) const {
    for (int i = 0; i < (int)groups.size(); ++i)
      if (groups[i].key == resultKey)
        return i;
    return -1;
  }
};

// Bits whose basis is non-zero along the reduced axis.
inline unsigned
reduceMaskFromBases(const std::vector<int32_t> &basesAlongAxis) {
  unsigned mask = 0;
  for (int b = 0; b < (int)basesAlongAxis.size(); ++b)
    if (basesAlongAxis[b] != 0)
      mask |= (1u << b);
  return mask;
}

// Every value of the masked bits, below `numWarps`.
inline std::vector<int> subsetsOf(unsigned mask, int numWarps) {
  std::vector<int> bits;
  for (int b = 0; (1 << b) < numWarps; ++b)
    if (mask & (1u << b))
      bits.push_back(b);
  std::vector<int> vals;
  for (int s = 0; s < (1 << bits.size()); ++s) {
    int v = 0;
    for (int i = 0; i < (int)bits.size(); ++i)
      if (s & (1 << i))
        v |= (1 << bits[i]);
    if (v < numWarps)
      vals.push_back(v);
  }
  return vals;
}

// Lane XOR steps from a lane mask, high bit first.
inline std::vector<ReduceStep> laneStepsFromMask(unsigned laneMask) {
  std::vector<ReduceStep> steps;
  for (int bit = 31; bit >= 0; --bit) {
    unsigned m = 1u << bit;
    if (laneMask & m)
      steps.push_back(ReduceStep{(int64_t)m});
  }
  return steps;
}

// Group registers by survivor coordinate, deduplicating registers whose full
// coordinates repeat: a replicated value folds once.
//
// `regCoords[r]` is register r's full coordinate vector.
inline std::vector<ReductionGroup>
groupSurvivors(const std::vector<CoordKey> &regCoords, int axis) {
  std::vector<CoordKey> seenFull;
  std::vector<ReductionGroup> groups;
  for (int r = 0; r < (int)regCoords.size(); ++r) {
    const CoordKey &full = regCoords[r];
    if (std::find(seenFull.begin(), seenFull.end(), full) != seenFull.end())
      continue;
    seenFull.push_back(full);
    CoordKey key = full.dropAxis(axis);
    auto it =
        std::find_if(groups.begin(), groups.end(),
                     [&](const ReductionGroup &g) { return g.key == key; });
    if (it == groups.end())
      groups.push_back(ReductionGroup{key, {r}});
    else
      it->sourceRegs.push_back(r);
  }
  // Deterministic order, sorted by key.
  std::sort(groups.begin(), groups.end(),
            [](const ReductionGroup &a, const ReductionGroup &b) {
              return a.key < b.key;
            });
  return groups;
}

} // namespace agpu

#endif // AGPU_REDUCTION_PLAN_H
