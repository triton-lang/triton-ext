// RebindPlan - the ops that emit no arithmetic, only a new name for a
// register that already exists.
//
// splat, unsplat, expand_dims, broadcast, join, split, trans. Each is the
// same walk: build a coordinate-to-register map for the source, then for
// each result register look up the source register whose coordinate matches
// after the op's transformation. `rebind` takes that transformation as a
// lambda, supplied by the caller from its own layout description.
//
// Output is `result register r takes source register k`. Turning that into
// names is the emitter's job.
#ifndef AGPU_REBIND_PLAN_H
#define AGPU_REBIND_PLAN_H

#include "agpu/core/Decline.h"

#include <cstdint>
#include <map>
#include <vector>

namespace agpu {

// A register's position in the tensor, one index per dimension.
using RegCoord = std::vector<int32_t>;

// `-1` means no source has that coordinate, which is a planning failure.
struct Rebind {
  std::vector<int> from; // indexed by result register
  int sourceIndex = 0;   // which source, for join/split

  bool complete() const {
    for (int f : from)
      if (f < 0)
        return false;
    return true;
  }
};

// Keyed on the coordinate itself: a hash collides once a dimension exceeds
// its bit budget.
using CoordIndex = std::map<RegCoord, int>;

inline CoordIndex indexByCoord(const std::vector<RegCoord> &coords) {
  CoordIndex out;
  for (std::size_t r = 0; r < coords.size(); ++r)
    out.emplace(coords[r], (int)r);
  return out;
}

// `toSource` maps a result coordinate to the source coordinate feeding it.
// Returning false leaves that entry at -1.
template <typename ToSourceFn>
inline Rebind rebind(const std::vector<RegCoord> &resultCoords,
                     const CoordIndex &sourceByCoord, ToSourceFn toSource) {
  Rebind out;
  out.from.assign(resultCoords.size(), -1);
  for (std::size_t r = 0; r < resultCoords.size(); ++r) {
    RegCoord want;
    if (!toSource(resultCoords[r], want))
      continue;
    const auto it = sourceByCoord.find(want);
    if (it != sourceByCoord.end())
      out.from[r] = it->second;
  }
  return out;
}

// An unfed result register means the layouts disagree: the op needs a data
// movement.
inline Decision rebindDecision(const Rebind &r) {
  if (r.from.empty())
    return Decision::declined("rebind", "no result registers");
  if (!r.complete())
    return Decision::declined(
        "rebind", "layouts disagree: a result register has no source");
  return Decision::emitted();
}
} // namespace agpu

#endif // AGPU_REBIND_PLAN_H
