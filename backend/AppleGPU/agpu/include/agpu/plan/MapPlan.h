// map_elementwise - a scalar body inlined once per group of registers.
//
// The op takes N source tensors, applies a scalar region to `pack` elements
// of each at a time and produces M result tensors. The two sides count
// differently:
//
//   block arguments are source-major  -> arg(s * pack + p)
//   register names are group-major    -> name[s][g * pack + p]
//
// Only `p` corresponds between them. Results interleave as `k * pack + p`.
#ifndef AGPU_MAP_PLAN_H
#define AGPU_MAP_PLAN_H

#include "agpu/core/Decline.h"

#include <cstdint>

namespace agpu {

struct MapFacts {
  int numSources = 0;
  int numResults = 0;
  int numRegisters = 0;    // per source tensor; all sources agree
  int pack = 1;            // elements consumed per body invocation
  bool multiBlock = false; // the region has control flow
};

struct MapPlan {
  MapFacts f;

  int groups() const { return f.pack ? f.numRegisters / f.pack : 0; }

  // Which register of source `s` binds to block argument `s * pack + p` on
  // the `g`th inlining.
  int sourceRegister(int g, int p) const { return g * f.pack + p; }

  int blockArgument(int s, int p) const { return s * f.pack + p; }

  int resultOperand(int k, int p) const { return k * f.pack + p; }

  int numBlockArguments() const { return f.numSources * f.pack; }
  int numResultOperands() const { return f.numResults * f.pack; }

  // A multi-block region has more than one terminator, so results come
  // through a declared capture per result element instead.
  bool needsCaptures() const { return f.multiBlock; }
  int numCaptures() const { return needsCaptures() ? numResultOperands() : 0; }

  bool usable() const {
    return f.numSources > 0 && f.numResults > 0 && f.pack > 0 &&
           f.numRegisters > 0 && f.numRegisters % f.pack == 0;
  }
};

// There is no partial-pack form.
inline Decision mapDecision(const MapPlan &p) {
  if (p.f.pack <= 0)
    return Decision::declined("map_elementwise", "pack must be positive");
  if (p.f.numRegisters <= 0)
    return Decision::declined("map_elementwise", "no registers to map");
  if (p.f.numRegisters % p.f.pack != 0)
    return Decision::declined("map_elementwise",
                              "register count is not a whole number of packs");
  if (p.f.numSources <= 0 || p.f.numResults <= 0)
    return Decision::declined("map_elementwise", "needs a source and a result");
  return Decision::emitted();
}

} // namespace agpu

#endif // AGPU_MAP_PLAN_H
