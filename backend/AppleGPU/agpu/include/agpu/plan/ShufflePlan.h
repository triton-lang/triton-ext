// ShufflePlan.h - a layout change that never leaves the warp.
//
// When every destination element already lives in the destination lane's own
// simdgroup, the conversion is a lane permutation: `simd_shuffle` moves it
// with no threadgroup traffic and no barrier.
#ifndef AGPU_SHUFFLE_PLAN_H
#define AGPU_SHUFFLE_PLAN_H

#include "agpu/core/Decline.h"
#include "agpu/core/Units.h"

#include <cstdint>
#include <map>
#include <utility>
#include <vector>

namespace agpu {

// One destination register: its value is `srcReg` of lane `srcLane[dstLane]`,
// all within the destination thread's own warp.
struct ShuffleStep {
  int srcReg = 0;
  std::vector<int32_t> srcLane; // indexed by destination lane

  // A plain register rebind: every lane reads its own.
  bool identity() const {
    for (std::size_t l = 0; l < srcLane.size(); ++l)
      if (srcLane[l] != (int32_t)l)
        return false;
    return true;
  }
};

// Whether one permutation is linear over GF(2): `srcLane[l]` is the xor of
// basis vectors over the set bits of `l`. Checked exhaustively, since a
// permutation can agree on the basis vectors and disagree elsewhere.
inline bool permIsLinear(const std::vector<int32_t> &perm,
                         std::vector<int32_t> &basis, int32_t &offset) {
  basis.clear();
  offset = 0;
  if (perm.empty())
    return false;

  // Affine: `lane ^ k` maps 0 to k.
  offset = perm[0];

  // basis[b] is where the lane with only bit b set lands, minus the offset.
  for (std::size_t b = 0; (1u << b) < perm.size(); ++b)
    basis.push_back(perm[1u << b] ^ offset);

  for (std::size_t l = 0; l < perm.size(); ++l) {
    int32_t acc = offset;
    for (std::size_t b = 0; b < basis.size(); ++b)
      if (l & (1u << b))
        acc ^= basis[b];
    if (acc != perm[l]) {
      basis.clear();
      offset = 0;
      return false;
    }
  }
  return true;
}

// Whether every step shares one permutation, so the lane index is computed
// once for all registers.
inline bool permIsUniform(const std::vector<ShuffleStep> &steps) {
  if (steps.size() < 2)
    return true;
  for (std::size_t i = 1; i < steps.size(); ++i)
    if (steps[i].srcLane != steps[0].srcLane)
      return false;
  return true;
}

struct ShufflePlan {
  std::vector<ShuffleStep> steps;
  bool uniformLanePerm = false;
  bool linearLanePerm = false;
  std::vector<int32_t> laneBasis;
  int32_t laneOffset = 0; // where lane 0 lands

  // The lane index is built once, so every shuffling register must want the
  // same permutation. Identity steps emit no shuffle and never read it.
  bool usable() const {
    if (steps.empty())
      return false;
    const std::vector<int32_t> *perm = nullptr;
    for (const ShuffleStep &s : steps) {
      if (s.identity())
        continue;
      if (perm && *perm != s.srcLane)
        return false;
      perm = &s.srcLane;
    }
    return true;
  }

  // The permutation the shuffling registers read, or null when none does.
  // Not `steps[0]`: it may be an identity step.
  const std::vector<int32_t> *shufflePerm() const {
    for (const ShuffleStep &s : steps)
      if (!s.identity())
        return &s.srcLane;
    return nullptr;
  }

  // Every step reads its own lane: the conversion is a rename.
  bool isRebind() const {
    for (const ShuffleStep &s : steps)
      if (!s.identity())
        return false;
    return !steps.empty();
  }

  // Shuffles the emitter actually issues.
  int64_t shuffleCount() const {
    int64_t n = 0;
    for (const ShuffleStep &s : steps)
      if (!s.identity())
        ++n;
    return n;
  }
};

// `laneMaps[r][l]` is the lane destination register `r` of lane `l` reads
// from; `srcRegs[r]` is which source register. A caller with an element from
// another warp passes nothing and gets an unusable plan.
inline ShufflePlan
planShuffle(const std::vector<int> &srcRegs,
            const std::vector<std::vector<int32_t>> &laneMaps) {
  ShufflePlan p;
  if (srcRegs.size() != laneMaps.size())
    return p;

  for (std::size_t r = 0; r < srcRegs.size(); ++r) {
    // A lane index outside the warp is not a shuffle at all.
    for (int32_t l : laneMaps[r])
      if (l < 0 || l >= (int32_t)kWarpSize)
        return ShufflePlan();
    p.steps.push_back(ShuffleStep{srcRegs[r], laneMaps[r]});
  }

  p.uniformLanePerm = permIsUniform(p.steps);
  if (const std::vector<int32_t> *perm = p.shufflePerm())
    p.linearLanePerm = permIsLinear(*perm, p.laneBasis, p.laneOffset);
  return p;
}

// `elems[r][l]` is the element register `r` of lane `l` holds, as a flat
// index into the tensor. For each destination register and lane, find the
// source (register, lane) holding that element.
//
// One destination register must read the same source register in every lane:
// `simd_shuffle` shuffles one variable across the warp.
inline ShufflePlan
planShuffleFromElems(const std::vector<std::vector<int64_t>> &srcElems,
                     const std::vector<std::vector<int64_t>> &dstElems) {
  if (srcElems.empty() || dstElems.empty())
    return ShufflePlan();

  // Which (register, lane) of the source holds each element.
  std::map<int64_t, std::pair<int, int32_t>> holder;
  for (std::size_t r = 0; r < srcElems.size(); ++r) {
    if (srcElems[r].size() != kWarpSize)
      return ShufflePlan();
    for (std::size_t l = 0; l < srcElems[r].size(); ++l)
      holder.emplace(srcElems[r][l], std::make_pair((int)r, (int32_t)l));
  }

  std::vector<int> srcRegs;
  std::vector<std::vector<int32_t>> laneMaps;
  for (const std::vector<int64_t> &wanted : dstElems) {
    if (wanted.size() != kWarpSize)
      return ShufflePlan();

    int reg = -1;
    std::vector<int32_t> lanes;
    for (int64_t want : wanted) {
      const auto it = holder.find(want);
      if (it == holder.end())
        return ShufflePlan(); // the element is in another warp
      if (reg < 0)
        reg = it->second.first;
      else if (reg != it->second.first)
        return ShufflePlan(); // one shuffle reads one variable
      lanes.push_back(it->second.second);
    }
    srcRegs.push_back(reg);
    laneMaps.push_back(std::move(lanes));
  }
  return planShuffle(srcRegs, laneMaps);
}

// The pool round trip is always correct, just slower.
inline Decision shuffleDecision(const ShufflePlan &p) {
  if (p.usable())
    return Decision::emitted();
  return Decision::declined("convertLayout", "elements cross a warp boundary");
}

} // namespace agpu

#endif // AGPU_SHUFFLE_PLAN_H
