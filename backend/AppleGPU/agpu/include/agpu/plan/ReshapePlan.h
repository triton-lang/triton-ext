// ReshapePlan.h - join, split, cat: two-source rebinds that emit nothing.
// Each result register names a source register and which source it came from.
// The one-source members of the family live in `RebindPlan.h`.
#ifndef AGPU_RESHAPE_PLAN_H
#define AGPU_RESHAPE_PLAN_H

#include "agpu/core/Decline.h"
#include "agpu/plan/RebindPlan.h"

#include <cstdint>
#include <vector>

namespace agpu {

// ── the interleaving pair ─────────────────────────────────────────────────

// `join` places two tensors side by side along a new trailing axis, so
// result register 2r comes from the left operand's register r and 2r+1
// from the right's. `split` is its inverse.
struct InterleavePlan {
  // For each result register: which operand (0 or 1) and which of its
  // registers.
  std::vector<std::pair<int, int>> from;
  bool usable = false;
};

// The trailing axis is the one `join` adds and `split` removes; its value
// selects the operand, the rest names the element.
struct InterleaveFacts {
  // Coordinates of the registers of one operand (join) or of the joined
  // tensor (split).
  std::vector<RegCoord> src;
  // Coordinates of the result's registers: the joined tensor for a join, one
  // half for a split.
  std::vector<RegCoord> dst;
};

// Result register r takes the trailing coordinate as its operand and the
// rest as the element to find in it. Both operands share a layout.
//
// Matched by coordinate: nothing requires the added axis to be the
// fastest-varying bit of the register index.
inline InterleavePlan planJoinFrom(const InterleaveFacts &f) {
  InterleavePlan p;
  if (f.dst.empty())
    return p;

  for (const RegCoord &d : f.dst) {
    if (d.empty())
      return {};
    const int32_t which = d.back();
    if (which < 0 || which > 1)
      return {};
    RegCoord want(d.begin(), d.end() - 1);

    int found = -1;
    for (std::size_t s = 0; s < f.src.size(); ++s)
      if (f.src[s] == want) {
        found = (int)s;
        break;
      }
    if (found < 0)
      return {};
    p.from.push_back({(int)which, found});
  }
  p.usable = true;
  return p;
}

// Half `which` of a joined tensor: the result's coordinate with `which`
// appended is the source's.
inline InterleavePlan planSplitFrom(const InterleaveFacts &f, int which) {
  InterleavePlan p;
  if (f.dst.empty() || which < 0 || which > 1)
    return p;

  for (const RegCoord &d : f.dst) {
    RegCoord want = d;
    want.push_back((int32_t)which);

    int found = -1;
    for (std::size_t s = 0; s < f.src.size(); ++s)
      if (f.src[s] == want) {
        found = (int)s;
        break;
      }
    if (found < 0)
      return {};
    p.from.push_back({0, found});
  }
  p.usable = true;
  return p;
}

inline Decision interleaveDecision(const InterleavePlan &p) {
  if (p.usable)
    return Decision::emitted();
  return Decision::declined("interleave",
                            "a result register has no source coordinate");
}

// ── unpacking two fp4 elements from one byte ──────────────────────────────

// Each i8 holds two e2m1 values: low nibble first along `axis`, high nibble
// second. The result is twice as long on that axis as the source.
struct Fp4Pick {
  int reg = 0;       // which source register holds the byte
  bool high = false; // the upper nibble
};

struct Fp4UnpackPlan {
  std::vector<Fp4Pick> from; // one per result register
  bool usable = false;
};

// A result element at coordinate c comes from the source element at c with
// `axis` halved and takes the high nibble when c[axis] is odd.
inline Fp4UnpackPlan planFp4Unpack(const InterleaveFacts &f, int axis) {
  Fp4UnpackPlan p;
  if (f.dst.empty() || axis < 0)
    return p;

  for (const RegCoord &d : f.dst) {
    if ((std::size_t)axis >= d.size())
      return {};
    RegCoord want = d;
    const int32_t along = want[(std::size_t)axis];
    if (along < 0)
      return {};
    want[(std::size_t)axis] = along / 2;

    int found = -1;
    for (std::size_t s = 0; s < f.src.size(); ++s)
      if (f.src[s] == want) {
        found = (int)s;
        break;
      }
    if (found < 0)
      return {};
    p.from.push_back({found, (along % 2) != 0});
  }
  p.usable = true;
  return p;
}

} // namespace agpu

#endif // AGPU_RESHAPE_PLAN_H
