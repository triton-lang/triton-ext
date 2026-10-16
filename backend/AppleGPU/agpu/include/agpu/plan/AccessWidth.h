// AccessWidth.h - how wide a load or store can go.
//
// A run of `w` consecutive registers is one w-wide access iff:
//
//   (a) the registers are adjacent along a single tensor dimension;
//   (b) the pointer is contiguous and w-aligned along that dimension;
//   (c) the run does not straddle a lane, warp or block boundary.
#ifndef AGPU_ACCESS_WIDTH_H
#define AGPU_ACCESS_WIDTH_H

#include "agpu/core/Decline.h"

#include <cstdint>
#include <vector>

namespace agpu {

// `bases[b][d]` is how far register bit `b` moves along dimension `d`.
using RegBases = std::vector<std::vector<int32_t>>;

// Condition (c)'s input: the xor of every runtime basis (lane, warp, block)
// along each dimension. The bases are disjoint powers of two here, so the xor
// is an or.
using RuntimeSpan = std::vector<int32_t>;

// What the pointer analysis knows about one dimension.
struct PtrInfo {
  int64_t contiguity = 1; // consecutive elements the pointer walks
  int64_t alignment = 1;  // elements the base address is aligned to
};

// One entry per tensor dimension, indexed by the dimension the run walks.
using PtrDims = std::vector<PtrInfo>;

// Metal's portable vector widths stop at 4.
inline constexpr int64_t kMaxAccessWidth = 4;

// Every element here has a `packed_` vector form, so only contiguity forces a
// demotion. 64-bit is excluded.
enum class VecElem {
  Packable,    // a packed_* form exists, so alignment can be relaxed
  Unsupported, // no vector access worth taking
};

inline VecElem vecElemOf(unsigned bits) {
  if (bits != 8 && bits != 16 && bits != 32)
    return VecElem::Unsupported;
  return VecElem::Packable;
}

// Conditions (a) and (c). Register bit `b` must move by `2^b` along one fixed
// dimension and by nothing along any other. A runtime basis with a low bit set
// along that dimension interleaves other threads' elements, so the run shrinks
// until every outside basis clears its bits. Length 1 with dim -1 means the
// registers are not adjacent.
struct RegRun {
  int64_t length = 1;
  int dim = -1;
};

inline RegRun longestRegRun(const RegBases &bases,
                            const RuntimeSpan &runtime = {}) {
  RegRun run;
  int dim = -1;
  int runLog2 = 0;
  for (std::size_t b = 0; b < bases.size(); ++b) {
    int hit = -1;
    bool bad = false;
    for (std::size_t d = 0; d < bases[b].size(); ++d) {
      const int32_t basis = bases[b][d];
      if (basis == 0)
        continue;
      if (hit >= 0 || basis != (std::int32_t(1) << b)) {
        bad = true;
        break;
      }
      hit = (int)d;
    }
    if (bad || hit < 0)
      break;
    if (dim >= 0 && hit != dim)
      break; // the run turned a corner
    dim = hit;
    ++runLog2;
  }

  if (runLog2 > 0) {
    int32_t outside = dim < (int)runtime.size() ? runtime[dim] : 0;
    for (std::size_t b = (std::size_t)runLog2; b < bases.size(); ++b)
      if (dim < (int)bases[b].size())
        outside |= bases[b][dim];
    while (runLog2 > 0 && (outside & ((1 << runLog2) - 1)))
      --runLog2;
  }

  run.length = (int64_t)1 << runLog2;
  run.dim = runLog2 > 0 ? dim : -1;
  return run;
}

// Condition (b): contiguity is a hard limit; alignment is relaxed through
// MSL's packed vectors, which align only to the element.
struct WidthLimit {
  int64_t width = 1;
  bool packed = false; // needs a packed_* vector type to be legal
};

inline WidthLimit limitFor(int64_t run, const PtrInfo &ptr, VecElem elem) {
  WidthLimit lim;
  if (elem == VecElem::Unsupported)
    return lim;

  int64_t width = run < kMaxAccessWidth ? run : kMaxAccessWidth;
  while (width > 1 && ptr.contiguity < width)
    width >>= 1;

  // Under-aligned for the vector type: use the packed spelling.
  if (width > 1 && ptr.alignment < width)
    lim.packed = true;
  lim.width = width;
  return lim;
}

// The whole answer for one access.
struct AccessPlan {
  int64_t width = 1;
  int dim = -1;
  bool packed = false;

  bool vectorised() const { return width > 1; }
};

// `ptr` is indexed by dimension; this picks the one the run walks.
inline AccessPlan planAccess(const RegBases &bases, const RuntimeSpan &runtime,
                             const PtrDims &ptr, VecElem elem) {
  AccessPlan p;
  if (elem == VecElem::Unsupported)
    return p;

  const RegRun run = longestRegRun(bases, runtime);
  if (run.dim < 0)
    return p;

  const PtrInfo pi = run.dim < (int)ptr.size() ? ptr[run.dim] : PtrInfo{};
  const WidthLimit lim = limitFor(run.length, pi, elem);
  p.width = lim.width;
  p.packed = lim.packed;
  p.dim = lim.width > 1 ? run.dim : -1;
  return p;
}

// Why an access could not be vectorised, for the decline channel.
inline Decision widthDecision(const AccessPlan &p, VecElem elem) {
  if (p.vectorised())
    return Decision::emitted();
  if (elem == VecElem::Unsupported)
    return Decision::declined("accessWidth",
                              "element width has no vector type");
  return Decision::declined("accessWidth", "registers are not contiguous");
}

} // namespace agpu

#endif // AGPU_ACCESS_WIDTH_H
