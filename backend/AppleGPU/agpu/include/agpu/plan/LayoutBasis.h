// LayoutBasis.h - a layout's bases, as integers.
//
// A distributed tensor assigns each (register, lane, warp, block) a
// coordinate. The assignment is linear over GF(2): each input bit
// contributes a basis vector and the coordinate is their xor. Register bits
// are known at compile time and fold to a constant; lane, warp and block
// bits are runtime ids.
#ifndef AGPU_LAYOUT_BASIS_H
#define AGPU_LAYOUT_BASIS_H

#include "agpu/core/CoordGuard.h"
#include "agpu/plan/AccessWidth.h"

#include <cstdint>
#include <vector>

namespace agpu {

// The bases of one input dimension along one output dimension. Index is the
// bit position; the value is what that bit adds to the coordinate.
using BasisRow = std::vector<int32_t>;

// A layout, reduced to what coordinate construction needs: the bases of each
// runtime input dimension, plus the register dimension that folds to a
// constant.
struct LayoutBasis {
  BasisRow reg;  // compile-time: selected by the register index
  BasisRow lane; // runtime: laneId
  BasisRow warp; // runtime: warpId
  // Runtime: threadgroup_position_in_grid.x. Empty for the
  // single-threadgroup case.
  BasisRow block;

  // The constant a given register contributes.
  int32_t registerConstant(int regIndex) const {
    int32_t v = 0;
    for (int b = 0; b < (int)reg.size(); ++b)
      if (regIndex & (std::int32_t(1) << b))
        v ^= reg[b];
    return v;
  }

  // A row of zeroes counts as absent.
  bool needsBlockId() const {
    for (int32_t b : block)
      if (b)
        return true;
    return false;
  }

  // Union of every runtime basis and whether they overlap. Overlapping rows
  // make the reachable set an xor lattice.
  struct RuntimeBits {
    int32_t mask = 0;
    bool disjoint = true;
  };
  RuntimeBits runtimeBits() const {
    RuntimeBits rb;
    for (const BasisRow *row : {&lane, &warp, &block})
      for (int32_t b : *row) {
        if (b & rb.mask)
          rb.disjoint = false;
        rb.mask |= b;
      }
    return rb;
  }

  // Whether the register's coordinate adds to the runtime part, given that
  // the layout xors into it. The real delta is
  //     (regConst(r) ^ runtime) - (regConst(0) ^ runtime)
  // which equals `regConst(r) - regConst(0)` only when the register bits and
  // the runtime bits are disjoint.
  bool registerDeltasAreAffine(int reg) const {
    const RuntimeBits rb = runtimeBits();
    return rb.disjoint && !(registerConstant(reg) & rb.mask);
  }

  // Every coordinate this register can reach, across all lanes, warps and
  // threadgroups. Overlapping bases return the whole dimension.
  CoordRange rangeOf(int reg, int dim, int64_t dimSize) const {
    const RuntimeBits rb = runtimeBits();
    const int32_t constPart = registerConstant(reg);
    if (!rb.disjoint || (constPart & rb.mask))
      return CoordRange{dim, 0, dimSize - 1};
    return CoordRange{dim, constPart, constPart | rb.mask};
  }
};

// The register bases in the shape `planAccess` reads: `[bit][dim]`, the
// transpose of one BasisRow per dimension.
inline RegBases regBasesOf(const std::vector<LayoutBasis> &dims) {
  std::size_t bits = 0;
  for (const LayoutBasis &lb : dims)
    bits = std::max(bits, lb.reg.size());
  RegBases out(bits, std::vector<int32_t>(dims.size(), 0));
  for (std::size_t d = 0; d < dims.size(); ++d)
    for (std::size_t b = 0; b < dims[d].reg.size(); ++b)
      out[b][d] = dims[d].reg[b];
  return out;
}

// Condition (c)'s input: everything the runtime ids contribute, per dimension.
inline RuntimeSpan runtimeSpanOf(const std::vector<LayoutBasis> &dims) {
  RuntimeSpan out(dims.size(), 0);
  for (std::size_t d = 0; d < dims.size(); ++d)
    for (const BasisRow *row : {&dims[d].lane, &dims[d].warp, &dims[d].block})
      for (int32_t b : *row)
        out[d] |= b;
  return out;
}

} // namespace agpu

#endif // AGPU_LAYOUT_BASIS_H
