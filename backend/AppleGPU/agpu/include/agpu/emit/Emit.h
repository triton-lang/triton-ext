// Emit.h - turning plans into MSL.
#ifndef AGPU_EMIT_H
#define AGPU_EMIT_H

#include "agpu/core/Names.h"
#include "agpu/emit/LayoutExpr.h"
#include "agpu/emit/primitives/CoordHoist.h"
#include "agpu/msl/Context.h"

namespace agpu {

struct CoordSource : ThreadNames {
  std::vector<LayoutBasis> dims; // one per output dimension

  // Null builds the expression inline. Set to share hoisted names across all
  // consumers of this source.
  CoordHoist *hoist = nullptr;

  msl::Expr *of(msl::Context &c, int reg, int dim) const {
    if (hoist)
      return hoist->coord(c, dims[dim], reg);
    return coordExpr(c, dims[dim], reg, laneId, warpId, blockId);
  }
};

} // namespace agpu

#endif // AGPU_EMIT_H
