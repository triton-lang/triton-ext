// make_range. Register `r` holds `start + coord(r)`, using LayoutExpr's
// coordinate expression.
#ifndef AGPU_EMIT_RANGE_H
#define AGPU_EMIT_RANGE_H

#include "agpu/emit/LayoutExpr.h"

namespace agpu {

inline msl::Expr *rangeElem(msl::Context &c, const LayoutBasis &lb, int reg,
                            int64_t start, const msl::Str &laneId,
                            const msl::Str &warpId,
                            const msl::Str &blockId = {}) {
  msl::Expr *coord = coordExpr(c, lb, reg, laneId, warpId, blockId);
  if (start == 0)
    return coord;
  return c.binary(msl::BinOp::Add, c.lit((int32_t)start), coord);
}

} // namespace agpu

#endif // AGPU_EMIT_RANGE_H
