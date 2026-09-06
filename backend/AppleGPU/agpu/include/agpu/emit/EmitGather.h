// EmitGather.h - reading each result register at an index-selected offset.
#ifndef AGPU_EMIT_GATHER_H
#define AGPU_EMIT_GATHER_H

#include "agpu/msl/Context.h"
#include "agpu/plan/ElemType.h"

#include <cassert>

namespace agpu {

// An out-of-range index is undefined in the IR, but must not read past the
// scratch tile. Tile extents are powers of two, so confinement is one AND.
inline msl::Expr *gatherIndexExpr(msl::Context &c, const msl::Str &index,
                                  int64_t extent) {
  assert(extent > 0 && (extent & (extent - 1)) == 0 &&
         "gather extents are powers of two");
  return c.binary(msl::BinOp::And, c.var(index), c.lit(extent - 1));
}

// `offsets[r]` is where result register `r` reads: an expression, since the
// gathered axis comes from a runtime index and the rest from the layout.
inline void emitGather(msl::Context &c, msl::Block &body,
                       const msl::Str &buffer,
                       const msl::SmallVec<msl::Expr *, 8> &offsets,
                       const msl::SmallVec<msl::Str, 8> &out, ElemType elem) {
  for (std::size_t r = 0; r < out.size() && r < offsets.size(); ++r)
    body.push_back(c.declStmt(mslTypeOf(elem), out[r],
                              c.subscript(c.var(buffer), offsets[r])));
}

} // namespace agpu

#endif // AGPU_EMIT_GATHER_H
