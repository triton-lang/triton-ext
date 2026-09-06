// Accessing through a memdesc handle.
//
// MemDesc holds where an element lives; this spells it in MSL. A compile-time
// coordinate folds to one literal subscript; a runtime one builds
// `base + sum(c_d * stride_d)` from the view's strides.
#ifndef AGPU_EMIT_MEMDESC_H
#define AGPU_EMIT_MEMDESC_H

#include "agpu/core/MemDesc.h"
#include "agpu/msl/Context.h"

#include <vector>

namespace agpu {

// The address of a compile-time coordinate: `buf[k]`, one literal.
inline msl::Expr *memDescElem(msl::Context &c, const MemDesc &m,
                              const TileView::Coord &at) {
  return c.subscript(c.var(m.buffer), c.lit(m.offsetOf(at)));
}

// The address of a coordinate whose components are runtime expressions.
// `at[d]` may be null, meaning that dimension is at zero.
inline msl::Expr *memDescElemAt(msl::Context &c, const MemDesc &m,
                                const std::vector<msl::Expr *> &at) {
  const TileView::Coord origin(at.size(), 0);
  msl::Expr *sum = c.lit(m.offsetOf(origin));

  for (std::size_t d = 0; d < at.size(); ++d) {
    if (!at[d])
      continue;
    const int64_t stride = m.view.strideAt((int)d);
    if (stride == 0)
      continue;
    msl::Expr *term =
        stride == 1 ? at[d] : c.binary(msl::BinOp::Mul, at[d], c.lit(stride));
    sum = c.binary(msl::BinOp::Add, sum, term);
  }
  return c.subscript(c.var(m.buffer), sum);
}

// The declaration a memdesc's buffer needs: `threadgroup T name[cosize]`.
inline msl::Stmt *memDescDecl(msl::Context &c, const MemDesc &m,
                              msl::Type elem) {
  return c.arrayDecl(elem.inAddrSpace(msl::AddrSpace::Threadgroup), m.buffer,
                     m.cosizeElems());
}

} // namespace agpu

#endif // AGPU_EMIT_MEMDESC_H
