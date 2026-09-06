// ub.poison. MSL has no undef, so poison is zero (null for pointers).
#ifndef AGPU_EMIT_POISON_H
#define AGPU_EMIT_POISON_H

#include "agpu/msl/Context.h"
#include "agpu/plan/Elementwise.h"

namespace agpu {

inline msl::Expr *poisonValue(msl::Context &c, ElemType elem) {
  const msl::Type t = mslTypeOf(elem);
  if (elem.kind == ElemType::Kind::Float)
    return c.litF(0.0, t);
  return c.lit(0, t);
}

inline msl::Expr *poisonPointer(msl::Context &c, msl::Type ptrType) {
  return c.litNull(std::move(ptrType));
}

inline msl::Stmt *poisonDecl(msl::Context &c, const msl::Str &dst,
                             ElemType elem) {
  return c.declStmt(mslTypeOf(elem), dst, poisonValue(c, elem));
}

inline msl::Stmt *poisonPointerDecl(msl::Context &c, const msl::Str &dst,
                                    msl::Type ptrType) {
  return c.declStmt(ptrType, dst, poisonPointer(c, ptrType));
}

} // namespace agpu

#endif // AGPU_EMIT_POISON_H
