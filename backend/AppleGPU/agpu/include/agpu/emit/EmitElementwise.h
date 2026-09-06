// EmitElementwise.h - the per-element operations, emitted.
//
// NaN compares false against everything including itself, so MSL's ordered
// operators need an explicit isnan() term for half the float predicates.
#ifndef AGPU_EMIT_ELEMENTWISE_H
#define AGPU_EMIT_ELEMENTWISE_H

#include "agpu/msl/Builtins.h"
#include "agpu/msl/Context.h"
#include "agpu/plan/Elementwise.h"

namespace agpu {

// Operand promotion and result type both come from `typesFor`: `divui` on an
// i32 reads both operands as uint and declares its result uint.
inline msl::Expr *ewExpr(msl::Context &c, EwOp op, ElemType elem, msl::Expr *a,
                         msl::Expr *b) {
  // MSL has no 1-bit arithmetic, so on i1 three operators differ from their
  // names.
  msl::BinOp bo;
  if (!spellingOf(op, elem, bo))
    return nullptr;

  const EwTypes t = typesFor(op, elem);
  if (!(t.operand == elem)) {
    const msl::Type to = mslTypeOf(t.operand);
    a = c.cast(to, a);
    b = c.cast(to, b);
  }
  return c.binary(bo, a, b);
}

} // namespace agpu

#endif // AGPU_EMIT_ELEMENTWISE_H
