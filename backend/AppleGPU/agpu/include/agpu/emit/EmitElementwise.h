// EmitElementwise.h - the per-element operations, emitted.
//
// NaN compares false against everything including itself, so MSL's ordered
// operators need an explicit isnan() term for half the float predicates.
#ifndef AGPU_EMIT_ELEMENTWISE_H
#define AGPU_EMIT_ELEMENTWISE_H

#include "agpu/msl/Builtins.h"
#include "agpu/msl/Context.h"
#include "agpu/plan/Elementwise.h"
#include "agpu/plan/MathFn.h"

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
  const ElemType want = evalWidthFor(t.operand);
  if (!(want == elem)) {
    const msl::Type to = mslTypeOf(want);
    a = c.cast(to, a);
    b = c.cast(to, b);
  }
  return c.binary(bo, a, b);
}

inline msl::Stmt *emitEw(msl::Context &c, EwOp op, ElemType elem,
                         const msl::Str &dst, msl::Expr *a, msl::Expr *b) {
  const EwTypes t = typesFor(op, elem);
  return c.declStmt(mslTypeOf(evalWidthFor(t.result)), dst,
                    ewExpr(c, op, elem, a, b));
}

// ── NaN ───────────────────────────────────────────────────────────────────

inline msl::Expr *isNanExpr(msl::Context &c, msl::Expr *v) {
  return c.call(msl::builtin::math::Isnan, {v});
}

inline msl::Expr *eitherIsNan(msl::Context &c, msl::Expr *a, msl::Expr *b) {
  return c.binary(msl::BinOp::LOr, isNanExpr(c, a), isNanExpr(c, b));
}

// ── float comparison ──────────────────────────────────────────────────────

// Operands are names: the guarded forms mention each operand twice (operator
// and isnan). Bind a computed value before calling.
inline msl::Expr *fcmpExpr(msl::Context &c, FCmp pred, const msl::Str &a,
                           const msl::Str &b) {
  const FCmpPlan p = planFCmp(pred);
  switch (p.kind) {
  case FCmpPlan::Kind::Constant:
    return c.litBool(p.constantValue);

  case FCmpPlan::Kind::NanTest: {
    msl::Expr *any = eitherIsNan(c, c.var(a), c.var(b));
    return p.wantNan ? any : c.unary(msl::UnOp::LNot, any);
  }

  case FCmpPlan::Kind::Relation: {
    msl::Expr *cmp = c.binary(p.op, c.var(a), c.var(b));
    switch (p.guard) {
    case NanGuard::None:
      return cmp;
    case NanGuard::MaskOut:
      return c.binary(
          msl::BinOp::LAnd,
          c.unary(msl::UnOp::LNot, eitherIsNan(c, c.var(a), c.var(b))), cmp);
    case NanGuard::OrIn:
      // NaN cases the ordered operator dropped go back in.
      return c.binary(msl::BinOp::LOr, eitherIsNan(c, c.var(a), c.var(b)), cmp);
    }
    return cmp;
  }
  }
  return nullptr;
}

inline msl::Stmt *emitFCmp(msl::Context &c, FCmp pred, const msl::Str &dst,
                           const msl::Str &a, const msl::Str &b) {
  return c.declStmt(mslTypeOf(i1()), dst, fcmpExpr(c, pred, a, b));
}

// ── min and max ───────────────────────────────────────────────────────────

// The NaN arm is `a + b`: it yields a NaN of the right type without naming
// one per float width.
inline msl::Expr *minMaxExprOf(msl::Context &c, MathFn2 fn, ElemType elem,
                               msl::Expr *a, msl::Expr *b, bool propagateNan) {
  msl::Expr *call = c.call(mathNameOf(fn), {a, b});
  if (!minMaxPropagatesNan(fn, elem, propagateNan))
    return call;
  return c.ternary(eitherIsNan(c, a, b), c.binary(msl::BinOp::Add, a, b), call);
}

inline msl::Expr *minMaxExpr(msl::Context &c, MathFn2 fn, ElemType elem,
                             const msl::Str &a, const msl::Str &b,
                             bool propagateNan) {
  return minMaxExprOf(c, fn, elem, c.var(a), c.var(b), propagateNan);
}

// ── the call-shaped math ──────────────────────────────────────────────────

// The operand type picks the spelling (`abs` is ambiguous on a float, `fabs`
// doesn't exist for integers) and whether the result needs narrowing back.
inline msl::Expr *mathExpr(msl::Context &c, MathFn fn, ElemType operand,
                           msl::Expr *v) {
  // fp8's `abs` is bit arithmetic (clear the sign bit); the call form does
  // not compile.
  if (const int64_t mask = mathBitMaskOf(fn, operand))
    return c.binary(msl::BinOp::And, v, c.lit(mask));

  msl::Expr *call = c.call(mathNameOf(fn, operand), {v});
  if (!mathResultNarrows(fn, operand))
    return call;
  return c.cast(mslTypeOf(operand), call);
}

inline msl::Stmt *emitMath(msl::Context &c, MathFn fn, ElemType elem,
                           const msl::Str &dst, msl::Expr *v) {
  return c.declStmt(mslTypeOf(mathResultType(fn, elem)), dst,
                    mathExpr(c, fn, elem, v));
}

inline msl::Expr *mathExpr(msl::Context &c, MathFn2 fn, msl::Expr *a,
                           msl::Expr *b) {
  return c.call(mathNameOf(fn), {a, b});
}

inline msl::Expr *mathExpr(msl::Context &c, MathFn3 fn, msl::Expr *a,
                           msl::Expr *b, msl::Expr *d) {
  return c.call(mathNameOf(fn), {a, b, d});
}

// `metal::clamp` drops NaN like min/max do. Under `propagateNan` the tested
// value is returned instead; it is a name, since it is mentioned twice.
inline msl::Expr *clampExpr(msl::Context &c, MathFn3 fn, ElemType elem,
                            const msl::Str &v, msl::Expr *lo, msl::Expr *hi,
                            bool propagateNan) {
  msl::Expr *call = mathExpr(c, fn, c.var(v), lo, hi);
  if (!math3PropagatesNan(fn, elem, propagateNan))
    return call;
  return c.ternary(isNanExpr(c, c.var(v)), c.var(v), call);
}

// ── select ────────────────────────────────────────────────────────────────

// A ternary: both arms are already in registers.
inline msl::Stmt *emitSelect(msl::Context &c, ElemType elem,
                             const msl::Str &dst, msl::Expr *cond,
                             msl::Expr *ifTrue, msl::Expr *ifFalse) {
  return c.declStmt(mslTypeOf(elem), dst, c.ternary(cond, ifTrue, ifFalse));
}

} // namespace agpu

#endif // AGPU_EMIT_ELEMENTWISE_H
