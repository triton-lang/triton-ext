// EmitEpilogue.h - the elementwise region folded into a dot.
//
// A chain of elementwise operations applied to the accumulator:
// `(acc * alpha + bias)`, `exp(acc)`. Folded into the fragment loop, so the
// values never round-trip through the pool.
#ifndef AGPU_EMIT_EPILOGUE_H
#define AGPU_EMIT_EPILOGUE_H

#include "agpu/emit/EmitElementwise.h"
#include "agpu/plan/EpilogueOps.h"

namespace agpu {

// One epilogue step: an operation name and its right operand if it has one.
struct EpilogueStep {
  std::string_view op;
  // The second operand, for a binary step; null for a unary one. Arity comes
  // from the table.
  msl::Expr *rhs = nullptr;
};

// The accumulator is fp32 until a step rounds it to the tensor's element, so
// the caller states what it holds. Max and Min are calls, spelled by
// epilogueExpr through MathFn2.
inline msl::BinOp epilogueOperator(EpilogueBinOp op) {
  switch (op) {
  case EpilogueBinOp::Sub:
    return msl::BinOp::Sub;
  case EpilogueBinOp::Mul:
    return msl::BinOp::Mul;
  case EpilogueBinOp::Div:
    return msl::BinOp::Div;
  default:
    break;
  }
  return msl::BinOp::Add;
}

// One step applied to `acc`, or null when the op is not an epilogue op.
inline msl::Expr *epilogueExpr(msl::Context &c, const EpilogueStep &step,
                               msl::Expr *acc, ElemType elem = f32()) {
  if (!acc)
    return nullptr;

  if (EpilogueBinOp bin; epilogueBinOpOf(step.op, bin)) {
    if (!step.rhs)
      return nullptr;
    switch (bin) {
    case EpilogueBinOp::Max:
    case EpilogueBinOp::MaxPropagate:
      return minMaxExprOf(c, MathFn2::Max, elem, acc, step.rhs,
                          bin == EpilogueBinOp::MaxPropagate);
    case EpilogueBinOp::Min:
    case EpilogueBinOp::MinPropagate:
      return minMaxExprOf(c, MathFn2::Min, elem, acc, step.rhs,
                          bin == EpilogueBinOp::MinPropagate);
    default:
      break;
    }
    return c.binary(epilogueOperator(bin), acc, step.rhs);
  }

  if (MathFn fn; epilogueUnaryFnOf(step.op, fn))
    return mathExpr(c, fn, elem, acc);

  return nullptr;
}

// A whole chain, applied in order: each step consumes the previous result.
inline msl::Expr *epilogueChain(msl::Context &c,
                                const std::vector<EpilogueStep> &steps,
                                msl::Expr *acc) {
  msl::Expr *cur = acc;
  for (const EpilogueStep &s : steps) {
    cur = epilogueExpr(c, s, cur);
    if (!cur)
      return nullptr;
  }
  return cur;
}

// Whether a chain can be folded into the fragment loop. A transcendental is
// correct to fold but costly: inside a K loop it runs once per step rather
// than once per output.
inline Decision epilogueDecision(const std::vector<EpilogueStep> &steps) {
  for (const EpilogueStep &s : steps) {
    if (!isEpilogueOp(s.op))
      return Decision::declined("epilogue", "operation cannot be fused");
    if (!isEpilogueBinary(s.op) && !isEpilogueUnary(s.op, /*fusableOnly=*/true))
      return Decision::declined("epilogue",
                                "operation is too costly to fold per step");
  }
  return Decision::emitted();
}

} // namespace agpu

#endif // AGPU_EMIT_EPILOGUE_H
