// EpilogueOps - one table for the ops a fused dot epilogue can render, read by
// both the recogniser and the renderer.
//
// A fused dot's drain folds a chain of these ops between its last MMA and its
// device store. An op folds when its second operand is readable at C's own
// coordinates: a splat, or a proven device window at the store's row and
// column starts. An op that moves elements across threads (reduce, trans, any
// layout change) never enters.
#ifndef AGPU_EPILOGUE_OPS_H
#define AGPU_EPILOGUE_OPS_H

#include "agpu/plan/Elementwise.h"
#include "agpu/plan/MathFn.h"

#include <cstddef>
#include <string_view>

namespace agpu {

// Spelling stays in the printer.
enum class EpilogueBinOp {
  Add,
  Sub,
  Mul,
  Div,
  // `metal::max`/`min` implement IEEE maxNum/minNum: a NaN operand is dropped.
  Max,
  Min,
  // The NaN-propagating variants, `arith.maximumf`/`minimumf`. The renderer
  // spells the guarded form; see `minMaxPropagatesNan`.
  MaxPropagate,
  MinPropagate,
};

struct EpilogueBinary {
  std::string_view op; // the MLIR operation name
  EpilogueBinOp bin;
};

inline constexpr EpilogueBinary kEpilogueBinary[] = {
    {"arith.addf", EpilogueBinOp::Add},
    {"arith.subf", EpilogueBinOp::Sub},
    {"arith.mulf", EpilogueBinOp::Mul},
    {"arith.divf", EpilogueBinOp::Div},
    {"tt.precise_divf", EpilogueBinOp::Div},
    {"arith.maxnumf", EpilogueBinOp::Max},
    {"arith.minnumf", EpilogueBinOp::Min},
    {"arith.maximumf", EpilogueBinOp::MaxPropagate},
    {"arith.minimumf", EpilogueBinOp::MinPropagate},
};

// Returns nullptr for an op this table does not cover.
inline const EpilogueBinary *epilogueBinaryFor(std::string_view op) {
  for (const EpilogueBinary &e : kEpilogueBinary)
    if (e.op == op)
      return &e;
  return nullptr;
}

// The recogniser's question.
inline bool isEpilogueBinary(std::string_view op) {
  return epilogueBinaryFor(op) != nullptr;
}

inline bool epilogueBinOpOf(std::string_view op, EpilogueBinOp &out) {
  if (const EpilogueBinary *e = epilogueBinaryFor(op)) {
    out = e->bin;
    return true;
  }
  return false;
}

inline constexpr std::size_t epilogueBinaryCount() {
  return sizeof(kEpilogueBinary) / sizeof(kEpilogueBinary[0]);
}

// ── the unary half ────────────────────────────────────────────────────────

// `fusable` is a policy: all of these have an MSL spelling.
// Folding into the fragment loop evaluates the op once per K step, so it
// costs kT evaluations per output.
struct EpilogueUnary {
  std::string_view op; // the MLIR operation name
  MathFn fn;           // what it lowers to
  bool fusable;        // may be folded into the fragment loop
};

inline constexpr EpilogueUnary kEpilogueUnary[] = {
    // Cheap and elementwise.
    {"math.absf", MathFn::Abs, true},
    {"math.floor", MathFn::Floor, true},
    {"math.ceil", MathFn::Ceil, true},
    {"math.trunc", MathFn::Trunc, true},
    {"math.round", MathFn::Round, true},
    {"math.roundeven", MathFn::RoundEven, true},

    // Transcendentals: correct to fold, but evaluated once per K step.
    {"math.exp", MathFn::Exp, false},
    {"math.exp2", MathFn::Exp2, false},
    {"math.log", MathFn::Log, false},
    {"math.log2", MathFn::Log2, false},
    {"math.sqrt", MathFn::Sqrt, false},
    {"math.rsqrt", MathFn::Rsqrt, false},
    {"math.tanh", MathFn::Tanh, false},
    {"math.erf", MathFn::Erf, false},
    {"math.cbrt", MathFn::Cbrt, false},
    {"math.sin", MathFn::Sin, false},
    {"math.cos", MathFn::Cos, false},
    {"math.tan", MathFn::Tan, false},
    {"math.asin", MathFn::Asin, false},
    {"math.acos", MathFn::Acos, false},
    {"math.atan", MathFn::Atan, false},
    {"math.sinh", MathFn::Sinh, false},
    {"math.cosh", MathFn::Cosh, false},
};

inline const EpilogueUnary *epilogueUnaryFor(std::string_view op) {
  for (const EpilogueUnary &e : kEpilogueUnary)
    if (e.op == op)
      return &e;
  return nullptr;
}

// The recogniser's question, with the policy applied.
inline bool isEpilogueUnary(std::string_view op, bool fusableOnly = true) {
  const EpilogueUnary *e = epilogueUnaryFor(op);
  return e && (!fusableOnly || e->fusable);
}

inline bool epilogueUnaryFnOf(std::string_view op, MathFn &out) {
  if (const EpilogueUnary *e = epilogueUnaryFor(op)) {
    out = e->fn;
    return true;
  }
  return false;
}

inline constexpr std::size_t epilogueUnaryCount() {
  return sizeof(kEpilogueUnary) / sizeof(kEpilogueUnary[0]);
}

// An epilogue op of either arity. Whether it should be folded is
// `isEpilogueUnary(op, true)`.
inline bool isEpilogueOp(std::string_view op) {
  return isEpilogueBinary(op) || isEpilogueUnary(op, /*fusableOnly=*/false);
}

} // namespace agpu

#endif // AGPU_EPILOGUE_OPS_H
