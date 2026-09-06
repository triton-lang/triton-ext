// Elementwise.h - the per-element operations and what type they produce.
#ifndef AGPU_ELEMENTWISE_H
#define AGPU_ELEMENTWISE_H

#include "agpu/core/Decline.h"
#include "agpu/msl/Ast.h"
#include "agpu/msl/Builtins.h"
#include "agpu/plan/ElemType.h"

#include <cstdint>
#include <vector>

namespace agpu {

// The elementwise operations and what each needs from its operands.
enum class EwOp {
  // Float and integer alike.
  Add,
  Sub,
  Mul,
  // Signedness-sensitive: the operation decides.
  DivS,
  DivU,
  RemS,
  RemU,
  // Float division and remainder. DivS/DivU promote their operands to a
  // signedness, which is meaningless for a float and `%` does not apply to
  // floats: RemF is `fmod`.
  DivF,
  RemF,
  // Bitwise.
  And,
  Or,
  Xor,
  Shl,
  ShrS,
  ShrU,
  // Comparisons, which produce bool whatever they consume.
  CmpEq,
  CmpNe,
  CmpLtS,
  CmpLtU,
  CmpLeS,
  CmpLeU,
  CmpGtS,
  CmpGtU,
  CmpGeS,
  CmpGeU,
};

// A property of the operation. MSL has one
// `/` and one `>>`, decided by the operand type, so `divui` on an i32 promotes
// both operands to uint and the result matches.
inline bool needsUnsigned(EwOp op) {
  switch (op) {
  case EwOp::DivU:
  case EwOp::RemU:
  case EwOp::ShrU:
  case EwOp::CmpLtU:
  case EwOp::CmpLeU:
  case EwOp::CmpGtU:
  case EwOp::CmpGeU:
    return true;
  default:
    return false;
  }
}

// fp8 is excluded: it has no MSL type and travels as a byte, so widening it
// would spell `uchar4(float, ...)` and truncate toward zero.
inline bool widensToF32(const ElemType &e) {
  return e.kind == ElemType::Kind::Float && e.bits == 16 &&
         (e.floatKind == FloatKind::Ieee || e.floatKind == FloatKind::Brain);
}

// MSL leaves the evaluation width of `bfloat a = b * c` unspecified: AGX2
// rounds once at f32, AGX3 at bf16. Both operands carry the ambiguity, so
// widening the result alone changes nothing.
inline ElemType evalWidthFor(ElemType elem) {
  return widensToF32(elem) ? f32() : elem;
}

inline bool isComparison(EwOp op) {
  switch (op) {
  case EwOp::CmpEq:
  case EwOp::CmpNe:
  case EwOp::CmpLtS:
  case EwOp::CmpLtU:
  case EwOp::CmpLeS:
  case EwOp::CmpLeU:
  case EwOp::CmpGtS:
  case EwOp::CmpGtU:
  case EwOp::CmpGeS:
  case EwOp::CmpGeU:
    return true;
  default:
    return false;
  }
}

struct EwTypes {
  ElemType result;
  ElemType operand;
};

inline EwTypes typesFor(EwOp op, ElemType elem) {
  EwTypes t;
  t.operand = elem;
  t.result = elem;

  // A comparison yields bool regardless of what it consumed.
  if (isComparison(op))
    t.result = i1();

  // i1 arithmetic stays i1: promoting to int makes `true + true` two.
  if (elem.kind == ElemType::Kind::Bool)
    return t;

  if (needsUnsigned(op)) {
    t.operand.isUnsigned = true;
    if (!isComparison(op))
      t.result.isUnsigned = true;
  }
  return t;
}

// Signed and unsigned variants map to the same operator; MSL distinguishes
// them by operand type.
struct EwSpelling {
  EwOp op;
  msl::BinOp binOp;

  bool intOnly = false;   // bitwise, shifts, integer remainder
  bool floatOnly = false; // float division
  const char *because = "";

  // MSL has no 1-bit arithmetic: `bool + bool` promotes to int, adds and
  // converts back, so `true + true` is `true`. i1 add and sub wrap mod 2
  // (xor), multiply is and.
  bool wrapsOnBool = false;
  msl::BinOp boolOp = msl::BinOp::Xor;

  msl::BinOp opFor(ElemType elem) const {
    if (wrapsOnBool && elem.kind == ElemType::Kind::Bool)
      return boolOp;
    return binOp;
  }
};

// One modulo for a register family, the rest constant deltas.
//
// Inductor stamps `tt.contiguity = extent` on index tensors it guarded with a
// safety modulo, asserting the values along that axis are consecutive across
// the whole tile. Every element's value is then register 0's plus its axis
// coordinate, so registers 1..N-1 are one add each.
//
// `ok` only when the registers vary along exactly one axis and the contiguity
// covers that axis whole.
struct RemFold {
  bool ok = false;
  int axis = -1;               // the contiguous axis the values track
  std::vector<int64_t> deltas; // per register, from register 0
};

inline RemFold planRemFold(const std::vector<std::vector<int64_t>> &coords,
                           const std::vector<int64_t> &shape,
                           int64_t contiguity) {
  RemFold f;
  if (coords.size() < 2)
    return f;
  int axis = -1;
  const std::vector<int64_t> &c0 = coords.front();
  for (const std::vector<int64_t> &cr : coords) {
    if (cr.size() != c0.size())
      return f;
    for (std::size_t d = 0; d < cr.size(); ++d)
      if (cr[d] != c0[d]) {
        if (axis >= 0 && axis != (int)d)
          return f;
        axis = (int)d;
      }
  }
  if (axis < 0 || axis >= (int)shape.size() || contiguity < shape[axis])
    return f;
  for (const std::vector<int64_t> &cr : coords)
    f.deltas.push_back(cr[axis] - c0[axis]);
  f.axis = axis;
  f.ok = true;
  return f;
}

// A value affine in its coordinates: every element equals some uniform base
// plus the dot product of `scales` with its coordinate vector. `tt.addptr`
// sums it into the pointer, where the load emitter turns it into one
// materialised base and literal subscripts.
struct AffineFamily {
  std::vector<int64_t> scales; // one per tensor axis; empty = no family
  bool ok() const { return !scales.empty(); }
};

inline AffineFamily uniformFamily(int rank) {
  AffineFamily f;
  f.scales.assign((std::size_t)std::max(rank, 0), 0);
  return f;
}

// How a binary op transforms the family. The bridge supplies each side's
// family, whether it is register-uniform (a uniform side is the all-zero
// family) and the uniform side's compile-time value where scaling needs it.
inline AffineFamily foldFamily(EwOp op, AffineFamily fa, bool aUniform,
                               AffineFamily fb, bool bUniform,
                               const int64_t *aConst, const int64_t *bConst,
                               int rank) {
  const AffineFamily none;
  if (rank <= 0)
    return none;
  if (!fa.ok() && aUniform)
    fa = uniformFamily(rank);
  if (!fb.ok() && bUniform)
    fb = uniformFamily(rank);
  if (!fa.ok() || !fb.ok() || (int)fa.scales.size() != rank ||
      (int)fb.scales.size() != rank)
    return none;
  const auto allZero = [](const AffineFamily &f) {
    for (int64_t v : f.scales)
      if (v)
        return false;
    return true;
  };
  AffineFamily out = uniformFamily(rank);
  switch (op) {
  case EwOp::Add:
    for (int d = 0; d < rank; ++d)
      out.scales[(std::size_t)d] =
          fa.scales[(std::size_t)d] + fb.scales[(std::size_t)d];
    return out;
  case EwOp::Sub:
    for (int d = 0; d < rank; ++d)
      out.scales[(std::size_t)d] =
          fa.scales[(std::size_t)d] - fb.scales[(std::size_t)d];
    return out;
  case EwOp::Mul:
    // Affine times affine is affine only when one side is a constant.
    if (allZero(fb) && bConst) {
      for (int d = 0; d < rank; ++d)
        out.scales[(std::size_t)d] = fa.scales[(std::size_t)d] * *bConst;
      return out;
    }
    if (allZero(fa) && aConst) {
      for (int d = 0; d < rank; ++d)
        out.scales[(std::size_t)d] = fb.scales[(std::size_t)d] * *aConst;
      return out;
    }
    return none;
  default:
    return none;
  }
}

inline constexpr EwSpelling kEwSpellings[] = {
    // op          operator         intOnly floatOnly  because  i1 operator
    {EwOp::Add, msl::BinOp::Add, false, false, "", true, msl::BinOp::Xor},
    {EwOp::Sub, msl::BinOp::Sub, false, false, "", true, msl::BinOp::Xor},
    {EwOp::Mul, msl::BinOp::Mul, false, false, "", true, msl::BinOp::And},
    {EwOp::DivS, msl::BinOp::Div, true, false,
     "integer division of a float; use DivF"},
    {EwOp::DivU, msl::BinOp::Div, true, false,
     "integer division of a float; use DivF"},
    {EwOp::DivF, msl::BinOp::Div, false, true, "float division of a non-float"},
    {EwOp::RemS, msl::BinOp::Rem, true, false, "integer remainder on a float"},
    {EwOp::RemU, msl::BinOp::Rem, true, false, "integer remainder on a float"},
    {EwOp::And, msl::BinOp::And, true, false, "bitwise op on a float"},
    {EwOp::Or, msl::BinOp::Or, true, false, "bitwise op on a float"},
    {EwOp::Xor, msl::BinOp::Xor, true, false, "bitwise op on a float"},
    {EwOp::Shl, msl::BinOp::Shl, true, false, "bitwise op on a float"},
    {EwOp::ShrS, msl::BinOp::Shr, true, false, "bitwise op on a float"},
    {EwOp::ShrU, msl::BinOp::Shr, true, false, "bitwise op on a float"},
    {EwOp::CmpEq, msl::BinOp::Eq},
    {EwOp::CmpNe, msl::BinOp::Ne},
    {EwOp::CmpLtS, msl::BinOp::Lt},
    {EwOp::CmpLtU, msl::BinOp::Lt},
    {EwOp::CmpLeS, msl::BinOp::Le},
    {EwOp::CmpLeU, msl::BinOp::Le},
    {EwOp::CmpGtS, msl::BinOp::Gt},
    {EwOp::CmpGtU, msl::BinOp::Gt},
    {EwOp::CmpGeS, msl::BinOp::Ge},
    {EwOp::CmpGeU, msl::BinOp::Ge},
    // RemF has no operator: `%` does not apply to floats. It is `fmod`,
    // routed to the math family by checkEw.
};

// The table row for an operation, or null when it has no spelling.
inline const EwSpelling *spellingRow(EwOp op) {
  for (const EwSpelling &s : kEwSpellings)
    if (s.op == op)
      return &s;
  return nullptr;
}

inline bool spellingOf(EwOp op, msl::BinOp &out) {
  const EwSpelling *s = spellingRow(op);
  if (!s)
    return false;
  out = s->binOp;
  return true;
}

// The same, for a given element type, which on i1 is not always the same
// operator.
inline bool spellingOf(EwOp op, ElemType elem, msl::BinOp &out) {
  const EwSpelling *s = spellingRow(op);
  if (!s)
    return false;
  out = s->opFor(elem);
  return true;
}

// ── float comparison ──────────────────────────────────────────────────────

// NaN compares false against everything including itself, so each relation
// exists in an ordered form (false if either operand is NaN) and an unordered
// form (true if either is). MSL's operators are the ordered ones except `!=`,
// so half of these need an explicit isnan() term.
enum class FCmp {
  False, // never true
  OEq,
  OGt,
  OGe,
  OLt,
  OLe,
  ONe,
  Ord, // neither is NaN
  UEq,
  UGt,
  UGe,
  ULt,
  ULe,
  UNe,
  Uno,  // either is NaN
  True, // always true
};

// What the emitter has to build around the bare operator.
enum class NanGuard {
  None,    // the bare operator already has the right sense
  MaskOut, // ordered predicate, unordered operator: !(isnan a || isnan b) &&
           // cmp
  OrIn,    // unordered predicate, ordered operator: isnan a || isnan b || cmp
};

// How a float comparison is emitted.
struct FCmpPlan {
  enum class Kind {
    Constant, // False / True: no comparison at all
    NanTest,  // Ord / Uno: only the isnan terms
    Relation, // an operator, possibly guarded
  };
  Kind kind = Kind::Relation;
  msl::BinOp op = msl::BinOp::Eq;  // Relation only
  NanGuard guard = NanGuard::None; // Relation only
  bool constantValue = false;      // Constant only
  bool wantNan = false;            // NanTest: Uno true, Ord false
};

inline FCmpPlan planFCmp(FCmp p) {
  FCmpPlan r;
  switch (p) {
  case FCmp::False:
    r.kind = FCmpPlan::Kind::Constant;
    r.constantValue = false;
    return r;
  case FCmp::True:
    r.kind = FCmpPlan::Kind::Constant;
    r.constantValue = true;
    return r;
  case FCmp::Ord:
    r.kind = FCmpPlan::Kind::NanTest;
    r.wantNan = false;
    return r;
  case FCmp::Uno:
    r.kind = FCmpPlan::Kind::NanTest;
    r.wantNan = true;
    return r;

  // The ordered relations map onto MSL's operators, except ONe: `!=` is the
  // one unordered operator MSL has.
  case FCmp::OEq:
    r.op = msl::BinOp::Eq;
    return r;
  case FCmp::OGt:
    r.op = msl::BinOp::Gt;
    return r;
  case FCmp::OGe:
    r.op = msl::BinOp::Ge;
    return r;
  case FCmp::OLt:
    r.op = msl::BinOp::Lt;
    return r;
  case FCmp::OLe:
    r.op = msl::BinOp::Le;
    return r;
  case FCmp::ONe:
    r.op = msl::BinOp::Ne;
    r.guard = NanGuard::MaskOut;
    return r;

  // Unordered relations need the NaN cases or'd back in, except UNe.
  case FCmp::UEq:
    r.op = msl::BinOp::Eq;
    r.guard = NanGuard::OrIn;
    return r;
  case FCmp::UGt:
    r.op = msl::BinOp::Gt;
    r.guard = NanGuard::OrIn;
    return r;
  case FCmp::UGe:
    r.op = msl::BinOp::Ge;
    r.guard = NanGuard::OrIn;
    return r;
  case FCmp::ULt:
    r.op = msl::BinOp::Lt;
    r.guard = NanGuard::OrIn;
    return r;
  case FCmp::ULe:
    r.op = msl::BinOp::Le;
    r.guard = NanGuard::OrIn;
    return r;
  case FCmp::UNe:
    r.op = msl::BinOp::Ne;
    return r;
  }
  return r;
}

// Whether an operation applies to an element type at all.
inline Decision checkEw(EwOp op, ElemType elem) {
  if (op == EwOp::RemF)
    return Decision::declined("elementwise", "float remainder is fmod");

  const EwSpelling *s = spellingRow(op);
  if (!s)
    return Decision::declined("elementwise", "no spelling for this operation");

  const bool isFloat = elem.kind == ElemType::Kind::Float;
  if ((s->intOnly && isFloat) || (s->floatOnly && !isFloat))
    return Decision::declined("elementwise", s->because);

  return Decision::emitted();
}

} // namespace agpu

#endif // AGPU_ELEMENTWISE_H
