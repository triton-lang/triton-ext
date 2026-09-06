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
