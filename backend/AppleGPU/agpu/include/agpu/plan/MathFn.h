// MathFn.h - the math intrinsics, their MSL names and what each accepts.
#ifndef AGPU_MATH_FN_H
#define AGPU_MATH_FN_H

#include "agpu/core/Decline.h"
#include "agpu/msl/Builtins.h"
#include "agpu/plan/ElemType.h"

#include <cstdint>

namespace agpu {

// ── the functions ─────────────────────────────────────────────────────────

// Math functions Metal provides directly.
enum class MathFn {
  Exp,
  Exp2,
  Exp10,
  Log,
  Log2,
  Log10,
  Sqrt,
  Rsqrt,
  Sin,
  Cos,
  Tanh,
  Abs,
  Floor,
  Ceil,
  Sign,
  Tan,
  Asin,
  Acos,
  Atan,
  Sinh,
  Cosh,
  // Round and RoundEven are different functions: half away from zero versus
  // half to even.
  Round,
  RoundEven,
  Trunc,
  Isnan,
  Isinf,
  // Metal has neither erf nor cbrt in either namespace; both lower to a
  // prelude helper.
  Erf,
  Cbrt,
  // Lets a test walk the enum and catch an enumerator with no spelling.
  Count,
};

// `Min`/`Max` live in this value-returning group because of that return
// value. Their NaN behaviour is a separate decision; see
// `minMaxPropagatesNan`.
enum class MathFn2 {
  Min,
  Max,
  Fmod,
  Pow,
  Atan2,
  Copysign,
  // Integer high-half multiply: the top word of a widening product.
  Mulhi,
  Count,
};

enum class MathFn3 {
  Fma,   // a*b + c, with one rounding
  Clamp, // min(max(v, lo), hi), as one call
  Count,
};

// ── the tables ────────────────────────────────────────────────────────────

enum class Accuracy {
  Exact,
  Tolerant,
};

inline constexpr const char *spell(msl::builtin::AccuracyPair s, Accuracy a) {
  return a == Accuracy::Exact ? s.precise : s.fast;
}

struct MathSpelling {
  MathFn fn;
  const char *name;
  bool floatOnly;
  bool returnsBool = false;
};

inline constexpr MathSpelling kMathSpellings[] = {
    {MathFn::Exp, spell(msl::builtin::accuracy::Exp, Accuracy::Exact), true},
    {MathFn::Exp2, spell(msl::builtin::accuracy::Exp2, Accuracy::Tolerant),
     true},
    {MathFn::Exp10, spell(msl::builtin::accuracy::Exp10, Accuracy::Exact),
     true},
    {MathFn::Log, spell(msl::builtin::accuracy::Log, Accuracy::Exact), true},
    {MathFn::Log2, spell(msl::builtin::accuracy::Log2, Accuracy::Exact), true},
    {MathFn::Log10, spell(msl::builtin::accuracy::Log10, Accuracy::Exact),
     true},
    {MathFn::Sqrt, spell(msl::builtin::accuracy::Sqrt, Accuracy::Exact), true},
    {MathFn::Rsqrt, spell(msl::builtin::accuracy::Rsqrt, Accuracy::Tolerant),
     true},
    {MathFn::Sin, spell(msl::builtin::accuracy::Sin, Accuracy::Exact), true},
    {MathFn::Cos, spell(msl::builtin::accuracy::Cos, Accuracy::Exact), true},
    {MathFn::Tanh, spell(msl::builtin::accuracy::Tanh, Accuracy::Exact), true},
    {MathFn::Abs, msl::builtin::math::Abs, false},
    {MathFn::Sign, msl::builtin::math::Sign, true},
    {MathFn::Floor, msl::builtin::math::Floor, true},
    {MathFn::Ceil, msl::builtin::math::Ceil, true},
    {MathFn::Isnan, msl::builtin::math::Isnan, true, true},
    {MathFn::Isinf, msl::builtin::math::Isinf, true, true},
    {MathFn::Erf, msl::builtin::helper::Erf, true},
    {MathFn::Cbrt, msl::builtin::helper::Cbrt, true},
    {MathFn::Tan, spell(msl::builtin::accuracy::Tan, Accuracy::Exact), true},
    {MathFn::Asin, spell(msl::builtin::accuracy::Asin, Accuracy::Exact), true},
    {MathFn::Acos, spell(msl::builtin::accuracy::Acos, Accuracy::Exact), true},
    {MathFn::Atan, spell(msl::builtin::accuracy::Atan, Accuracy::Exact), true},
    {MathFn::Sinh, spell(msl::builtin::accuracy::Sinh, Accuracy::Exact), true},
    {MathFn::Cosh, spell(msl::builtin::accuracy::Cosh, Accuracy::Exact), true},
    {MathFn::Round, msl::builtin::math::Round, true},
    {MathFn::RoundEven, msl::builtin::math::RoundEven, true},
    {MathFn::Trunc, msl::builtin::math::Trunc, true},
};

// `metal::abs` has a float and an integer overload, so a `bfloat` converts to
// either and the call is ambiguous. `fabs` has no integer overload.
struct MathFloatSpelling {
  MathFn fn;
  const char *name;
};

inline constexpr MathFloatSpelling kMathFloatSpellings[] = {
    {MathFn::Abs, msl::builtin::math::Fabs},
};

struct MathSpelling2 {
  MathFn2 fn;
  const char *name;
  bool floatOnly;
};

inline constexpr MathSpelling2 kMathSpellings2[] = {
    {MathFn2::Min, msl::builtin::math::Min, false},
    {MathFn2::Max, msl::builtin::math::Max, false},
    {MathFn2::Fmod, spell(msl::builtin::accuracy::Fmod, Accuracy::Exact), true},
    {MathFn2::Pow, spell(msl::builtin::accuracy::Pow, Accuracy::Exact), true},
    {MathFn2::Atan2, spell(msl::builtin::accuracy::Atan2, Accuracy::Exact),
     true},
    {MathFn2::Copysign, msl::builtin::math::Copysign, true},
    {MathFn2::Mulhi, msl::builtin::math::Mulhi, false},
};

// ── one operand ───────────────────────────────────────────────────────────

inline const char *mathNameOf(MathFn fn) {
  for (const MathSpelling &s : kMathSpellings)
    if (s.fn == fn)
      return s.name;
  return nullptr;
}

// fp8 has no arithmetic here; a value is the `uchar` holding its encoding.
// `abs` is the exception: clearing the sign bit is exact in the encoding.
//
// Returns the mask, or zero when the op is an ordinary call.
inline int64_t mathBitMaskOf(MathFn fn, ElemType operand) {
  const bool isFp8 = operand.kind == ElemType::Kind::Float && operand.bits == 8;
  if (fn != MathFn::Abs || !isFp8)
    return 0;
  // Both fp8 encodings put the sign in the top bit.
  return 0x7f;
}

inline const char *mathNameOf(MathFn fn, ElemType operand) {
  if (operand.kind == ElemType::Kind::Float)
    for (const MathFloatSpelling &s : kMathFloatSpellings)
      if (s.fn == fn)
        return s.name;
  return mathNameOf(fn);
}

// What a math call produces, which is not always the operand type.
inline ElemType mathResultType(MathFn fn, ElemType operand) {
  for (const MathSpelling &s : kMathSpellings)
    if (s.fn == fn)
      return s.returnsBool ? i1() : operand;
  return operand;
}

// Metal's math functions take and return `float`, so a `half` or `bfloat`
// argument promotes and the wider result does not convert back implicitly.
inline bool mathResultNarrows(MathFn fn, ElemType operand) {
  return operand.kind == ElemType::Kind::Float && operand.bits < 32 &&
         mathResultType(fn, operand) == operand;
}

inline Decision checkMath(MathFn fn, ElemType elem) {
  for (const MathSpelling &s : kMathSpellings) {
    if (s.fn != fn)
      continue;
    if (s.floatOnly && elem.kind != ElemType::Kind::Float)
      return Decision::declined("math", "float-only function on a non-float");
    return Decision::emitted();
  }
  return Decision::declined("math", "no spelling for this function");
}

// ── two operands ──────────────────────────────────────────────────────────

inline const char *mathNameOf(MathFn2 fn) {
  for (const MathSpelling2 &s : kMathSpellings2)
    if (s.fn == fn)
      return s.name;
  return nullptr;
}

inline Decision checkMath2(MathFn2 fn, ElemType elem) {
  for (const MathSpelling2 &s : kMathSpellings2) {
    if (s.fn != fn)
      continue;
    if (s.floatOnly && elem.kind != ElemType::Kind::Float)
      return Decision::declined("math2", "float-only function on a non-float");
    if (fn == MathFn2::Mulhi && elem.kind != ElemType::Kind::Int)
      return Decision::declined("math2", "high-half multiply of a non-integer");
    return Decision::emitted();
  }
  return Decision::declined("math2", "no spelling for this function");
}

// `metal::min`/`max` return the other operand when one is NaN (IEEE
// minNum/maxNum). A reduction wants a NaN anywhere in the input to reach the
// output.
inline bool minMaxPropagatesNan(MathFn2 fn, ElemType elem, bool wantPropagate) {
  if (fn != MathFn2::Min && fn != MathFn2::Max)
    return false;
  return wantPropagate && elem.kind == ElemType::Kind::Float;
}

// ── three operands ────────────────────────────────────────────────────────

// `clamp` is `min(max(...))` in one call, so it drops a NaN the same way.
// `tt.clampf` carries a `propagateNan` attribute. `fma` never wants this.
inline bool math3PropagatesNan(MathFn3 fn, ElemType elem, bool wantPropagate) {
  if (fn != MathFn3::Clamp)
    return false;
  return wantPropagate && elem.kind == ElemType::Kind::Float;
}

inline const char *mathNameOf(MathFn3 fn) {
  switch (fn) {
  case MathFn3::Fma:
    return msl::builtin::math::Fma;
  case MathFn3::Clamp:
    return msl::builtin::math::Clamp;
  case MathFn3::Count:
    break;
  }
  return nullptr;
}

inline Decision checkMath3(MathFn3 fn, ElemType elem) {
  if (fn == MathFn3::Fma && elem.kind != ElemType::Kind::Float)
    return Decision::declined("math3", "fused multiply-add of a non-float");
  return Decision::emitted();
}

} // namespace agpu

#endif // AGPU_MATH_FN_H
