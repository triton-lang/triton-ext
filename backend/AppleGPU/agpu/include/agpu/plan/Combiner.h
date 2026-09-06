// Combiner.h - the reduce/scan combiners Metal folds in one call.
#ifndef AGPU_COMBINER_H
#define AGPU_COMBINER_H

#include "agpu/msl/Builtins.h"
#include "agpu/plan/ElemType.h"

namespace agpu {

enum class Combiner {
  Generic,
  AddF,
  MulF,
  AddI,
  MulI,
  MinF,
  MaxF,
  MinS,
  MaxS,
  MinU,
  MaxU,
  AndI,
  OrI,
  XorI,
  // Lets a test walk the enum and catch a combiner with no row.
  Count,
};

// Metal's simd folds take 32-bit and 16-bit scalars. bfloat and the 64-bit
// widths have no overload, and unlike a shuffle there is no bitcast that
// preserves the arithmetic.
inline bool simdFoldable(ElemType e) {
  if (e.kind == ElemType::Kind::Float)
    return (e.bits == 32 || e.bits == 16) && e.floatKind != FloatKind::Brain;
  return e.kind == ElemType::Kind::Int && e.bits <= 32 && e.bits >= 16;
}

struct CombinerSpelling {
  Combiner fn;
  const char *reduce;
  const char *prefixInclusive;
  const char *prefixExclusive;
  bool floatOnly;
  bool signedOnly;
  bool unsignedOnly;
};

inline constexpr CombinerSpelling kCombinerSpellings[] = {
    {Combiner::AddF, msl::builtin::simd::Sum,
     msl::builtin::simd::PrefixInclusiveSum,
     msl::builtin::simd::PrefixExclusiveSum, true, false, false},
    {Combiner::MulF, msl::builtin::simd::Product,
     msl::builtin::simd::PrefixInclusiveProduct,
     msl::builtin::simd::PrefixExclusiveProduct, true, false, false},
    {Combiner::AddI, msl::builtin::simd::Sum,
     msl::builtin::simd::PrefixInclusiveSum,
     msl::builtin::simd::PrefixExclusiveSum, false, false, false},
    {Combiner::MulI, msl::builtin::simd::Product,
     msl::builtin::simd::PrefixInclusiveProduct,
     msl::builtin::simd::PrefixExclusiveProduct, false, false, false},
    {Combiner::MinF, msl::builtin::simd::Min, nullptr, nullptr, true, false,
     false},
    {Combiner::MaxF, msl::builtin::simd::Max, nullptr, nullptr, true, false,
     false},
    {Combiner::MinS, msl::builtin::simd::Min, nullptr, nullptr, false, true,
     false},
    {Combiner::MaxS, msl::builtin::simd::Max, nullptr, nullptr, false, true,
     false},
    {Combiner::MinU, msl::builtin::simd::Min, nullptr, nullptr, false, false,
     true},
    {Combiner::MaxU, msl::builtin::simd::Max, nullptr, nullptr, false, false,
     true},
    {Combiner::AndI, msl::builtin::simd::And, nullptr, nullptr, false, false,
     false},
    {Combiner::OrI, msl::builtin::simd::Or, nullptr, nullptr, false, false,
     false},
    {Combiner::XorI, msl::builtin::simd::Xor, nullptr, nullptr, false, false,
     false},
};

inline const CombinerSpelling *combinerSpelling(Combiner fn, ElemType e) {
  if (fn == Combiner::Generic || !simdFoldable(e))
    return nullptr;
  const bool isFloat = e.kind == ElemType::Kind::Float;
  for (const CombinerSpelling &s : kCombinerSpellings) {
    if (s.fn != fn)
      continue;
    if (s.floatOnly != isFloat)
      return nullptr;
    if (s.signedOnly && e.isUnsigned)
      return nullptr;
    if (s.unsignedOnly && !e.isUnsigned)
      return nullptr;
    return &s;
  }
  return nullptr;
}

inline const char *simdReduceFn(Combiner fn, ElemType e) {
  const CombinerSpelling *s = combinerSpelling(fn, e);
  return s ? s->reduce : nullptr;
}

inline const char *simdPrefixInclusiveFn(Combiner fn, ElemType e) {
  const CombinerSpelling *s = combinerSpelling(fn, e);
  return s ? s->prefixInclusive : nullptr;
}

inline const char *simdPrefixExclusiveFn(Combiner fn, ElemType e) {
  const CombinerSpelling *s = combinerSpelling(fn, e);
  return s ? s->prefixExclusive : nullptr;
}

} // namespace agpu

#endif // AGPU_COMBINER_H
