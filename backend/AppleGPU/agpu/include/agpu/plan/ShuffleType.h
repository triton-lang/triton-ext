// ShuffleType.h - how a value crosses lanes when its own type cannot.
//
// `ShufflePlan.h` decides whether a layout change stays inside the warp;
// this decides what type the value travels as.
//
// `simd_shuffle` and friends are constrained to 32-bit scalars and vectors
// of them. Anything else is a compile error at the call, so the value
// travels as a supported type and comes back. A shuffle moves bits without
// interpreting them, so the round trip is exact.
#ifndef AGPU_SHUFFLE_TYPE_H
#define AGPU_SHUFFLE_TYPE_H

#include "agpu/msl/Ast.h"

namespace agpu {

enum class ShuffleForm {
  // A type the hardware already moves.
  Direct,
  // 64 bits as a `uint2`: both halves shuffle by the same amount and one
  // `as_type` round-trips the pair.
  SplitU32Pair,
  // Bool travels as a byte, by conversion.
  BoolAsU8,
  // Narrower than 32 bits: travels as the same-width integer, so a half's
  // bits arrive unrounded.
  NarrowAsBits,
};

inline ShuffleForm shuffleFormOf(msl::Scalar s) {
  switch (s) {
  case msl::Scalar::I64:
  case msl::Scalar::U64:
    return ShuffleForm::SplitU32Pair;
  case msl::Scalar::Bool:
    return ShuffleForm::BoolAsU8;
  case msl::Scalar::BF16:
  case msl::Scalar::F16:
  case msl::Scalar::I16:
  case msl::Scalar::U16:
  case msl::Scalar::I8:
  case msl::Scalar::U8:
    return ShuffleForm::NarrowAsBits;
  default:
    return ShuffleForm::Direct;
  }
}

// Same width, so no float is rounded on the way.
inline msl::Scalar shuffleBitsOf(msl::Scalar s) {
  switch (s) {
  case msl::Scalar::BF16:
  case msl::Scalar::F16:
  case msl::Scalar::I16:
  case msl::Scalar::U16:
    return msl::Scalar::U16;
  case msl::Scalar::I8:
  case msl::Scalar::U8:
    return msl::Scalar::U8;
  default:
    return s;
  }
}

} // namespace agpu

#endif // AGPU_SHUFFLE_TYPE_H
