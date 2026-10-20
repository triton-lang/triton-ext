// Convert.h - how one type becomes another and what rounds.
//
// Most casts are `static_cast`. Two families are not: narrowing f32 to half or
// bfloat has a rounding mode and fp8 has no MSL type, so every fp8 conversion
// is a pack or unpack.
#ifndef AGPU_TYPE_CONVERT_H
#define AGPU_TYPE_CONVERT_H

#include "agpu/core/Decline.h"
#include "agpu/plan/ElemType.h"

#include <optional>

namespace agpu {

// How the IR asked for the result to be rounded.
enum class Rounding {
  Default, // the IR did not say: MSL's own rule applies
  RTNE,    // round to nearest, ties to even
  RTZ,     // round toward zero
};

// The fp8 encoding of a type Metal has no type for, or nothing.
inline std::optional<FloatKind> fp8KindOf(ElemType e) {
  if (e.kind != ElemType::Kind::Float || e.bits != 8)
    return std::nullopt;
  switch (e.floatKind) {
  case FloatKind::E4M3:
  case FloatKind::E5M2:
  case FloatKind::E4B8:
  case FloatKind::E5B16:
    return e.floatKind;
  default:
    return std::nullopt;
  }
}

// How a conversion is performed.
enum class ConvertKind {
  None,       // same type: nothing to emit
  Cast,       // a plain static_cast
  NarrowRtne, // f32 -> half/bfloat, round-to-nearest-even, via a helper
  NarrowRtz,  // f32 -> half/bfloat with round-toward-zero, via a helper
  Fp8Pack,    // f32 -> fp8
  Fp8Unpack,  // fp8 -> f32
  Unsupported,
};

struct ConvertPlan {
  ConvertKind kind = ConvertKind::Cast;
  // The fp8 side's encoding, for the pack and unpack kinds.
  FloatKind fp8 = FloatKind::Ieee;

  // The helper name depends on it.
  ElemType to;

  // The narrowing and packing helpers take f32, so a 16-bit source widens
  // first. f16 and bf16 are subsets of f32, so this is exact.
  bool widensOperand = false;

  // Fp8Unpack only: what the f32 the unpacker returns becomes, chosen the
  // way any f32 -> `to` conversion is.
  ConvertKind narrows = ConvertKind::None;

  ConvertPlan narrowing() const {
    ConvertPlan n;
    n.kind = narrows;
    n.to = to;
    return n;
  }

  bool needsHelper() const {
    return kind == ConvertKind::NarrowRtz || kind == ConvertKind::NarrowRtne ||
           kind == ConvertKind::Fp8Pack || kind == ConvertKind::Fp8Unpack;
  }
  bool usable() const { return kind != ConvertKind::Unsupported; }
};

// Narrowing a float is the only case where the rounding mode is observable.
inline bool narrowsFloat(ElemType from, ElemType to) {
  return from.kind == ElemType::Kind::Float &&
         to.kind == ElemType::Kind::Float && from.bits == 32 && to.bits == 16;
}

// fp8 is checked first: MSL cannot cast it at all.
inline ConvertPlan planConvert(ElemType from, ElemType to, Rounding r) {
  ConvertPlan p;
  p.to = to;
  const std::optional<FloatKind> srcFp8 = fp8KindOf(from);
  const std::optional<FloatKind> dstFp8 = fp8KindOf(to);

  if (srcFp8 && dstFp8) {
    // fp8 to fp8 would be unpack-then-pack. Nothing asks for it yet.
    p.kind = ConvertKind::Unsupported;
    return p;
  }
  // The packers work in f32; f16/bf16 reach it exactly.
  const bool fromReachesF32 =
      from.kind == ElemType::Kind::Float && from.bits <= 32;
  const bool toReachesF32 = to.kind == ElemType::Kind::Float && to.bits <= 32;

  if (dstFp8) {
    p.kind = fromReachesF32 ? ConvertKind::Fp8Pack : ConvertKind::Unsupported;
    p.fp8 = *dstFp8;
    p.widensOperand = from.bits < 32;
    return p;
  }
  if (srcFp8) {
    p.kind = toReachesF32 ? ConvertKind::Fp8Unpack : ConvertKind::Unsupported;
    p.fp8 = *srcFp8;
    if (toReachesF32 && to.bits < 32)
      p.narrows = planConvert(f32(), to, r).kind;
    return p;
  }

  if (from == to) {
    p.kind = ConvertKind::None;
    return p;
  }

  // MSL offers no way to ask for RTZ, so that always needs the helper. RTNE
  // asks for it explicitly.
  if (narrowsFloat(from, to)) {
    if (r == Rounding::RTZ) {
      p.kind = ConvertKind::NarrowRtz;
      return p;
    }
    // Default too for bfloat: AGX3 miscompiles `(bfloat)x` as `(half)x`.
    if (r == Rounding::RTNE || to.floatKind == FloatKind::Brain) {
      p.kind = ConvertKind::NarrowRtne;
      return p;
    }
  }

  p.kind = ConvertKind::Cast;
  return p;
}

inline Decision convertDecision(const ConvertPlan &p) {
  if (p.usable())
    return Decision::emitted();
  return Decision::declined("convert", "no conversion between these types");
}

} // namespace agpu

#endif // AGPU_TYPE_CONVERT_H
