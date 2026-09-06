// EmitConvert.h - one type conversion, emitted from its plan.
#ifndef AGPU_EMIT_CONVERT_H
#define AGPU_EMIT_CONVERT_H

#include "agpu/emit/Prelude.h"
#include "agpu/msl/Context.h"
#include "agpu/plan/TypeConvert.h"

namespace agpu {

// The helper a conversion calls, or empty when it needs none.
inline msl::Str convertHelperName(const ConvertPlan &p) {
  Helper h;
  if (!convertHelper(p, h))
    return {};
  return msl::Str(helperName(h));
}

// Returns the value unchanged when the conversion is a no-op.
inline msl::Expr *convertExpr(msl::Context &c, const ConvertPlan &p,
                              msl::Expr *value, ElemType to) {
  switch (p.kind) {
  case ConvertKind::None:
    return value;

  case ConvertKind::Cast:
    return c.cast(mslTypeOf(to), value);

  case ConvertKind::NarrowRtz:
  case ConvertKind::NarrowRtne:
  case ConvertKind::Fp8Pack: {
    // The helper takes f32; a narrower operand widens first.
    msl::Expr *in = value;
    if (p.widensOperand)
      in = c.cast(msl::Type::scalar(msl::Scalar::F32), value);
    return c.call(convertHelperName(p), {in});
  }

  case ConvertKind::Fp8Unpack: {
    // fp8 travels as a byte; the helper takes raw storage and returns f32.
    msl::Expr *call = c.call(convertHelperName(p), {value});
    if (to.kind == ElemType::Kind::Float && to.bits < 32)
      return c.cast(mslTypeOf(to), call);
    return call;
  }

  case ConvertKind::Unsupported:
    return nullptr;
  }
  return nullptr;
}

// A whole tensor's worth: one declaration per register.
inline msl::SmallVec<msl::Str, 8>
emitConvert(msl::Context &c, msl::Block &body, const ConvertPlan &p,
            const msl::SmallVec<msl::Str, 8> &srcNames,
            const msl::SmallVec<msl::Str, 8> &dstNames, ElemType to) {
  msl::SmallVec<msl::Str, 8> out;
  if (!p.usable())
    return out;

  if (p.kind == ConvertKind::None)
    return srcNames;

  for (std::size_t r = 0; r < srcNames.size(); ++r) {
    body.push_back(c.declStmt(mslTypeOf(to), dstNames[r],
                              convertExpr(c, p, c.var(srcNames[r]), to)));
    out.push_back(dstNames[r]);
  }
  return out;
}

} // namespace agpu

#endif // AGPU_EMIT_CONVERT_H
