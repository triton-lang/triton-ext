// VectorSpelling.h - how a wide access is spelled. Shared by loads/stores
// (EmitMove.h) and pool staging (Emit.h).
#ifndef AGPU_EMIT_VECTOR_SPELLING_H
#define AGPU_EMIT_VECTOR_SPELLING_H

#include "agpu/msl/Context.h"
#include "agpu/msl/Printer.h"
#include "agpu/plan/Elementwise.h"

namespace agpu {

// The constructor a vector literal names: `float4(a, b, c, d)`. Never the
// packed spelling; MSL constructs the plain type and converts on assignment.
inline msl::Str vecCtorName(ElemType elem, int64_t width) {
  return msl::Str(msl::spell(mslTypeOf(elem).scalarKind())) +
         std::to_string(width);
}

// The pointer cast a wide access needs: `*(device vecN *)p`.
inline msl::Type vectorTypeOf(ElemType elem, int64_t width, bool packed) {
  const msl::Scalar s = mslTypeOf(elem).scalarKind();
  return packed ? msl::Type::packedVector(s, (int)width)
                : msl::Type::vector(s, (int)width);
}

// A `width`-wide lvalue over an element lvalue the caller already has:
// `*(space vecN *)&elem`.
inline msl::Expr *wideLValue(msl::Context &c, msl::Expr *elemLValue,
                             ElemType elem, int64_t width, bool packed,
                             msl::AddrSpace space,
                             unsigned quals = msl::Type::QualNone) {
  return c.deref(
      c.cast(vectorTypeOf(elem, width, packed).pointerTo(space, quals),
             c.addrOf(elemLValue)));
}

} // namespace agpu

#endif // AGPU_EMIT_VECTOR_SPELLING_H
