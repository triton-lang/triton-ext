// AgpuTypes - MLIR types in, agpu::ElemType out. Reads only: MSL spelling,
// precision loss and access width all live in agpu/.
#ifndef AGPU_BRIDGE_TYPES_H
#define AGPU_BRIDGE_TYPES_H

#include "agpu/plan/Elementwise.h"

#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include <optional>

namespace mlir::triton::applegpu::bridge {

// Unwraps both tensor and pointer wrappers: `tensor<64x!tt.ptr<f16>>` is f16.
inline Type elementScalarType(Type t) {
  if (auto rt = dyn_cast<RankedTensorType>(t))
    t = rt.getElementType();
  if (auto pt = dyn_cast<triton::PointerType>(t))
    t = pt.getPointeeType();
  return t;
}

// Nothing for a type this backend has no representation for. f64 is
// representable; narrowsSilently downstream reports that Metal computes it
// in f32, which is distinct from having no spelling at all.
inline std::optional<agpu::ElemType> elemTypeOf(Type t) {
  t = elementScalarType(t);

  if (t.isF32())
    return agpu::f32();
  if (t.isF64())
    return agpu::f64();
  if (t.isF16())
    return agpu::f16();
  if (t.isBF16())
    return agpu::bf16();
  if (auto it = dyn_cast<IntegerType>(t)) {
    const unsigned bits = it.getWidth();
    if (bits == 1)
      return agpu::i1();
    if (bits == 8 || bits == 16 || bits == 32 || bits == 64)
      // MLIR integers are signless; the operation decides signedness.
      return agpu::ElemType{agpu::ElemType::Kind::Int, bits, false};
  }

  return std::nullopt;
}
} // namespace mlir::triton::applegpu::bridge

#endif // AGPU_BRIDGE_TYPES_H
