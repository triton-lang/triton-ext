// AgpuTypes - MLIR types in, agpu::ElemType out. Reads only: MSL spelling,
// precision loss and access width all live in agpu/.
#ifndef AGPU_BRIDGE_TYPES_H
#define AGPU_BRIDGE_TYPES_H

#include "agpu/plan/DeviceFn.h"
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

// fp8 encodings are not interchangeable: e4m3's top slot is NaN, e5m2 has a
// real infinity and the FNUZ pair has neither.
inline std::optional<agpu::FloatKind> fp8FloatKindOf(Type t) {
  if (isa<Float8E4M3FNType>(t))
    return agpu::FloatKind::E4M3;
  if (isa<Float8E5M2Type>(t))
    return agpu::FloatKind::E5M2;
  if (isa<Float8E4M3FNUZType>(t))
    return agpu::FloatKind::E4B8;
  if (isa<Float8E5M2FNUZType>(t))
    return agpu::FloatKind::E5B16;
  return std::nullopt;
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
  if (const std::optional<agpu::FloatKind> k = fp8FloatKindOf(t))
    return agpu::ElemType{agpu::ElemType::Kind::Float, 8, false, *k};

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

// The one case where axis analysis reports divisibility in bytes rather
// than elements.
inline bool isTensorOfPointers(Type t) {
  auto rt = dyn_cast<RankedTensorType>(t);
  return rt && isa<triton::PointerType>(rt.getElementType());
}

// Distinct from elemTypeOf, which unwraps to the pointee (i1 for
// !tt.ptr<i1>); every pointer here is 64 bits whatever it points to.
inline bool isPointerLike(Type t) {
  if (auto rt = dyn_cast<RankedTensorType>(t))
    t = rt.getElementType();
  return isa<triton::PointerType>(t);
}

// For a pointer, the address's own type (64 bits). Declaring a `!tt.ptr<i32>`
// register from the pointee would truncate the address to 32 bits.
inline std::optional<agpu::ElemType> heldTypeOf(Type t) {
  if (!isPointerLike(t))
    return elemTypeOf(t);

  const std::optional<agpu::ElemType> pointee = elemTypeOf(t);
  if (!pointee)
    return std::nullopt;

  agpu::ElemType p;
  p.kind = agpu::ElemType::Kind::Pointer;
  p.bits = 64;
  p.isUnsigned = true;
  p.pointee = agpu::mslTypeOf(*pointee).scalarKind();
  p.addrSpace = agpu::msl::AddrSpace::Device;
  return p;
}

// For a pointer tensor, what moves is the per-register offset. It is i64:
// recorded offsets mix `int` sums and `long` casts.
inline std::optional<agpu::ElemType> movedTypeOf(Type t) {
  if (isPointerLike(t))
    return agpu::i64();
  return elemTypeOf(t);
}

// regCount is passed in: it is a layout question. Uses elemTypeOf (the
// pointee): the caller applies the pointer itself, so heldTypeOf here would
// spell `device device float **`.
inline std::optional<agpu::DeviceValue> deviceValueOf(Type t,
                                                      int64_t regCount) {
  if (regCount <= 0)
    return std::nullopt;
  const std::optional<agpu::ElemType> elem = elemTypeOf(t);
  if (!elem)
    return std::nullopt;
  agpu::DeviceValue v;
  v.elem = *elem;
  v.isPointer = isPointerLike(t);
  v.regCount = regCount;
  return v;
}

} // namespace mlir::triton::applegpu::bridge

#endif // AGPU_BRIDGE_TYPES_H
