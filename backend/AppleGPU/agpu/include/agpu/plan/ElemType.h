// ElemType.h - the element types and their MSL spelling.
#ifndef AGPU_ELEM_TYPE_H
#define AGPU_ELEM_TYPE_H

#include "agpu/msl/Ast.h"

#include <cstdint>

namespace agpu {

enum class FloatKind {
  Ieee,  // f32, f16; the width says which
  Brain, // bf16
};

// The element type of a value, as the IR gives it.
struct ElemType {
  enum class Kind { Int, Float, Bool, Pointer };
  Kind kind = Kind::Int;
  unsigned bits = 32;
  bool isUnsigned = false;               // integers only: the IR's signedness
  FloatKind floatKind = FloatKind::Ieee; // floats only

  // What a pointer points at, so it can be spelled `device T *`. A
  // pointer-typed value declared as an integer takes the pointee's width, so a
  // select between two `!tt.ptr<i32>` would truncate a 64-bit address to 32.
  msl::Scalar pointee = msl::Scalar::I32;
  msl::AddrSpace addrSpace = msl::AddrSpace::Device;

  bool isPointer() const { return kind == Kind::Pointer; }

  bool operator==(const ElemType &o) const {
    if (kind != o.kind || bits != o.bits || isUnsigned != o.isUnsigned ||
        floatKind != o.floatKind)
      return false;
    // The pointee is part of the type only when there is one.
    return !isPointer() || (pointee == o.pointee && addrSpace == o.addrSpace);
  }
};

// Rounded up and never zero: an i1 is one bit wide and one byte stored.
inline int64_t byteWidthOf(const ElemType &e) {
  return (int64_t)((e.bits + 7u) / 8u);
}

inline ElemType i32() { return {ElemType::Kind::Int, 32, false}; }
inline ElemType f32() { return {ElemType::Kind::Float, 32, false}; }
inline ElemType f16() { return {ElemType::Kind::Float, 16, false}; }
inline ElemType bf16() {
  return {ElemType::Kind::Float, 16, false, FloatKind::Brain};
}
inline ElemType i1() { return {ElemType::Kind::Bool, 1, false}; }

// Only f64. Metal has no double, so a kernel written in f64 computes in f32.
// `mslTypeOf` is a spelling function and cannot report it, so callers holding
// a decline log ask separately.
inline bool narrowsSilently(ElemType e) {
  return e.kind == ElemType::Kind::Float && e.bits == 64;
}

// The type this actually becomes, which for f64 is not the type asked for.
inline ElemType f64() { return {ElemType::Kind::Float, 64, false}; }

// The MSL spelling of an element type.
inline msl::Type mslTypeOf(ElemType e) {
  using S = msl::Scalar;
  switch (e.kind) {
  case ElemType::Kind::Bool:
    return msl::Type::scalar(S::Bool);
  case ElemType::Kind::Float:
    if (e.floatKind == FloatKind::Brain)
      return msl::Type::scalar(S::BF16);
    // fp8 has no MSL type; it travels as a byte.
    if (e.bits == 8)
      return msl::Type::scalar(S::U8);
    // 64 lands on F32: Metal has no double.
    return msl::Type::scalar(e.bits == 16 ? S::F16 : S::F32);
  case ElemType::Kind::Pointer:
    return msl::Type::scalar(e.pointee).pointerTo(e.addrSpace);
  case ElemType::Kind::Int:
    break;
  }
  switch (e.bits) {
  case 8:
    return msl::Type::scalar(e.isUnsigned ? S::U8 : S::I8);
  case 16:
    return msl::Type::scalar(e.isUnsigned ? S::U16 : S::I16);
  case 64:
    return msl::Type::scalar(e.isUnsigned ? S::U64 : S::I64);
  default:
    return msl::Type::scalar(e.isUnsigned ? S::U32 : S::I32);
  }
}

} // namespace agpu

#endif // AGPU_ELEM_TYPE_H
