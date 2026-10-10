// PointerBind - what an axis analysis says about a pointer, in the units the
// width planner works in.
//
// `AccessWidth.h`'s `PtrInfo` counts contiguity and alignment in elements; an
// axis analysis reports divisibility in bytes. Reading a tensor-of-pointers'
// byte divisibility as elements vectorises loads whose odd rows are misaligned.
#ifndef AGPU_POINTER_BIND_H
#define AGPU_POINTER_BIND_H

#include "agpu/plan/AccessWidth.h"
#include "agpu/plan/Elementwise.h"

#include <algorithm>
#include <cstdint>

namespace agpu {

// What an axis analysis reports about one dimension of a pointer, filled
// straight from `getContiguity` / `getDivisibility` with no conversion.
struct AxisReport {
  int64_t contiguityElems = 1;
  int64_t divisibilityBytes = 1;

  // Only a tensor of pointers reports divisibility in bytes. Elsewhere the
  // number is already in elements.
  bool isTensorOfPointers = false;
};

// The planner's view, converted.
inline PtrInfo ptrInfoFrom(const AxisReport &a, ElemType elem) {
  PtrInfo out;
  out.contiguity = std::max<int64_t>(a.contiguityElems, 1);

  if (!a.isTensorOfPointers) {
    out.alignment = std::max<int64_t>(a.divisibilityBytes, 1);
    return out;
  }

  const int64_t bytesPerElem = byteWidthOf(elem);
  out.alignment = std::max<int64_t>(a.divisibilityBytes / bytesPerElem, 1);
  return out;
}

} // namespace agpu

#endif // AGPU_POINTER_BIND_H
