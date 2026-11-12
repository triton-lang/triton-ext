// Axis analysis units: contiguity is in elements, divisibility is in bytes.
#include "agpu/bind/PointerBind.h"
#include "fixtures.h"
#include "harness.h"

using namespace agpu;

using agpu_test::contiguousBases;

namespace {

AccessPlan planAccess(const RegBases &bases, const PtrInfo &p, VecElem elem) {
  return agpu::planAccess(bases, /*runtime=*/{}, PtrDims(2, p), elem);
}

} // namespace

int main() {
  CASE("the f16 odd-stride case: bytes read as elements vectorises a "
       "misaligned load");
  {
    AxisReport a;
    a.contiguityElems = 8;
    a.divisibilityBytes = 2;
    a.isTensorOfPointers = true;

    const PtrInfo p = ptrInfoFrom(a, f16());
    CHECK_EQ(p.alignment, (int64_t)1);
    CHECK_EQ(p.contiguity, (int64_t)8);
    CHECK(p.alignment != a.divisibilityBytes);

    PtrInfo wrong;
    wrong.contiguity = 8;
    wrong.alignment = 2;

    const AccessPlan packedGood =
        planAccess(contiguousBases(3), p, VecElem::Packable);
    const AccessPlan packedBad =
        planAccess(contiguousBases(3), wrong, VecElem::Packable);
    CHECK_EQ(packedGood.width, packedBad.width);
    CHECK_EQ(packedGood.packed, packedBad.packed);
  }

  CASE("on the packed path the divergence is the type, at a higher "
       "divisibility");
  {
    AxisReport a;
    a.contiguityElems = 8;
    a.divisibilityBytes = 4;
    a.isTensorOfPointers = true;

    const PtrInfo p = ptrInfoFrom(a, f16());
    CHECK_EQ(p.alignment, (int64_t)2);

    PtrInfo wrong;
    wrong.contiguity = 8;
    wrong.alignment = 4;

    const AccessPlan good =
        planAccess(contiguousBases(3), p, VecElem::Packable);
    const AccessPlan bad =
        planAccess(contiguousBases(3), wrong, VecElem::Packable);

    CHECK_EQ(good.width, (int64_t)4);
    CHECK_EQ(bad.width, (int64_t)4);
    CHECK(good.packed);
    CHECK(!bad.packed);
  }

  CASE("f32 halves its divisibility too");
  {
    AxisReport a;
    a.contiguityElems = 16;
    a.divisibilityBytes = 16;
    a.isTensorOfPointers = true;
    CHECK_EQ(ptrInfoFrom(a, f32()).alignment, (int64_t)4);
  }

  CASE("an 8-bit element converts to itself");
  {
    AxisReport a;
    a.contiguityElems = 4;
    a.divisibilityBytes = 4;
    a.isTensorOfPointers = true;
    CHECK_EQ(ptrInfoFrom(a, e4m3()).alignment, a.divisibilityBytes);
  }

  CASE("a value that is not a tensor of pointers is already in elements");
  {
    AxisReport a;
    a.contiguityElems = 8;
    a.divisibilityBytes = 8;
    a.isTensorOfPointers = false;
    CHECK_EQ(ptrInfoFrom(a, f32()).alignment, a.divisibilityBytes);
  }

  CASE("the conversion never reports better alignment than it was told");
  {
    for (ElemType e : {f32(), f16(), bf16(), e4m3(), e5m2(), i32()})
      for (int64_t bytes : {1, 2, 4, 8, 16, 32, 128}) {
        AxisReport a;
        a.contiguityElems = 8;
        a.divisibilityBytes = bytes;
        a.isTensorOfPointers = true;
        const PtrInfo p = ptrInfoFrom(a, e);
        CHECK(p.alignment >= 1);
        CHECK(p.alignment <= bytes);
      }
  }

  CASE("a zero or negative report floors to 1");
  {
    AxisReport a;
    a.contiguityElems = 0;
    a.divisibilityBytes = 0;
    a.isTensorOfPointers = true;
    const PtrInfo p = ptrInfoFrom(a, f32());
    CHECK_EQ(p.contiguity, (int64_t)1);
    CHECK_EQ(p.alignment, (int64_t)1);
  }

  CASE("contiguity is never converted, whatever the element");
  {
    for (ElemType e : {f32(), f16(), e4m3()}) {
      AxisReport a;
      a.contiguityElems = 16;
      a.divisibilityBytes = 64;
      a.isTensorOfPointers = true;
      CHECK_EQ(ptrInfoFrom(a, e).contiguity, (int64_t)16);
    }
  }

  return ::agpu_test::report("PointerBind");
}
