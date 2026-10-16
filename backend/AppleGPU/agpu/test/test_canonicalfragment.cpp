// CanonicalFragment tests.
#include "agpu/plan/CanonicalFragment.h"
#include "harness.h"

using namespace agpu;

int main() {
  CASE("MSL type spelling comes from the kind");
  {
    CanonicalFragment f(FragmentKind::Simdgroup8x8);
    CHECK_EQ(f.mslType("half"), std::string("simdgroup_half8x8"));
    CHECK_EQ(f.mslType("float"), std::string("simdgroup_float8x8"));
    CHECK_EQ(f.mslType("bfloat"), std::string("simdgroup_bfloat8x8"));
  }

  CASE("an unrecognised fragment spells nothing and measures zero");
  {
    CanonicalFragment none;
    CHECK(!bool(none));
    CHECK_EQ(none.dim(), 0);
    CHECK_EQ(none.lanes(), 0);
    CHECK(none.mslType("half").empty());
  }

  CASE("the 8x8 fragment is 64 elements over the warp");
  {
    CHECK_EQ(kSimdgroup8x8.dim(), kSgFragDim);
    CHECK_EQ(kSimdgroup8x8.lanes(), kWarpSize);
    CHECK_EQ(kSimdgroup8x8.elemsPerLane(), kSgFragDim * kSgFragDim / kWarpSize);
  }

  return ::agpu_test::report("CanonicalFragment");
}
