// The bank-conflict pad.
#include "agpu/core/Padding.h"
#include "agpu/plan/DotPlan.h"
#include "harness.h"

using namespace agpu;

int main() {
  CASE("the rule is TileLang's: whole 256-bit rows pad by 128 bits");
  {
    CHECK_EQ(padElemsFor(64, 4), 4);
    CHECK_EQ(padElemsFor(64, 2), 8);
    CHECK_EQ(padElemsFor(128, 2), 8);
    CHECK_EQ(padElemsFor(48, 4), 4);
    CHECK_EQ(padElemsFor(33, 2), 0);
    CHECK_EQ(padElemsFor(12, 2), 0);
    CHECK_EQ(padElemsFor(0, 4), 0);
    CHECK_EQ(padElemsFor(64, 0), 0);
    CHECK_EQ(padElemsFor(-8, 4), 0);
  }

  CASE("a staged view's stride carries the pad; its extent does not");
  {
    const TileView v = stagedTileView(60, 64, 4);
    CHECK_EQ(v.extentAt(0), 64);
    CHECK_EQ(v.extentAt(1), 64);
    CHECK_EQ(v.strideAt(0), 68);
    CHECK_EQ(v.strideAt(1), 1);
    CHECK_EQ(v.offsetOf({1, 0}), 68);
  }

  CASE("reservation and addressing are the same number");
  {
    const TileView v = stagedTileView(64, 64, 2);
    CHECK_EQ(stagedTileBytes(64, 64, 2).count(), v.cosizeElems() * 2);
    CHECK(v.offsetOf({63, 63}) < v.cosizeElems());
  }

  CASE("planStageBytes pads both operands through the one rule");
  {
    DotFacts f;
    f.M = 64;
    f.N = 64;
    f.K = 64;
    f.aElemBytes = 4;
    f.bElemBytes = 4;
    const StageBytes s = planStageBytes(f);
    CHECK_EQ(s.a.count(), stagedTileBytes(64, 64, 4).count());
    CHECK_EQ(s.b.count(), stagedTileBytes(64, 64, 4).count());
    CHECK_EQ(s.a.count(), (63 * 68 + 64) * 4);
  }

  return ::agpu_test::report("Padding");
}
