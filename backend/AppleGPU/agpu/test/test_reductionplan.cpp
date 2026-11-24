#include "agpu/plan/ReductionPlan.h"
#include "harness.h"

using namespace agpu;

namespace {
std::vector<int32_t> keyOf(const CoordKey &k) { return k.coords(); }
} // namespace

int main() {
  // ── CoordKey identity ──────────────────────────────────────────────────
  CASE("typed key: equality is componentwise");
  {
    CHECK(CoordKey({1, 2, 3}) == CoordKey({1, 2, 3}));
    CHECK(CoordKey({1, 2, 3}) != CoordKey({1, 3, 2}));
    CHECK(CoordKey({12, 3}) != CoordKey({1, 23}));
  }

  CASE("dropAxis removes exactly one dimension");
  {
    CoordKey full({4, 7, 2});
    CHECK_EQ(keyOf(full.dropAxis(0)), std::vector<int32_t>({7, 2}));
    CHECK_EQ(keyOf(full.dropAxis(1)), std::vector<int32_t>({4, 2}));
    CHECK_EQ(keyOf(full.dropAxis(2)), std::vector<int32_t>({4, 7}));
    CHECK_EQ(full.dropAxis(1).rank(), 2);
  }

  // ── the invariant emitReduce assumes ───────────────────────────────────
  CASE("group key and result key are the same value");
  {
    const int axis = 1;
    std::vector<CoordKey> src = {
        CoordKey({0, 0}), CoordKey({0, 1}), CoordKey({0, 2}),
        CoordKey({1, 0}), CoordKey({1, 1}), CoordKey({1, 2}),
    };
    auto groups = groupSurvivors(src, axis);
    CHECK_EQ(groups.size(), 2u);
    CHECK_EQ(keyOf(groups[0].key), std::vector<int32_t>({0}));
    CHECK_EQ(keyOf(groups[1].key), std::vector<int32_t>({1}));
    CHECK_EQ(groups[0].sourceRegs, std::vector<int>({0, 1, 2}));
    CHECK_EQ(groups[1].sourceRegs, std::vector<int>({3, 4, 5}));

    ReductionPlan plan;
    plan.reducedAxis = axis;
    plan.groups = groups;
    CHECK_EQ(plan.groupFor(CoordKey({1})), 1);
    CHECK_EQ(plan.groupFor(CoordKey({0})), 0);
    CHECK_EQ(plan.groupFor(CoordKey({9})), -1);
  }

  CASE("reducing the outer axis still round-trips");
  {
    const int axis = 0;
    std::vector<CoordKey> src = {
        CoordKey({0, 0}),
        CoordKey({1, 0}),
        CoordKey({0, 1}),
        CoordKey({1, 1}),
    };
    auto groups = groupSurvivors(src, axis);
    CHECK_EQ(groups.size(), 2u);
    CHECK_EQ(keyOf(groups[0].key), std::vector<int32_t>({0}));
    CHECK_EQ(groups[0].sourceRegs, std::vector<int>({0, 1}));
    CHECK_EQ(keyOf(groups[1].key), std::vector<int32_t>({1}));
    CHECK_EQ(groups[1].sourceRegs, std::vector<int>({2, 3}));
  }

  CASE("replicated registers fold exactly once");
  {
    std::vector<CoordKey> src = {
        CoordKey({0, 0}),
        CoordKey({0, 1}),
        CoordKey({0, 0}),
    };
    auto groups = groupSurvivors(src, 1);
    CHECK_EQ(groups.size(), 1u);
    CHECK_EQ(groups[0].sourceRegs, std::vector<int>({0, 1}));
  }

  CASE("rank-3: dropping the middle axis");
  {
    std::vector<CoordKey> src = {
        CoordKey({0, 0, 0}),
        CoordKey({0, 1, 0}),
        CoordKey({0, 0, 1}),
        CoordKey({0, 1, 1}),
    };
    auto groups = groupSurvivors(src, 1);
    CHECK_EQ(groups.size(), 2u);
    CHECK_EQ(keyOf(groups[0].key), std::vector<int32_t>({0, 0}));
    CHECK_EQ(keyOf(groups[1].key), std::vector<int32_t>({0, 1}));
  }

  // ── lane steps ─────────────────────────────────────────────────────────
  CASE("lane XOR steps are emitted high bit first");
  {
    auto steps = laneStepsFromMask(0b10101);
    CHECK_EQ(steps.size(), 3u);
    CHECK_EQ(steps[0].xorOffset, 16);
    CHECK_EQ(steps[1].xorOffset, 4);
    CHECK_EQ(steps[2].xorOffset, 1);
  }

  CASE("full 32-lane reduction is five steps");
  {
    auto steps = laneStepsFromMask(0b11111);
    CHECK_EQ(steps.size(), 5u);
    CHECK_EQ(steps[0].xorOffset, 16);
    CHECK_EQ(steps[4].xorOffset, 1);
  }

  CASE("no lane bits: lane-local reduction");
  {
    CHECK(laneStepsFromMask(0).empty());
  }

  // ── mask derivation, mirroring reduceMask ──────────────────────────────
  CASE("reduceMask: bits whose basis moves the axis");
  {
    CHECK_EQ(reduceMaskFromBases({1, 2, 0, 0}), 0b0011u);
    CHECK_EQ(reduceMaskFromBases({0, 0, 4, 8}), 0b1100u);
    CHECK_EQ(reduceMaskFromBases({0, 0, 0, 0}), 0u);
  }

  // ── warp subsets, mirroring subsetsOf ──────────────────────────────────
  CASE("warp subset: every value of the masked bits");
  {
    CHECK_EQ(subsetsOf(0b011, 8), std::vector<int>({0, 1, 2, 3}));
    CHECK_EQ(subsetsOf(0b101, 8), std::vector<int>({0, 1, 4, 5}));
    CHECK_EQ(subsetsOf(0, 8), std::vector<int>({0}));
  }

  CASE("warp subset clamps to numWarps");
  {
    auto v = subsetsOf(0b111, 4);
    for (int x : v)
      CHECK(x < 4);
    CHECK_EQ(v, std::vector<int>({0, 1, 2, 3}));
  }

  // ── cross-warp classification ──────────────────────────────────────────
  CASE("crossWarp is false for a lane-local reduction");
  {
    ReductionPlan p;
    p.warpSubset = subsetsOf(0, 8);
    CHECK(!p.crossWarp());
    CHECK_EQ(p.warpSubset.size(), 1u);
  }

  CASE("crossWarp is true when the axis spans warps");
  {
    ReductionPlan p;
    p.warpSubset = subsetsOf(0b011, 8);
    p.scratch = ScratchLayout{8 * 32, 32};
    CHECK(p.crossWarp());
    CHECK_EQ(p.scratch.slotFor(3, 7), 3 * 32 + 7);
    CHECK_EQ(p.scratch.slotsPerOperand, 256);
  }

  // ── multi-operand ──────────────────────────────────────────────────────
  CASE("multi-operand: the topology is shared, scratch is per operand");
  {
    ReductionPlan p;
    p.warpSubset = subsetsOf(0b1, 4);
    p.scratch = ScratchLayout{4 * 32, 32};
    CHECK(p.crossWarp());
    CHECK_EQ(p.scratch.slotsPerOperand, 128);
  }

  return ::agpu_test::report("ReductionPlan");
}
