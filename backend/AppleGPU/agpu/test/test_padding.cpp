// The bank-conflict pad.
#include "agpu/plan/DotPlan.h"
#include "fixtures.h"
#include "harness.h"

using namespace agpu;
using agpu_test::basis;

int main() {
  CASE("one simdgroup access takes as many passes as its busiest bank");
  {
    std::vector<int64_t> words, quads, rows;
    for (int64_t l = 0; l < kWarpSize; ++l) {
      words.push_back(l * 4);
      quads.push_back(l * 16);
      rows.push_back(l * 128);
    }
    CHECK_EQ(cost::bankPasses(words, 4), 1);
    CHECK_EQ(cost::bankPasses(quads, 16), 16);
    CHECK_EQ(cost::bankPasses(rows, 4), 32);
  }

  CASE("fragment reads alone pad whole-line rows by half a line and leave "
       "spread rows unpadded");
  {
    CHECK_EQ(stagedPadFor(64, 64, 4), 4);
    CHECK_EQ(stagedPadFor(64, 64, 2), 8);
    CHECK_EQ(stagedPadFor(64, 128, 2), 8);
    CHECK_EQ(stagedPadFor(64, 48, 4), 4);
    CHECK_EQ(stagedPadFor(64, 24, 2), 0);
    CHECK_EQ(stagedPadFor(64, 8, 4), 0);
    CHECK_EQ(stagedPadFor(0, 64, 4), 0);
    CHECK_EQ(stagedPadFor(64, 64, 0), 0);
  }

  CASE("stores that run along a row keep the read pad");
  {
    // Four elements a lane, eight lanes a row, four rows a simdgroup.
    const StageTraffic rowRuns{
        {basis({}, {0, 0, 0, 1, 2}), basis({1, 2}, {4, 8, 16, 0, 0})}, 4};
    CHECK_EQ(stagedPadFor(64, 32, 4, rowRuns), 4);
  }

  CASE("stores that run down a column pick the pitch that spreads the lanes");
  {
    // A transposed operand: each lane holds four rows of one column, and the
    // simdgroup's lanes stand four rows apart. At the read pad (pitch 36)
    // they share two banks; at pitch 38 eight.
    const StageTraffic colRuns{{basis({1, 2, 0, 0}, {4, 8, 16, 32, 64}),
                                basis({0, 0, 8, 16}, {0, 0, 0, 0, 0})},
                               8};
    CHECK_EQ(stagedPadFor(128, 32, 4, colRuns), 6);
    const TileView v = stagedTileView(128, 32, 4, true, colRuns);
    CHECK_EQ(v.strideAt(0), 38);
    CHECK_EQ(stagedTileBytes(128, 32, 4, true, colRuns).count(),
             v.cosizeElems() * 4);
  }

  CASE("a pitch that saves store passes only as fast as it adds read passes "
       "keeps the smaller pad");
  {
    // Sixteen lanes four rows apart, a lane bit across the columns. Stored
    // by eight warps and read by two per fragment, the spread pitch wins;
    // stored by four and read by four, it ties and the smaller pad stays.
    const std::vector<LayoutBasis> down{basis({1, 2, 0}, {4, 8, 16, 32, 0}),
                                        basis({0, 0, 16}, {0, 0, 0, 0, 1})};
    CHECK_EQ(stagedPadFor(64, 32, 4, {down, 8, 64}), 6);
    const std::vector<LayoutBasis> down4{basis({1, 2, 0, 0}, {4, 8, 16, 32, 0}),
                                         basis({0, 0, 8, 16}, {0, 0, 0, 0, 1})};
    CHECK_EQ(stagedPadFor(64, 32, 4, {down4, 4, 128}), 4);
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

  CASE("planStageBytes pads each operand for its own writers");
  {
    DotFacts f;
    f.M = 128;
    f.N = 64;
    f.K = 32;
    f.aElemBytes = 4;
    f.bElemBytes = 4;
    f.numWarps = 8;
    CHECK_EQ(planStageBytes(f).a.count(), (127 * 36 + 32) * 4);
    f.aStageDims = {basis({1, 2, 0, 0}, {4, 8, 16, 32, 64}),
                    basis({0, 0, 8, 16}, {0, 0, 0, 0, 0})};
    const StageBytes s = planStageBytes(f);
    CHECK_EQ(s.a.count(), (127 * 38 + 32) * 4);
    CHECK(stagedAView(f) == stagedTileView(128, 32, 4, true,
                                           {f.aStageDims, f.numWarps,
                                            stagedFragmentReads(f, true)}));
    CHECK_EQ(s.b.count(), stagedTileBytes(32, 64, 4).count());
  }

  return ::agpu_test::report("Padding");
}
