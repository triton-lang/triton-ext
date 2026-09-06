#include "agpu/emit/primitives/FragLane.h"
#include "agpu/plan/ReadbackPlan.h"
#include "harness.h"

using namespace agpu;

namespace {

// A 32x64 result under warpsPerCTA [4,1].
std::vector<LayoutBasis> mmaDims(int64_t regRowBits, int64_t regColBits) {
  LayoutBasis row, col;
  row.lane = {0, 1, 2, 0, 4};
  col.lane = {2, 0, 0, 4, 0};
  col.reg.push_back(1);
  for (int64_t b = 0; b < regColBits; ++b) {
    row.reg.push_back(0);
    col.reg.push_back((int32_t)(kSgFragDim << b));
  }
  for (int64_t b = 0; b < regRowBits; ++b) {
    row.reg.push_back((int32_t)(kSgFragDim << b));
    col.reg.push_back(0);
  }
  row.warp = {8, 16};
  col.warp = {0, 0};
  return {row, col};
}

std::vector<WarpSlot> cover(int64_t miCount, int64_t niCount) {
  std::vector<WarpSlot> s;
  int acc = 0;
  for (int64_t m = 0; m < miCount; ++m)
    for (int64_t n = 0; n < niCount; ++n)
      s.push_back({SlotCoord::affine(miCount == 1 ? 1 : 0, m),
                   SlotCoord::fixed(n), acc++});
  return s;
}

} // namespace

int main() {
  CASE("the basis constants agree with the emitted fragment map");
  {
    for (int64_t lane = 0; lane < 32; ++lane) {
      int32_t r = 0, c = 0;
      for (int b = 0; b < 5; ++b)
        if (lane & (int64_t(1) << b)) {
          r ^= kFragLaneRowBasis[b];
          c ^= kFragLaneColBasis[b];
        }
      CHECK_EQ(r, (int32_t)fragLaneRow(lane));
      CHECK_EQ(c, (int32_t)fragLaneCol(lane, 0));
    }
  }

  CASE("a cover that matches the layout reads back by rename");
  {
    const ReadbackPlan p = planReadback(mmaDims(0, 3), cover(1, 8), 16, 4);
    CHECK(p.rename());
    CHECK_EQ((int64_t)p.regs.size(), 16);
    for (int64_t r = 0; r < 16; ++r) {
      CHECK_EQ(p.regs[(std::size_t)r].acc, r / 2);
      CHECK_EQ(p.regs[(std::size_t)r].elem, r % 2);
    }
  }

  CASE("a cover that splits a row against the layout stays on the pool");
  {
    CHECK(!planReadback(mmaDims(0, 3), cover(2, 4), 16, 4).rename());
  }

  CASE("a block-varying coordinate is never a rename");
  {
    std::vector<LayoutBasis> dims = mmaDims(0, 3);
    dims[0].block = {8};
    CHECK(!planReadback(dims, cover(1, 8), 16, 4).rename());
  }

  CASE("a lane map that is not the fragment's is never a rename");
  {
    std::vector<LayoutBasis> dims = mmaDims(0, 3);
    dims[0].lane = {1, 2, 4, 8, 16};
    CHECK(!planReadback(dims, cover(1, 8), 16, 4).rename());
  }

  CASE("a register basis inside a fragment is never a rename");
  {
    std::vector<LayoutBasis> dims = mmaDims(0, 3);
    dims[1].reg[1] = 2;
    CHECK(!planReadback(dims, cover(1, 8), 16, 4).rename());
  }

  CASE("no slots and no registers decline gracefully");
  {
    CHECK(!planReadback(mmaDims(0, 3), {}, 16, 4).rename());
    CHECK(!planReadback(mmaDims(0, 3), cover(1, 8), 0, 4).rename());
    CHECK(!planReadback({}, cover(1, 8), 16, 4).rename());
  }

  CASE("a window names only its registers, at tile-local fragments");
  {
    ReadbackWindow w;
    w.rowHi = 32;
    w.colLo = 32;
    w.colHi = 64;
    const ReadbackPlan p = planReadback(mmaDims(0, 3), cover(1, 4), 16, 4, w);
    CHECK(p.rename());
    CHECK_EQ((int64_t)p.regs.size(), 16);
    for (int64_t r = 0; r < 8; ++r)
      CHECK_EQ(p.regs[(std::size_t)r].acc, -1);
    for (int64_t r = 8; r < 16; ++r) {
      CHECK_EQ(p.regs[(std::size_t)r].acc, (r - 8) / 2);
      CHECK_EQ(p.regs[(std::size_t)r].elem, r % 2);
    }
  }

  CASE("a window that cuts through a warp basis stays on the pool");
  {
    ReadbackWindow w;
    w.rowLo = 8;
    w.rowHi = 24;
    w.colHi = 64;
    CHECK(!planReadback(mmaDims(0, 3), cover(1, 8), 16, 4, w).rename());
  }

  CASE("a batch axis needs a window that names its slice");
  {
    std::vector<LayoutBasis> dims = mmaDims(0, 3);
    dims.insert(dims.begin(), LayoutBasis{});
    CHECK(!planReadback(dims, cover(1, 8), 16, 4).rename());
    ReadbackWindow w;
    w.rowHi = 32;
    w.colHi = 64;
    w.batch = 0;
    CHECK(planReadback(dims, cover(1, 8), 16, 4, w).rename());
    w.batch = 1;
    const ReadbackPlan other = planReadback(dims, cover(1, 8), 16, 4, w);
    CHECK(other.rename());
    CHECK_EQ(other.regs[0].acc, -1);
  }

  return ::agpu_test::report("ReadbackPlan");
}
