#include "agpu/plan/WarpSlots.h"
#include "harness.h"

#include <set>

using namespace agpu;

namespace {

bool coversExactlyOnce(const WarpProgram &p, const WarpGrid &g) {
  std::set<std::pair<int64_t, int64_t>> seen;
  int64_t count = 0;
  for (int64_t b = 0; b < p.blockCount(g.numWarps); ++b)
    for (int64_t w = 0; w < g.numWarps; ++w) {
      const std::optional<int64_t> only = p.guardWarp(b);
      if (only && *only != w)
        continue;
      for (const WarpSlot &s : p.slots(w, g.mT, g.nT, g.numWarps)) {
        if (!seen.insert({s.mi.at(w), s.ni.at(w)}).second)
          return false;
        ++count;
      }
    }
  return count == g.mT * g.nT && (int64_t)seen.size() == g.mT * g.nT;
}

} // namespace

int main() {
  CASE("KStep distinguishes rolled from unrolled by construction");
  {
    KStep u = KStep::unrolled(3);
    KStep::Offset uo = u.kOffset(32);
    CHECK(!uo.fromLoopVar);
    CHECK_EQ(uo.constant, 3 * kSgFragDim * 32);

    KStep r = KStep::rolled();
    KStep::Offset ro = r.kOffset(32);
    CHECK(ro.fromLoopVar);
    CHECK_EQ(ro.scale, 32);
    CHECK_EQ(ro.constant, 0);
  }

  CASE("effectiveWarps clamps to the fragment count");
  {
    CHECK_EQ(effectiveWarps(8, 4), 4);
    CHECK_EQ(effectiveWarps(4, 16), 4);
    CHECK_EQ(effectiveWarps(8, 8), 8);
    CHECK_EQ(effectiveWarps(8, 0), 1);
  }

  CASE("a fragment index maps to its row and column, row-major");
  {
    CHECK_EQ(fragRowOf(11, 8), 1);
    CHECK_EQ(fragColOf(11, 8), 3);
    CHECK_EQ(fragRowOf(0, 8), 0);
    CHECK_EQ(fragColOf(7, 8), 7);
  }

  CASE("64x128 tile, 4 warps");
  {
    WarpGrid g;
    g.mT = 64 / kSgFragDim;  // 8
    g.nT = 128 / kSgFragDim; // 16
    g.numWarps = 4;
    CHECK(coversExactlyOnce(planWarpProgram(g), g));
  }

  CASE("64x64 tile, 8 warps");
  {
    WarpGrid g;
    g.mT = 8;
    g.nT = 8;
    g.numWarps = 8;
    CHECK(coversExactlyOnce(planWarpProgram(g), g));
  }

  CASE("a cover the plan chose is taken when exact, else the scan decides");
  {
    WarpGrid g;
    g.mT = 4;
    g.nT = 4;
    g.numWarps = 4;
    const WarpProgram scanned = planWarpProgram(g);
    CHECK_EQ(scanned.miCount, 2);
    CHECK_EQ(scanned.niCount, 2);

    g.cover = {1, 4};
    const WarpProgram chosen = planWarpProgram(g);
    CHECK(chosen.form == WarpForm::Parameterised);
    CHECK_EQ(chosen.miCount, 1);
    CHECK_EQ(chosen.niCount, 4);
    CHECK(coversExactlyOnce(chosen, g));

    g.cover = {3, 1};
    const WarpProgram ignored = planWarpProgram(g);
    CHECK_EQ(ignored.miCount, 2);
    CHECK_EQ(ignored.niCount, 2);
  }

  CASE("exhaustive: every legal power-of-two shape covers exactly once");
  {
    int checked = 0;
    for (int64_t mT = 1; mT <= 16; mT *= 2)
      for (int64_t nT = 1; nT <= 16; nT *= 2)
        for (int64_t nw = 1; nw <= 16; nw *= 2) {
          if ((mT * nT) % nw)
            continue;
          WarpGrid g;
          g.mT = mT;
          g.nT = nT;
          g.numWarps = nw;
          CHECK(coversExactlyOnce(planWarpProgram(g), g));
          ++checked;
        }
    CHECK(checked > 20);
  }

  CASE("one warp folds affine coordinates to constants");
  {
    // A single warp's id is identically zero.
    WarpGrid g;
    g.mT = 2;
    g.nT = 2;
    g.numWarps = 1;
    const WarpProgram p = planWarpProgram(g);
    std::set<std::pair<int64_t, int64_t>> seen;
    for (const WarpSlot &s : p.slots(0, g.mT, g.nT, g.numWarps)) {
      CHECK(s.mi.isConst());
      CHECK(s.ni.isConst());
      CHECK(seen.insert({s.mi.at(0), s.ni.at(0)}).second);
    }
    CHECK_EQ((int64_t)seen.size(), g.nFrag());
  }

  return ::agpu_test::report("WarpSlots");
}
