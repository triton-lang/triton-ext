#include "agpu/plan/RebindPlan.h"
#include "fixtures.h"
#include "harness.h"

#include <vector>

using namespace agpu;

using agpu_test::coordsOfShape;

namespace {

// `order[d]` is the source axis that becomes result axis d, MLIR's convention.
Rebind transposeBy(const std::vector<RegCoord> &res,
                   const std::vector<RegCoord> &src,
                   const std::vector<int> &order) {
  return rebind(res, indexByCoord(src),
                [&order](const RegCoord &rc, RegCoord &want) {
                  if (rc.size() != order.size())
                    return false;
                  want.assign(rc.size(), 0);
                  for (std::size_t d = 0; d < rc.size(); ++d)
                    want[(std::size_t)order[d]] = rc[d];
                  return true;
                });
}

} // namespace

int main() {
  CASE("the walk finds the source whose coordinate the map names");
  {
    const std::vector<RegCoord> src = coordsOfShape({4, 3});
    const std::vector<RegCoord> res = coordsOfShape({4, 1, 3});
    const Rebind r =
        rebind(res, indexByCoord(src), [](const RegCoord &rc, RegCoord &want) {
          want = {rc[0], rc[2]};
          return true;
        });
    CHECK(rebindDecision(r).ok());
    for (std::size_t n = 0; n < res.size(); ++n)
      CHECK_EQ(src[(std::size_t)r.from[n]], (RegCoord{res[n][0], res[n][2]}));
  }

  CASE("a map returning false leaves the register unclaimed");
  {
    // Unclaimed registers stay at -1 for the caller to fill from the other
    // walk.
    const std::vector<RegCoord> src = coordsOfShape({4});
    const std::vector<RegCoord> res = coordsOfShape({4, 2});
    int claimedTotal = 0;
    for (int which = 0; which < 2; ++which) {
      const Rebind r = rebind(res, indexByCoord(src),
                              [which](const RegCoord &rc, RegCoord &want) {
                                if (rc[1] != which)
                                  return false;
                                want = {rc[0]};
                                return true;
                              });
      CHECK(!r.complete());
      for (std::size_t n = 0; n < res.size(); ++n)
        if (r.from[n] >= 0) {
          ++claimedTotal;
          CHECK_EQ(res[n][1], which);
          CHECK_EQ(src[(std::size_t)r.from[n]], (RegCoord{res[n][0]}));
        }
    }
    // Between them, exactly once each.
    CHECK_EQ(claimedTotal, (int)res.size());
  }

  CASE("trans reads `order` as MLIR does: order[d] is the source axis");
  {
    const RegCoord shape = {2, 3, 4};
    const std::vector<int> order = {1, 2, 0};
    const std::vector<RegCoord> src = coordsOfShape(shape);
    const std::vector<RegCoord> res = coordsOfShape({3, 4, 2});

    const Rebind r = transposeBy(res, src, order);
    CHECK(rebindDecision(r).ok());

    for (std::size_t n = 0; n < res.size(); ++n) {
      RegCoord want(3, 0);
      for (int d = 0; d < 3; ++d)
        want[(std::size_t)order[(std::size_t)d]] = res[n][(std::size_t)d];
      CHECK_EQ(src[(std::size_t)r.from[n]], want);
    }
  }

  CASE("a transpose composed with its inverse is the identity");
  {
    const std::vector<RegCoord> a = coordsOfShape({2, 3, 4});
    const std::vector<RegCoord> b = coordsOfShape({3, 4, 2});

    const Rebind ab = transposeBy(b, a, {1, 2, 0});
    const Rebind ba = transposeBy(a, b, {2, 0, 1});
    CHECK(rebindDecision(ab).ok());
    CHECK(rebindDecision(ba).ok());

    for (std::size_t n = 0; n < a.size(); ++n)
      CHECK_EQ(ab.from[(std::size_t)ba.from[n]], (int)n);
  }

  CASE("a 2-D transpose is its own inverse, which is why it hides the bug");
  {
    const std::vector<RegCoord> sq = coordsOfShape({3, 3});
    const Rebind r = transposeBy(sq, sq, {1, 0});
    CHECK(rebindDecision(r).ok());
    for (std::size_t n = 0; n < sq.size(); ++n)
      CHECK_EQ(r.from[(std::size_t)r.from[n]], (int)n);
  }

  CASE("a result register with no source declines");
  {
    const std::vector<RegCoord> src = coordsOfShape({2});
    const std::vector<RegCoord> res = coordsOfShape({4});
    const Rebind r =
        rebind(res, indexByCoord(src), [](const RegCoord &rc, RegCoord &want) {
          want = rc;
          return true;
        });
    CHECK(!r.complete());
    const Decision d = rebindDecision(r);
    CHECK(d.isDecline());
    CHECK(!d.isBug());
    CHECK_EQ(r.from[0], 0);
    CHECK(r.from[3] < 0);
  }

  CASE("an empty rebinding declines outright");
  {
    CHECK(rebindDecision(Rebind{}).isDecline());
  }

  CASE("a coordinate index does not collide the way a packed key would");
  {
    const std::vector<RegCoord> wide = coordsOfShape({1, 70000});
    const CoordIndex idx = indexByCoord(wide);
    CHECK_EQ(idx.size(), wide.size());
    CHECK_EQ(idx.at(RegCoord{0, 69999}), 69999);
    CHECK_EQ(idx.at(RegCoord{0, 65536}), 65536);
  }

  return ::agpu_test::report("RebindPlan");
}
