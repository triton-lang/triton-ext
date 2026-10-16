// Coordinate expressions, emitted once each.
#include "agpu/bind/LayoutBind.h"
#include "agpu/msl/Printer.h"
#include "fixtures.h"
#include "harness.h"
#include "render.h"

#include <set>
#include <sstream>

using namespace agpu;
using agpu_test::basis;
using agpu_test::render;

namespace {

std::string unhoisted(const LayoutBasis &lb, int reg) {
  msl::Context c;
  return render(coordExpr(c, lb, reg, "lane", "warp", "tgpos.x"));
}

LayoutBasis laneWarp() {
  return basis(/*reg=*/{64}, /*lane=*/{1, 2, 4, 8, 16}, /*warp=*/{32});
}

} // namespace

int main() {
  CASE("two coordinates share a name IFF they would print the same");
  {
    const std::vector<LayoutBasis> layouts = {
        laneWarp(),
        basis({32}, {1, 2, 4, 8, 16}),
        basis({}, {1, 2, 4, 8, 16}, {32, 64}),
        basis({16}, {1, 2}, {4}, {8}),
    };

    msl::Context c;
    CoordHoist h{ThreadNames{}};

    std::vector<std::pair<std::string, std::string>> seen; // (text, name)
    for (const LayoutBasis &lb : layouts)
      for (int reg = 0; reg < 4; ++reg) {
        const std::string want = unhoisted(lb, reg);
        const std::string got = render(h.coord(c, lb, reg));
        seen.push_back({want, got});
      }

    for (std::size_t i = 0; i < seen.size(); ++i)
      for (std::size_t j = 0; j < seen.size(); ++j) {
        const bool sameText = seen[i].first == seen[j].first;
        const bool sameName = seen[i].second == seen[j].second;
        CHECK_EQ(sameText, sameName);
      }
  }

  CASE("the second ask returns the same cached name");
  {
    msl::Context c;
    CoordHoist h{ThreadNames{}};
    const LayoutBasis lb = laneWarp();

    const std::string first = render(h.coord(c, lb, 0));
    const std::string second = render(h.coord(c, lb, 0));
    CHECK_EQ(first, second);
    CHECK_EQ(h.decls.size(), (std::size_t)1);
    CHECK_EQ(h.distinct(), (std::size_t)1);

    const std::string decls = render(h.decls);
    CHECK_HAS(decls, "lane & 31");
    CHECK_HAS(decls, first);
  }

  CASE("different registers of one layout do not share");
  {
    msl::Context c;
    CoordHoist h{ThreadNames{}};
    const LayoutBasis lb = laneWarp();
    const std::string a = render(h.coord(c, lb, 0));
    const std::string b = render(h.coord(c, lb, 1));
    CHECK(a != b);
    CHECK_EQ(h.decls.size(), (std::size_t)2);
  }

  CASE("two layouts with the same bases do share");
  {
    msl::Context c;
    CoordHoist h{ThreadNames{}};
    const LayoutBasis a = laneWarp();
    const LayoutBasis b = laneWarp();
    CHECK_EQ(render(h.coord(c, a, 0)), render(h.coord(c, b, 0)));
    CHECK_EQ(h.decls.size(), (std::size_t)1);
  }

  CASE("a key cannot be confused by concatenation");
  {
    const LayoutBasis a{{}, {1, 2}, {}, {}};
    const LayoutBasis b{{}, {12}, {}, {}};
    CHECK(coordKey(a, 0) != coordKey(b, 0));
    const LayoutBasis lane{{}, {1}, {}, {}};
    const LayoutBasis warp{{}, {}, {1}, {}};
    CHECK(coordKey(lane, 0) != coordKey(warp, 0));
  }

  CASE("a coordinate that folds to a literal is not hoisted");
  {
    msl::Context c;
    CoordHoist h{ThreadNames{}};
    const LayoutBasis constant{/*reg=*/{4}, /*lane=*/{0, 0, 0, 0, 0},
                               /*warp=*/{}, /*block=*/{}};
    CHECK_EQ(render(h.coord(c, constant, 1)), std::string("4"));
    CHECK(h.decls.empty());
    CHECK_EQ(h.distinct(), (std::size_t)0);
  }

  CASE("a block-distributed coordinate hoists too, reading the block id");
  {
    msl::Context c;
    CoordHoist h{ThreadNames{}};
    const LayoutBasis lb{{}, {1, 2, 4}, {}, {8, 16}};
    CHECK(h.coord(c, lb, 0) != nullptr);
    const std::string decls = render(h.decls);
    CHECK_HAS(decls, "tgpos.x");
  }

  CASE("declarations come out in creation order");
  {
    msl::Context c;
    CoordHoist h{ThreadNames{}};
    const std::string first = render(h.coord(c, laneWarp(), 0));
    const std::string second = render(h.coord(c, laneWarp(), 1));
    const std::string decls = render(h.decls);
    CHECK(decls.find(first) < decls.find(second));
  }

  return ::agpu_test::report("LayoutBind");
}
