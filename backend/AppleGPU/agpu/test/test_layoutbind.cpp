// Coordinate expressions, emitted once each.
#include "agpu/bind/LayoutBind.h"
#include "agpu/msl/Printer.h"
#include "harness.h"
#include "render.h"

#include <set>
#include <sstream>

using namespace agpu;
using agpu_test::render;

namespace {

std::string renderBlock(const msl::Block &b) {
  std::ostringstream os;
  msl::Printer p(os);
  p.printBlock(b);
  return os.str();
}

std::string unhoisted(const LayoutBasis &lb, int reg) {
  msl::Context c;
  return render(coordExpr(c, lb, reg, "lane", "warp", "tgpos.x"));
}

LayoutBasis laneWarp() {
  return LayoutBasis{/*reg=*/{64}, /*lane=*/{1, 2, 4, 8, 16},
                     /*warp=*/{32}, /*block=*/{}};
}

} // namespace

int main() {
  CASE("two coordinates share a name IFF they would print the same");
  {
    const std::vector<LayoutBasis> layouts = {
        laneWarp(),
        LayoutBasis{{32}, {1, 2, 4, 8, 16}, {}, {}},
        LayoutBasis{{}, {1, 2, 4, 8, 16}, {32, 64}, {}},
        LayoutBasis{{16}, {1, 2}, {4}, {8}},
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

    const std::string decls = renderBlock(h.decls);
    CHECK(decls.find("lane & 31") != std::string::npos);
    CHECK(decls.find(first) != std::string::npos);
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
    const std::string decls = renderBlock(h.decls);
    CHECK(decls.find("tgpos.x") != std::string::npos);
  }

  CASE("declarations come out in creation order");
  {
    msl::Context c;
    CoordHoist h{ThreadNames{}};
    const std::string first = render(h.coord(c, laneWarp(), 0));
    const std::string second = render(h.coord(c, laneWarp(), 1));
    const std::string decls = renderBlock(h.decls);
    CHECK(decls.find(first) < decls.find(second));
  }

  CASE("LayoutSource hands over exactly the four dimensions");
  {
    LayoutSource s;
    s.reg = {64};
    s.lane = {1, 2, 4, 8, 16};
    s.warp = {32};
    const LayoutBasis lb = s.basis();
    CHECK_EQ(lb.reg, s.reg);
    CHECK_EQ(lb.lane, s.lane);
    CHECK_EQ(lb.warp, s.warp);
    CHECK(lb.block.empty());
    CHECK(!lb.needsBlockId());
  }

  return ::agpu_test::report("LayoutBind");
}
