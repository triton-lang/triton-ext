// LayoutExpr tests: coordinate expressions from layout bases.
#include "agpu/emit/LayoutExpr.h"
#include "agpu/msl/Printer.h"
#include "fixtures.h"
#include "harness.h"
#include "render.h"

#include <sstream>

using namespace agpu;
using agpu_test::basis;
using agpu_test::render;

namespace {

struct ExprRow {
  const char *what;
  LayoutBasis b;
  int reg2Index;
  const char *want;
};

struct RangeRow {
  const char *what;
  LayoutBasis b;
  int reg2Index;
  std::int32_t extent, lo, hi;
};

} // namespace

int main() {
  CASE("a coordinate spells exactly the bits its bases reach");
  {
    // An identity run collapses to one mask. Anything else is an explicit
    // bit term, and the terms xor together. The run test is
    // basis(k) == 1<<k on the bit's own position.
    const ExprRow rows[] = {
        {"a full identity run is one mask", basis({}, {1, 2, 4, 8, 16}), 0,
         "lane & 31"},
        {"a partial identity run masks only its own bits",
         basis({}, {0, 0, 4, 8, 16}), 0, "lane & 28"},
        {"a non-identity basis becomes an explicit bit term",
         basis({}, {0, 0, 0, 0, 1}), 0, "lane >> 4 & 1"},
        {"mixed runs and singletons combine by xor", basis({}, {1, 2, 0, 0, 1}),
         0, "lane & 3 ^ lane >> 4 & 1"},
        {"a register-only coordinate is a bare constant", basis({8, 16}, {}), 0,
         "0"},
        {"register 1 of a register-only coordinate", basis({8, 16}, {}), 1,
         "8"},
        {"register 2 of a register-only coordinate", basis({8, 16}, {}), 2,
         "16"},
        {"a register past the bases keeps counting", basis({8, 16}, {}), 3,
         "24"},
        {"a register constant joins the runtime terms",
         basis({32}, {1, 2, 4, 0, 0}), 1, "32 ^ lane & 7"},
        {"register 0 contributes no constant", basis({32}, {1, 2, 4, 0, 0}), 0,
         "lane & 7"},
        {"lane and warp both contribute", basis({}, {1, 2, 0, 0, 0}, {1, 2}), 0,
         "lane & 3 ^ warp & 3"},
        {"a shifted warp mapping is not an identity run", basis({}, {}, {4, 8}),
         0, "(warp & 1) * 4 ^ (warp >> 1 & 1) * 8"},
        {"an empty layout is the zero coordinate", basis({}, {}), 0, "0"},
    };

    for (const ExprRow &r : rows) {
      SUBCASE(r.what);
      msl::Context c;
      CHECK_EQ(render(coordExpr(c, r.b, r.reg2Index, "lane", "warp")),
               std::string(r.want));
    }
  }

  CASE("a range is exact when the bases are disjoint, conservative when not");
  {
    // Bases sharing a bit make the reachable set an xor lattice, which no
    // interval describes: the range widens to the whole dimension.
    const RangeRow rows[] = {
        {"disjoint bases give an exact range", basis({8}, {1, 2, 4, 0, 0}), 0,
         64, 0, 7},
        {"the register constant shifts the exact range",
         basis({8}, {1, 2, 4, 0, 0}), 1, 64, 8, 15},
        {"overlapping bases fall back to the whole dimension",
         basis({}, {1, 1}), 0, 64, 0, 63},
        {"a register constant overlapping the runtime mask is conservative",
         basis({1}, {1, 2, 0, 0, 0}), 1, 64, 0, 63},
        {"a block-distributed range is wider than lane alone",
         basis({}, {1, 2, 4, 0, 0}, {}, {8, 16, 32}), 0, 256, 0, 63},
        {"the same layout without block bases stays narrow",
         basis({}, {1, 2, 4, 0, 0}), 0, 256, 0, 7},
        {"block bases join the disjointness test too",
         basis({}, {1, 2}, {}, {2}), 0, 64, 0, 63},
    };

    for (const RangeRow &r : rows) {
      SUBCASE(r.what);
      const CoordRange got = r.b.rangeOf(r.reg2Index, 0, r.extent);
      CHECK_EQ(got.lo, r.lo);
      CHECK_EQ(got.hi, r.hi);
    }
  }

  CASE("a coordinate reads the block id and every threadgroup differs");
  {
    msl::Context c;
    LayoutBasis b;
    b.lane = {1, 2, 4, 0, 0};
    b.block = {8, 16, 32};
    CHECK(b.needsBlockId());

    const std::string s = render(coordExpr(c, b, 0, "lane", "warp", "tgpos.x"));
    CHECK_HAS(s, "tgpos.x");
    CHECK_HAS(s, "lane & 7");
  }

  CASE("a layout with no block bases demands no block id");
  {
    msl::Context c;
    LayoutBasis b;
    b.lane = {1, 2, 4, 0, 0};
    CHECK(!b.needsBlockId());
    CHECK_EQ(render(coordExpr(c, b, 0, "lane", "warp")),
             render(coordExpr(c, b, 0, "lane", "warp", "tgpos.x")));
  }

  CASE("a row of zero bases is not a block dimension");
  {
    LayoutBasis b;
    b.lane = {1, 2};
    b.block = {0, 0, 0};
    CHECK(!b.needsBlockId());
    CHECK_EQ(b.rangeOf(0, 0, 64).hi, 3);
  }

  CASE("every threadgroup reaches a distinct coordinate, by simulation");
  {
    // Eight threadgroups times eight lanes must cover 0..63 once each.
    LayoutBasis b;
    b.lane = {1, 2, 4, 0, 0};
    b.block = {8, 16, 32};

    const auto coordOf = [&](int lane, int blk) {
      int32_t v = 0;
      for (std::size_t i = 0; i < b.lane.size(); ++i)
        if (lane & (1 << i))
          v ^= b.lane[i];
      for (std::size_t i = 0; i < b.block.size(); ++i)
        if (blk & (1 << i))
          v ^= b.block[i];
      return v;
    };

    bool seen[64] = {};
    for (int blk = 0; blk < 8; ++blk)
      for (int lane = 0; lane < 8; ++lane) {
        const int32_t coord = coordOf(lane, blk);
        CHECK(coord >= 0 && coord < 64);
        CHECK(!seen[coord]);
        seen[coord] = true;
      }
    for (bool v : seen)
      CHECK(v);

    const CoordRange r = b.rangeOf(0, 0, 64);
    CHECK_EQ(r.lo, 0);
    CHECK_EQ(r.hi, 63);
  }

  CASE("range and expression describe the same register");
  {
    msl::Context c;
    LayoutBasis b;
    b.reg = {16, 32};
    b.lane = {1, 2, 4, 8, 0};
    for (int reg = 0; reg < 4; ++reg) {
      CoordRange r = b.rangeOf(reg, 0, 64);
      const int32_t base = b.registerConstant(reg);
      CHECK_EQ(r.lo, base);
      CHECK_EQ(r.hi, base + 15);
      std::string s = render(coordExpr(c, b, reg, "lane", "warp"));
      if (base == 0)
        CHECK_EQ(s, std::string("lane & 15"));
      else
        CHECK_EQ(s, std::to_string(base) + " ^ lane & 15");
    }
  }

  return ::agpu_test::report("LayoutExpr");
}
