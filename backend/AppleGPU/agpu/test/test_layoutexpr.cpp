// LayoutExpr tests: coordinate expressions from layout bases.
#include "agpu/emit/LayoutExpr.h"
#include "agpu/msl/Printer.h"
#include "harness.h"
#include "render.h"

#include <sstream>

using namespace agpu;
using agpu_test::render;

int main() {
  CASE("a full identity run is one mask");
  {
    msl::Context c;
    LayoutBasis b;
    b.lane = {1, 2, 4, 8, 16};
    CHECK_EQ(render(coordExpr(c, b, 0, "lane", "warp")),
             std::string("lane & 31"));
  }

  CASE("a partial identity run masks only its own bits");
  {
    msl::Context c;
    LayoutBasis b;
    b.lane = {0, 0, 4, 8, 16};
    CHECK_EQ(render(coordExpr(c, b, 0, "lane", "warp")),
             std::string("lane & 28"));
  }

  CASE("a non-identity basis becomes an explicit bit term");
  {
    msl::Context c;
    LayoutBasis b;
    b.lane = {0, 0, 0, 0, 1};
    CHECK_EQ(render(coordExpr(c, b, 0, "lane", "warp")),
             std::string("lane >> 4 & 1"));
  }

  CASE("mixed runs and singletons combine by xor");
  {
    msl::Context c;
    LayoutBasis b;
    b.lane = {1, 2, 0, 0, 1};
    CHECK_EQ(render(coordExpr(c, b, 0, "lane", "warp")),
             std::string("lane & 3 ^ lane >> 4 & 1"));
  }

  CASE("a register-only coordinate is a bare constant");
  {
    msl::Context c;
    LayoutBasis b;
    b.reg = {8, 16};
    CHECK_EQ(render(coordExpr(c, b, 0, "lane", "warp")), std::string("0"));
    CHECK_EQ(render(coordExpr(c, b, 1, "lane", "warp")), std::string("8"));
    CHECK_EQ(render(coordExpr(c, b, 2, "lane", "warp")), std::string("16"));
    CHECK_EQ(render(coordExpr(c, b, 3, "lane", "warp")), std::string("24"));
  }

  CASE("a register constant joins the runtime terms");
  {
    msl::Context c;
    LayoutBasis b;
    b.reg = {32};
    b.lane = {1, 2, 4, 0, 0};
    CHECK_EQ(render(coordExpr(c, b, 1, "lane", "warp")),
             std::string("32 ^ lane & 7"));
    CHECK_EQ(render(coordExpr(c, b, 0, "lane", "warp")),
             std::string("lane & 7"));
  }

  CASE("lane and warp both contribute");
  {
    msl::Context c;
    LayoutBasis b;
    b.lane = {1, 2, 0, 0, 0};
    b.warp = {1, 2};
    CHECK_EQ(render(coordExpr(c, b, 0, "lane", "warp")),
             std::string("lane & 3 ^ warp & 3"));
  }

  CASE("a shifted warp mapping is not an identity run");
  {
    msl::Context c;
    LayoutBasis b;
    // Run test is basis(k) == 1<<k on the bit's own position.
    b.warp = {4, 8};
    CHECK_EQ(render(coordExpr(c, b, 0, "lane", "warp")),
             std::string("(warp & 1) * 4 ^ (warp >> 1 & 1) * 8"));
  }

  CASE("an empty layout is the zero coordinate");
  {
    msl::Context c;
    LayoutBasis b;
    CHECK_EQ(render(coordExpr(c, b, 0, "lane", "warp")), std::string("0"));
  }

  CASE("disjoint bases give an exact range");
  {
    LayoutBasis b;
    b.reg = {8};
    b.lane = {1, 2, 4, 0, 0};
    CoordRange r0 = b.rangeOf(0, 0, 64);
    CHECK_EQ(r0.lo, 0);
    CHECK_EQ(r0.hi, 7);
    CoordRange r1 = b.rangeOf(1, 0, 64);
    CHECK_EQ(r1.lo, 8);
    CHECK_EQ(r1.hi, 15);
  }

  CASE("overlapping bases fall back to the whole dimension");
  {
    // Bases sharing a bit make the reachable set an xor lattice.
    LayoutBasis b;
    b.lane = {1, 1};
    CoordRange r = b.rangeOf(0, 0, 64);
    CHECK_EQ(r.lo, 0);
    CHECK_EQ(r.hi, 63);
  }

  CASE("a register constant overlapping the runtime mask is conservative");
  {
    LayoutBasis b;
    b.reg = {1};
    b.lane = {1, 2, 0, 0, 0};
    CoordRange r = b.rangeOf(1, 0, 64);
    CHECK_EQ(r.lo, 0);
    CHECK_EQ(r.hi, 63);
  }

  CASE("a block-distributed range is wider than lane and warp alone");
  {
    LayoutBasis b;
    b.lane = {1, 2, 4, 0, 0};
    b.block = {8, 16, 32};

    const CoordRange r = b.rangeOf(0, 0, 256);
    CHECK_EQ(r.lo, 0);
    CHECK_EQ(r.hi, 63);

    LayoutBasis noBlock = b;
    noBlock.block = {};
    CHECK_EQ(noBlock.rangeOf(0, 0, 256).hi, 7);
  }

  CASE("block bases join the disjointness test too");
  {
    LayoutBasis b;
    b.lane = {1, 2};
    b.block = {2};
    const CoordRange r = b.rangeOf(0, 0, 64);
    CHECK_EQ(r.lo, 0);
    CHECK_EQ(r.hi, 63);
  }

  CASE("a coordinate reads the block id and every threadgroup differs");
  {
    msl::Context c;
    LayoutBasis b;
    b.lane = {1, 2, 4, 0, 0};
    b.block = {8, 16, 32};
    CHECK(b.needsBlockId());

    const std::string s = render(coordExpr(c, b, 0, "lane", "warp", "tgpos.x"));
    CHECK(s.find("tgpos.x") != std::string::npos);
    CHECK(s.find("lane & 7") != std::string::npos);
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
