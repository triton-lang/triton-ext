// make_range: the element IS its coordinate.
#include "agpu/emit/EmitRange.h"
#include "agpu/msl/Printer.h"
#include "harness.h"
#include "render.h"

#include <sstream>

using namespace agpu;
using agpu_test::render;

namespace {

LayoutBasis laneMajor() {
  return LayoutBasis{/*reg=*/{32, 64, 128}, /*lane=*/{1, 2, 4, 8, 16},
                     /*warp=*/{}, /*block=*/{}};
}

} // namespace

int main() {
  CASE("a range element is exactly the coordinate the address path builds");
  {
    msl::Context c;
    const LayoutBasis lb = laneMajor();
    for (int reg = 0; reg < 8; ++reg)
      CHECK_EQ(render(rangeElem(c, lb, reg, 0, "lane", "warp")),
               render(coordExpr(c, lb, reg, "lane", "warp")));
  }

  CASE("a start adds; it does not merge into the register constant");
  {
    // Register 1 contributes 32 and start is 8, so `40 ^ (lane & 31)` is
    // wrong: 8 lies inside the lane mask 31 and at lane 8 the xor cancels the
    // start. Merging is sound only when the start is disjoint from every
    // variable bit.
    msl::Context c;
    const LayoutBasis lb = laneMajor();
    CHECK_EQ(render(rangeElem(c, lb, 1, 8, "lane", "warp")),
             std::string("8 + (32 ^ lane & 31)"));
  }

  CASE("a zero start costs nothing");
  {
    msl::Context c;
    const LayoutBasis lb = laneMajor();
    CHECK_EQ(render(rangeElem(c, lb, 0, 0, "lane", "warp")),
             std::string("lane & 31"));
  }

  CASE("a purely register-determined range is a bare literal");
  {
    msl::Context c;
    const LayoutBasis lb{/*reg=*/{1, 2}, /*lane=*/{0, 0, 0, 0, 0},
                         /*warp=*/{}, /*block=*/{}};
    CHECK_EQ(render(rangeElem(c, lb, 3, 100, "lane", "warp")),
             std::string("103"));
    CHECK_EQ(render(rangeElem(c, lb, 0, 0, "lane", "warp")), std::string("0"));
  }

  CASE("a warp-strided range reads the warp id, one term per bit");
  {
    // The run-collapse fires only on identity bases (basis(k) == 1<<k). These
    // are shifted, so each bit stays its own term.
    msl::Context c;
    const LayoutBasis lb{/*reg=*/{}, /*lane=*/{1, 2, 4, 8, 16},
                         /*warp=*/{32, 64}, /*block=*/{}};
    CHECK_EQ(render(rangeElem(c, lb, 0, 0, "lane", "warp")),
             std::string("lane & 31 ^ (warp & 1) * 32 ^ (warp >> 1 & 1) * 64"));
  }

  CASE("consecutive values across lanes, which is what the op means");
  {
    const LayoutBasis lb = laneMajor();
    const int64_t start = 5;
    bool seen[64] = {};
    for (int reg = 0; reg < 2; ++reg)
      for (int lane = 0; lane < 32; ++lane) {
        const int64_t coord = lb.registerConstant(reg) ^ lane;
        const int64_t v = start + coord;
        CHECK(v >= start && v < start + 64);
        CHECK(!seen[v - start]);
        seen[v - start] = true;
      }
    for (bool b : seen)
      CHECK(b);
  }

  return ::agpu_test::report("EmitRange");
}
