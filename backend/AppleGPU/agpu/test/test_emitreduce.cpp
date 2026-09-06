// Reduction emission: plan -> local fold -> lane shuffles -> cross-warp.
#include "agpu/emit/EmitReduce.h"
#include "agpu/msl/Printer.h"
#include "fixtures.h"
#include "harness.h"
#include "render.h"

#include <set>
#include <sstream>

using namespace agpu;
using agpu_test::countOf;
using agpu_test::render;

namespace {

constexpr int64_t kNumWarps = 8;

CombineFn adder(msl::Context &c) {
  auto counter = std::make_shared<int>(0);
  return [&c, counter](msl::Block &body, const msl::SmallVec<msl::Str, 4> &a,
                       const msl::SmallVec<msl::Str, 4> &b) {
    msl::SmallVec<msl::Str, 4> out;
    for (std::size_t k = 0; k < a.size(); ++k) {
      const msl::Str n = "sum" + std::to_string((*counter)++);
      body.push_back(
          c.declStmt(msl::Context::f32(), n,
                     c.binary(msl::BinOp::Add, c.var(a[k]), c.var(b[k]))));
      out.push_back(n);
    }
    return out;
  };
}

ReductionPlan onePlan(int regs, unsigned laneMask, int warpBits = 0,
                      int64_t numWarps = kNumWarps) {
  ReductionPlan p;
  ReductionGroup g;
  g.key = CoordKey({0});
  for (int r = 0; r < regs; ++r)
    g.sourceRegs.push_back(r);
  p.groups.push_back(g);
  p.laneSteps = laneStepsFromMask(laneMask);
  p.warpSubset = subsetsOf(warpBits, numWarps);
  p.warpMask = (unsigned)warpBits;
  if (p.crossWarp())
    p.scratch = ScratchLayout{numWarps * 32, 32};
  return p;
}

msl::SmallVec<msl::SmallVec<msl::Str, 8>, 4> sources(int nOp, int regs) {
  msl::SmallVec<msl::SmallVec<msl::Str, 8>, 4> s;
  for (int k = 0; k < nOp; ++k) {
    msl::SmallVec<msl::Str, 8> names;
    for (int r = 0; r < regs; ++r)
      names.push_back("v" + std::to_string(k) + "_" + std::to_string(r));
    s.push_back(names);
  }
  return s;
}

} // namespace

int main() {
  ReduceNames nm;
  nm.scratch = {"scr0", "scr1"};

  CASE("a cross-warp reduction addresses the pool and declares nothing");
  {
    // The scratch is a pool region the caller resolved: a reduction can sit
    // inside a device function and Metal admits threadgroup declarations
    // only in a kernel.
    msl::Context c;
    msl::Block body;
    ReductionPlan p = onePlan(1, 0, /*warpBits=*/0b11, kNumWarps);
    CHECK(p.crossWarp());
    CHECK(p.scratch.slotsPerOperand > 0);
    emitReduce(c, body, p, kNumWarps, sources(1, 1), nm, adder(c));

    const std::string out = render(body);
    CHECK(out.find("threadgroup float") == std::string::npos);
    CHECK(out.find("scr0[") != std::string::npos);
  }

  CASE("each operand publishes through its own region");
  {
    // An argmax publishes a float and an int; one shared buffer would give
    // one of them the other's width.
    msl::Context c;
    msl::Block body;
    ReductionPlan p = onePlan(1, 0, /*warpBits=*/0b1, kNumWarps);
    p.elems = {f32(), i32()};
    emitReduce(c, body, p, kNumWarps, sources(2, 1), nm, adder(c));
    const std::string out = render(body);
    CHECK(out.find("scr0[") != std::string::npos);
    CHECK(out.find("scr1[") != std::string::npos);
  }

  CASE("a lane-local reduction declares no buffer");
  {
    msl::Context c;
    msl::Block body;
    ReductionPlan p = onePlan(1, 0b11111, /*warpBits=*/0, kNumWarps);
    CHECK(!p.crossWarp());
    emitReduce(c, body, p, kNumWarps, sources(1, 1), nm, adder(c));
    CHECK(render(body).find("threadgroup") == std::string::npos);
  }

  CASE("an integer reduction keeps an integer accumulator");
  {
    msl::Context c;
    msl::Block body;
    ReductionPlan p = onePlan(2, 0b11111, /*warpBits=*/0b1, kNumWarps);
    p.elems = {i32()};
    emitReduce(c, body, p, kNumWarps, sources(1, 2), nm, adder(c));
    const std::string out = render(body);
    CHECK(out.find("int acc") != std::string::npos);
    CHECK(out.find("int peer") != std::string::npos);
    CHECK(out.find("float acc") == std::string::npos);
    CHECK(out.find("float peer") == std::string::npos);
  }

  CASE("an unset element type still means f32, so existing callers are safe");
  {
    msl::Context c;
    msl::Block body;
    ReductionPlan p = onePlan(2, 0b11111, /*warpBits=*/0b1, kNumWarps);
    CHECK(p.elems.empty());
    emitReduce(c, body, p, kNumWarps, sources(1, 2), nm, adder(c));
    const std::string out = render(body);
    CHECK(out.find("float acc") != std::string::npos);
  }

  CASE("each operand declares its own type");
  {
    msl::Context c;
    msl::Block body;
    ReductionPlan p = onePlan(1, 0b11111, /*warpBits=*/0, kNumWarps);
    p.elems = {f32(), i32()};
    emitReduce(c, body, p, kNumWarps, sources(2, 1), nm, adder(c));
    const std::string out = render(body);
    CHECK(out.find("float acc0_0") != std::string::npos);
    CHECK(out.find("int acc0_1") != std::string::npos);
  }

  CASE("a reduce whose operands disagree on layout emits nothing");
  {
    msl::Context c;
    msl::Block body;
    ReductionPlan p = onePlan(2, 0b11111, /*warpBits=*/0, kNumWarps);
    p.regsPerOperand = {2, 1};
    CHECK(!p.operandsShareLayout());
    auto out = emitReduce(c, body, p, kNumWarps, sources(2, 2), nm, adder(c));
    CHECK(out.empty());
    CHECK(body.empty());
  }

  CASE("a reduce whose name arrays disagree emits nothing either");
  {
    msl::Context c;
    msl::Block body;
    ReductionPlan p = onePlan(2, 0b11111, /*warpBits=*/0, kNumWarps);
    msl::SmallVec<msl::SmallVec<msl::Str, 8>, 4> ragged;
    ragged.push_back({"a0", "a1"});
    ragged.push_back({"b0"});
    auto out = emitReduce(c, body, p, kNumWarps, ragged, nm, adder(c));
    CHECK(out.empty());
    CHECK(body.empty());
  }

  CASE("agreeing operands still emit, so the check is not a veto");
  {
    msl::Context c;
    msl::Block body;
    ReductionPlan p = onePlan(2, 0b11111, /*warpBits=*/0, kNumWarps);
    p.regsPerOperand = {2, 2};
    CHECK(p.operandsShareLayout());
    auto out = emitReduce(c, body, p, kNumWarps, sources(2, 2), nm, adder(c));
    CHECK(!out.empty());
    CHECK(!body.empty());
  }

  CASE("a single register needs no combine at all");
  {
    msl::Context c;
    msl::Block body;
    emitReduce(c, body, onePlan(1, 0), kNumWarps, sources(1, 1), nm, adder(c));
    CHECK_EQ(render(body), std::string("float acc0_0 = v0_0;\n"));
  }

  CASE("registers fold in order, one combine each");
  {
    msl::Context c;
    msl::Block body;
    emitReduce(c, body, onePlan(4, 0), kNumWarps, sources(1, 4), nm, adder(c));
    const std::string out = render(body);
    CHECK_EQ(countOf(out, "sum"), 3 * 2); // decl + use per combine
    CHECK(out.find("float acc0_0 = v0_0;") != std::string::npos);
    CHECK(out.find("v0_1") != std::string::npos);
    CHECK(out.find("v0_3") != std::string::npos);
  }

  CASE("one XOR shuffle per planned lane step, high bit first");
  {
    msl::Context c;
    msl::Block body;
    // laneMask 0b10101 -> offsets 16, 4, 1.
    emitReduce(c, body, onePlan(1, 0b10101), kNumWarps, sources(1, 1), nm,
               adder(c));
    const std::string out = render(body);
    CHECK_EQ(countOf(out, "simd_shuffle_xor"), 3);
    const std::size_t p16 = out.find("simd_shuffle_xor(acc0_0, 16u)");
    const std::size_t p4 = out.find("simd_shuffle_xor(acc0_0, 4u)");
    const std::size_t p1 = out.find("simd_shuffle_xor(acc0_0, 1u)");
    CHECK(p16 != std::string::npos);
    CHECK(p4 != std::string::npos);
    CHECK(p1 != std::string::npos);
    CHECK(p16 < p4);
    CHECK(p4 < p1);
  }

  CASE("a full 32-lane reduction is five shuffles");
  {
    msl::Context c;
    msl::Block body;
    emitReduce(c, body, onePlan(1, 0b11111), kNumWarps, sources(1, 1), nm,
               adder(c));
    CHECK_EQ(countOf(render(body), "simd_shuffle_xor"), 5);
  }

  CASE("a lane-local reduction emits no shuffle");
  {
    msl::Context c;
    msl::Block body;
    emitReduce(c, body, onePlan(2, 0), kNumWarps, sources(1, 2), nm, adder(c));
    CHECK_EQ(countOf(render(body), "simd_shuffle_xor"), 0);
  }

  CASE("a lane-only reduction touches no scratch");
  {
    msl::Context c;
    msl::Block body;
    ReductionPlan p = onePlan(1, 0b11111, /*warpBits=*/0);
    CHECK(!p.crossWarp());
    emitReduce(c, body, p, kNumWarps, sources(1, 1), nm, adder(c));
    const std::string out = render(body);
    CHECK_EQ(countOf(out, "scr"), 0);
    CHECK_EQ(countOf(out, "threadgroup_barrier"), 0);
  }

  CASE("a cross-warp reduction publishes, barriers, then combines");
  {
    msl::Context c;
    msl::Block body;
    // warpBits 0b011 over 8 warps -> the subset {0,1,2,3}.
    ReductionPlan p = onePlan(1, 0, /*warpBits=*/0b011);
    CHECK(p.crossWarp());
    CHECK_EQ(p.warpSubset.size(), 4u);
    emitReduce(c, body, p, kNumWarps, sources(1, 1), nm, adder(c));
    const std::string out = render(body);

    CHECK(out.find("scr0[warp * 32 + lane] = acc0_0;") != std::string::npos);
    CHECK_EQ(countOf(out, "threadgroup_barrier"), 3);
    CHECK_EQ(countOf(out, "sum"), 3 * 2); // three peers combined

    // The anchor clears the reduced bits: `warp & ~0b011` over 8 warps is
    // `warp & 4`. Every warp reads its own subset.
    CHECK(out.find("float accw0_0 = scr0[(warp & 4) * 32 + lane];") !=
          std::string::npos);
    CHECK(out.find("scr0[(warp & 4) * 32 + lane + 32]") != std::string::npos);
    CHECK(out.find("scr0[(warp & 4) * 32 + lane + 64]") != std::string::npos);
    CHECK(out.find("scr0[(warp & 4) * 32 + lane + 96]") != std::string::npos);
  }

  CASE("the barrier separates publish from read");
  {
    msl::Context c;
    msl::Block body;
    emitReduce(c, body, onePlan(1, 0, 0b001), kNumWarps, sources(1, 1), nm,
               adder(c));
    const std::string out = render(body);
    const std::size_t bar0 = out.find("threadgroup_barrier");
    const std::size_t pub = out.find("scr0[warp * 32 + lane] =");
    const std::size_t bar1 = out.find("threadgroup_barrier", pub);
    const std::size_t rd = out.find("scr0[(warp & 6) * 32 + lane + 32]");
    CHECK(bar0 < pub);
    CHECK(pub < bar1);
    CHECK(bar1 < rd);
  }

  CASE("a barrier closes the scratch reads before the pool can reuse them");
  {
    msl::Context c;
    msl::Block body;
    emitReduce(c, body, onePlan(1, 0, 0b001), kNumWarps, sources(1, 1), nm,
               adder(c));
    const std::string out = render(body);
    const std::size_t rd = out.rfind("scr0[");
    CHECK(rd != std::string::npos);
    CHECK(out.find("threadgroup_barrier", rd) != std::string::npos);
  }

  CASE("every warp combines its own subset");
  {
    // warpSubset holds XOR offsets: for each warp, the slots it reads are the
    // warps that differ from it only in the mask.
    const int64_t nw = 8;
    for (int warpBits : {0b001, 0b011, 0b100}) {
      ReductionPlan p = onePlan(1, 0, warpBits, nw);
      for (int64_t w = 0; w < nw; ++w) {
        const int64_t anchor = w & (int64_t)p.anchorMask(nw);
        std::set<int64_t> reads;
        for (int off : p.warpSubset)
          reads.insert(anchor + off);

        // Exactly the warps congruent to w outside the mask.
        std::set<int64_t> want;
        for (int64_t o = 0; o < nw; ++o)
          if ((o & ~(int64_t)warpBits & (nw - 1)) == anchor)
            want.insert(o);
        CHECK(reads == want);
        CHECK(reads.count(w) == 1u);
      }
    }
  }

  CASE("scratch is sized for the whole warp count");
  {
    // The publish writes at warp*32 from all warps, so a reservation sized
    // to the subset is written past by the warps outside it.
    ReductionPlan p = onePlan(1, 0, 0b001, 8);
    CHECK_EQ(p.scratch.slotsPerOperand, 8 * 32);
    CHECK_EQ(p.scratch.slotFor(7, 31), 8 * 32 - 1);
    CHECK(p.scratch.slotFor(7, 31) < p.scratch.slotsPerOperand);
  }

  CASE("every operand travels the same topology");
  {
    msl::Context c;
    msl::Block body;
    emitReduce(c, body, onePlan(2, 0b1), kNumWarps, sources(2, 2), nm,
               adder(c));
    const std::string out = render(body);
    CHECK(out.find("float acc0_0 = v0_0;") != std::string::npos);
    CHECK(out.find("float acc0_1 = v1_0;") != std::string::npos);
    CHECK_EQ(countOf(out, "simd_shuffle_xor"), 2);
  }

  CASE("each survivor group gets its own accumulator");
  {
    msl::Context c;
    ReductionPlan p;
    std::vector<CoordKey> coords = {CoordKey({0, 0}), CoordKey({0, 1}),
                                    CoordKey({1, 0}), CoordKey({1, 1})};
    p.groups = groupSurvivors(coords, /*axis=*/1);
    CHECK_EQ(p.groups.size(), 2u);

    msl::Block body;
    auto res = emitReduce(c, body, p, kNumWarps, sources(1, 4), nm, adder(c));
    CHECK_EQ(res.size(), 2u);
    const std::string out = render(body);
    CHECK(out.find("float acc0_0 = v0_0;") != std::string::npos);
    CHECK(out.find("float acc1_0 = v0_2;") != std::string::npos);
    CHECK_EQ(countOf(out, "sum"), 2 * 2);
  }

  CASE("results come back in plan order, keyed by the plan");
  {
    msl::Context c;
    ReductionPlan p;
    std::vector<CoordKey> coords = {CoordKey({0, 0}), CoordKey({0, 1}),
                                    CoordKey({1, 0}), CoordKey({1, 1})};
    p.groups = groupSurvivors(coords, 1);
    msl::Block body;
    auto res = emitReduce(c, body, p, kNumWarps, sources(1, 4), nm, adder(c));
    CHECK_EQ(p.groupFor(CoordKey({0})), 0);
    CHECK_EQ(p.groupFor(CoordKey({1})), 1);
    CHECK_EQ(res[0][0], std::string("acc0_0"));
    CHECK_EQ(res[1][0], std::string("acc1_0"));
  }

  CASE("a full reduction reads as a script");
  {
    msl::Context c;
    msl::Block body;
    emitReduce(c, body, onePlan(4, 0b11111, 0b001, 8), kNumWarps, sources(1, 4),
               nm, adder(c));
    const std::string out = render(body);
    // local fold: 3 combines, lane: 5 shuffles, warp: publish + 1 peer.
    CHECK_EQ(countOf(out, "simd_shuffle_xor"), 5);
    CHECK_EQ(countOf(out, "threadgroup_barrier"), 3);
    CHECK_EQ(countOf(out, "float sum"), 3 + 5 + 1);
  }

  return ::agpu_test::report("EmitReduce");
}
