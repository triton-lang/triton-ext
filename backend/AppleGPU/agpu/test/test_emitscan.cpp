// Prefix scan emission.
#include "agpu/emit/EmitScan.h"
#include "agpu/msl/Printer.h"
#include "fixtures.h"
#include "harness.h"
#include "render.h"

#include <sstream>

using namespace agpu;
using agpu_test::countOf;
using agpu_test::render;

namespace {

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

ScanFacts facts(std::vector<AxisBit> lane, std::vector<AxisBit> warp = {},
                int64_t numWarps = 1, int64_t regs = 1) {
  ScanFacts f;
  f.laneBits = std::move(lane);
  f.warpBits = std::move(warp);
  f.numWarps = numWarps;
  f.regCount = regs;
  // Registers along the axis. See `offAxisRegs` for the other case.
  for (int64_t b = 0, s = 1; (1 << b) < regs; ++b, s *= 2)
    f.regBits.push_back({(int)b, (int32_t)s});
  return f;
}

// Same facts with registers across the axis: independent scans.
ScanFacts offAxisRegs(std::vector<AxisBit> lane, std::vector<AxisBit> warp,
                      int64_t numWarps, int64_t regs) {
  ScanFacts f = facts(std::move(lane), std::move(warp), numWarps, regs);
  f.regBits.clear();
  return f;
}

// Five lane bits: the axis covers the whole warp.
std::vector<AxisBit> wholeWarp() {
  return {{0, 1}, {1, 2}, {2, 4}, {3, 8}, {4, 16}};
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
  ScanNames nm;
  nm.scratch = {"scr0", "scr1"};

  // ── it shuffles up ─────────────────────────────────────────────────────

  CASE("the lane ladder shuffles up");
  {
    msl::Context c;
    msl::Block body;
    ScanPlan p = planScan(facts({{0, 1}, {1, 2}}));
    emitScan(c, body, p, 1, sources(1, 1), nm, adder(c));
    const std::string out = render(body);
    // Two ladder rungs plus the prefix fold.
    CHECK_EQ(countOf(out, "simd_shuffle_up"), 2 + 1);
    CHECK_EQ(countOf(out, "simd_shuffle_xor"), 0);
  }

  CASE("the deltas increase by powers of two");
  {
    msl::Context c;
    msl::Block body;
    ScanPlan p = planScan(facts({{0, 1}, {1, 2}, {2, 4}}));
    emitScan(c, body, p, 1, sources(1, 1), nm, adder(c));
    const std::string out = render(body);
    // The ladder runs on the cross-lane accumulator `sax`.
    const std::size_t d1 = out.find("simd_shuffle_up(sax0, 1u)");
    const std::size_t d2 = out.find("simd_shuffle_up(sax0, 2u)");
    const std::size_t d4 = out.find("simd_shuffle_up(sax0, 4u)");
    CHECK(d1 != std::string::npos);
    CHECK(d2 != std::string::npos);
    CHECK(d4 != std::string::npos);
    CHECK(d1 < d2);
    CHECK(d2 < d4);
  }

  CASE("every rung is guarded by the lane's position along the axis");
  {
    // Lanes below the delta have no source; shuffle_up leaves them undefined.
    // The guard masks to the axis bits: these two bits are a four-lane axis
    // replicated eight times across the warp.
    msl::Context c;
    msl::Block body;
    ScanPlan p = planScan(facts({{0, 1}, {1, 2}}));
    emitScan(c, body, p, 1, sources(1, 1), nm, adder(c));
    const std::string out = render(body);
    CHECK(out.find("if (lane & 3 >= 1)") != std::string::npos ||
          out.find("if ((lane & 3) >= 1)") != std::string::npos);
    CHECK(out.find("if (lane & 3 >= 2)") != std::string::npos ||
          out.find("if ((lane & 3) >= 2)") != std::string::npos);
    // Never the bare lane id.
    CHECK(out.find("if (lane >= 1)") == std::string::npos);
  }

  CASE("an axis filling the warp needs no mask");
  {
    // With five lane bits `lane & 31` is the lane id.
    msl::Context c;
    msl::Block body;
    ScanPlan p = planScan(facts({{0, 1}, {1, 2}, {2, 4}, {3, 8}, {4, 16}}));
    emitScan(c, body, p, 1, sources(1, 1), nm, adder(c));
    const std::string out = render(body);
    CHECK(out.find("if (lane >= 1)") != std::string::npos);
    CHECK(out.find("lane & 31") == std::string::npos);
  }

  // ── the local pass keeps every partial ─────────────────────────────────

  CASE("each register yields its own running total");
  {
    msl::Context c;
    msl::Block body;
    ScanPlan p = planScan(facts({}, {}, 1, 4));
    auto res = emitScan(c, body, p, 1, sources(1, 4), nm, adder(c))[0];
    CHECK_EQ(res.size(), 4u);
    // Three combines for four registers.
    CHECK_EQ(countOf(render(body), "float sum"), 3);
  }

  CASE("the first register's result is the value itself");
  {
    msl::Context c;
    msl::Block body;
    ScanPlan p = planScan(facts({}, {}, 1, 3));
    auto res = emitScan(c, body, p, 1, sources(1, 3), nm, adder(c))[0];
    // No cross-lane phase, so register 0's accumulator keeps its seed.
    CHECK(render(body).find("float sa0_0 = v0_0;") != std::string::npos);
    CHECK_EQ(res[0], std::string("sa0_0"));
  }

  CASE("more registers than one window still plans several usable scans");
  {
    // The layout puts the tail windows at different axis positions, so each
    // window is its own scan.
    ScanFacts f = facts({{0, 1}, {1, 2}}, {}, 1, /*regs=*/8);
    const ScanPlan p = planScan(f);
    CHECK(p.usable);
    CHECK(p.windowRegs < 8);       // more than one window
    CHECK_EQ(8 % p.windowRegs, 0); // and they divide the registers evenly

    // A window boundary is where a fresh scan seeds.
    CHECK(p.startsWindow(0));
    CHECK(p.startsWindow(p.windowRegs));
    for (int64_t r = 1; r < p.windowRegs; ++r)
      CHECK(!p.startsWindow(r));
  }

  CASE("a scan whose window covers every register folds straight through");
  {
    msl::Context c;
    msl::Block body;
    ScanPlan p = planScan(facts({}, {}, 1, /*regs=*/4));
    CHECK_EQ(p.windowRegs, 4);
    emitScan(c, body, p, 1, sources(1, 4), nm, adder(c));
    const std::string out = render(body);
    // Three combines and no register keeps its seed unfolded.
    CHECK_EQ(countOf(out, "float sum"), 3);
    CHECK(out.find("sa0_1 = sum") != std::string::npos);
    CHECK(out.find("sa0_3 = sum") != std::string::npos);
  }

  CASE("every register gets its own result");
  {
    // Contract, from EmitMSLScanReduce.cpp:355-495: `accs[k][r]` is
    // per-register, the local pass combines r-1 into r in place, the lane
    // ladder runs on a separate variable seeded from the last register and
    // the lane prefix is folded back into every register.
    msl::Context c;
    msl::Block body;
    ScanPlan p = planScan(facts({}, {}, 1, 4));
    auto res = emitScan(c, body, p, 1, sources(1, 4), nm, adder(c))[0];
    CHECK_EQ(res.size(), 4u);

    int distinct = 0;
    for (std::size_t i = 0; i < res.size(); ++i) {
      bool seen = false;
      for (std::size_t j = 0; j < i; ++j)
        seen = seen || res[j] == res[i];
      distinct += seen ? 0 : 1;
    }
    CHECK_EQ(distinct, 4);
  }

  CASE("the cross-lane result reaches every register");
  {
    msl::Context c;
    msl::Block body;
    ScanPlan p = planScan(facts({{0, 1}}, {}, 1, 2));
    CHECK(p.usable);
    auto res = emitScan(c, body, p, 1, sources(1, 2), nm, adder(c))[0];
    const std::string out = render(body);

    // A separate accumulator, seeded from the last register.
    CHECK(out.find("float sax0 = sa0_1;") != std::string::npos);
    CHECK_EQ(countOf(out, "if ("), 2); // the ladder rung and the fold
    for (const msl::Str &r : res)
      CHECK(out.find(r + " = ") != std::string::npos);
  }

  CASE("a single register needs no local combine");
  {
    msl::Context c;
    msl::Block body;
    emitScan(c, body, planScan(facts({}, {}, 1, 1)), 1, sources(1, 1), nm,
             adder(c));
    CHECK_EQ(countOf(render(body), "float sum"), 0);
  }

  // ── cross-warp is a prefix ─────────────────────────────────────────────

  CASE("a warp scan publishes from its last lane");
  {
    msl::Context c;
    msl::Block body;
    ScanPlan p = planScan(facts(wholeWarp(), {{0, 32}}, 4));
    emitScan(c, body, p, 4, sources(1, 1), nm, adder(c));
    const std::string out = render(body);
    CHECK(out.find("if (lane == 31)") != std::string::npos);
    CHECK(out.find("scr0[warp * 32]") != std::string::npos);
  }

  CASE("the combine takes the earlier element first");
  {
    // Argument group 1 is the earlier element, group 2 the later and
    // shuffle_up fetches from the lane before, so the peer leads. Uses
    // subtraction: a commutative combine cannot tell the orders apart.
    msl::Context c;
    msl::Block body;
    ScanPlan p = planScan(facts({{0, 1}, {1, 2}}));
    const CombineFn sub = [&c](msl::Block &into,
                               const msl::SmallVec<msl::Str, 4> &lhs,
                               const msl::SmallVec<msl::Str, 4> &rhs) {
      const msl::Str n = "d";
      into.push_back(
          c.declStmt(mslTypeOf(f32()), n,
                     c.binary(msl::BinOp::Sub, c.var(lhs[0]), c.var(rhs[0]))));
      return msl::SmallVec<msl::Str, 4>{n};
    };
    emitScan(c, body, p, 1, sources(1, 1), nm, sub);
    const std::string out = render(body);

    // Peer minus accumulator.
    CHECK(out.find("sp0_0 - sax0") != std::string::npos);
    CHECK(out.find("sax0 - sp0_0") == std::string::npos);
  }

  CASE("registers across the axis are independent scans");
  {
    // A 16x16 column scan gives a thread two registers holding two columns of
    // one row. Each is emitted as its own whole scan.
    msl::Context c;
    msl::Block body;
    ScanPlan p = planScan(offAxisRegs({{3, 1}, {4, 2}}, {{0, 4}}, 4, 2));
    auto res = emitScan(c, body, p, 4, sources(1, 2), nm, adder(c));
    const std::string out = render(body);

    CHECK_EQ(res[0].size(), 2u);
    // Two ladders, one per register.
    CHECK_EQ(countOf(out, "simd_shuffle_up"), 2 * 3);
    // Both registers take the cross-warp carry.
    CHECK(countOf(out, "scarryc0") > 0);
    CHECK_EQ(countOf(out, "scarryc1"), countOf(out, "scarryc0"));
    // One pool region between them, declared by the pool.
    CHECK(countOf(out, "scr0[") > 0);
    CHECK_EQ(countOf(out, "threadgroup float"), 0);
  }

  CASE("a partial-warp axis publishes per lane, with no elected publisher");
  {
    // One lane bit leaves lanes 2..31 holding other elements, so a single
    // published value would give them all lane 0's prefix.
    msl::Context c;
    msl::Block body;
    ScanPlan p = planScan(facts({{0, 1}}, {{0, 2}}, 4));
    emitScan(c, body, p, 4, sources(1, 1), nm, adder(c));
    const std::string out = render(body);
    CHECK(out.find("scr0[warp * 32 + lane]") != std::string::npos);
    CHECK(out.find("if (lane == ") == std::string::npos);
  }

  CASE("each warp takes the warps before it");
  {
    // Warp w reads [0, w), so the guard is `warp > n`. An axis spanning all
    // four warps needs two warp bits.
    msl::Context c;
    msl::Block body;
    ScanPlan p = planScan(facts({{0, 1}}, {{0, 2}, {1, 4}}, 4));
    CHECK_EQ((int)p.carryWarps(4).size(), 4);
    emitScan(c, body, p, 4, sources(1, 1), nm, adder(c));
    const std::string out = render(body);
    CHECK(countOf(out, "> 0)") >= 1);
    CHECK(countOf(out, "> 1)") >= 1);
    CHECK(countOf(out, "> 2)") >= 1);
    // Never the last warp: nothing follows it.
    CHECK(countOf(out, "> 3)") == 0);
  }

  CASE("a scan whose axis spans some warps carries only across those");
  {
    // An axis over one warp bit splits the threadgroup into independent
    // scans; a warp outside the mask holds a different output element.
    msl::Context c;
    msl::Block body;
    ScanPlan p = planScan(facts({{0, 1}}, {{0, 2}}, 4));
    CHECK_EQ(p.warpMask, 1u);
    CHECK_EQ((int)p.carryWarps(4).size(), 2); // not 4
    emitScan(c, body, p, 4, sources(1, 1), nm, adder(c));
    const std::string out = render(body);

    // One carry step.
    CHECK_EQ(countOf(out, "scarry"), 2); // one declaration, one read

    // The peer slot is anchored to the executing warp's own subset.
    CHECK(out.find("(warp & 2)") != std::string::npos);
    // The guard compares the position within that subset.
    CHECK(out.find("(warp & 1) > 0") != std::string::npos);
  }

  CASE("the cross-warp carry reaches the lane the prefix guard rejects");
  {
    // The lane prefix is guarded, the warp carry applies to every lane.
    // Carrying before the shuffle puts the warp's contribution inside the
    // guard, so lane 0 of every warp but the first loses it.
    msl::Context c;
    msl::Block body;
    ScanPlan p = planScan(facts({{0, 1}}, {{0, 2}}, 4, 2));
    CHECK(p.crossWarp);
    emitScan(c, body, p, 4, sources(1, 2), nm, adder(c));
    const std::string out = render(body);

    // The carry comes after the guarded lane-prefix fold.
    const std::size_t prefix = out.find("sap0");
    const std::size_t carry = out.find("scarry");
    CHECK(prefix != std::string::npos);
    CHECK(carry != std::string::npos);
    CHECK(prefix < carry);

    // One fold, reaching every register.
    CHECK_EQ(countOf(out, "float scarry0 ="), 1);
    const std::string afterCarry = out.substr(carry);
    CHECK(afterCarry.find("sa0_0 = ") != std::string::npos);
    CHECK(afterCarry.find("sa0_1 = ") != std::string::npos);
  }

  CASE("the warp total is published exactly once");
  {
    // Publishing per register would put a barrier inside the carry loop and
    // a barrier under divergent control flow is undefined in Metal.
    msl::Context c;
    msl::Block body;
    ScanPlan p = planScan(facts(wholeWarp(), {{0, 32}}, 4, 4));
    emitScan(c, body, p, 4, sources(1, 4), nm, adder(c));
    const std::string out = render(body);
    CHECK_EQ(countOf(out, "if (lane == 31)"), 1);
    CHECK_EQ(countOf(out, "threadgroup_barrier"), 3);
  }

  CASE("a scattered warp mask carries correctly too");
  {
    // carryWarps yields subset values; `anchor + w` reconstructs the
    // absolute id.
    msl::Context c;
    msl::Block body;
    // Bits 0 and 2, strides 2 and 4. Subset values are {0, 1, 4, 5}.
    ScanPlan p = planScan(facts({{0, 1}}, {{0, 2}, {2, 4}}, 8));
    CHECK(p.usable);
    CHECK_EQ(p.warpMask, 5u);
    CHECK_EQ(p.anchorMask(8), 2u);
    CHECK_EQ((int)p.carryWarps(8).size(), 4);
    emitScan(c, body, p, 8, sources(1, 1), nm, adder(c));
    const std::string out = render(body);

    // Four subset members: the earliest seeds the carry, two fold in under
    // their guards, the last is dropped.
    CHECK_EQ(countOf(out, "float scarry0 ="), 1);
    CHECK_EQ(countOf(out, "float scarry1_0 ="), 1);
    CHECK_EQ(countOf(out, "float scarry4_0 ="), 1);

    // The anchor keeps the bit the axis does not traverse.
    CHECK(out.find("(warp & 2)") != std::string::npos);
    // Guards on the subset values {0, 1, 4}.
    CHECK(out.find("(warp & 5) > 4") != std::string::npos);
    CHECK(out.find("(warp & 5) > 2") == std::string::npos);
  }

  // ── the scratch buffer ─────────────────────────────────────────────────

  CASE("a cross-warp scan addresses the pool and declares nothing");
  {
    // The carry publishes through a pool region the caller resolved: a scan
    // can sit inside a device function and Metal admits threadgroup
    // declarations only in a kernel.
    msl::Context c;
    msl::Block body;
    ScanPlan p = planScan(facts({{0, 1}}, {{0, 2}}, 4));
    CHECK(p.crossWarp);
    CHECK(p.scratch.slotsPerOperand > 0);
    emitScan(c, body, p, 4, sources(1, 1), nm, adder(c));
    const std::string out = render(body);
    CHECK(out.find("threadgroup float") == std::string::npos);
    CHECK(out.find("scr0[") != std::string::npos);
  }

  CASE("a lane-local scan declares no buffer");
  {
    msl::Context c;
    msl::Block body;
    ScanPlan p = planScan(facts({{0, 1}}));
    CHECK(!p.crossWarp);
    emitScan(c, body, p, 1, sources(1, 1), nm, adder(c));
    CHECK(render(body).find("threadgroup") == std::string::npos);
  }

  // ── multi-operand agreement ────────────────────────────────────────────

  CASE("a scan whose name arrays disagree emits nothing");
  {
    // `regs` comes from operand 0 and indexes every operand, so a shorter
    // array would read past its end.
    msl::Context c;
    msl::Block body;
    ScanPlan p = planScan(facts({{0, 1}}, {}, 1, 2));
    msl::SmallVec<msl::SmallVec<msl::Str, 8>, 4> ragged;
    ragged.push_back({"a0", "a1"});
    ragged.push_back({"b0"});
    auto out = emitScan(c, body, p, 1, ragged, nm, adder(c));
    CHECK(out.empty());
    CHECK(body.empty());
  }

  CASE("every operand's registers come back");
  {
    // A scan carrying a value and an index computes both and hands back both.
    msl::Context c;
    msl::Block body;
    ScanFacts f = facts({}, {}, 1, 3);
    f.elems = {f32(), f32()};
    f.regsPerOperand = {3, 3};
    ScanPlan p = planScan(f);
    CHECK(p.usable);

    auto res = emitScan(c, body, p, 1, sources(2, 3), nm, adder(c));
    CHECK_EQ(res.size(), 2u);
    CHECK_EQ(res[0].size(), 3u);
    CHECK_EQ(res[1].size(), 3u);

    // One variable per operand per register: `acc[k][r]`.
    for (const msl::Str &a : res[0])
      for (const msl::Str &b : res[1])
        CHECK(a != b);

    const std::string out = render(body);
    for (const msl::Str &n : res[0])
      CHECK(out.find(n) != std::string::npos);
    for (const msl::Str &n : res[1])
      CHECK(out.find(n) != std::string::npos);
  }

  // ── the element type ───────────────────────────────────────────────────

  CASE("an integer scan keeps an integer accumulator");
  {
    msl::Context c;
    msl::Block body;
    ScanFacts f = facts({{0, 1}}, {{0, 2}}, 4);
    f.elems = {i32()};
    ScanPlan p = planScan(f);
    emitScan(c, body, p, 4, sources(1, 1), nm, adder(c));
    const std::string out = render(body);
    // Only the emitter's own declarations: accumulator, peer, carry.
    CHECK(out.find("int sa") != std::string::npos);
    CHECK(out.find("int sp") != std::string::npos);
    CHECK(out.find("float sa") == std::string::npos);
    CHECK(out.find("float sp") == std::string::npos);
  }

  CASE("an unset element type still means f32");
  {
    msl::Context c;
    msl::Block body;
    ScanPlan p = planScan(facts({{0, 1}}));
    CHECK(p.elems.empty());
    emitScan(c, body, p, 1, sources(1, 1), nm, adder(c));
    CHECK(render(body).find("float sa") != std::string::npos);
  }

  // ── reverse ────────────────────────────────────────────────────────────

  CASE("a reverse scan shuffles down and inverts its guard");
  {
    // Reversing the register walk order does not change which lane
    // simd_shuffle_up reads from; the lane ladder needs the other builtin.
    msl::Context c;
    msl::Block body;
    ScanFacts f = facts({{0, 1}, {1, 2}});
    f.reverse = true;
    ScanPlan p = planScan(f);
    emitScan(c, body, p, 1, sources(1, 1), nm, adder(c));
    const std::string out = render(body);

    // Two rungs plus the prefix fold, no upward shuffle.
    CHECK_EQ(countOf(out, "simd_shuffle_down"), 2 + 1);
    CHECK_EQ(countOf(out, "simd_shuffle_up"), 0);

    // Forward guards `>= delta`, reverse guards `<= top - delta`. The top is
    // the axis's, which these two bits put at 3.
    CHECK(out.find("<= 2") != std::string::npos); // delta 1
    CHECK(out.find("<= 1") != std::string::npos); // delta 2
    CHECK(out.find(">= ") == std::string::npos);
  }

  CASE("a reverse warp scan publishes its total from lane 0");
  {
    // The ladder finishes at the opposite end, so the warp's total is there.
    msl::Context c;
    msl::Block body;
    ScanFacts f = facts(wholeWarp(), {{0, 32}}, 4);
    f.reverse = true;
    ScanPlan p = planScan(f);
    emitScan(c, body, p, 4, sources(1, 1), nm, adder(c));
    const std::string out = render(body);
    CHECK(out.find("if (lane == 0)") != std::string::npos);
    CHECK(out.find("if (lane == 31)") == std::string::npos);
  }

  CASE("a reverse scan carries from the warps after it");
  {
    // The prefix runs the other way, so warp w takes (w, numWarps).
    msl::Context c;
    msl::Block body;
    ScanFacts f = facts({{0, 1}}, {{0, 2}, {1, 4}}, 4);
    f.reverse = true;
    ScanPlan p = planScan(f);
    emitScan(c, body, p, 4, sources(1, 1), nm, adder(c));
    const std::string out = render(body);
    CHECK(countOf(out, "< 2)") >= 1);
    CHECK(countOf(out, "> 0)") == 0);
    CHECK(countOf(out, "> 2)") == 0);
    // Not `< 0)`: reversed, position 0 is the far end, so that block is dead.
    CHECK(countOf(out, "< 0)") == 0);
  }

  CASE("forward and reverse emit different text, so neither is a no-op");
  {
    msl::Context c;
    ScanFacts ff = facts({{0, 1}, {1, 2}}, {{0, 4}}, 4);
    ScanFacts rf = ff;
    rf.reverse = true;

    msl::Block a, b;
    emitScan(c, a, planScan(ff), 4, sources(1, 1), nm, adder(c));
    emitScan(c, b, planScan(rf), 4, sources(1, 1), nm, adder(c));
    CHECK(render(a) != render(b));
  }

  CASE("a lane-local scan touches no scratch and no barrier");
  {
    msl::Context c;
    msl::Block body;
    ScanPlan p = planScan(facts({{0, 1}, {1, 2}}, {}, 8));
    CHECK(!p.crossWarp);
    emitScan(c, body, p, 8, sources(1, 1), nm, adder(c));
    const std::string out = render(body);
    CHECK_EQ(countOf(out, "scr"), 0);
    CHECK_EQ(countOf(out, "barrier"), 0);
  }

  CASE("the publish is barriered on both sides");
  {
    msl::Context c;
    msl::Block body;
    ScanPlan p = planScan(facts({{0, 1}}, {{0, 2}}, 2));
    emitScan(c, body, p, 2, sources(1, 1), nm, adder(c));
    const std::string out = render(body);
    // Two around the publish, and a third closing the reads so the pool cannot
    // reuse the scratch under them.
    CHECK_EQ(countOf(out, "threadgroup_barrier"), 3);
    // Either slot form: `scr0[warp * 32]` or `scr0[warp * 32 + lane]`.
    const std::size_t pub = out.find("scr0[warp * 32");
    CHECK(pub != std::string::npos);
    CHECK(out.find("threadgroup_barrier") < pub);
    CHECK(out.find("threadgroup_barrier", pub) != std::string::npos);
  }

  // ── multi-operand ──────────────────────────────────────────────────────

  CASE("every operand travels the same ladder");
  {
    msl::Context c;
    msl::Block body;
    ScanPlan p = planScan(facts({{0, 1}}));
    emitScan(c, body, p, 1, sources(2, 1), nm, adder(c));
    const std::string out = render(body);
    // One accumulator per (operand, register).
    CHECK(out.find("float sa0_0 = v0_0;") != std::string::npos);
    CHECK(out.find("float sa1_0 = v1_0;") != std::string::npos);
    // One ladder rung and one prefix fold per operand.
    CHECK_EQ(countOf(out, "simd_shuffle_up"), (1 + 1) * 2);
  }

  // ── declining ──────────────────────────────────────────────────────────

  CASE("a declined plan emits nothing");
  {
    msl::Context c;
    msl::Block body;
    ScanPlan p = planScan(facts({{0, 1}, {2, 4}}));
    CHECK(!p.usable);
    auto res = emitScan(c, body, p, 1, sources(1, 1), nm, adder(c));
    CHECK_EQ(body.size(), 0u);
    CHECK_EQ(res.size(), 0u);
  }

  return ::agpu_test::report("EmitScan");
}
