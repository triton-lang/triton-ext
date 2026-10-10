// CoordGuard tests. Cases mirror the three existing implementations this type
// replaces: panelStageGuard (EmitMSLDot.cpp:2386), bandRoundTrip's `banded`
// (EmitMSLMemory.cpp:135) and `inBand` (EmitMSLMemory.cpp:694).
#include "agpu/core/CoordGuard.h"
#include "harness.h"

using namespace agpu;
using Op = GuardTerm::Op;

static int kindOf(const CoordGuard &g) { return static_cast<int>(g.kind()); }
static const int kDead = static_cast<int>(CoordGuard::Kind::Dead);
static const int kUnguarded = static_cast<int>(CoordGuard::Kind::Unguarded);
static const int kNeeded = static_cast<int>(CoordGuard::Kind::Needed);

int main() {
  const int ROW = 0, COL = 1;

  // ── the three outcomes, which panelStageGuard conflates into two ───────
  CASE("wholly inside: no test emitted");
  {
    CoordGuard g = planGuard({{ROW, 8, 15}}, {{ROW, 0, 32}});
    CHECK_EQ(kindOf(g), kUnguarded);
    CHECK(g.terms().empty());
  }

  CASE("wholly outside: register is skipped entirely");
  {
    CoordGuard g = planGuard({{ROW, 40, 47}}, {{ROW, 0, 32}});
    CHECK_EQ(kindOf(g), kDead);
    CHECK(g.isDead());
  }

  CASE("straddling the low bound: one Ge term");
  {
    CoordGuard g = planGuard({{ROW, 24, 39}}, {{ROW, 32, 64}});
    CHECK_EQ(kindOf(g), kNeeded);
    CHECK_EQ(g.terms().size(), 1u);
    CHECK(g.terms()[0] == (GuardTerm{ROW, Op::Ge, 32}));
  }

  CASE("straddling the high bound: one Lt term");
  {
    CoordGuard g = planGuard({{ROW, 24, 39}}, {{ROW, 0, 32}});
    CHECK_EQ(g.terms().size(), 1u);
    CHECK(g.terms()[0] == (GuardTerm{ROW, Op::Lt, 32}));
  }

  CASE("straddling both bounds: two terms");
  {
    CoordGuard g = planGuard({{ROW, 0, 63}}, {{ROW, 16, 32}});
    CHECK_EQ(g.terms().size(), 2u);
    CHECK(g.terms()[0] == (GuardTerm{ROW, Op::Ge, 16}));
    CHECK(g.terms()[1] == (GuardTerm{ROW, Op::Lt, 32}));
  }

  // ── two axes, as panelStageGuard's A/B staging uses ────────────────────
  CASE("two axes, both inside");
  {
    CoordGuard g =
        planGuard({{ROW, 8, 15}, {COL, 0, 7}}, {{ROW, 0, 32}, {COL, 0, 32}});
    CHECK(g.isUnguarded());
  }

  CASE("two axes, one straddles");
  {
    CoordGuard g =
        planGuard({{ROW, 8, 15}, {COL, 24, 39}}, {{ROW, 0, 32}, {COL, 0, 32}});
    CHECK_EQ(g.terms().size(), 1u);
    CHECK(g.terms()[0] == (GuardTerm{COL, Op::Lt, 32}));
  }

  CASE("two axes, second disjoint: dead regardless of the first");
  {
    CoordGuard g =
        planGuard({{ROW, 8, 15}, {COL, 64, 71}}, {{ROW, 0, 32}, {COL, 0, 32}});
    CHECK(g.isDead());
    CHECK(g.terms().empty());
  }

  CASE("first axis disjoint short-circuits");
  {
    CoordGuard g =
        planGuard({{ROW, 99, 99}, {COL, 0, 7}}, {{ROW, 0, 32}, {COL, 0, 32}});
    CHECK(g.isDead());
  }

  // ── the real panel staging pattern, EmitMSLDot.cpp:2452-2482 ───────────
  CASE("A-stage: 4 registers across a 2-panel K split");
  {
    int inCount = 0, deadCount = 0, guardedCount = 0;
    for (int r = 0; r < 8; ++r) {
      CoordGuard g = planGuard({{ROW, 8 * r, 8 * r + 7}, {COL, 0, 63}},
                               {{ROW, 0, 32}, {COL, 0, 32}});
      if (g.isDead())
        ++deadCount;
      else if (g.isUnguarded())
        ++inCount;
      else
        ++guardedCount;
    }
    CHECK_EQ(deadCount, 4);
    CHECK_EQ(guardedCount, 4);
    CHECK_EQ(inCount, 0);
  }

  // ── rank-3 batch as a degenerate window, no separate branch ────────────
  CASE("batch filter is just another window");
  {
    const int BATCH = 0;
    CoordGuard in = planGuard({{BATCH, 2, 2}}, {batchWindow(BATCH, 2)});
    CHECK(in.isUnguarded());
    CoordGuard out = planGuard({{BATCH, 2, 2}}, {batchWindow(BATCH, 1)});
    CHECK(out.isDead());
    CoordGuard span = planGuard({{BATCH, 0, 3}}, {batchWindow(BATCH, 2)});
    CHECK_EQ(span.terms().size(), 2u);
    CHECK(span.terms()[0] == (GuardTerm{BATCH, Op::Ge, 2}));
    CHECK(span.terms()[1] == (GuardTerm{BATCH, Op::Lt, 3}));
  }

  // ── the band case, EmitMSLMemory.cpp:694 `inBand` ──────────────────────
  CASE("row band: registers partition across bands");
  {
    const int64_t rowsTotal = 64, bandRows = 16;
    int participating = 0;
    for (int64_t r0 = 0; r0 < rowsTotal; r0 += bandRows) {
      CoordGuard g = planGuard({{ROW, 20, 27}}, {{ROW, r0, r0 + bandRows}});
      if (!g.isDead())
        ++participating;
    }
    CHECK_EQ(participating, 1);
  }

  CASE("row band: a register spanning two bands is guarded in both");
  {
    CoordGuard b0 = planGuard({{ROW, 12, 19}}, {{ROW, 0, 16}});
    CoordGuard b1 = planGuard({{ROW, 12, 19}}, {{ROW, 16, 32}});
    CHECK(b0.needsTest());
    CHECK(b1.needsTest());
    CHECK(b0.terms()[0] == (GuardTerm{ROW, Op::Lt, 16}));
    CHECK(b1.terms()[0] == (GuardTerm{ROW, Op::Ge, 16}));
  }

  // ── boundary exactness ─────────────────────────────────────────────────
  CASE("half-open windows: hi is exclusive, range hi is inclusive");
  {
    CHECK(planGuard({{ROW, 0, 31}}, {{ROW, 0, 32}}).isUnguarded());
    CHECK(planGuard({{ROW, 0, 32}}, {{ROW, 0, 32}}).needsTest());
    CHECK(planGuard({{ROW, 32, 32}}, {{ROW, 0, 32}}).isDead());
    CHECK(planGuard({{ROW, 31, 31}}, {{ROW, 0, 32}}).isUnguarded());
  }

  CASE("single-element window");
  {
    CHECK(planGuard({{ROW, 5, 5}}, {{ROW, 5, 6}}).isUnguarded());
    CHECK(planGuard({{ROW, 4, 6}}, {{ROW, 5, 6}}).needsTest());
    CHECK(planGuard({{ROW, 6, 6}}, {{ROW, 5, 6}}).isDead());
  }

  return ::agpu_test::report("CoordGuard");
}
