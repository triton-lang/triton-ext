// Loads and stores.
#include "agpu/emit/EmitMove.h"
#include "agpu/msl/Printer.h"
#include "fixtures.h"
#include "harness.h"
#include "render.h"

#include <sstream>

using namespace agpu;
using agpu_test::countOf;
using agpu_test::render;

namespace {

RegBases runBases(int n) {
  RegBases b;
  for (int i = 0; i < n; ++i)
    b.push_back({0, 1 << i});
  return b;
}

MoveFacts loadOf(int64_t regs, int bits, int64_t contig, int64_t align) {
  MoveFacts f;
  f.regCount = regs;
  f.elemBits = bits;
  f.ptr = PtrDims(2, PtrInfo{contig, align});
  int n = 0;
  while ((1 << (n + 1)) <= regs)
    ++n;
  f.bases = runBases(n);
  return f;
}

struct TestSite {
  msl::Context &c;
  MoveSite site;

  TestSite(msl::Context &c, int64_t regs, bool mask = false, bool other = false,
           bool broadcastMask = false)
      : c(c) {
    site.elem = [&c](int64_t r) {
      return c.deref(c.var("p" + std::to_string(r)));
    };
    if (mask)
      site.guard = [&c, broadcastMask](int64_t r) -> msl::Expr * {
        return c.var(broadcastMask ? msl::Str("m") : "m" + std::to_string(r));
      };
    if (other)
      site.other = [&c](int64_t r) { return c.var("o" + std::to_string(r)); };
    for (int64_t r = 0; r < regs; ++r)
      site.values.push_back("v" + std::to_string(r));
  }
};

} // namespace

int main() {
  const ElemType elem = f32();

  // ── spelling ───────────────────────────────────────────────────────────

  CASE("a vector constructor is spelled according to its type");
  {
    CHECK_EQ(vecCtorName(f32(), 4), std::string("float4"));
    CHECK_EQ(vecCtorName(f16(), 2), std::string("half2"));
    CHECK_EQ(vecCtorName(i32(), 4), std::string("int4"));

    const ElemType i64{ElemType::Kind::Int, 64, false};
    const ElemType u8{ElemType::Kind::Int, 8, true};
    CHECK_EQ(vecCtorName(i64, 2), std::string("long2"));
    CHECK_EQ(vecCtorName(u8, 4), std::string("uchar4"));

    for (ElemType e : {f32(), f16(), i32(), i64, u8})
      CHECK_EQ(vecCtorName(e, 2),
               std::string(msl::spell(mslTypeOf(e).scalarKind())) + "2");
  }

  // ── coherence ──────────────────────────────────────────────────────────

  CASE("a coherent access says so, on every path it takes");
  {
    msl::Context c;

    MoveFacts f = loadOf(4, 32, 4, 16);
    f.coherent = true;
    msl::Block wide;
    emitMove(c, wide, f, planMove(f), TestSite(c, 4).site, elem);
    CHECK(render(wide).find("device coherent(device) float4 *") !=
          std::string::npos);

    MoveFacts s = loadOf(3, 32, 1, 4);
    s.coherent = true;
    msl::Block scalar;
    emitMove(c, scalar, s, planMove(s), TestSite(c, 3).site, elem);
    CHECK(render(scalar).find("device coherent(device) float *") !=
          std::string::npos);

    MoveFacts m = loadOf(2, 32, 1, 4);
    m.coherent = true;
    m.hasMask = true;
    msl::Block masked;
    emitMove(c, masked, m, planMove(m), TestSite(c, 2, /*mask=*/true).site,
             elem);
    CHECK(render(masked).find("device coherent(device) float *") !=
          std::string::npos);
  }

  CASE("an ordinary access pays nothing for the coherent path");
  {
    msl::Context c;
    MoveFacts f = loadOf(3, 32, 1, 4);
    CHECK(!f.coherent);
    msl::Block body;
    emitMove(c, body, f, planMove(f), TestSite(c, 3).site, elem);
    const std::string out = render(body);
    CHECK(out.find("coherent") == std::string::npos);
    CHECK(out.find("float v0 = *p0;") != std::string::npos);
  }

  // ── vectorisation ──────────────────────────────────────────────────────

  CASE("a contiguous aligned load issues one wide access per run");
  {
    msl::Context c;
    msl::Block body;
    MoveFacts f = loadOf(4, 32, 4, 4);
    MovePlan p = planMove(f);
    CHECK_EQ(p.width(), 4);
    emitMove(c, body, f, p, TestSite(c, 4).site, elem);
    const std::string out = render(body);
    CHECK_EQ(countOf(out, "float4"), 2);
    CHECK_EQ(countOf(out, "v0_w["), 4);
    CHECK(out.find("float v1 = v0_w[1];") != std::string::npos);
  }

  CASE("a register count that does not divide the width stays scalar");
  {
    MoveFacts f = loadOf(4, 32, 4, 4);
    f.regCount = 6;
    MovePlan p = planMove(f);
    CHECK_EQ(p.width(), 1);
    CHECK_EQ(p.runs.runs, 6);
  }

  CASE("an underaligned float run goes wide through a packed type");
  {
    msl::Context c;
    msl::Block body;
    MoveFacts f = loadOf(4, 32, 4, 2);
    MovePlan p = planMove(f);
    CHECK_EQ(p.width(), 4);
    CHECK(p.runs.packed);
    emitMove(c, body, f, p, TestSite(c, 4).site, elem);
    CHECK(render(body).find("packed_float4") != std::string::npos);
  }

  CASE("a scalar load reads each element where it lies");
  {
    msl::Context c;
    msl::Block body;
    MoveFacts f = loadOf(2, 32, 1, 1);
    MovePlan p = planMove(f);
    CHECK(!p.vectorised());
    emitMove(c, body, f, p, TestSite(c, 2).site, elem);
    const std::string out = render(body);
    CHECK(out.find("float v0 = *p0;") != std::string::npos);
    CHECK(out.find("float v1 = *p1;") != std::string::npos);
  }

  // ── stores ─────────────────────────────────────────────────────────────

  CASE("a store writes through the pointer");
  {
    msl::Context c;
    msl::Block body;
    MoveFacts f = loadOf(2, 32, 1, 1);
    f.isStore = true;
    emitMove(c, body, f, planMove(f), TestSite(c, 2).site, elem);
    const std::string out = render(body);
    CHECK(out.find("*p0 = v0;") != std::string::npos);
    CHECK_EQ(countOf(out, "float v"), 0);
  }

  CASE("a wide store builds the vector in place and writes it once");
  {
    msl::Context c;
    msl::Block body;
    MoveFacts f = loadOf(4, 32, 4, 4);
    f.isStore = true;
    MovePlan p = planMove(f);
    emitMove(c, body, f, p, TestSite(c, 4).site, elem);
    const std::string out = render(body);
    CHECK(out.find("float4(v0, v1, v2, v3)") != std::string::npos);
    CHECK_EQ(countOf(out, "float4(v0"), 1);
  }

  // ── masks ──────────────────────────────────────────────────────────────

  CASE("a masked load initialises every register before the mask");
  {
    // The mask is a runtime value, so a lane it excludes still reads the
    // register's name.
    msl::Context c;
    msl::Block body;
    MoveFacts f = loadOf(2, 32, 1, 1);
    f.hasMask = true;
    emitMove(c, body, f, planMove(f), TestSite(c, 2, /*mask=*/true).site, elem);
    const std::string out = render(body);
    CHECK(out.find("float v0 = 0") != std::string::npos);
    CHECK(out.find("if (m0)") != std::string::npos);
  }

  CASE("an unmasked load initialises nothing");
  {
    MoveFacts f = loadOf(2, 32, 1, 1);
    CHECK(planMove(f).init == MaskedInit::None);
  }

  CASE("an `other` value seeds the registers");
  {
    msl::Context c;
    msl::Block body;
    MoveFacts f = loadOf(2, 32, 1, 1);
    f.hasMask = true;
    f.hasOther = true;
    emitMove(c, body, f, planMove(f),
             TestSite(c, 2, /*mask=*/true, /*other=*/true).site, elem);
    CHECK(render(body).find("float v0 = o0;") != std::string::npos);
  }

  CASE("a store initialises nothing");
  {
    MoveFacts f = loadOf(4, 32, 4, 4);
    f.isStore = true;
    f.hasMask = true;
    CHECK(planMove(f).init == MaskedInit::None);
  }

  // ── the mask fast path ─────────────────────────────────────────────────

  CASE("too few registers to pay for a peel stay individually guarded");
  {
    MoveFacts f = loadOf(2, 32, 2, 2);
    f.hasMask = true;
    CHECK(!planMove(f).peel);
  }

  CASE("enough registers peel an all-true fast path");
  {
    // A predicated device load stalls the memory pipe on its predicate. The
    // peel lets the interior-tile case issue as one unconditional batch.
    msl::Context c;
    msl::Block body;
    MoveFacts f = loadOf(4, 32, 4, 4);
    f.hasMask = true;
    MovePlan p = planMove(f);
    CHECK(p.peel);
    emitMove(c, body, f, p, TestSite(c, 4, /*mask=*/true).site, elem);
    const std::string out = render(body);
    CHECK(out.find("else") != std::string::npos);
    CHECK(out.find("packed_float4") != std::string::npos ||
          out.find("float4") != std::string::npos);
    CHECK_EQ(countOf(out, "if (m"), 4 + 1);
    for (int r = 0; r < 4; ++r)
      CHECK(out.find("if (m" + std::to_string(r) + ")") != std::string::npos);
  }

  CASE("the peel guard spans every register in one shot");
  {
    // Splitting it per run would re-serialise the runs against each other.
    msl::Context c;
    msl::Block body;
    MoveFacts f = loadOf(8, 32, 4, 4);
    f.hasMask = true;
    MovePlan p = planMove(f);
    CHECK_EQ(p.width(), 4);
    CHECK_EQ(p.runs.runs, 2);
    emitMove(c, body, f, p, TestSite(c, 8, /*mask=*/true).site, elem);
    const std::string out = render(body);
    CHECK_EQ(countOf(out, "} else {"), 1);
    CHECK(out.find("m0 && m1") != std::string::npos);
  }

  CASE("a broadcast mask contributes a single shared term");
  {
    msl::Context c;
    msl::Block body;
    MoveFacts f = loadOf(4, 32, 4, 4);
    f.hasMask = true;
    emitMove(c, body, f, planMove(f),
             TestSite(c, 4, /*mask=*/true, /*other=*/false,
                      /*broadcastMask=*/true)
                 .site,
             elem);
    const std::string out = render(body);
    CHECK(out.find("if (m)") != std::string::npos);
    CHECK(out.find("m && m") == std::string::npos);
  }

  CASE("stores that alias one address keep every guard");
  {
    // A single-element output folds every offset to the same address; the
    // masked element must win, so aliasing can never relax the guards.
    msl::Context c;
    msl::Block body;
    MoveFacts f = loadOf(8, 32, 1, 1);
    f.isStore = true;
    f.hasMask = true;
    MovePlan p = planMove(f);
    CHECK(p.peel);
    CHECK(!p.vectorised());
    TestSite t(c, 8, /*mask=*/true);
    t.site.elem = [&c](int64_t) { return c.subscript(c.var("out"), c.lit(0)); };
    emitMove(c, body, f, p, t.site, elem);
    const std::string out = render(body);
    CHECK_EQ(countOf(out, "if (m"), 8 + 1);
    CHECK(out.find("m6 && m7") != std::string::npos);
    CHECK_EQ(countOf(out, "out[0] = v"), 16);
    for (int r = 0; r < 8; ++r)
      CHECK(out.find("if (m" + std::to_string(r) + ") out[0] = v" +
                     std::to_string(r)) != std::string::npos);
  }

  // ── a mask the layout decides ──────────────────────────────────────────

  CASE("a bound the layout decides classifies every register");
  {
    // An `R0_BLOCK` of 1024 over a 768-wide row: the last eight registers sit
    // past the bound for every lane, the rest never reach it.
    MoveFacts f = loadOf(32, 32, 4, 16);
    f.hasMask = true;
    f.bound.known = true;
    f.bound.dim = 0;
    f.bound.limit = 768;
    f.bound.dimSize = 1024;
    f.bound.basis.reg = {1, 2, 128, 256, 512};
    f.bound.basis.lane = {4, 8, 16, 32, 64};

    MovePlan p = planMove(f);
    CHECK(!p.guards.empty());
    CHECK(!p.peel);
    for (int64_t r = 0; r < 24; ++r) {
      CHECK(!p.guards.testedAt(r));
      CHECK(!p.guards.deadAt(r));
    }
    for (int64_t r = 24; r < 32; ++r)
      CHECK(p.guards.deadAt(r));
  }

  CASE("registers the bound excludes emit no access at all");
  {
    msl::Context c;
    msl::Block body;
    MoveFacts f = loadOf(32, 32, 4, 16);
    f.hasMask = true;
    f.bound.known = true;
    f.bound.dim = 0;
    f.bound.limit = 768;
    f.bound.dimSize = 1024;
    f.bound.basis.reg = {1, 2, 128, 256, 512};
    f.bound.basis.lane = {4, 8, 16, 32, 64};

    MovePlan p = planMove(f);
    CHECK_EQ(p.width(), 4);
    emitMove(c, body, f, p, TestSite(c, 32, /*mask=*/true).site, elem);
    const std::string out = render(body);
    // Six whole runs survive, none of them guarded, and the dead eight leave
    // only the initialiser that defines their names.
    CHECK_EQ(countOf(out, "_w = *"), 6);
    CHECK_EQ(countOf(out, "if (m"), 0);
    CHECK(out.find("float v24 = 0") != std::string::npos);
    CHECK(out.find("*p24") == std::string::npos);
  }

  CASE("a register the lanes carry across the bound keeps its guard");
  {
    // Every register spans the whole lane range here, so no register is
    // decided either way and the access keeps the mask it had.
    msl::Context c;
    msl::Block body;
    MoveFacts f = loadOf(4, 32, 4, 16);
    f.hasMask = true;
    f.bound.known = true;
    f.bound.dim = 0;
    f.bound.limit = 10;
    f.bound.dimSize = 16;
    f.bound.basis.reg = {1, 2};
    f.bound.basis.lane = {4, 8};

    MovePlan p = planMove(f);
    for (int64_t r = 0; r < 4; ++r)
      CHECK(p.guards.testedAt(r));
    CHECK(!runIsUnguarded(p.guards, 0, 4));
    CHECK(!runIsDead(p.guards, 0, 4));
    emitMove(c, body, f, p, TestSite(c, 4, /*mask=*/true).site, elem);
    const std::string out = render(body);
    CHECK_EQ(countOf(out, "if (m"), 4);
    CHECK(out.find("_w = *") == std::string::npos);
  }

  CASE("a second runtime term leaves the mask alone");
  {
    // A store that elects one writer ANDs the election into the guard, so the
    // layout no longer decides the whole condition.
    MoveFacts f = loadOf(32, 32, 4, 16);
    f.isStore = true;
    f.hasMask = true;
    f.guardHasRuntimeTerm = true;
    f.bound.known = true;
    f.bound.dim = 0;
    f.bound.limit = 768;
    f.bound.dimSize = 1024;
    f.bound.basis.reg = {1, 2, 128, 256, 512};
    f.bound.basis.lane = {4, 8, 16, 32, 64};

    MovePlan p = planMove(f);
    CHECK(p.guards.empty());
    CHECK(p.peel);
  }

  CASE("an unrecognised mask keeps the peel it had");
  {
    MoveFacts f = loadOf(8, 32, 4, 16);
    f.hasMask = true;
    CHECK(!f.bound.known);
    MovePlan p = planMove(f);
    CHECK(p.guards.empty());
    CHECK(p.peel);
  }

  // ── declining ──────────────────────────────────────────────────────────

  CASE("a scalar access declines and is no bug");
  {
    MoveFacts f = loadOf(4, 32, 1, 1);
    MovePlan p = planMove(f);
    Decision d = moveDecision(p, f);
    CHECK(d.isDecline());
    CHECK(!d.isBug());
    CHECK_EQ(d.where(), std::string("emitLoad"));
  }

  CASE("a store names itself when it declines");
  {
    // 64-bit has no vector form.
    MoveFacts f = loadOf(4, 64, 1, 1);
    f.isStore = true;
    Decision d = moveDecision(planMove(f), f);
    CHECK_EQ(d.where(), std::string("emitStore"));
    CHECK_EQ(d.why(), std::string("element width has no vector type"));
  }

  return ::agpu_test::report("EmitMove");
}
