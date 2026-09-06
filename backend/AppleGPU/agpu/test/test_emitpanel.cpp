// Panel emission: schedule -> phases -> MSL.
#include "agpu/emit/EmitPanel.h"
#include "agpu/msl/Analysis.h"
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

DotFacts gemm(int64_t M, int64_t N, int64_t K) {
  DotFacts f;
  f.M = M;
  f.N = N;
  f.K = K;
  f.aElemBytes = 2;
  f.bElemBytes = 2;
  f.numWarps = 1;
  return f;
}

const std::string kZeroAcc = kSimdgroup8x8.zeroCtor("float");

WarpSlot slotAt(int64_t mi, int64_t ni, int acc) {
  return {SlotCoord::fixed(mi), SlotCoord::fixed(ni), acc};
}

OperandSource stagedA(const PanelTile &t) {
  return {PanelNames{}.poolA, Stride(t.aView().strideAt(0))};
}

} // namespace

int main() {
  PanelNames nm;

  CASE("the accumulator is declared zeroed, once, outside the warp blocks");
  {
    msl::Context c;
    DotFacts f = gemm(16, 16, 16);
    PanelSchedule s = planPanelSchedule(f, panelCost(16, 16, 16, 2, kAccBytes));
    const WarpGrid g = panelWarpGrid(s.tiles[0], f.numWarps);
    msl::Block body;
    emitPanelAccumDecls(c, body, s.tiles[0], nm, planWarpProgram(g), g);
    const std::string out = render(body);
    CHECK(out.find("simdgroup_float8x8(0.0f)") != std::string::npos);
    CHECK(out.find("make_filled") == std::string::npos);
    CHECK_EQ(countOf(out, "simdgroup_load"), 0);
    CHECK_EQ(countOf(out, "if ("), 0);
  }

  CASE("a later K panel accumulates into the same registers, no pool trip");
  {
    msl::Context c;
    DotFacts f = gemm(16, 16, 32);
    PanelSchedule s = planPanelSchedule(f, panelCost(16, 16, 16, 2, kAccBytes));
    CHECK_EQ(s.size(), 2);
    CHECK_EQ(panelAccName(s.tiles[0], nm, 0), panelAccName(s.tiles[1], nm, 0));
    msl::Block body;
    emitPanelMma(c, body, s.tiles[1], nm, stagedA(s.tiles[1]),
                 {slotAt(0, 0, 0)}, false);
    const std::string out = render(body);
    CHECK(out.find(kZeroAcc + "(0.0f)") == std::string::npos);
    CHECK(out.find("simdgroup_load(acc") == std::string::npos);
    CHECK(out.find(", pC") == std::string::npos);
    CHECK(out.find("simdgroup_multiply_accumulate(acc0") != std::string::npos);
  }

  CASE("only the drain stores C back");
  {
    msl::Context c;
    DotFacts f = gemm(16, 16, 32);
    PanelSchedule s = planPanelSchedule(f, panelCost(16, 16, 16, 2, kAccBytes));
    CHECK(!s.tiles.empty());
    for (const PanelTile &t : s.tiles) {
      msl::Block body;
      emitPanelMma(c, body, t, nm, stagedA(t), {slotAt(0, 0, 0)}, false);
      CHECK_EQ(countOf(render(body), "simdgroup_store"), 0);
    }
    msl::Block drain;
    emitAccumStore(c, drain, s.tiles[1], nm, slotAt(0, 0, 0),
                   panelAccName(s.tiles[1], nm, 0));
    const std::string out = render(drain);
    CHECK(out.find("simdgroup_store(acc") != std::string::npos);
    CHECK(out.find(", pC") != std::string::npos);
  }

  CASE("unrolled K emits one MMA per step");
  {
    msl::Context c;
    DotFacts f = gemm(16, 16, 32);
    PanelSchedule s = planPanelSchedule(f, panelCost(16, 16, 32, 2, kAccBytes));
    msl::Block body;
    emitPanelMma(c, body, s.tiles[0], nm, stagedA(s.tiles[0]),
                 {slotAt(0, 0, 0)}, /*rollK=*/false);
    const std::string out = render(body);
    CHECK_EQ(countOf(out, "simdgroup_multiply_accumulate"), 4); // 32/8
    CHECK_EQ(countOf(out, "for ("), 0);
  }

  CASE("rolled K emits one loop with one MMA");
  {
    msl::Context c;
    DotFacts f = gemm(16, 16, 32);
    PanelSchedule s = planPanelSchedule(f, panelCost(16, 16, 32, 2, kAccBytes));
    msl::Block body;
    emitPanelMma(c, body, s.tiles[0], nm, stagedA(s.tiles[0]),
                 {slotAt(0, 0, 0)}, /*rollK=*/true);
    const std::string out = render(body);
    CHECK_EQ(countOf(out, "simdgroup_multiply_accumulate"), 1);
    CHECK_EQ(countOf(out, "for ("), 1);
    CHECK(out.find("kv < 32") != std::string::npos);
    CHECK(out.find("kv += 8") != std::string::npos);
    CHECK_EQ(countOf(out, kZeroAcc + "(0.0f)"), 0);
    CHECK_EQ(countOf(out, "simdgroup_store"), 0);
  }

  CASE("the K term is the only difference between the forms");
  {
    msl::Context c;
    DotFacts f = gemm(16, 16, 16);
    PanelSchedule s = planPanelSchedule(f, panelCost(16, 16, 16, 2, kAccBytes));
    msl::Block un;
    emitPanelMma(c, un, s.tiles[0], nm, stagedA(s.tiles[0]), {slotAt(0, 0, 0)},
                 false);
    msl::Block ro;
    emitPanelMma(c, ro, s.tiles[0], nm, stagedA(s.tiles[0]), {slotAt(0, 0, 0)},
                 true);
    const std::string u = render(un), r = render(ro);
    CHECK_EQ(countOf(u, "simdgroup_multiply_accumulate"), 2);
    CHECK_EQ(countOf(r, "simdgroup_multiply_accumulate"), 1);
    CHECK_EQ(countOf(u, "for ("), 0);
    CHECK_EQ(countOf(r, "for ("), 1);
    for (const std::string &s2 : {u, r}) {
      CHECK_EQ(countOf(s2, kZeroAcc + "(0.0f)"), 0);
      CHECK_EQ(countOf(s2, "simdgroup_store"), 0);
    }
  }

  CASE("A steps by 8 along its innermost axis");
  {
    msl::Context c;
    DotFacts f = gemm(16, 16, 32);
    PanelSchedule s = planPanelSchedule(f, panelCost(16, 16, 32, 2, kAccBytes));
    msl::Block body;
    emitPanelMma(c, body, s.tiles[0], nm, stagedA(s.tiles[0]),
                 {slotAt(0, 0, 0)}, false);
    const std::string out = render(body);
    CHECK(out.find("simdgroup_load(fa0, pA, 40)") != std::string::npos);
    CHECK(out.find("simdgroup_load(fa2, pA + 8, 40)") != std::string::npos);
    CHECK(out.find("simdgroup_load(fa4, pA + 16, 40)") != std::string::npos);
  }

  CASE("B steps by the row stride, because K is its row axis");
  {
    msl::Context c;
    DotFacts f = gemm(16, 16, 32);
    PanelSchedule s = planPanelSchedule(f, panelCost(16, 16, 32, 2, kAccBytes));
    msl::Block body;
    emitPanelMma(c, body, s.tiles[0], nm, stagedA(s.tiles[0]),
                 {slotAt(0, 0, 0)}, false);
    const std::string out = render(body);
    CHECK(out.find("simdgroup_load(fb1, pB, 24)") != std::string::npos);
    CHECK(out.find("simdgroup_load(fb3, pB + 192, 24)") != std::string::npos);
  }

  CASE("a second slot addresses a different fragment");
  {
    msl::Context c;
    DotFacts f = gemm(16, 16, 16);
    PanelSchedule s = planPanelSchedule(f, panelCost(16, 16, 16, 2, kAccBytes));
    msl::Block body;
    emitPanelMma(c, body, s.tiles[0], nm, stagedA(s.tiles[0]),
                 {slotAt(1, 1, 1)}, false);
    emitAccumStore(c, body, s.tiles[0], nm, slotAt(1, 1, 1),
                   panelAccName(s.tiles[0], nm, 1));
    const std::string out = render(body);
    CHECK(out.find("pA + 192") != std::string::npos);
    CHECK(out.find("pB + 8") != std::string::npos);
    CHECK(out.find("simdgroup_store(acc1, pC + 168, 20)") != std::string::npos);
  }

  CASE("a ragged tile addresses its own smaller extent");
  {
    msl::Context c;
    DotFacts f = gemm(24, 16, 16);
    PanelSchedule s = planPanelSchedule(f, panelCost(16, 16, 16, 2, kAccBytes));
    CHECK_EQ(s.size(), 2);
    const PanelTile &tail = s.tiles[1];
    CHECK_EQ(tail.m.size(), 8);
    msl::Block body;
    emitPanelMma(c, body, tail, nm, stagedA(tail), {slotAt(0, 0, 0)}, false);
    emitAccumStore(c, body, tail, nm, slotAt(0, 0, 0),
                   panelAccName(tail, nm, 0));
    const std::string tail_out = render(body);
    CHECK(tail_out.find("simdgroup_store(acc") != std::string::npos);
    CHECK(tail_out.find(", pC, 20)") != std::string::npos);
  }

  CASE("each operand's guard reads its own layout");
  {
    msl::Context c;
    DotFacts f = gemm(16, 16, 16);
    PanelSchedule s = planPanelSchedule(f, panelCost(16, 16, 16, 2, kAccBytes));

    auto sourceOn = [](const msl::Str &id) {
      CoordSource cs;
      cs.laneId = id;
      LayoutBasis d0, d1;
      d0.lane = {1, 2, 0, 0, 0};
      d1.lane = {0, 0, 4, 8, 16};
      cs.dims = {d0, d1};
      return cs;
    };

    auto act = planStage(0, {{0, 0, 31}, {1, 0, 31}}, {{0, 0, 16}, {1, 0, 16}},
                         {0, 0});
    PanelInputs in;
    in.a = stagedA(s.tiles[0]);
    in.aActions = {*act};
    in.aNames = {"a0"};
    in.bActions = {*act};
    in.bNames = {"b0"};
    in.cActions = {*act};
    in.cNames = {"r0"};
    in.cBases = {""};
    PanelCoords pc{sourceOn("laneA"), sourceOn("laneB"), sourceOn("laneC")};
    msl::Block body;
    const WarpGrid g = panelWarpGrid(s.tiles[0], f.numWarps);
    emitPanelTile(c, body, s.tiles[0], nm, in, pc, g, planWarpProgram(g));
    const std::string out = render(body);

    CHECK(out.find("= a0") != std::string::npos);
    CHECK(out.find("laneA") != std::string::npos);
    CHECK(out.find("laneB") != std::string::npos);
    const std::size_t aAt = out.find("pA[");
    const std::size_t bAt = out.find("pB[");
    CHECK(aAt != std::string::npos && bAt != std::string::npos);
    CHECK(out.rfind("laneA", aAt) != std::string::npos);
    CHECK(out.rfind("laneB", bAt) != std::string::npos);
  }

  CASE("forAll is the one-liner for operands that share a layout");
  {
    CoordSource cs;
    LayoutBasis d;
    d.lane = {1, 2, 0, 0, 0};
    cs.dims = {d, d};
    const PanelCoords pc = PanelCoords::forAll(cs);
    CHECK_EQ(pc.a.laneId, pc.b.laneId);
    CHECK_EQ(pc.b.laneId, pc.c.laneId);
    CHECK_EQ(pc.a.dims.size(), (std::size_t)2);
  }

  CASE("a whole tile is barrier-separated phases");
  {
    msl::Context c;
    DotFacts f = gemm(16, 16, 16);
    PanelSchedule s = planPanelSchedule(f, panelCost(16, 16, 16, 2, kAccBytes));

    CoordSource cs;
    LayoutBasis row, col;
    row.lane = {1, 2, 0, 0, 0};
    col.lane = {0, 0, 4, 8, 16};
    cs.dims = {row, col};

    auto a0 =
        planStage(0, {{0, 0, 7}, {1, 0, 15}}, {{0, 0, 16}, {1, 0, 16}}, {0, 0});
    auto b0 = planStage(0, {{0, 0, 15}, {1, 0, 15}}, {{0, 0, 16}, {1, 0, 16}},
                        {0, 0});

    PanelInputs in;
    in.a = stagedA(s.tiles[0]);
    in.aActions = {*a0};
    in.aNames = {"a0"};
    in.bActions = {*b0};
    in.bNames = {"b0"};

    msl::Block body;
    const WarpGrid g = panelWarpGrid(s.tiles[0], f.numWarps);
    emitPanelTile(c, body, s.tiles[0], nm, in, PanelCoords::forAll(cs), g,
                  planWarpProgram(g));
    const std::string out = render(body);

    CHECK_EQ(countOf(out, "threadgroup_barrier"), 5);
    const std::size_t pa = out.find("= a0");
    const std::size_t pb = out.find("= b0");
    const std::size_t mma = out.find("simdgroup_multiply_accumulate");
    CHECK(pa < pb);
    CHECK(pb < mma);
  }

  CASE("the caller decides barrier order");
  {
    msl::Context c;
    DotFacts f = gemm(16, 16, 16);
    PanelSchedule s = planPanelSchedule(f, panelCost(16, 16, 16, 2, kAccBytes));
    CoordSource cs;
    LayoutBasis row;
    row.lane = {1, 2, 0, 0, 0};
    cs.dims = {row, row};
    auto a0 =
        planStage(0, {{0, 0, 7}, {1, 0, 15}}, {{0, 0, 16}, {1, 0, 16}}, {0, 0});
    msl::Block body;
    emitStage(c, body, s.tiles[0].aView(), nm.poolA, {*a0}, {"a0"}, cs, f16());
    CHECK_EQ(countOf(render(body), "threadgroup_barrier"), 0);
  }

  CASE("readback accumulates onto the incoming value");
  {
    msl::Context c;
    DotFacts f = gemm(16, 16, 16);
    PanelSchedule s = planPanelSchedule(f, panelCost(16, 16, 16, 2, kAccBytes));
    CoordSource cs;
    LayoutBasis row, col;
    row.lane = {1, 2, 0, 0, 0};
    col.lane = {0, 0, 4, 8, 16};
    cs.dims = {row, col};

    auto c0 =
        planStage(0, {{0, 0, 7}, {1, 0, 15}}, {{0, 0, 16}, {1, 0, 16}}, {0, 0});
    msl::Block body;
    emitReadback(c, body, s.tiles[0].cView(), nm.poolC, {*c0}, {"r0"},
                 {"base0"}, cs, f32(), f32());
    const std::string out = render(body);
    CHECK(out.find("r0 = pC[") != std::string::npos);
    CHECK(out.find("+ base0;") != std::string::npos);
  }

  CASE("readback without an incoming value is a plain load");
  {
    msl::Context c;
    DotFacts f = gemm(16, 16, 16);
    PanelSchedule s = planPanelSchedule(f, panelCost(16, 16, 16, 2, kAccBytes));
    CoordSource cs;
    LayoutBasis row;
    row.lane = {1, 2, 0, 0, 0};
    cs.dims = {row, row};
    auto c0 =
        planStage(0, {{0, 0, 7}, {1, 0, 15}}, {{0, 0, 16}, {1, 0, 16}}, {0, 0});
    msl::Block body;
    emitReadback(c, body, s.tiles[0].cView(), nm.poolC, {*c0}, {"r0"}, {""}, cs,
                 f32(), f32());
    const std::string out = render(body);
    CHECK(out.find("r0 = pC[") != std::string::npos);
    CHECK(out.find("] +") == std::string::npos);
  }

  CASE("readback elides the bounds it can prove");
  {
    msl::Context c;
    DotFacts f = gemm(16, 16, 16);
    PanelSchedule s = planPanelSchedule(f, panelCost(16, 16, 16, 2, kAccBytes));
    CoordSource cs;
    LayoutBasis row, col;
    row.lane = {1, 2, 0, 0, 0};
    col.lane = {0, 0, 4, 8, 16};
    cs.dims = {row, col};

    auto inside =
        planStage(0, {{0, 0, 7}, {1, 0, 15}}, {{0, 0, 16}, {1, 0, 16}}, {0, 0});
    msl::Block body;
    emitReadback(c, body, s.tiles[0].cView(), nm.poolC, {*inside}, {"r0"},
                 {"b0"}, cs, f32(), f32());
    CHECK_EQ(countOf(render(body), "if ("), 0);

    auto straddles = planStage(0, {{0, 0, 31}, {1, 0, 15}},
                               {{0, 0, 16}, {1, 0, 16}}, {0, 0});
    msl::Block b2;
    emitReadback(c, b2, s.tiles[0].cView(), nm.poolC, {*straddles}, {"r0"},
                 {"b0"}, cs, f32(), f32());
    const std::string out = render(b2);
    CHECK_EQ(countOf(out, "if ("), 1);
    CHECK_EQ(countOf(out, "&&"), 0); // one term
  }

  CASE("a ragged tile withholds the surplus its edge fragment computed");
  {
    msl::Context c;
    DotFacts f = gemm(60, 64, 64);
    PanelSchedule s = planPanelSchedule(f, panelCost(64, 64, 64, 2, kAccBytes));
    const PanelTile &t = s.tiles[0];
    CHECK(t.raggedM());
    CHECK(!t.raggedN());
    CHECK_EQ(t.cView().extentAt(0), 64); // addressed
    CHECK_EQ(t.m.size(), 60);            // real

    CoordSource cs;
    LayoutBasis row, col;
    row.lane = {1, 2, 4, 8, 16};
    col.lane = {0, 0, 0, 0, 0};
    cs.dims = {row, col};

    auto edge =
        planStage(0, {{0, 56, 63}, {1, 0, 63}}, t.readbackWindows(), {56, 0});
    CHECK(edge.has_value());
    CHECK(edge->guard.needsTest());

    msl::Block body;
    emitReadback(c, body, t.cView(), nm.poolC, {*edge}, {"r0"}, {""}, cs, f32(),
                 f32());
    const std::string out = render(body);
    CHECK_EQ(countOf(out, "if ("), 1);
    CHECK(out.find("60") != std::string::npos);
    CHECK(out.find("< 64") == std::string::npos);
  }

  CASE("an aligned tile pays nothing for the ragged path");
  {
    msl::Context c;
    DotFacts f = gemm(64, 64, 64);
    PanelSchedule s = planPanelSchedule(f, panelCost(64, 64, 64, 2, kAccBytes));
    const PanelTile &t = s.tiles[0];
    CHECK(!t.ragged());

    CoordSource cs;
    LayoutBasis row, col;
    row.lane = {1, 2, 4, 8, 16};
    col.lane = {0, 0, 0, 0, 0};
    cs.dims = {row, col};

    auto a =
        planStage(0, {{0, 56, 63}, {1, 0, 63}}, t.readbackWindows(), {56, 0});
    CHECK(a.has_value());
    CHECK(a->guard.isUnguarded());

    msl::Block body;
    emitReadback(c, body, t.cView(), nm.poolC, {*a}, {"r0"}, {""}, cs, f32(),
                 f32());
    CHECK_EQ(countOf(render(body), "if ("), 0);
  }

  CASE("a register wholly past the edge is dropped outright");
  {
    DotFacts f = gemm(60, 64, 64);
    PanelSchedule s = planPanelSchedule(f, panelCost(64, 64, 64, 2, kAccBytes));
    const PanelTile &t = s.tiles[0];

    auto dead =
        planStage(0, {{0, 60, 63}, {1, 0, 63}}, t.readbackWindows(), {60, 0});
    CHECK(!dead.has_value());
  }

  CASE("a non-final K panel emits no readback");
  {
    msl::Context c;
    DotFacts f = gemm(16, 16, 32);
    PanelSchedule s = planPanelSchedule(f, panelCost(16, 16, 16, 2, kAccBytes));
    CoordSource cs;
    LayoutBasis row;
    row.lane = {1, 2, 0, 0, 0};
    cs.dims = {row, row};

    PanelInputs in;
    in.a = stagedA(s.tiles[0]);
    auto c0 =
        planStage(0, {{0, 0, 7}, {1, 0, 15}}, {{0, 0, 16}, {1, 0, 16}}, {0, 0});
    in.cActions = {*c0};
    in.cNames = {"r0"};
    in.cBases = {"b0"};

    const WarpGrid g = panelWarpGrid(s.tiles[0], f.numWarps);
    msl::Block first;
    emitPanelTile(c, first, s.tiles[0], nm, in, PanelCoords::forAll(cs), g,
                  planWarpProgram(g));
    CHECK(!s.tiles[0].finalK);
    CHECK_EQ(countOf(render(first), "r0 ="), 0);

    msl::Block last;
    emitPanelTile(c, last, s.tiles[1], nm, in, PanelCoords::forAll(cs), g,
                  planWarpProgram(g));
    CHECK(s.tiles[1].finalK);
    CHECK_EQ(countOf(render(last), "+ b0"), 1);
  }

  CASE("a multi-panel walk emits every tile");
  {
    msl::Context c;
    DotFacts f = gemm(32, 16, 32);
    Panel p = panelCost(16, 16, 16, 2, kAccBytes);
    PanelSchedule s = planPanelSchedule(f, p);
    CHECK_EQ(s.size(), 4); // 2 M panels x 1 N x 2 K

    msl::Block body;
    for (const PanelTile &t : s.tiles) {
      const WarpGrid g = panelWarpGrid(t, f.numWarps);
      if (t.k.lo == 0)
        emitPanelAccumDecls(c, body, t, nm, planWarpProgram(g), g);
      emitPanelMma(c, body, t, nm, stagedA(t), {slotAt(0, 0, 0)}, false);
      if (t.finalK)
        emitAccumStore(c, body, t, nm, slotAt(0, 0, 0), panelAccName(t, nm, 0));
    }
    const std::string out = render(body);
    CHECK_EQ(countOf(out, kZeroAcc + "(0.0f)"), 8);   // 2 positions x 4 slots
    CHECK_EQ(countOf(out, "simdgroup_store(acc"), 2); // one drain per position

    std::set<std::string> declared;
    std::istringstream lines(out);
    for (std::string line; std::getline(lines, line);) {
      const std::size_t at = line.find("simdgroup_float8x8 acc");
      if (at == std::string::npos)
        continue;
      const std::size_t nameAt = line.find("acc", at);
      const std::size_t end = line.find_first_of(" =;", nameAt);
      CHECK(declared.insert(line.substr(nameAt, end - nameAt)).second);
    }
    CHECK_EQ(declared.size(), (std::size_t)8);
  }

  CASE("a batch slice is part of the tile's name");
  {
    DotFacts f = gemm(16, 16, 16);
    f.rank = 3;
    f.Bd = 2;
    PanelSchedule s = planPanelSchedule(f, panelCost(16, 16, 16, 2, kAccBytes));
    CHECK_EQ(s.size(), 2);
    CHECK_EQ(tileTag(s.tiles[0]), msl::Str{});
    CHECK_EQ(tileTag(s.tiles[1]), msl::Str{"_s1"});
  }

  CASE("the fragment supplies its own type names");
  {
    CHECK_EQ(kSimdgroup8x8.mslType("float"), std::string("simdgroup_float8x8"));
    CHECK_EQ(kSimdgroup8x8.mslType("bfloat"),
             std::string("simdgroup_bfloat8x8"));

    msl::Context c;
    PanelNames nm;
    nm.opElem = "bfloat";
    DotFacts f = gemm(16, 16, 16);
    PanelSchedule s = planPanelSchedule(f, panelCost(16, 16, 16, 2, kAccBytes));
    msl::Block body;
    const WarpGrid g = panelWarpGrid(s.tiles[0], f.numWarps);
    emitPanelAccumDecls(c, body, s.tiles[0], nm, planWarpProgram(g), g);
    emitPanelMma(c, body, s.tiles[0], nm, stagedA(s.tiles[0]),
                 {slotAt(0, 0, 0)}, false);
    const std::string out = render(body);
    CHECK(out.find("simdgroup_bfloat8x8") != std::string::npos);
    CHECK(out.find("simdgroup_half8x8") == std::string::npos);
    CHECK(out.find("simdgroup_float8x8") != std::string::npos); // the acc
  }

  CASE("an affine slot spells the warp term into every address");
  {
    msl::Context c;
    DotFacts f = gemm(16, 16, 16);
    PanelSchedule s = planPanelSchedule(f, panelCost(16, 16, 16, 2, kAccBytes));
    msl::Block body;
    const WarpSlot s0{SlotCoord::fixed(0), SlotCoord::affine(1, 0), 0};
    emitPanelMma(c, body, s.tiles[0], nm, stagedA(s.tiles[0]), {s0}, false);
    emitAccumStore(c, body, s.tiles[0], nm, s0,
                   panelAccName(s.tiles[0], nm, 0));
    const std::string out = render(body);
    CHECK(out.find("simdgroup_load(fb1, pB + warp * 8, 24)") !=
          std::string::npos);
    CHECK(out.find("simdgroup_store(acc0, pC + warp * 8, 20)") !=
          std::string::npos);
  }

  CASE("a two-axis cover: one block, warp row and column as div and mod");
  {
    msl::Context c;
    DotFacts f = gemm(16, 16, 16);
    PanelSchedule s = planPanelSchedule(f, panelCost(16, 16, 16, 2, kAccBytes));
    const WarpGrid g = panelWarpGrid(s.tiles[0], 4);
    const WarpProgram prog = planWarpProgram(g);
    CHECK(prog.form == WarpForm::Parameterised);
    msl::Block body;
    PanelInputs in;
    in.a = stagedA(s.tiles[0]);
    emitPanelTile(c, body, s.tiles[0], nm, in, PanelCoords::forAll({}), g,
                  prog);
    const std::string out = render(body);
    CHECK(out.find("if (warp ==") == std::string::npos);
    CHECK(out.find("simdgroup_load(fa0, pA + warp / 2 % 2 * 8 * 24, 24)") !=
          std::string::npos);
    CHECK(out.find("simdgroup_load(fb1, pB + warp % 2 * 8, 24)") !=
          std::string::npos);
    CHECK(out.find("simdgroup_store(acc0, pC + (warp / 2 % 2 * 8 * 20 + "
                   "warp % 2 * 8), 20)") != std::string::npos);
  }

  CASE("an affine cover: one unguarded block serves every warp");
  {
    msl::Context c;
    DotFacts f = gemm(16, 32, 16);
    PanelSchedule s = planPanelSchedule(f, panelCost(16, 32, 16, 2, kAccBytes));
    const WarpGrid g = panelWarpGrid(s.tiles[0], 4);
    const WarpProgram prog = planWarpProgram(g);
    CHECK(prog.form == WarpForm::Parameterised);
    msl::Block body;
    PanelInputs in;
    in.a = stagedA(s.tiles[0]);
    emitPanelTile(c, body, s.tiles[0], nm, in, PanelCoords::forAll({}), g,
                  prog);
    const std::string out = render(body);
    CHECK(out.find("if (warp ==") == std::string::npos);
    CHECK(out.find("simdgroup_store(acc0, pC + (warp / 2 % 2 * 8 * 36 + "
                   "warp % 2 * 8), 36)") != std::string::npos);
    CHECK(out.find("simdgroup_store(acc1, pC + (warp / 2 % 2 * 8 * 36 + "
                   "(warp % 2 + 2) * 8), 36)") != std::string::npos);
  }

  CASE("a device A tile reads fragments at its global corner");
  {
    msl::Context c;
    DotFacts f = gemm(32, 16, 16);
    PanelSchedule s = planPanelSchedule(f, panelCost(16, 16, 16, 2, kAccBytes));
    CHECK_EQ(s.tiles[1].m.lo, 16);

    OperandSource dev;
    dev.buffer = "dA";
    dev.leadingDim = Stride::runtime("ldA");
    dev.rowOrigin = s.tiles[1].m.lo;
    dev.colOrigin = s.tiles[1].k.lo;
    msl::Block body;
    emitPanelMma(c, body, s.tiles[1], nm, dev, {slotAt(0, 0, 0)}, false);
    const std::string out = render(body);
    CHECK(out.find("simdgroup_load(fa_16_0_0_0, dA + 16 * ldA, ldA)") !=
          std::string::npos);
    CHECK(out.find("dA + (16 * ldA + 8)") != std::string::npos);
    CHECK(out.find(", pB,") != std::string::npos);
  }

  CASE("a device-A tile has no StageA phase and loses its barrier");
  {
    msl::Context c;
    DotFacts f = gemm(16, 16, 16);
    f.aDirect = true;
    PanelSchedule s = planPanelSchedule(f, panelCost(16, 16, 16, 2, kAccBytes,
                                                     /*aStaged=*/false));
    const PanelTile &t = s.tiles[0];
    CHECK(t.aDirect);

    PanelInputs in;
    in.a.buffer = "dA";
    in.a.leadingDim = Stride::runtime("ldA");
    auto b0 = planStage(0, {{0, 0, 15}, {1, 0, 15}}, {{0, 0, 16}, {1, 0, 16}},
                        {0, 0});
    in.bActions = {*b0};
    in.bNames = {"b0"};
    CoordSource cs;
    LayoutBasis row;
    row.lane = {1, 2, 0, 0, 0};
    cs.dims = {row, row};
    const WarpGrid g = panelWarpGrid(t, f.numWarps);
    msl::Block body;
    emitPanelTile(c, body, t, nm, in, PanelCoords::forAll(cs), g,
                  planWarpProgram(g));
    const std::string out = render(body);
    CHECK_EQ(countOf(out, "threadgroup_barrier"), 4);
    CHECK_EQ(countOf(out, "pA["), 0);
    CHECK(out.find("simdgroup_load(fa0, dA, ldA)") != std::string::npos);
  }

  CASE("predictPanelDotSize matches a measured emission of the panel walk");
  {
    struct Shape {
      int64_t M, N, K, warps, Bd;
    };
    const Shape shapes[] = {{32, 48, 32, 4, 1},
                            {24, 40, 24, 2, 1},
                            {16, 16, 48, 3, 1},
                            {32, 32, 32, 4, 2}};
    for (const Shape &sh : shapes)
      for (const bool rollK : {false, true}) {
        msl::Context c;
        DotFacts f = gemm(sh.M, sh.N, sh.K);
        f.numWarps = sh.warps;
        if (sh.Bd > 1) {
          f.rank = 3;
          f.Bd = sh.Bd;
        }
        const Panel p = panelCost(16, 16, 16, 2, kAccBytes);
        const PanelSchedule s = planPanelSchedule(f, p);
        CHECK(!s.empty());
        msl::Block body;
        FragReuse reuse(body);
        for (const PanelTile &t : s.tiles) {
          const WarpGrid grid = panelWarpGrid(t, warpsFor(f), f.numWarps);
          const WarpProgram prog = planWarpProgram(grid);
          if (t.k.lo == 0)
            emitPanelAccumDecls(c, body, t, nm, prog, grid);
          emitWarpBlocks(
              c, body, prog, grid, nm.warpId,
              [&](msl::Block &inner, const std::vector<WarpSlot> &slots,
                  int64_t w) {
                emitPanelMma(c, inner, t, nm, stagedA(t), slots, rollK,
                             reuse.shareFor(t, prog.guardWarp(w).value_or(-1)));
              });
        }
        const msl::FuncSize m = msl::measure(body);
        const PanelMmaSize pr = predictPanelDotSize(f, p, rollK);
        CHECK_EQ(m.decls, pr.decls);
        CHECK_EQ(m.fragDecls, pr.fragDecls);
        CHECK_EQ(m.mma, pr.mma);
      }
  }

  CASE("a renaming tile names its fragments and drains nothing");
  {
    msl::Context c;
    DotFacts f = gemm(16, 32, 16);
    f.numWarps = 4;
    LayoutBasis row, col;
    row.lane = {0, 1, 2, 0, 4};
    col.lane = {2, 0, 0, 4, 0};
    row.reg = {0, 0};
    col.reg = {1, 16};
    row.warp = {0, 8};
    col.warp = {8, 0};
    f.cDims = {row, col};
    f.cRegs = 4;
    PanelSchedule s = planPanelSchedule(f, panelCost(16, 32, 16, 2, kAccBytes));
    PanelTile t = s.tiles[0];
    t.cover = {1, 2};
    t.renameC = true;
    const WarpGrid g = panelWarpGrid(t, 4);
    const WarpProgram prog = planWarpProgram(g);
    CHECK_EQ(prog.miCount, 1);
    CHECK_EQ(prog.niCount, 2);
    PanelInputs in;
    in.a = stagedA(t);
    in.cNames = {"c0", "c1", "c2", "c3"};
    in.cRename = panelTileReadback(f, t);
    CHECK(in.cRename.rename());
    msl::Block body;
    emitPanelTile(c, body, t, nm, in, PanelCoords::forAll({}), g, prog);
    const std::string out = render(body);
    CHECK(out.find("c0 = acc0.thread_elements()[0];") != std::string::npos);
    CHECK(out.find("c1 = acc0.thread_elements()[1];") != std::string::npos);
    CHECK(out.find("c2 = acc1.thread_elements()[0];") != std::string::npos);
    CHECK(out.find("simdgroup_store") == std::string::npos);
    CHECK(out.find(nm.poolC + "[") == std::string::npos);
    CHECK_EQ(countOf(out, "threadgroup_barrier"), 1);

    t.renameC = false;
    msl::Block pooled;
    emitPanelTile(c, pooled, t, nm, in, PanelCoords::forAll({}), g, prog);
    const std::string via = render(pooled);
    CHECK_EQ(countOf(via, "simdgroup_store"), 2);
    CHECK_EQ(countOf(via, "threadgroup_barrier"), 3);
  }

  return ::agpu_test::report("EmitPanel");
}
