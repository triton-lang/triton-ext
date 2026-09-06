// PanelSchedule tests.
#include "agpu/plan/PanelSchedule.h"
#include "harness.h"

using namespace agpu;

namespace {

DotFacts gemm(int64_t M, int64_t N, int64_t K, int64_t Bd = 1) {
  DotFacts f;
  f.M = M;
  f.N = N;
  f.K = K;
  f.Bd = Bd;
  f.rank = Bd > 1 ? 3 : 2;
  f.aElemBytes = 2;
  f.bElemBytes = 2;
  f.numWarps = 4;
  return f;
}

} // namespace

int main() {
  CASE("a tile that fits in one panel is a single entry");
  {
    DotFacts f = gemm(64, 64, 64);
    Panel p = panelCost(64, 64, 64, 2, kAccBytes);
    PanelSchedule s = planPanelSchedule(f, p);
    CHECK_EQ(s.size(), 1);
    CHECK(s.tiles[0].finalK);
    CHECK(s.tiles[0].m == (Range{0, 64}));
    CHECK(s.tiles[0].k == (Range{0, 64}));
  }

  CASE("panel counts multiply out");
  {
    DotFacts f = gemm(128, 128, 128);
    Panel p = panelCost(64, 64, 64, 2, kAccBytes);
    PanelSchedule s = planPanelSchedule(f, p);
    CHECK_EQ(s.size(), 2 * 2 * 2);
    CHECK_EQ(s.readbackCount(), 4); // one per (m, n) position
  }

  CASE("K is innermost, so C accumulates in place");
  {
    DotFacts f = gemm(64, 64, 128);
    Panel p = panelCost(64, 64, 64, 2, kAccBytes);
    PanelSchedule s = planPanelSchedule(f, p);
    CHECK_EQ(s.size(), 2);
    CHECK_EQ(s.tiles[0].k.lo, 0);
    CHECK_EQ(s.tiles[1].k.lo, 64);
    CHECK(!s.tiles[0].finalK);
    CHECK(s.tiles[1].finalK);
    CHECK(s.tiles[0].m == s.tiles[1].m);
    CHECK(s.tiles[0].n == s.tiles[1].n);
  }

  CASE("batch is outermost");
  {
    DotFacts f = gemm(64, 64, 64, /*Bd=*/3);
    Panel p = panelCost(64, 64, 64, 2, kAccBytes);
    PanelSchedule s = planPanelSchedule(f, p);
    CHECK_EQ(s.size(), 3);
    CHECK_EQ(s.tiles[0].batch, 0);
    CHECK_EQ(s.tiles[1].batch, 1);
    CHECK_EQ(s.tiles[2].batch, 2);
  }

  CASE("a batched tile's product frame carries the batch axis at stride 0");
  {
    // The pool holds one slice: the batch coordinate adds nothing to the
    // address.
    DotFacts f = gemm(64, 32, 32, /*Bd=*/3);
    Panel p = panelCost(64, 32, 32, 2, kAccBytes);
    PanelSchedule s = planPanelSchedule(f, p);
    const PanelTile &t = s.tiles[1];
    CHECK(t.batched);

    const TileView av = t.aStagedView();
    CHECK_EQ(av.rank(), 3);
    CHECK_EQ(av.extentAt(0), 1);
    CHECK_EQ(av.strideAt(0), 0);
    const TileView plain = t.aView().originAt({t.m.lo, t.k.lo});
    CHECK_EQ(av.strideAt(1), plain.strideAt(0));
    CHECK_EQ(av.origin(), plain.origin());
  }

  CASE("a batched tile's windows confine every operand to its slice");
  {
    DotFacts f = gemm(64, 32, 32, /*Bd=*/3);
    Panel p = panelCost(64, 32, 32, 2, kAccBytes);
    PanelSchedule s = planPanelSchedule(f, p);
    const PanelTile &t = s.tiles[2];

    const std::vector<CoordWindow> a = t.aWindows();
    CHECK_EQ((int)a.size(), 3);
    CHECK_EQ(a[0].dim, 0);
    CHECK_EQ(a[0].lo, 2);
    CHECK_EQ(a[0].hi, 3); // the degenerate batch window
    CHECK_EQ(a[1].dim, 1);
    CHECK_EQ(a[1].hi, 64); // m
    CHECK_EQ(a[2].dim, 2);
    CHECK_EQ(a[2].hi, 32); // k
    CHECK_EQ((int)t.bWindows().size(), 3);
    CHECK_EQ((int)t.readbackWindows().size(), 3);

    DotFacts f2 = gemm(64, 32, 32);
    PanelSchedule s2 = planPanelSchedule(f2, p);
    const std::vector<CoordWindow> a2 = s2.tiles[0].aWindows();
    CHECK_EQ((int)a2.size(), 2);
    CHECK_EQ(a2[0].dim, 0);
    CHECK_EQ(a2[0].hi, 64);
  }

  CASE("a ragged final panel is still emitted, just clipped");
  {
    DotFacts f = gemm(100, 64, 64);
    Panel p = panelCost(64, 64, 64, 2, kAccBytes);
    PanelSchedule s = planPanelSchedule(f, p);
    CHECK_EQ(s.size(), 2);
    CHECK(s.tiles[0].m == (Range{0, 64}));
    CHECK(s.tiles[1].m == (Range{64, 100})); // a short tail of 36 rows
    CHECK_EQ(s.tiles[1].m.size(), 36);
  }

  CASE("a ragged tile addresses whole fragments while knowing its real extent");
  {
    // The MMA reads and writes whole 8x8 fragments: the view rounds up to 40
    // rows while the tile's extent stays 36.
    DotFacts f = gemm(100, 64, 64);
    Panel p = panelCost(64, 64, 64, 2, kAccBytes);
    PanelSchedule s = planPanelSchedule(f, p);
    const PanelTile &tail = s.tiles[1];

    CHECK_EQ(tail.m.size(), 36);            // what is real
    CHECK_EQ(tail.aView().extentAt(0), 40); // what is addressed
    CHECK_EQ(tail.cView().extentAt(0), 40);
    CHECK_EQ(tail.cView().cosizeElems(), 39 * (64 + 4) + 64);
    CHECK(tail.ragged());
    CHECK(tail.raggedM());
    CHECK(!tail.raggedN());

    const std::vector<CoordWindow> w = tail.readbackWindows();
    CHECK_EQ((int)w.size(), 2);
    CHECK_EQ(w[0].hi, 36);
    CHECK_EQ(w[1].hi, 64);
  }

  CASE("an aligned tile's view is its extent, with nothing rounded");
  {
    DotFacts f = gemm(128, 64, 64);
    Panel p = panelCost(64, 64, 64, 2, kAccBytes);
    PanelSchedule s = planPanelSchedule(f, p);
    for (const PanelTile &t : s.tiles) {
      CHECK(!t.ragged());
      CHECK_EQ(t.cView().extentAt(0), t.m.size());
      CHECK_EQ(t.cView().extentAt(1), t.n.size());
    }
  }

  CASE("a ragged K panel still completes the contraction");
  {
    DotFacts f = gemm(64, 64, 96);
    Panel p = panelCost(64, 64, 64, 2, kAccBytes);
    PanelSchedule s = planPanelSchedule(f, p);
    CHECK_EQ(s.size(), 2);
    CHECK_EQ(s.tiles[1].k.size(), 32);
    CHECK(s.tiles[1].finalK);
    CHECK_EQ(s.tiles[0].k.size() + s.tiles[1].k.size(), 96);
  }

  CASE("every output element is covered exactly once");
  {
    for (auto [M, N, K] : {std::tuple<int64_t, int64_t, int64_t>{128, 128, 128},
                           {100, 64, 64},
                           {64, 100, 96},
                           {256, 128, 64}}) {
      DotFacts f = gemm(M, N, K);
      Panel p = panelCost(64, 64, 64, 2, kAccBytes);
      PanelSchedule s = planPanelSchedule(f, p);
      CHECK(tilesCoverOutput(f, s));
    }
  }

  CASE("C is read back exactly once per output position");
  {
    DotFacts f = gemm(128, 128, 192);
    Panel p = panelCost(64, 64, 64, 2, kAccBytes);
    PanelSchedule s = planPanelSchedule(f, p);
    CHECK(readbackIsExactlyOncePerPosition(f, s, p));
    CHECK_EQ(s.size(), 2 * 2 * 3);
    CHECK_EQ(s.readbackCount(), 4);
  }

  CASE("every K panel is visited once per position");
  {
    DotFacts f = gemm(128, 128, 192);
    Panel p = panelCost(64, 64, 64, 2, kAccBytes);
    PanelSchedule s = planPanelSchedule(f, p);
    CHECK(contractionIsComplete(f, s, p));
  }

  CASE("the invariants hold across a sweep of shapes");
  {
    int checked = 0;
    for (int64_t M = 64; M <= 192; M += 32)
      for (int64_t N = 64; N <= 192; N += 32)
        for (int64_t K = 64; K <= 128; K += 32) {
          DotFacts f = gemm(M, N, K);
          Panel p = panelCost(64, 64, 64, 2, kAccBytes);
          PanelSchedule s = planPanelSchedule(f, p);
          CHECK(tilesCoverOutput(f, s));
          CHECK(readbackIsExactlyOncePerPosition(f, s, p));
          CHECK(contractionIsComplete(f, s, p));
          ++checked;
        }
    CHECK(checked >= 45);
  }

  CASE("rank-3 covers every batch");
  {
    DotFacts f = gemm(64, 64, 64, /*Bd=*/4);
    Panel p = panelCost(64, 64, 64, 2, kAccBytes);
    PanelSchedule s = planPanelSchedule(f, p);
    CHECK(tilesCoverOutput(f, s));
    CHECK_EQ(s.readbackCount(), 4);
  }

  CASE("a non-final tile has three phases, a final one five");
  {
    DotFacts f = gemm(64, 64, 128);
    Panel p = panelCost(64, 64, 64, 2, kAccBytes);
    PanelSchedule s = planPanelSchedule(f, p);
    CHECK_EQ(phasesOf(s.tiles[0]).size(), 3u);
    CHECK_EQ(phasesOf(s.tiles[1]).size(), 5u);
    CHECK(phasesOf(s.tiles[1])[3] == PanelPhase::Drain);
    CHECK(phasesOf(s.tiles[1]).back() == PanelPhase::Readback);
  }

  CASE("phase order is fixed and visible");
  {
    DotFacts f = gemm(64, 64, 64);
    Panel p = panelCost(64, 64, 64, 2, kAccBytes);
    auto ph = phasesOf(planPanelSchedule(f, p).tiles[0]);
    CHECK(ph[0] == PanelPhase::StageA);
    CHECK(ph[1] == PanelPhase::StageB);
    CHECK(ph[2] == PanelPhase::Mma);
    CHECK(ph[3] == PanelPhase::Drain);
    CHECK(ph[4] == PanelPhase::Readback);
    CHECK_EQ(ph.size(), 5u);
  }

  CASE("a drain forfeits the resident B, so the next tile restages");
  {
    DotFacts f = gemm(128, 64, 64);
    Panel p = panelCost(64, 64, 64, 2, kAccBytes);
    PanelSchedule s = planPanelSchedule(f, p);
    CHECK_EQ(s.size(), 2);
    CHECK(s.tiles[0].finalK);
    CHECK(s.tiles[0].stageB);
    CHECK(s.tiles[1].stageB);
    CHECK(s.tiles[0].n == s.tiles[1].n);
    CHECK(s.tiles[0].k == s.tiles[1].k);
    CHECK(s.tiles[0].m.lo != s.tiles[1].m.lo);
  }

  CASE("a K panel walk restages B every tile");
  {
    DotFacts f = gemm(64, 64, 128);
    Panel p = panelCost(64, 64, 64, 2, kAccBytes);
    PanelSchedule s = planPanelSchedule(f, p);
    CHECK_EQ(s.size(), 2);
    CHECK(s.tiles[0].stageB);
    CHECK(s.tiles[1].stageB);
  }

  CASE("a batch boundary restages B even at the same n and k");
  {
    DotFacts f = gemm(64, 64, 64, /*Bd=*/2);
    Panel p = panelCost(64, 64, 64, 2, kAccBytes);
    PanelSchedule s = planPanelSchedule(f, p);
    CHECK_EQ(s.size(), 2);
    CHECK(s.tiles[0].stageB);
    CHECK(s.tiles[1].stageB);
  }

  CASE("fragment counts follow the tile, ragged included");
  {
    DotFacts f = gemm(100, 64, 64);
    Panel p = panelCost(64, 64, 64, 2, kAccBytes);
    PanelSchedule s = planPanelSchedule(f, p);
    CHECK_EQ(s.tiles[0].mFrags(), 8);
    CHECK_EQ(s.tiles[0].nFrags(), 8);
    CHECK_EQ(s.tiles[0].kSteps(), 8);
    // 36 rows is five fragments: four whole and one edge at full 8x8 width.
    CHECK_EQ(s.tiles[1].mFrags(), 5);
    CHECK_EQ(s.tiles[1].mFrags() * kSgFragDim, 40);
    CHECK(s.tiles[1].mFrags() * kSgFragDim >= s.tiles[1].m.size());
  }

  CASE("every real row of a ragged tile is inside some fragment");
  {
    for (int64_t rows = 1; rows <= 130; ++rows) {
      DotFacts f = gemm(rows, 64, 64);
      Panel p = panelCost(64, 64, 64, 2, kAccBytes);
      PanelSchedule s = planPanelSchedule(f, p);
      for (const PanelTile &t : s.tiles) {
        CHECK(t.mFrags() * kSgFragDim >= t.m.size());
        CHECK((t.mFrags() - 1) * kSgFragDim < t.m.size());
        CHECK_EQ(t.cView().extentAt(0), t.mFrags() * kSgFragDim);
      }
    }
  }

  CASE("a tile's views carry the pad in the stride and the cost matches");
  {
    // simdgroup ops address a base pointer and a leading dimension, so bank
    // spread is a longer stride.
    DotFacts f = gemm(64, 64, 64);
    Panel pan = panelCost(64, 64, 64, 2, kAccBytes);
    PanelSchedule s = planPanelSchedule(f, pan);
    const PanelTile &t = s.tiles[0];
    CHECK_EQ(t.aView().strideAt(0), 64 + padElemsFor(64, 2));
    CHECK_EQ(t.aView().offsetOf({1, 0}), 72); // row 1 starts one pitch on
    CHECK_EQ(t.cView().strideAt(0), 64 + padElemsFor(64, kAccBytes));
    CHECK_EQ(pan.aBytes.count(), t.aView().cosizeElems() * 2);
  }

  CASE("an empty panel yields an empty schedule");
  {
    DotFacts f = gemm(64, 64, 64);
    PanelSchedule s = planPanelSchedule(f, panelCost(0, 64, 64, 2, kAccBytes));
    CHECK(s.empty());
  }

  CASE("a plan that chose Panel produces a usable schedule");
  {
    DotFacts f = gemm(256, 256, 256);
    Plan plan = planDot(f, Bytes(kTGResidentBudgetBytes));
    CHECK(plan.kind == Plan::Kind::Panel);
    PanelSchedule s = planPanelSchedule(f, plan.panel().panel);
    CHECK(!s.empty());
    CHECK(tilesCoverOutput(f, s));
    CHECK(readbackIsExactlyOncePerPosition(f, s, plan.panel().panel));
    CHECK_EQ(s.size(), plan.panel().tiles());
  }

  CASE("a later tile's staged views land its corner at the buffer's start");
  {
    // Staged views carry the tile's position as an origin.
    DotFacts f = gemm(64, 64, 32);
    PanelSchedule s = planPanelSchedule(f, panelCost(32, 32, 32, 2, kAccBytes));
    int later = 0;
    for (const PanelTile &t : s.tiles) {
      CHECK_EQ(t.aStagedView().offsetOf({t.m.lo, t.k.lo}), 0);
      CHECK_EQ(t.bStagedView().offsetOf({t.k.lo, t.n.lo}), 0);
      CHECK_EQ(t.cStagedView().offsetOf({t.m.lo, t.n.lo}), 0);
      CHECK_EQ(t.cStagedView().offsetOf({t.m.lo, t.n.lo + 1}), 1);
      CHECK_EQ(t.cStagedView().offsetOf({t.m.lo + 1, t.n.lo}),
               t.cView().strideAt(0));
      if (t.m.lo || t.n.lo || t.k.lo)
        ++later;
    }
    CHECK(later > 0); // the property is vacuous on the (0,0,0) tile
  }

  CASE("panelWarpGrid is the tile's fragment counts, A always staged");
  {
    DotFacts f = gemm(64, 64, 32);
    PanelSchedule s = planPanelSchedule(f, panelCost(40, 48, 32, 4, kAccBytes));
    const PanelTile &t = s.tiles[0];
    const WarpGrid g = panelWarpGrid(t, 4);
    CHECK_EQ(g.mT, t.mFrags());
    CHECK_EQ(g.nT, t.nFrags());
    CHECK_EQ(g.numWarps, 4);
    CHECK(!g.aDirect);
    const PanelTile &tail = s.tiles.back();
    CHECK(tail.mFrags() < t.mFrags()); // 64 % 40 leaves a 24-row tail
    CHECK_EQ(panelWarpGrid(tail, 4).mT, tail.mFrags());
  }

  CASE("a device-A tile has no StageA phase and its panel costs no A bytes");
  {
    DotFacts f = gemm(64, 64, 64);
    f.aDirect = true;
    const Panel pan =
        planPanel(64, 64, 64, 2, kAccBytes, Bytes(12288), /*aStaged=*/false);
    CHECK_EQ(pan.aBytes.count(), 0);
    PanelSchedule s = planPanelSchedule(f, pan);
    for (const PanelTile &t : s.tiles) {
      CHECK(t.aDirect);
      const std::vector<PanelPhase> ph = phasesOf(t);
      for (PanelPhase p : ph)
        CHECK(p != PanelPhase::StageA);
      CHECK(ph.front() == (t.stageB ? PanelPhase::StageB : PanelPhase::Mma));
    }

    const Panel staged = planPanel(64, 64, 64, 2, kAccBytes, Bytes(12288));
    CHECK(pan.mp * pan.np >= staged.mp * staged.np);
    CHECK(pan.total() <= Bytes(12288));
  }

  return ::agpu_test::report("PanelSchedule");
}
