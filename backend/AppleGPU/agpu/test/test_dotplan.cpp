// DotPlan tests: strategy selection, panel shrinking, pool reservation.
#include "agpu/plan/DotPassSchedule.h"
#include "agpu/plan/DotPlan.h"
#include "agpu/plan/PanelSchedule.h"
#include "harness.h"

using namespace agpu;

namespace {

DotFacts gemm(int64_t M, int64_t N, int64_t K, int64_t warps = 4) {
  DotFacts f;
  f.M = M;
  f.N = N;
  f.K = K;
  f.aElemBytes = 2;
  f.bElemBytes = 2;
  f.numWarps = warps;
  return f;
}

const Bytes kBudget{kTGResidentBudgetBytes};

// C under `apple_mma` with warpsPerCTA [gM, gN]: warps interleave at
// fragment granularity, low warp bits along columns, repetitions above.
void landIn(DotFacts &f, int64_t gM, int64_t gN) {
  LayoutBasis row, col;
  row.lane = {0, 1, 2, 0, 4};
  col.lane = {2, 0, 0, 4, 0};
  row.reg.push_back(0);
  col.reg.push_back(1);
  for (int64_t s = kSgFragDim; s < gN * kSgFragDim; s <<= 1) {
    row.warp.push_back(0);
    col.warp.push_back((int32_t)s);
  }
  for (int64_t s = kSgFragDim; s < gM * kSgFragDim; s <<= 1) {
    row.warp.push_back((int32_t)s);
    col.warp.push_back(0);
  }
  for (int64_t s = gN * kSgFragDim; s < f.nT() * kSgFragDim; s <<= 1) {
    row.reg.push_back(0);
    col.reg.push_back((int32_t)s);
  }
  for (int64_t s = gM * kSgFragDim; s < f.mT() * kSgFragDim; s <<= 1) {
    row.reg.push_back((int32_t)s);
    col.reg.push_back(0);
  }
  f.cDims = {row, col};
  f.cRegs = 2 * (f.mT() / gM) * (f.nT() / gN);
}

int kindOf(const Plan &p) { return static_cast<int>(p.kind); }
const int kScalar = static_cast<int>(Plan::Kind::Scalar);
const int kPanel = static_cast<int>(Plan::Kind::Panel);
const int kDirect = static_cast<int>(Plan::Kind::Direct);
const int kFused = static_cast<int>(Plan::Kind::Fused);
const int kUnsupported = static_cast<int>(Plan::Kind::Unsupported);

} // namespace

int main() {
  // ── units ──────────────────────────────────────────────────────────────

  CASE("bytes and elements do not mix silently");
  {
    Bytes b(1024);
    CHECK_EQ(b.inElems(4).count(), 256);
    CHECK_EQ(b.inElems(2).count(), 512);
    CHECK_EQ((b + Bytes(1024)).count(), 2048);
    CHECK(minBytes(Bytes(10), Bytes(20)) == Bytes(10));
    CHECK(maxBytes(Bytes(10), Bytes(20)) == Bytes(20));
  }

  CASE("residency is a step function of the declared pool");
  {
    CHECK_EQ(tgResidency(16384), 4);
    CHECK_EQ(tgResidency(22000), 2);
    CHECK_EQ(tgResidency(32768), 2);
    CHECK_EQ(tgResidency(8192), 8);
    CHECK_EQ(tgPoolForResidency(4), 16384);
  }

  // ── strategy selection ─────────────────────────────────────────────────

  CASE("an integer accumulator takes the scalar path");
  {
    DotFacts f = gemm(64, 64, 64);
    f.intAcc = true;
    Plan p = planDot(f, kBudget);
    CHECK_EQ(kindOf(p), kScalar);
    CHECK(p.pool.cNeed == p.stage.ab());
    CHECK(p.pool.cReserve() == Bytes(0));
    CHECK(!p.cThroughPool());
    CHECK_EQ(p.cPoolRegion().bytes, 0);
    CHECK(p.scalar().acc == i32());
  }

  CASE("the scalar reservation reads straight off its payload");
  {
    DotFacts f = gemm(64, 64, 64);
    f.intAcc = true;
    Plan p = planDot(f, kBudget);
    CHECK_EQ(kindOf(p), kScalar);
    CHECK(std::holds_alternative<ScalarParams>(p.params));
    CHECK(std::get<ScalarParams>(p.params).intAcc);
    CHECK(p.facts.intAcc);
  }

  CASE("a shape ragged in M or N still gets emitted");
  {
    CHECK_EQ(kindOf(planDot(gemm(60, 64, 64), kBudget)), kDirect);
    CHECK_EQ(kindOf(planDot(gemm(64, 63, 64), kBudget)), kDirect);
    CHECK_EQ(kindOf(planDot(gemm(60, 63, 64), kBudget)), kDirect);
  }

  CASE("the fragment grid rounds up, so the edge is computed");
  {
    DotFacts f = gemm(60, 63, 64);
    CHECK_EQ(f.mT(), 8);
    CHECK_EQ(f.nT(), 8);
    CHECK(f.raggedM());
    CHECK(f.raggedN());
    CHECK(f.ragged());

    DotFacts aligned = gemm(64, 64, 64);
    CHECK_EQ(aligned.mT(), 8);
    CHECK(!aligned.ragged());
  }

  CASE("C is reserved in whole fragments, since that is what the MMA writes");
  {
    Plan ragged = planDot(gemm(28, 32, 32), kBudget);
    Plan whole = planDot(gemm(32, 32, 32), kBudget);
    CHECK_EQ(kindOf(ragged), kDirect);
    CHECK(ragged.stage.ab() == whole.stage.ab());
    CHECK(ragged.pool.cReserve() == whole.pool.cReserve());

    CHECK(ragged.pool.cReserve() >=
          stagedTileBytes(28, fragAlignedExtent(32), kAccBytes));
  }

  CASE("a ragged K is declined");
  {
    CHECK_EQ(kindOf(planDot(gemm(64, 64, 12), kBudget)), kUnsupported);
    CHECK_EQ(kindOf(planDot(gemm(0, 64, 64), kBudget)), kUnsupported);
  }

  CASE("a tile that fits takes the direct path");
  {
    Plan p = planDot(gemm(64, 64, 64), kBudget);
    CHECK_EQ(kindOf(p), kDirect);
    CHECK(p.stage.ab() <= kBudget);
  }

  CASE("a register-resident accumulator takes the fused path");
  {
    DotFacts f = gemm(64, 64, 64);
    f.fusedAcc = true;
    Plan p = planDot(f, kBudget);
    CHECK_EQ(kindOf(p), kFused);
    CHECK_EQ(p.fused().fragsPerWarp, 16);
  }

  CASE("C's staging region holds accumulators, whatever the output type is");
  {
    for (const int64_t outBytes : {int64_t(2), int64_t(4)}) {
      DotFacts f = gemm(64, 64, 64);
      f.cElemBytes = outBytes;
      const Plan p = planDot(f, kBudget);
      CHECK_EQ(p.cPoolElem().bits, 32u);
      CHECK(p.cPoolElem().kind == ElemType::Kind::Float);
      CHECK_EQ((int64_t)(p.cPoolElem().bits / 8u), kAccBytes);
    }
  }

  CASE("a fused C overlays the staged operands at the pool's base");
  {
    DotFacts f = gemm(64, 64, 64);
    f.fusedAcc = true;
    Plan p = planDot(f, kBudget);
    CHECK_EQ(kindOf(p), kFused);
    const Plan::CPoolRegion c = p.cPoolRegion();
    CHECK(c.overlaysOperands);
    CHECK_EQ(c.bytes, stagedTileBytes(64, fragAlignedExtent(64), kAccBytes,
                                      p.padStagedC())
                          .count());
    CHECK_EQ(p.pool.reserved().count(),
             std::max(p.stage.ab().count(), c.bytes));
    CHECK(p.pool.reserved() <= kBudget);
  }

  CASE("a fused pad that costs a residency class is dropped");
  {
    DotFacts f = gemm(64, 64, 64);
    f.fusedAcc = true;
    const Bytes padded(stagedTileBytes(64, fragAlignedExtent(64), kAccBytes));
    const Bytes plain(
        stagedTileBytes(64, fragAlignedExtent(64), kAccBytes, false));
    CHECK(padded <= kBudget);
    CHECK_EQ(plain.count(), tgPoolForResidency(4));
    CHECK(tgResidency(padded.count()) < tgResidency(plain.count()));
    CHECK(!planDot(f, kBudget).padStagedC());

    CHECK(fusedPadWorthCarrying(Bytes(4336), Bytes(4000), kBudget));
  }

  CASE("a fused C that fits only unpadded drops the pad but keeps the fusion");
  {
    DotFacts f = gemm(64, 128, 64);
    f.fusedAcc = true;
    CHECK(stagedTileBytes(64, fragAlignedExtent(128), kAccBytes) > kBudget);
    CHECK(stagedTileBytes(64, fragAlignedExtent(128), kAccBytes, false) <=
          kBudget);
    Plan p = planDot(f, kBudget);
    CHECK_EQ(kindOf(p), kFused);
    CHECK(!p.fused().stagePad);
    CHECK(!p.padStagedC());
    CHECK_EQ(
        p.cPoolRegion().bytes,
        stagedTileBytes(64, fragAlignedExtent(128), kAccBytes, false).count());
    CHECK_EQ(p.stage.ab().count(), planStageBytes(f, false).ab().count());
    CHECK(p.pool.reserved() <= kBudget);
  }

  CASE("a fused C too large to cross the pool at either pitch demotes");
  {
    DotFacts f = gemm(64, 136, 64);
    f.fusedAcc = true;
    CHECK(stagedTileBytes(64, fragAlignedExtent(136), kAccBytes, false) >
          kBudget);
    Plan p = planDot(f, kBudget);
    CHECK_EQ(kindOf(p), kDirect);
    CHECK(!p.cPoolRegion().overlaysOperands);
  }

  CASE("a proven store makes the fused drain direct: no C in the pool");
  {
    DotFacts f = gemm(64, 64, 64);
    f.fusedAcc = true;
    f.cDirect = true;
    Plan p = planDot(f, kBudget);
    CHECK_EQ(kindOf(p), kFused);
    CHECK(p.storesCDirect());
    CHECK(p.fused().cDirect);
    CHECK_EQ(p.pool.reserved().count(),
             p.stage.ab().count() + p.edgeScratch.count());
    CHECK_EQ(p.cPoolRegion().bytes, 0);
    CHECK(!p.cPoolRegion().overlaysOperands);
  }

  CASE("a live init closes the direct drain: the pool readback adds it");
  {
    DotFacts f = gemm(64, 64, 64);
    f.fusedAcc = true;
    f.cDirect = true;
    f.cInitNonzero = true;
    Plan p = planDot(f, kBudget);
    CHECK_EQ(kindOf(p), kFused);
    CHECK(!p.storesCDirect());
    CHECK(p.cThroughPool());
    CHECK(p.cPoolRegion().bytes > 0);
  }

  CASE("a direct-drain C too big for the pool still fuses");
  {
    DotFacts f = gemm(64, 136, 64);
    f.fusedAcc = true;
    f.cDirect = true;
    Plan p = planDot(f, kBudget);
    CHECK_EQ(kindOf(p), kFused);
    CHECK(p.storesCDirect());
    CHECK(p.pool.reserved() <= kBudget);
  }

  CASE("an unfused C never overlays");
  {
    Plan p = planDot(gemm(64, 64, 64), kBudget);
    CHECK_EQ(kindOf(p), kDirect);
    CHECK(!p.cPoolRegion().overlaysOperands);
    CHECK_EQ(p.cPoolRegion().bytes, p.pool.cReserve().count());
  }

  CASE("operands overflowing the budget force panels");
  {
    Plan p = planDot(gemm(256, 256, 256), kBudget);
    CHECK_EQ(kindOf(p), kPanel);
    CHECK(p.panel().panel.total() <= kBudget);
  }

  // ── panel geometry ─────────────────────────────────────────────────────

  CASE("a panel fits the budget and stays fragment-aligned");
  {
    Panel p = planPanel(256, 256, 256, 2, kAccBytes, kBudget);
    CHECK(p.total() <= kBudget);
    CHECK_EQ(p.mp % kSgFragDim, 0);
    CHECK_EQ(p.np % kSgFragDim, 0);
    CHECK_EQ(p.kp % kSgFragDim, 0);
  }

  CASE("M and N shrink before K");
  {
    Panel p = planPanel(256, 256, 64, 2, kAccBytes, kBudget);
    CHECK_EQ(p.kp, 64);
    CHECK(p.mp < 256 || p.np < 256);
  }

  CASE("the larger extent shrinks first, keeping the panel square");
  {
    Panel p = planPanel(512, 64, 64, 2, kAccBytes, kBudget);
    CHECK(p.mp < 512);
    CHECK_EQ(p.np, 64);
  }

  CASE("M and N shrink only until the drained C fits, K absorbs the rest");
  {
    Panel p = planPanel(64, 64, 1024, 2, kAccBytes, Bytes(2048));
    CHECK(p.cBytes <= Bytes(2048));
    CHECK(p.total() <= Bytes(2048));
    CHECK(p.mp * p.np > kSgFragDim * kSgFragDim);
    CHECK(p.kp < 1024);
    CHECK_EQ(p.kp % kSgFragDim, 0);
  }

  CASE("a shrunk K stays fragment-aligned for any K");
  {
    for (int64_t K : {24, 40, 48, 96, 128, 256, 1024})
      for (int64_t budget : {256, 384, 512, 1024, 2048}) {
        Panel p = planPanel(64, 64, K, 2, kAccBytes, Bytes(budget));
        CHECK_EQ(p.kp % kSgFragDim, 0);
        CHECK(p.kp >= kSgFragDim);
        CHECK(p.kp / kSgFragDim >= 1);
      }
  }

  CASE("panel bytes come from the staged-tile formula the views address by");
  {
    Panel p = planPanel(256, 256, 256, 2, kAccBytes, kBudget);
    CHECK_EQ(p.aBytes.count(),
             stagedTileBytes(p.mp, p.kp, 2, p.stagePad).count());
    CHECK_EQ(
        p.bBytes.count(),
        stagedTileBytes(p.kp, fragAlignedExtent(p.np), 2, p.stagePad).count());
    CHECK_EQ(p.cBytes.count(), stagedTileBytes(p.mp, fragAlignedExtent(p.np),
                                               kAccBytes, p.stagePad)
                                   .count());
  }

  CASE("the pad is dropped when it costs whole panel tiles");
  {
    Panel p = planPanel(64, 128, 128, 4, kAccBytes, kBudget);
    CHECK(!p.stagePad);
    CHECK_EQ(p.mp, 64);
    CHECK_EQ(p.np, 128);

    Panel q = planPanel(32, 32, 32, 2, kAccBytes, kBudget);
    CHECK(q.stagePad);
    CHECK_EQ(panelTiles(32, 32, 32, q), 1);
  }

  CASE("panel counts cover the whole tile");
  {
    Plan p = planDot(gemm(256, 256, 256), kBudget);
    const PanelParams &pp = p.panel();
    CHECK(pp.panelsM * pp.panel.mp >= 256);
    CHECK(pp.panelsN * pp.panel.np >= 256);
    CHECK(pp.panelsK * pp.panel.kp >= 256);
    CHECK(pp.tiles() > 1);
  }

  // ── pool reservation ───────────────────────────────────────────────────

  CASE("C sits alongside the operands when it fits");
  {
    Plan p = planDot(gemm(64, 64, 64), kBudget);
    CHECK(p.pool.reserved() <= kBudget);
    CHECK(p.pool.cNeed > p.stage.ab());
  }

  CASE("C is banded when it does not fit alongside");
  {
    Plan p = planDot(gemm(128, 128, 32), kBudget);
    CHECK_EQ(kindOf(p), kDirect);
    CHECK(p.pool.reserved() <= kBudget);
    CHECK(p.direct().bandRows >= 1);
    CHECK(p.direct().bandRows < 128);
  }

  CASE("a direct store with no ragged arm reserves only the edge scratch");
  {
    DotFacts f = gemm(64, 64, 64);
    f.fusedAcc = true;
    f.cDirect = true;
    f.cFallback = false;
    Plan p = planDot(f, kBudget);
    CHECK(p.edgeScratch > Bytes(0));
    CHECK(p.pool.cNeed == p.stage.ab() + p.edgeScratch);
    CHECK(p.pool.cReserve() == p.edgeScratch);
  }

  CASE("a direct store with a ragged arm still reserves");
  {
    DotFacts f = gemm(64, 64, 64);
    f.fusedAcc = true;
    f.cDirect = true;
    f.cFallback = true;
    Plan p = planDot(f, kBudget);
    CHECK(p.pool.cNeed > Bytes(0));
  }

  CASE("a reservation of zero is raised to something nameable");
  {
    DotFacts f = gemm(64, 64, 64);
    f.fusedAcc = true;
    f.cDirect = true;
    f.cFallback = false;
    f.aInPlace = true;
    f.bInPlace = true;
    Plan p = planDot(f, kBudget);
    CHECK_EQ(p.stage.ab().count(), 0);
    CHECK_EQ(p.pool.cNeed.count(), p.edgeScratch.count());
    CHECK(p.pool.reserved() > Bytes(0));
  }

  CASE("a real reservation is not raised");
  {
    Plan p = planDot(gemm(64, 64, 64), kBudget);
    CHECK(p.pool.cNeed.count() > kMinPoolPtrBytes);
  }

  CASE("the reservation never exceeds the budget");
  {
    for (int64_t M : {32, 64, 128, 256})
      for (int64_t N : {32, 64, 128, 256})
        for (int64_t K : {32, 64, 128, 256}) {
          Plan p = planDot(gemm(M, N, K), kBudget);
          CHECK(p.pool.reserved() <= kBudget);
        }
  }

  CASE("a direct dot always has room for a row of C");
  {
    for (int64_t M : {64, 128, 256})
      for (int64_t N : {64, 128, 256})
        for (int64_t K : {32, 64, 128, 256}) {
          Plan p = planDot(gemm(M, N, K), kBudget);
          if (p.kind != Plan::Kind::Direct)
            continue;
          const Bytes oneRow(N * kAccBytes);
          CHECK(p.stage.ab() + oneRow <= kBudget);
        }
  }

  // ── the residency consequence ──────────────────────────────────────────

  CASE("an in-place operand costs nothing and buys residency");
  {
    DotFacts staged = gemm(64, 64, 64);
    DotFacts inPlace = staged;
    inPlace.aInPlace = true;
    inPlace.bInPlace = true;

    Plan ps = planDot(staged, kBudget);
    Plan pi = planDot(inPlace, kBudget);
    CHECK(pi.stage.ab() == Bytes(0));
    CHECK(pi.pool.reserved() < ps.pool.reserved());
    CHECK(pi.pool.residency() >= ps.pool.residency());
  }

  CASE("a device-direct A operand is not staged, where directness survives");
  {
    DotFacts f = gemm(128, 64, 64);
    f.aElemBytes = f.bElemBytes = 4;
    f.aDirect = true;
    Plan p = planDot(f, kBudget);
    CHECK(p.facts.aDirect);
    CHECK(p.stage.a == Bytes(0));
    CHECK(p.stage.b > Bytes(0));
  }

  // ── padding ────────────────────────────────────────────────────────────

  CASE("staged operands carry the bank pad when the budget is indifferent");
  {
    DotFacts f = gemm(32, 32, 64);
    Plan p = planDot(f, kBudget);
    CHECK_EQ(kindOf(p), kDirect);
    CHECK(p.direct().stagePad);
    CHECK_EQ(p.stage.a.count(), (31 * (64 + 8) + 64) * 2);
    CHECK_EQ(p.stage.a.count(), stagedTileBytes(32, 64, 2).count());
  }

  CASE("rows that already walk the banks get no pad");
  {
    DotFacts f = gemm(64, 64, 24);
    Plan p = planDot(f, kBudget);
    CHECK_EQ(padElemsFor(24, 2), 0);
    CHECK_EQ(p.stage.a.count(), 64 * 24 * 2);
  }

  CASE("a shape the pad pushes past the budget panels instead");
  {
    DotFacts f = gemm(128, 128, 64);
    Plan p = planDot(f, Bytes(20000));
    CHECK_EQ(kindOf(p), kPanel);
    CHECK(p.pool.reserved() <= Bytes(20000));
  }

  // ── the banded C readback ──────────────────────────────────────────────

  CASE("a C too large to sit beside the operands is banded, in fragments");
  {
    DotFacts f = gemm(128, 64, 64);
    f.aElemBytes = f.bElemBytes = 4;
    f.aDirect = true;
    Plan p = planDot(f, kBudget);
    CHECK_EQ(kindOf(p), kDirect);
    const DirectParams &dp = p.direct();
    CHECK(dp.bandRows > 0);
    CHECK(dp.bandRows < fragAlignedExtent(f.M));
    CHECK_EQ(dp.bandRows % kSgFragDim, 0);
    CHECK(p.pool.cReserve().count() >= dp.bandRows * 68 * kAccBytes);
    CHECK(p.pool.reserved() <= kBudget);
  }

  CASE("a C that fits whole is one band and reserves the whole tile");
  {
    DotFacts f = gemm(32, 32, 32);
    f.aDirect = true;
    Plan p = planDot(f, kBudget);
    CHECK_EQ(kindOf(p), kDirect);
    CHECK(p.direct().bandRows >= fragAlignedExtent(f.M));
  }

  CASE("a device-readable A stages anyway when the whole dot fits with it");
  {
    DotFacts f = gemm(64, 64, 64);
    f.aDirect = true;
    Plan p = planDot(f, kBudget);
    CHECK_EQ(kindOf(p), kDirect);
    CHECK(!p.facts.aDirect);
    CHECK(p.stage.a > Bytes(0));
    CHECK(p.direct().bandRows >= fragAlignedExtent(f.M));

    DotFacts g = gemm(128, 64, 64);
    g.aElemBytes = g.bElemBytes = 4;
    g.aDirect = true;
    Plan q = planDot(g, kBudget);
    CHECK_EQ(kindOf(q), kDirect);
    CHECK(q.facts.aDirect);
    CHECK_EQ(q.stage.a.count(), 0);
  }

  CASE("a loop-carried dot keeps its device-direct A");
  {
    DotFacts f = gemm(64, 64, 32);
    f.aElemBytes = f.bElemBytes = 4;
    f.aDirect = true;
    Plan single = planDot(f, kBudget);
    CHECK(!single.facts.aDirect);
    f.carriedAcc = true;
    Plan looped = planDot(f, kBudget);
    CHECK(looped.facts.aDirect);
    CHECK_EQ(looped.stage.a.count(), 0);
  }

  CASE("a shape only the plain pitch admits goes direct-unpadded");
  {
    DotFacts f = gemm(64, 56, 64);
    f.aElemBytes = f.bElemBytes = 4;
    Plan p = planDot(f, kBudget);
    CHECK_EQ(kindOf(p), kDirect);
    CHECK(!p.direct().stagePad);
    CHECK(p.direct().bandRows > 0);
    CHECK(p.pool.reserved() <= kBudget);
    CHECK_EQ(p.stage.b.count(), 64 * 56 * 4);
  }

  CASE("the pad is dropped when it alone would band a whole-tile C");
  {
    DotFacts f = gemm(64, 64, 64);
    f.aElemBytes = f.bElemBytes = 4;
    f.aDirect = true;
    Plan p = planDot(f, kBudget);
    CHECK_EQ(kindOf(p), kDirect);
    CHECK(!p.direct().stagePad);
    CHECK(p.direct().bandRows >= 64);
    CHECK_EQ(p.stage.b.count(), 64 * 64 * 4);
    CHECK(p.pool.reserved() <= kBudget);

    DotFacts g = gemm(128, 64, 64);
    g.aElemBytes = g.bElemBytes = 4;
    g.aDirect = true;
    Plan q = planDot(g, kBudget);
    CHECK_EQ(kindOf(q), kDirect);
    CHECK(q.direct().stagePad);
    CHECK(q.direct().bandRows < fragAlignedExtent(g.M));
  }

  // ── warp tiling flows into the plan ────────────────────────────────────

  CASE("the fragment grid is derived once");
  {
    DotFacts f = gemm(64, 128, 64);
    CHECK_EQ(f.mT(), 8);
    CHECK_EQ(f.nT(), 16);
    CHECK_EQ(f.nFrag(), 128);
    CHECK_EQ(f.kT(), 8);
  }

  CASE("direct and fused both carry their fragment share");
  {
    DotFacts f = gemm(64, 128, 64, 4);
    Plan d = planDot(f, kBudget);
    CHECK_EQ(kindOf(d), kDirect);
    CHECK_EQ(d.direct().fragsPerWarp, 32);

    DotFacts g = gemm(64, 64, 64, 4);
    g.fusedAcc = true;
    Plan u = planDot(g, kBudget);
    CHECK_EQ(kindOf(u), kFused);
    CHECK_EQ(u.fused().fragsPerWarp, 16);
  }

  // ── the payload separation, finding A.1.5.1 ────────────────────────────

  CASE("a strategy's fields are unreachable from another strategy");
  {
    DotFacts f = gemm(64, 64, 64);
    f.fusedAcc = true;
    Plan p = planDot(f, kBudget);
    CHECK_EQ(kindOf(p), kFused);
    CHECK(!p.fused().cDirect);
    CHECK_EQ(p.panel().panel.mp, 0);
    CHECK_EQ(p.panel().tiles(), 0);
  }

  // ── batched (rank-3) ───────────────────────────────────────────────────

  CASE("a batched MMA dot walks the panel schedule even when a slice fits");
  {
    DotFacts f = gemm(32, 32, 32);
    f.rank = 3;
    f.Bd = 2;
    Plan p = planDot(f, kBudget);
    CHECK_EQ(kindOf(p), kPanel);
    CHECK_EQ(p.panel().panel.mp, 32);
    CHECK_EQ(p.panel().panel.np, 32);
    CHECK_EQ(p.panel().panel.kp, 32);
    CHECK_EQ(planPanelSchedule(f, p.panel().panel).size(), 2);
  }

  CASE("a rank-3 axis of extent 1 is still batched");
  {
    DotFacts f = gemm(32, 32, 32);
    f.rank = 3;
    f.Bd = 1;
    CHECK(f.batched());
    CHECK_EQ(kindOf(planDot(f, kBudget)), kPanel);
  }

  CASE("a batched dot reads and writes nothing in place");
  {
    DotFacts f = gemm(64, 64, 64);
    f.rank = 3;
    f.Bd = 4;
    f.aDirect = true;
    f.cDirect = true;
    Plan p = planDot(f, kBudget);
    CHECK(!p.facts.aDirect);
    CHECK(!p.facts.cDirect);
  }

  CASE("a batched scalar dot stays scalar and stages every slice");
  {
    DotFacts f = gemm(32, 32, 512);
    f.rank = 3;
    f.Bd = 4;
    f.aElemBytes = f.bElemBytes = 1;
    f.intAcc = true;
    CHECK(!liftsToFloatMma(f));
    Plan p = planDot(f, kBudget);
    CHECK_EQ(kindOf(p), kScalar);
    const TileView a = stagedOperandView(f, f.M, f.K, f.aElemBytes);
    CHECK_EQ(a.rank(), 3);
    CHECK_EQ(a.extentAt(0), 4);
    CHECK_EQ(a.strideAt(0), stagedTileView(32, 512, 1).cosizeElems());
    CHECK(p.pool.stagedAB ==
          stagedOperandBytes(f, f.M, f.K, 1) +
              stagedOperandBytes(f, f.K, fragAlignedExtent(f.N), 1));
  }

  CASE("a batched scalar dot drops the pad when only the plain pitch fits");
  {
    DotFacts f = gemm(32, 32, 64);
    f.rank = 3;
    f.Bd = 8;
    f.aElemBytes = f.bElemBytes = 1;
    f.intAcc = true;
    f.fusedAcc = true;
    CHECK(!liftsToFloatMma(f));
    Plan p = planDot(f, kBudget);
    CHECK_EQ(kindOf(p), kScalar);
    CHECK(!p.scalar().stagePad);
    CHECK(!p.padStagedC());
    CHECK(p.pool.reserved() <= kBudget);

    DotFacts f2 = gemm(64, 64, 64);
    f2.aElemBytes = f2.bElemBytes = 1;
    f2.intAcc = true;
    f2.fusedAcc = true;
    CHECK(planDot(f2, kBudget).scalar().stagePad);
  }

  // ── the integer lift ───────────────────────────────────────────────────

  CASE("an i8 dot within the exactness bound lifts to the float MMA");
  {
    DotFacts f = gemm(64, 64, 64);
    f.aElemBytes = f.bElemBytes = 1;
    f.intAcc = true;
    CHECK(liftsToFloatMma(f));
    Plan p = planDot(f, kBudget);
    CHECK(p.intThroughFloat);
    CHECK(kindOf(p) != kScalar);
    CHECK_EQ(p.facts.aElemBytes, 4);
    CHECK_EQ(p.facts.bElemBytes, 4);
    CHECK(!p.facts.intAcc);
  }

  CASE("each exactness hole keeps the integer dot off the lift");
  {
    DotFacts f = gemm(64, 64, 64);
    f.aElemBytes = f.bElemBytes = 1;
    f.intAcc = true;
    CHECK(liftsToFloatMma(f));

    DotFacts pastK = f;
    pastK.K = kIntLiftMaxK * 2;
    CHECK(!liftsToFloatMma(pastK));

    DotFacts wide = f;
    wide.aElemBytes = 2;
    CHECK(!liftsToFloatMma(wide));

    DotFacts fused = f;
    fused.fusedAcc = true;
    CHECK(!liftsToFloatMma(fused));

    DotFacts inPlace = f;
    inPlace.aInPlace = true;
    CHECK(!liftsToFloatMma(inPlace));

    CHECK(!planDot(pastK, kBudget).intThroughFloat);
    CHECK(planDot(pastK, kBudget).kind == Plan::Kind::Scalar);
  }

  CASE("a lifted dot closes every arm typed by the IR's element");
  {
    DotFacts f = gemm(64, 64, 64);
    f.aElemBytes = f.bElemBytes = 1;
    f.intAcc = true;
    f.aDirect = true;
    f.cDirect = true;
    Plan p = planDot(f, kBudget);
    CHECK(p.intThroughFloat);
    CHECK(!p.facts.aDirect);
    CHECK(!p.facts.cDirect);
    CHECK(!p.storesCDirect());
  }

  CASE("an unbatched staged operand view is the staged tile itself");
  {
    DotFacts f = gemm(64, 32, 32);
    CHECK(stagedOperandView(f, f.M, f.K, 2) == stagedTileView(64, 32, 2));
  }

  // ── the pass schedule ──────────────────────────────────────────────────

  CASE(
      "a fused pass neither declares nor drains; a single-shot pass does both");
  {
    DotFacts f = gemm(64, 64, 64);
    Plan direct = planDot(f, kBudget);
    DotPassSchedule single = DotPassSchedule::of(direct);
    CHECK(single.declareAccums);
    CHECK(single.drain == DotPassSchedule::Drain::Pool);

    f.fusedAcc = true;
    Plan fused = planDot(f, kBudget);
    DotPassSchedule pass = DotPassSchedule::of(fused);
    CHECK(!pass.declareAccums);
    CHECK(!pass.drainsC());
  }

  CASE("a renamed readback drains without touching the pool");
  {
    DotFacts f = gemm(64, 64, 64);
    landIn(f, 2, 2);
    Plan p = planDot(f, kBudget);
    CHECK(p.facts.cCostsPoolNothing());
    CHECK(!p.cThroughPool());
    CHECK(p.readsBackByRename());
    CHECK_EQ((int64_t)p.readback.regs.size(), f.cRegs);
    CHECK(DotPassSchedule::of(p).drain == DotPassSchedule::Drain::Rename);
    CHECK_EQ(p.pool.cReserve().count(), 0);
    CHECK_EQ(p.cBandRows(), 64);
  }

  CASE("the cover follows the layout while the extra loads fit the round trip");
  {
    DotFacts f = gemm(64, 64, 64);
    landIn(f, 4, 1);
    Plan p = planDot(f, kBudget);
    CHECK(p.readsBackByRename());
    CHECK((p.cover == WarpCover{2, 8}));
    CHECK_EQ((int64_t)p.readback.regs.size(), f.cRegs);

    DotFacts wide = gemm(64, 64, 64, 8);
    landIn(wide, 8, 1);
    Plan pooled = planDot(wide, kBudget);
    CHECK_EQ(kindOf(pooled), kDirect);
    CHECK(!pooled.readsBackByRename());
    CHECK(!pooled.cover.set());
    CHECK(pooled.cThroughPool());
    CHECK(DotPassSchedule::of(pooled).drain == DotPassSchedule::Drain::Pool);
  }

  CASE("a layout no exact cover matches stays a pool readback");
  {
    DotFacts f = gemm(64, 64, 64);
    landIn(f, 4, 1);
    f.cDims[0].warp = {8, 0};
    f.cDims[1].warp = {0, 8};
    Plan p = planDot(f, kBudget);
    CHECK(!p.readsBackByRename());
    CHECK(p.cThroughPool());
  }

  CASE("the rename is decided on the plan's own emitted cover");
  {
    DotFacts f = gemm(32, 32, 32);
    f.aElemBytes = f.bElemBytes = 4;
    f.aDirect = true;
    landIn(f, 4, 1);
    Plan p = planDot(f, kBudget);
    CHECK(!p.facts.aDirect);
    CHECK(p.readsBackByRename());
    CHECK((p.cover == WarpCover{1, 4}));

    landIn(f, 2, 2);
    Plan renamed = planDot(f, kBudget);
    CHECK(renamed.readsBackByRename());
    CHECK((renamed.cover == WarpCover{2, 2}));

    f.carriedAcc = true;
    Plan direct = planDot(f, kBudget);
    CHECK(direct.facts.aDirect);
    CHECK(!direct.readsBackByRename());
  }

  CASE("idle hardware warps refuse the rename");
  {
    DotFacts f = gemm(8, 16, 32);
    landIn(f, 1, 2);
    f.cDims[0].warp.push_back(0);
    f.cDims[1].warp.push_back(0);
    Plan p = planDot(f, kBudget);
    CHECK(warpsFor(p.facts) < p.facts.numWarps);
    CHECK(!p.readsBackByRename());
  }

  CASE("an integer dot the lift refuses never claims a rename");
  {
    DotFacts ints = gemm(32, 32, 32);
    landIn(ints, 2, 2);
    ints.intAcc = true;
    Plan p = planDot(ints, kBudget);
    CHECK_EQ(kindOf(p), kScalar);
    CHECK(!p.facts.cRename);
    CHECK(!p.readsBackByRename());
  }

  CASE("a batched dot renames through its panel walk, one slice per tile");
  {
    DotFacts f = gemm(32, 32, 32);
    landIn(f, 2, 2);
    f.cDims.insert(f.cDims.begin(), LayoutBasis{});
    f.rank = 3;
    f.Bd = 2;
    Plan p = planDot(f, kBudget);
    CHECK_EQ(kindOf(p), kPanel);
    CHECK(p.readsBackByRename());
    int slices = 0;
    forEachPanelTile(p.facts, p.panel().panel, [&](const PanelTile &t) {
      if (t.finalK && t.renameC)
        ++slices;
    });
    CHECK_EQ(slices, 2);
  }

  CASE("a panel carves a C region exactly when it drains through the pool");
  {
    DotFacts f = gemm(32, 128, 64);
    f.aElemBytes = f.bElemBytes = 4;
    landIn(f, 1, 4);
    Plan renamed = planDot(f, kBudget);
    CHECK_EQ(kindOf(renamed), kPanel);
    CHECK(renamed.readsBackByRename());
    CHECK(!renamed.cThroughPool());
    CHECK_EQ(renamed.cPoolRegion().bytes, 0);

    f.cDims[0].warp = {0, 0};
    f.cDims[1].warp = {8, 0};
    Plan pooled = planDot(f, kBudget);
    CHECK_EQ(kindOf(pooled), kPanel);
    CHECK(!pooled.readsBackByRename());
    CHECK(pooled.cThroughPool());
    CHECK(pooled.cPoolRegion().bytes > 0);
  }

  CASE("a fused dot renames its drain after the loop");
  {
    DotFacts f = gemm(64, 64, 64);
    landIn(f, 2, 2);
    f.fusedAcc = true;
    Plan p = planDot(f, kBudget);
    CHECK_EQ(kindOf(p), kFused);
    CHECK(p.readsBackByRename());
    CHECK(!p.cThroughPool());
    CHECK_EQ(p.cPoolRegion().bytes, 0);
    CHECK(DotPassSchedule::of(p).drain == DotPassSchedule::Drain::None);
  }

  CASE("a fused loop takes only the cover that costs no extra loads");
  {
    DotFacts f = gemm(32, 32, 32);
    f.aElemBytes = f.bElemBytes = 4;
    f.fusedAcc = true;
    landIn(f, 4, 1);
    CHECK(!planDot(f, kBudget).readsBackByRename());
    landIn(f, 2, 2);
    CHECK(planDot(f, kBudget).readsBackByRename());
  }

  CASE("a lifted integer dot renames like a float one");
  {
    DotFacts f = gemm(32, 32, 32);
    f.aElemBytes = f.bElemBytes = 1;
    f.intAcc = true;
    landIn(f, 2, 2);
    Plan p = planDot(f, kBudget);
    CHECK(p.intThroughFloat);
    CHECK(p.readsBackByRename());
  }

  CASE("a panel renames when every tile sits on the layout's warp grid");
  {
    DotFacts f = gemm(128, 128, 768);
    landIn(f, 2, 2);
    Plan p = planDot(f, kBudget);
    CHECK_EQ(kindOf(p), kPanel);
    CHECK((p.panel().panel.renameWarps == WarpCover{2, 2}));
    CHECK(p.readsBackByRename());
    CHECK(!p.cThroughPool());
    CHECK_EQ(p.cPoolRegion().bytes, 0);
    int renamed = 0, pooled = 0;
    forEachPanelTile(p.facts, p.panel().panel, [&](const PanelTile &t) {
      if (!t.finalK)
        return;
      CHECK((t.cover == WarpCover{t.mFrags() / 2, t.nFrags() / 2}));
      (t.renameC ? renamed : pooled) += 1;
    });
    CHECK_EQ(renamed, 2);
    CHECK_EQ(pooled, 0);
  }

  CASE("a panel whose pitch splits the layout's warp grid stays on the pool");
  {
    DotFacts f = gemm(128, 256, 768, 8);
    landIn(f, 2, 4);
    Plan p = planDot(f, kBudget);
    CHECK_EQ(kindOf(p), kPanel);
    CHECK_EQ(p.panel().panel.np % 32, 24);
    CHECK(!p.panel().panel.renameWarps.set());
    CHECK(!p.readsBackByRename());
    CHECK(p.cThroughPool());
  }

  CASE("a rename never bands, even where the pool could not hold C whole");
  {
    DotFacts f = gemm(64, 64, 64);
    f.aElemBytes = f.bElemBytes = 4;
    Plan pooled = planDot(f, kBudget);
    CHECK_EQ(kindOf(pooled), kPanel);

    landIn(f, 2, 2);
    Plan p = planDot(f, kBudget);
    CHECK_EQ(kindOf(p), kDirect);
    CHECK(p.readsBackByRename());
    CHECK_EQ(p.cBandRows(), 64);
    CHECK(p.pool.reserved() <= kBudget);
  }

  CASE("the plan explains why a dot went unfused, beyond just its choice");
  {
    DotFacts big = gemm(128, 128, 128);
    big.fusedAcc = true;
    const Plan panelled = planDot(big, kBudget);
    CHECK(!panelled.fit.operandsAndBand);
    const std::string r = dotPlanReport(panelled.facts, panelled.fit);
    CHECK(r.find("128x128x128") != std::string::npos);
    CHECK(r.find("fusedAcc=y") != std::string::npos);
    CHECK(r.find("opsAndBand=n") != std::string::npos);

    DotFacts small = gemm(64, 64, 64);
    small.fusedAcc = true;
    const Plan fused = planDot(small, kBudget);
    CHECK(fused.kind == Plan::Kind::Fused);
    const std::string rf = dotPlanReport(fused.facts, fused.fit);
    CHECK(rf.find("registers across the K loop") != std::string::npos);
    CHECK(rf.find("wholeC=y") != std::string::npos);
  }

  CASE("the fit the plan reports is the fit it chose from");
  {
    DotFacts f = gemm(64, 64, 64);
    f.fusedAcc = true;
    const Plan p = planDot(f, kBudget);
    CHECK(p.fit.wholeC);
    CHECK(p.fit.operandsAndBand);

    const Plan starved = planDot(f, Bytes{1024});
    CHECK(!starved.fit.operandsAndBand);
    CHECK(starved.kind == Plan::Kind::Panel);
  }

  return ::agpu_test::report("DotPlan");
}
