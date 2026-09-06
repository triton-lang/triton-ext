// The dot entry point: facts in, statements out.
#include "agpu/emit/EmitDot.h"
#include "agpu/msl/Printer.h"
#include "fixtures.h"
#include "harness.h"
#include "render.h"

#include <sstream>

using namespace agpu;
using agpu_test::countOf;
using agpu_test::render;

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

DotInputs inputsFor() {
  DotInputs in;
  in.a = {"pA", 64};
  in.b = {"pB", 64};
  return in;
}

const Bytes kBudget{kTGResidentBudgetBytes};

} // namespace

int main() {
  // ── the strategy reaches the emitter ───────────────────────────────────

  CASE("a direct dot emits its MMA grid, driven by the plan");
  {
    msl::Context c;
    msl::Block body;
    const Plan p = planDot(gemm(64, 64, 64), kBudget);
    CHECK(p.kind == Plan::Kind::Direct);

    Decision d = emitDot(c, body, p, inputsFor());
    CHECK(d.ok());
    const std::string out = render(body);
    CHECK(countOf(out, "simdgroup_multiply_accumulate") > 0);
    CHECK(countOf(out, "simdgroup_float8x8(0.0f)") > 0);
  }

  CASE("a shape that panels emits its own panel tiles");
  {
    // Same call, different strategy: the budget is too small for the operands.
    msl::Context c;
    msl::Block body;
    const Plan p = planDot(gemm(256, 256, 64), Bytes(8192));
    CHECK(p.kind == Plan::Kind::Panel);

    DotInputs in = inputsFor();
    in.tileInputs = [](const PanelTile &) {
      PanelInputs pi;
      return pi;
    };
    Decision d = emitDot(c, body, p, in);
    CHECK(d.ok());
    // A panel walk barriers between phases; a direct grid does not.
    CHECK(countOf(render(body), "threadgroup_barrier") > 0);
  }

  CASE("the two strategies emit different programs for the same shape");
  {
    msl::Context c;
    msl::Block direct, panel;

    const Plan pd = planDot(gemm(64, 64, 64), kBudget);
    emitDot(c, direct, pd, inputsFor());

    const Plan pp = planDot(gemm(256, 256, 64), Bytes(8192));
    DotInputs in = inputsFor();
    in.tileInputs = [](const PanelTile &) {
      PanelInputs pi;
      return pi;
    };
    emitDot(c, panel, pp, in);

    CHECK(render(direct) != render(panel));
  }

  // ── the declines ───────────────────────────────────────────────────────

  CASE("an unsupported shape declines with the plan's own reason");
  {
    // K is summed, so a partial K fragment would add operand elements that
    // do not exist.
    msl::Context c;
    msl::Block body;
    const Plan p = planDot(gemm(64, 64, 12), kBudget);
    CHECK(p.kind == Plan::Kind::Unsupported);

    Decision d = emitDot(c, body, p, inputsFor());
    CHECK(d.isDecline());
    CHECK(!d.isBug());
    CHECK(body.empty());
  }

  CASE("an integer accumulator past the lift bound emits a per-thread K loop");
  {
    // There is no integer MMA on Apple GPUs; `simdgroup_matrix<int>` is a
    // static_assert. Within the exactness bound an i8 dot lifts to the float
    // MMA (see test_dotplan); past it the plan is a scalar K loop.
    msl::Context c;
    msl::Block body;
    DotFacts f = gemm(32, 64, 512);
    f.aElemBytes = f.bElemBytes = 1;
    f.intAcc = true;
    CHECK(!liftsToFloatMma(f));
    const Plan p = planDot(f, kBudget);
    CHECK(p.kind == Plan::Kind::Scalar);

    DotInputs in = inputsFor();
    in.readbackFor = [](const Range &rows) {
      ReadbackInputs back;
      back.actions.push_back(
          StageAction{0, 1, false, {rows.lo, 0}, CoordGuard::unguarded()});
      back.actions.push_back(
          StageAction{1, 1, false, {rows.lo, 1}, CoordGuard::unguarded()});
      back.names.push_back("out0");
      back.names.push_back("out1");
      back.bases.push_back("");
      back.bases.push_back("cin1"); // a carried accumulator to add onto
      return back;
    };
    LayoutBasis row, col;
    row.lane = {1, 2, 4, 8, 16};
    col.reg = {1};
    CoordSource cs;
    cs.dims = {row, col};
    in.coords = PanelCoords::forAll(cs);

    Decision d = emitDot(c, body, p, in);
    CHECK(d.ok());
    const std::string out = render(body);
    CHECK_EQ(countOf(out, "simdgroup"), 0);

    // One K loop per result register, at the operand extent K.
    CHECK_EQ(countOf(out, "< 512;"), 2);

    // Both elements widen before the multiply; an i8 product overflows at K=2.
    CHECK(countOf(out, "(int)pA[") > 0);
    CHECK(countOf(out, "(int)pB[") > 0);

    // A carried accumulator starts at the incoming value.
    CHECK(countOf(out, "out0 = 0") > 0);
    CHECK(countOf(out, "out1 = cin1") > 0);
  }

  CASE("a batched scalar dot selects each operand slice by the batch coord");
  {
    // The staged buffers hold every slice, batch-major. No batch loop, just
    // one more address term from the result register's batch coordinate.
    msl::Context c;
    msl::Block body;
    DotFacts f = gemm(8, 8, 512);
    f.rank = 3;
    f.Bd = 2;
    f.aElemBytes = f.bElemBytes = 1;
    f.intAcc = true;
    CHECK(!liftsToFloatMma(f)); // K past the lift bound
    const Plan p = planDot(f, kBudget);
    CHECK(p.kind == Plan::Kind::Scalar);

    DotInputs in = inputsFor();
    in.a.sliceStride = 400;
    in.b.sliceStride = 500;
    in.readbackFor = [](const Range &) {
      ReadbackInputs back;
      back.actions.push_back(
          StageAction{0, 1, false, {0, 0, 0}, CoordGuard::unguarded()});
      back.names.push_back("out0");
      back.bases.push_back("");
      return back;
    };
    // The batch coordinate varies with the lane; a constant one folds away.
    LayoutBasis batch, row, col;
    batch.lane = {1};
    row.lane = {2, 4, 8};
    col.lane = {16, 1, 2};
    CoordSource cs;
    cs.dims = {batch, row, col};
    in.coords = PanelCoords::forAll(cs);

    Decision d = emitDot(c, body, p, in);
    CHECK(d.ok());
    const std::string out = render(body);
    // The batch term scales each source's own slice stride; row and column
    // come from the axes behind the batch.
    CHECK(countOf(out, "* 400") > 0);
    CHECK(countOf(out, "* 500") > 0);
    CHECK_EQ(countOf(out, "simdgroup"), 0);
  }

  CASE("a panel plan with no tile builder asserts as a caller bug");
  {
    msl::Context c;
    msl::Block body;
    const Plan p = planDot(gemm(256, 256, 64), Bytes(8192));
    Decision d = emitDot(c, body, p, inputsFor());
    CHECK(!d.ok());
    CHECK(!d.isDecline());
  }

  CASE("every plan has a Decision, like every other plan in the tree");
  {
    CHECK(dotDecision(planDot(gemm(64, 64, 64), kBudget)).ok());
    CHECK(dotDecision(planDot(gemm(64, 64, 12), kBudget)).isDecline());
  }

  // ── the grid comes from the plan ───────────────────────────────────────

  CASE("the warp grid is read directly off the facts");
  {
    const Plan p = planDot(gemm(64, 128, 64), kBudget);
    const WarpGrid g = gridOf(p);
    CHECK_EQ(g.mT, p.facts.mT());
    CHECK_EQ(g.nT, p.facts.nT());
    CHECK_EQ(g.numWarps, warpsFor(p.facts));
  }

  CASE("a ragged shape reaches the emitter through the same call");
  {
    msl::Context c;
    msl::Block body;
    const Plan p = planDot(gemm(60, 64, 64), kBudget);
    CHECK(p.kind == Plan::Kind::Direct);
    CHECK(p.facts.ragged());

    Decision d = emitDot(c, body, p, inputsFor());
    CHECK(d.ok());
    CHECK(countOf(render(body), "simdgroup_multiply_accumulate") > 0);
  }

  CASE("rolling is supplied by the caller as an input");
  {
    // emitKernel decides it from the size budget and passes it down.
    msl::Context c;
    msl::Block unrolled, rolled;
    const Plan p = planDot(gemm(64, 64, 64), kBudget);

    DotInputs in = inputsFor();
    emitDot(c, unrolled, p, in);
    in.rollK = true;
    emitDot(c, rolled, p, in);

    CHECK(render(unrolled) != render(rolled));
    CHECK(countOf(render(rolled), "for (") > 0);
    CHECK_EQ(countOf(render(unrolled), "for ("), 0);
  }

  // ── the fused loop bracket ─────────────────────────────────────────────

  CASE("a fused loop declares, hosts, drains: in that order, once each");
  {
    msl::Context c;
    msl::Block body;
    DotFacts f = gemm(64, 64, 64);
    f.fusedAcc = true;
    const Plan p = planDot(f, kBudget);
    CHECK(p.kind == Plan::Kind::Fused);

    auto readbackFor = [](const Range &rows) {
      ReadbackInputs back;
      back.actions.push_back(
          StageAction{0, 1, false, {rows.lo, 0}, CoordGuard::unguarded()});
      back.names.push_back("out0");
      back.names.push_back("out1");
      back.bases.push_back("");
      back.bases.push_back("");
      return back;
    };
    CoordSource cs;
    LayoutBasis row, col;
    row.lane = {1, 2, 4, 8, 16};
    cs.dims = {row, col};

    DirectNames nm;
    const Decision d =
        emitFusedLoop(c, body, p, nm, readbackFor, cs, {}, {}, [&]() {
          body.push_back(c.exprStmt(c.var("THE_LOOP")));
          return Decision::emitted();
        });
    CHECK(d.ok());
    const std::string out = render(body);

    // Every accumulator index declared once and zeroed.
    const std::string zero = " = " + kSimdgroup8x8.zeroCtor("float") + "(0.0f)";
    CHECK_EQ(countOf(out, zero), 16); // 64 fragments over 4 warps
    for (int i = 0; i < 16; ++i)
      CHECK_EQ(
          countOf(out, "simdgroup_float8x8 acc" + std::to_string(i) + " ="), 1);

    // Result registers and fragments before the loop, stores and readback
    // after it, each phase once.
    const std::size_t decl = out.find("float out0 = 0.0f");
    const std::size_t frag = out.find(zero);
    const std::size_t loop = out.find("THE_LOOP");
    const std::size_t store = out.find("simdgroup_store");
    const std::size_t read = out.find("out0 =", loop);
    CHECK(decl != std::string::npos && decl < loop);
    CHECK(frag != std::string::npos && frag < loop);
    CHECK(loop < store);
    CHECK(store < read);
    CHECK_EQ(countOf(out, "simdgroup_store"), 16);

    // Two barriers, owned by the bracket: one before the stores, since this
    // plan's C overlays the operand pool the loop was still reading and one
    // before the readback, since registers come from slots other warps stored.
    CHECK(p.cPoolRegion().overlaysOperands);
    CHECK_EQ(countOf(out, "threadgroup_barrier"), 2);
    const std::size_t b1 = out.find("threadgroup_barrier");
    const std::size_t b2 = out.find("threadgroup_barrier", b1 + 1);
    CHECK(loop < b1 && b1 < store);
    CHECK(store < b2 && b2 < read);
  }

  CASE("a fused plan's own emission neither declares nor drains");
  {
    // The same direct emitter, told by the pass schedule that the bracket
    // owns both ends: no zeroed fragments, no stores, no pool touch.
    msl::Context c;
    msl::Block body;
    DotFacts f = gemm(64, 64, 64);
    f.fusedAcc = true;
    const Plan p = planDot(f, kBudget);
    CHECK(p.kind == Plan::Kind::Fused);

    const Decision d = emitDot(c, body, p, inputsFor());
    CHECK(d.ok());
    const std::string out = render(body);
    CHECK(countOf(out, "simdgroup_multiply_accumulate") > 0);
    CHECK_EQ(countOf(out, "simdgroup_float8x8(0.0f)"), 0);
    CHECK_EQ(countOf(out, "simdgroup_store"), 0);
    CHECK_EQ(countOf(out, "threadgroup_barrier"), 0);
  }

  CASE("a direct drain stores fragments to the device window, unfenced");
  {
    // The plan proved the store's window, so the drain is the store:
    // fragments straight to the device tensor at its runtime leading
    // dimension. No pool, no readback registers, no barriers.
    msl::Context c;
    msl::Block body;
    DotFacts f = gemm(64, 64, 64);
    f.fusedAcc = true;
    f.cDirect = true;
    const Plan p = planDot(f, kBudget);
    CHECK(p.storesCDirect());

    DeviceStoreTarget t;
    t.base = "cptr";
    t.leadingDim = Stride::runtime("ldc");
    t.rowStart = c.var("rs");
    t.colStart = c.var("cs");

    DirectNames nm;
    const Decision d = emitFusedLoop(c, body, p, nm, {}, {}, t, {}, [&]() {
      body.push_back(c.exprStmt(c.var("THE_LOOP")));
      return Decision::emitted();
    });
    CHECK(d.ok());
    const std::string out = render(body);

    CHECK_EQ(countOf(out, "simdgroup_store"), 16);
    CHECK_EQ(countOf(out, "threadgroup_barrier"), 0);
    CHECK_EQ(countOf(out, "float out"), 0); // no readback registers
    CHECK_EQ(countOf(out, "pC"), 0);        // the pool is never named
    // Every store addresses the window: base, both starts and the runtime
    // leading dimension as both the row scale and the store's own stride.
    CHECK_EQ(countOf(out, "cptr + "), 16);
    CHECK_EQ(countOf(out, "rs"), 16);
    CHECK_EQ(countOf(out, "cs"), 16);
    CHECK_EQ(countOf(out, "ldc"), 32);
    // Each accumulator stored exactly once, after the loop.
    for (int i = 0; i < 16; ++i)
      CHECK_EQ(countOf(out, "simdgroup_store(acc" + std::to_string(i) + ","),
               1);
    CHECK(out.find("THE_LOOP") < out.find("simdgroup_store"));
    // The fragments are still declared before the loop, once each.
    const std::string zero = " = " + kSimdgroup8x8.zeroCtor("float") + "(0.0f)";
    CHECK_EQ(countOf(out, zero), 16);
    CHECK(out.find(zero) < out.find("THE_LOOP"));
  }

  CASE("a direct drain without its window asserts as a caller bug");
  {
    msl::Context c;
    msl::Block body;
    DotFacts f = gemm(64, 64, 64);
    f.fusedAcc = true;
    f.cDirect = true;
    const Plan p = planDot(f, kBudget);
    const Decision d = emitFusedLoop(c, body, p, DirectNames{}, {}, {}, {}, {},
                                     [&]() { return Decision::emitted(); });
    CHECK(!d.ok());
    CHECK(!d.isDecline());
  }

  CASE("an unpadded fused plan drains at the plain pitch throughout");
  {
    // 64x128 f32 C fits the pool only unpadded. Stores and readback both use
    // the plain leading dimension, via `cStagedView` reading `padStagedC`.
    msl::Context c;
    msl::Block body;
    DotFacts f = gemm(64, 128, 64);
    f.fusedAcc = true;
    const Plan p = planDot(f, kBudget);
    CHECK(p.kind == Plan::Kind::Fused);
    CHECK(!p.padStagedC());
    CHECK_EQ(p.cStagedView().strideAt(0), 128);

    auto readbackFor = [](const Range &) {
      ReadbackInputs back;
      back.actions.push_back(
          StageAction{0, 1, false, {0, 0}, CoordGuard::unguarded()});
      back.names.push_back("out0");
      back.bases.push_back("");
      return back;
    };
    CoordSource cs;
    cs.dims = {LayoutBasis{}, LayoutBasis{}};
    const Decision d =
        emitFusedLoop(c, body, p, DirectNames{}, readbackFor, cs, {}, {},
                      [&]() { return Decision::emitted(); });
    CHECK(d.ok());
    const std::string out = render(body);
    // Store stride is the plain 128; the padded 132 must appear nowhere.
    CHECK(countOf(out, ", 128)") > 0);
    CHECK_EQ(countOf(out, "132"), 0);
  }

  CASE("a small fused dot fences idle warps; its fragments stay visible");
  {
    // 8x16 on a four-warp launch: two fragments, two effective warps. The
    // MMAs and the drain run under `warp < 2`; the fragment declarations do
    // not, or the guarded loop body could not see them.
    msl::Context c;
    msl::Block body;
    DotFacts f = gemm(8, 16, 64);
    f.fusedAcc = true;
    f.cDirect = true;
    const Plan p = planDot(f, kBudget);
    CHECK(p.kind == Plan::Kind::Fused);
    CHECK(gridOf(p).guardsIdleWarps());

    DeviceStoreTarget t;
    t.base = "cptr";
    t.leadingDim = Stride::runtime("ldc");

    const Decision d =
        emitFusedLoop(c, body, p, DirectNames{}, {}, {}, t, {}, [&]() {
          body.push_back(c.exprStmt(c.var("THE_LOOP")));
          return Decision::emitted();
        });
    CHECK(d.ok());
    const std::string out = render(body);
    const std::string zero = " = " + kSimdgroup8x8.zeroCtor("float") + "(0.0f)";
    // One declaration per accumulator index, before the loop, unguarded: the
    // parameterised block's single slot serves both warps.
    CHECK_EQ(countOf(out, zero), 1);
    CHECK(out.find(zero) < out.find("THE_LOOP"));
    CHECK(out.find("if (warp") == std::string::npos ||
          out.find(zero) < out.find("if (warp"));
    // No store outside `warp < 2` and the one store is spelled in the warp id.
    CHECK_EQ(countOf(out, "simdgroup_store"), 1);
    CHECK(out.find("warp % 2 * 8") != std::string::npos);
    CHECK(countOf(out, "if (warp < 2)") > 0);
    CHECK(out.find("if (warp < 2)", out.find("THE_LOOP")) <
          out.find("simdgroup_store"));
  }

  CASE("a declined loop body declines the bracket, emitting no drain");
  {
    msl::Context c;
    msl::Block body;
    DotFacts f = gemm(64, 64, 64);
    f.fusedAcc = true;
    const Plan p = planDot(f, kBudget);

    auto readbackFor = [](const Range &) {
      ReadbackInputs back;
      back.actions.push_back(
          StageAction{0, 1, false, {0, 0}, CoordGuard::unguarded()});
      back.names.push_back("out0");
      back.bases.push_back("");
      return back;
    };
    CoordSource cs;
    cs.dims = {LayoutBasis{}, LayoutBasis{}};
    const Decision d =
        emitFusedLoop(c, body, p, DirectNames{}, readbackFor, cs, {}, {},
                      [&]() { return Decision::declined("test", "no loop"); });
    CHECK(!d.ok());
    CHECK_EQ(countOf(render(body), "simdgroup_store"), 0);
  }

  return ::agpu_test::report("EmitDot");
}
