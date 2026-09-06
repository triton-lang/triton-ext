// The direct dot path: four arms, one emitter.
#include "agpu/emit/EmitDirect.h"
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

const std::string kZeroAcc = kSimdgroup8x8.zeroCtor("float") + "(0.0f)";

OperandSource operandFrom(msl::Str buffer, int64_t leadingDim,
                          OperandSource::FragAxis axis, int64_t bandFrags = 0) {
  OperandSource s;
  s.buffer = std::move(buffer);
  s.leadingDim = leadingDim;
  s.fragAxis = axis;
  s.bandFrags = bandFrags;
  return s;
}

DirectInputs gemmInputs(int64_t kT, int64_t lda, int64_t ldb,
                        bool rollK = false) {
  DirectInputs in;
  in.a = operandFrom("pA", lda, OperandSource::FragAxis::Rows);
  in.b = operandFrom("pB", ldb, OperandSource::FragAxis::Cols);
  in.kT = kT;
  in.rollK = rollK;
  return in;
}

std::multiset<std::pair<int64_t, int64_t>> coverage(const WarpProgram &p,
                                                    const WarpGrid &g) {
  std::multiset<std::pair<int64_t, int64_t>> out;
  for (int64_t w = 0; w < g.numWarps; ++w) {
    const int64_t block = p.form == WarpForm::Parameterised ? 0 : w;
    for (const WarpSlot &s : p.slots(block, g.mT, g.nT, g.numWarps))
      out.insert({s.mi.at(w), s.ni.at(w)});
  }
  return out;
}

} // namespace

int main() {
  DirectNames nm;

  // ── SlotCoord ──────────────────────────────────────────────────────────

  CASE("a coordinate is constant or affine and knows which");
  {
    CHECK(SlotCoord::fixed(3).isConst());
    CHECK(!SlotCoord::affine(2, 1).isConst());
    CHECK_EQ(SlotCoord::fixed(3).at(7), 3);
    CHECK_EQ(SlotCoord::affine(2, 1).at(3), 7);
  }

  CASE("coordinates are compared as integers");
  {
    CHECK(SlotCoord::affine(2, 1) == SlotCoord::affine(2, 1));
    CHECK(!(SlotCoord::affine(2, 1) == SlotCoord::affine(1, 2)));
  }

  // ── the warp cover, as a cost comparison ───────────────────────────────

  CASE("a device-direct A pulls the cover toward fewer A fragments");
  {
    WarpGrid g{8, 2, 4, /*aDirect=*/true};
    WarpProgram p = planWarpProgram(g);
    CHECK(p.form == WarpForm::Parameterised);
    CHECK_EQ(p.miCount, 2);
    CHECK_EQ(p.niCount, 2);
  }

  CASE("staged operands take the near-square cover");
  {
    WarpGrid g{4, 4, 4};
    WarpProgram p = planWarpProgram(g);
    CHECK(p.form == WarpForm::Parameterised);
    CHECK_EQ(p.miCount, 2);
    CHECK_EQ(p.niCount, 2);
  }

  CASE("a flat grid still splits both axes when that loads less");
  {
    WarpGrid g{2, 8, 4};
    WarpProgram p = planWarpProgram(g);
    CHECK(p.form == WarpForm::Parameterised);
    CHECK_EQ(p.miCount, 2);
    CHECK_EQ(p.niCount, 2);
  }

  CASE("no exact cover falls back to one guarded block per warp");
  {
    WarpGrid g{3, 5, 4};
    WarpProgram p = planWarpProgram(g);
    CHECK(p.form == WarpForm::PerWarp);
    CHECK_EQ(p.blockCount(4), 4);
  }

  CASE("a program always covers the grid exactly once");
  {
    const WarpGrid grids[] = {
        {8, 2, 4, true}, {4, 4, 4}, {2, 8, 4}, {3, 5, 4}, {4, 4, 2}, {6, 3, 3},
    };
    for (const WarpGrid &g : grids) {
      auto cov = coverage(planWarpProgram(g), g);
      CHECK_EQ((int64_t)cov.size(), g.nFrag());
      for (int64_t mi = 0; mi < g.mT; ++mi)
        for (int64_t ni = 0; ni < g.nT; ++ni)
          CHECK_EQ((int)cov.count({mi, ni}), 1);
    }
  }

  // ── one emitter for every arm ──────────────────────────────────────────

  CASE("the parameterised form emits a single shared block");
  {
    msl::Context c;
    msl::Block body;
    WarpGrid g{4, 4, 4};
    WarpProgram p = planWarpProgram(g);
    emitDirectDot(c, body, p, g, gemmInputs(2, 32, 64),
                  TileView::rowMajor({32, 32}), nm, 0, {}, {});
    const std::string out = render(body);
    CHECK_EQ(countOf(out, "if (warp =="), 0);
    CHECK(out.find("warp") != std::string::npos);
  }

  CASE("the program says which warp a block serves, so the emitter need not "
       "ask");
  {
    WarpGrid par{4, 4, 4};
    CHECK(!planWarpProgram(par).guardWarp(0).has_value());

    WarpGrid per{3, 5, 4};
    WarpProgram p = planWarpProgram(per);
    CHECK(p.guardWarp(2).has_value());
    CHECK_EQ(*p.guardWarp(2), 2);
  }

  CASE("the per-warp form emits one guarded block per warp");
  {
    msl::Context c;
    msl::Block body;
    WarpGrid g{3, 5, 4};
    WarpProgram p = planWarpProgram(g);
    emitDirectDot(c, body, p, g, gemmInputs(1, 8, 40),
                  TileView::rowMajor({24, 40}), nm, 0, {}, {});
    const std::string out = render(body);
    CHECK_EQ(countOf(out, "if (warp == 0)"), 1);
    CHECK_EQ(countOf(out, "if (warp == 3)"), 1);
  }

  CASE("both forms run the same emitter, over the slots the program hands out");
  {
    const WarpGrid grids[] = {{4, 4, 4}, {3, 5, 4}};
    for (const WarpGrid &g : grids) {
      msl::Context c;
      msl::Block body;
      const WarpProgram p = planWarpProgram(g);
      emitDirectDot(c, body, p, g, gemmInputs(2, 16, 40),
                    TileView::rowMajor({g.mT * 8, g.nT * 8}), nm, 0, {}, {});

      int64_t emitted = 0;
      for (int64_t w = 0; w < p.blockCount(g.numWarps); ++w)
        emitted += (int64_t)p.slots(w, g.mT, g.nT, g.numWarps).size();
      CHECK_EQ(countOf(render(body), "simdgroup_multiply_accumulate"),
               (int)(emitted * 2));
    }
  }

  // ── the fragment cache ─────────────────────────────────────────────────

  CASE("an A fragment is loaded exactly once per row");
  {
    msl::Context c;
    msl::Block body;
    std::vector<WarpSlot> slots;
    for (int64_t mi = 0; mi < 2; ++mi)
      for (int64_t ni = 0; ni < 3; ++ni)
        slots.push_back(
            {SlotCoord::fixed(mi), SlotCoord::fixed(ni), (int)slots.size()});
    int counter = 0;
    emitDirectMma(c, body, slots, gemmInputs(1, 8, 24), nm, counter);
    const std::string out = render(body);
    CHECK_EQ(countOf(out, "simdgroup_load"), 2 + 3);
    CHECK_EQ(countOf(out, "simdgroup_multiply_accumulate"), 6);
  }

  CASE("a fragment cached under one warp expression is found under the same "
       "one");
  {
    msl::Context c;
    msl::Block body;
    std::vector<WarpSlot> slots = {
        {SlotCoord::affine(2, 0), SlotCoord::fixed(0), 0},
        {SlotCoord::affine(2, 0), SlotCoord::fixed(1), 1},
    };
    int counter = 0;
    emitDirectMma(c, body, slots, gemmInputs(1, 8, 16), nm, counter);
    CHECK_EQ(countOf(render(body), "simdgroup_load"), 3);
  }

  CASE("distinct K steps do not share a fragment");
  {
    msl::Context c;
    msl::Block body;
    std::vector<WarpSlot> slots = {
        {SlotCoord::fixed(0), SlotCoord::fixed(0), 0}};
    int counter = 0;
    emitDirectMma(c, body, slots, gemmInputs(4, 32, 8), nm, counter);
    const std::string out = render(body);
    CHECK_EQ(countOf(out, "simdgroup_load"), 8);
    CHECK_EQ(countOf(out, "simdgroup_multiply_accumulate"), 4);
  }

  // ── rolled K ───────────────────────────────────────────────────────────

  CASE("rolling K emits a single shared loop body");
  {
    msl::Context c;
    std::vector<WarpSlot> slots = {
        {SlotCoord::fixed(0), SlotCoord::fixed(0), 0},
        {SlotCoord::fixed(1), SlotCoord::fixed(0), 1}};

    msl::Block rolled;
    int c1 = 0;
    emitDirectMma(c, rolled, slots, gemmInputs(8, 64, 8, /*rollK=*/true), nm,
                  c1);
    msl::Block unrolled;
    int c2 = 0;
    emitDirectMma(c, unrolled, slots, gemmInputs(8, 64, 8, false), nm, c2);

    CHECK_EQ(countOf(render(rolled), "for ("), 1);
    CHECK(measure(rolled).fragDecls < measure(unrolled).fragDecls);
    CHECK_EQ(measure(rolled).fragDecls, 3);
    CHECK_EQ(measure(unrolled).fragDecls, 3 * 8);
  }

  CASE("the rolled loop counts in fragment steps to kT*8");
  {
    msl::Context c;
    msl::Block body;
    std::vector<WarpSlot> slots = {
        {SlotCoord::fixed(0), SlotCoord::fixed(0), 0}};
    int counter = 0;
    emitDirectMma(c, body, slots, gemmInputs(4, 32, 8, true), nm, counter);
    const std::string out = render(body);
    CHECK(out.find("kv < 32") != std::string::npos);
    CHECK(out.find("kv += 8") != std::string::npos);
  }

  CASE("the K term is the only difference between the two forms");
  {
    msl::Context c;
    std::vector<WarpSlot> slots = {
        {SlotCoord::fixed(1), SlotCoord::fixed(0), 0}};
    msl::Block rolled, unrolled;
    int c1 = 0, c2 = 0;
    emitDirectMma(c, rolled, slots, gemmInputs(2, 16, 8, true), nm, c1);
    emitDirectMma(c, unrolled, slots, gemmInputs(2, 16, 8, false), nm, c2);
    CHECK(render(rolled).find("kv") != std::string::npos);
    CHECK(render(unrolled).find("kv") == std::string::npos);
  }

  // ── accumulators and stores ────────────────────────────────────────────

  CASE("every slot gets one zeroed accumulator and one store");
  {
    msl::Context c;
    msl::Block body;
    WarpGrid g{2, 2, 4};
    WarpProgram p = planWarpProgram(g);
    emitDirectDot(c, body, p, g, gemmInputs(1, 8, 16),
                  TileView::rowMajor({16, 16}), nm, 0, {}, {});
    const std::string out = render(body);
    CHECK_EQ(countOf(out, kZeroAcc), countOf(out, "simdgroup_store"));
  }

  CASE("a constant slot's store address folds to a literal");
  {
    msl::Context c;
    msl::Block body;
    std::vector<WarpSlot> slots = {
        {SlotCoord::fixed(1), SlotCoord::fixed(1), 0}};
    emitAccumStores(c, body, slots, TileView::rowMajor({32, 32}), nm, 0);
    CHECK(render(body).find("pC + 264") != std::string::npos);
  }

  CASE("an affine slot's store address keeps the warp term");
  {
    msl::Context c;
    msl::Block body;
    std::vector<WarpSlot> slots = {
        {SlotCoord::affine(2, 1), SlotCoord::fixed(0), 0}};
    emitAccumStores(c, body, slots, TileView::rowMajor({32, 32}), nm, 0);
    CHECK(render(body).find("warp") != std::string::npos);
  }

  // ── the fused path is the same emitter ─────────────────────────────────

  CASE("a fused dot accumulates into accumulators declared outside its K loop");
  {
    msl::Context c;
    std::vector<WarpSlot> slots = {
        {SlotCoord::fixed(0), SlotCoord::fixed(0), 0},
        {SlotCoord::fixed(1), SlotCoord::fixed(0), 1}};

    msl::Block body;
    emitAccumDecls(c, body, slots, nm);
    const int declsBeforeLoop = (int)body.size();

    msl::Block step;
    int counter = 0;
    emitDirectMma(c, step, slots, gemmInputs(2, 16, 8), nm, counter);

    const std::string inner = render(step);
    CHECK_EQ(countOf(inner, kZeroAcc), 0);
    CHECK_EQ(countOf(render(body), kZeroAcc), 2);
    CHECK_EQ(declsBeforeLoop, 2);
    CHECK_EQ(countOf(inner, "simdgroup_multiply_accumulate"), 2 * 2);
  }

  CASE("a banded device operand reads band-relative rows");
  {
    OperandSource banded =
        operandFrom("dA", 64, OperandSource::FragAxis::Rows, /*bandFrags=*/2);
    CHECK_EQ(banded.rowOf(SlotCoord::fixed(0)).constant, 0);
    CHECK_EQ(banded.rowOf(SlotCoord::fixed(1)).constant, 1);
    CHECK_EQ(banded.rowOf(SlotCoord::fixed(2)).constant, 0);
    CHECK_EQ(banded.rowOf(SlotCoord::fixed(5)).constant, 1);

    OperandSource plain = operandFrom("pA", 64, OperandSource::FragAxis::Rows);
    CHECK_EQ(plain.rowOf(SlotCoord::fixed(5)).constant, 5);
  }

  CASE("two slots on one banded row share a fragment");
  {
    msl::Context c;
    msl::Block body;
    std::vector<WarpSlot> slots = {
        {SlotCoord::fixed(0), SlotCoord::fixed(0), 0},
        {SlotCoord::fixed(2), SlotCoord::fixed(1), 1}};
    DirectInputs in;
    in.a =
        operandFrom("dA", 64, OperandSource::FragAxis::Rows, /*bandFrags=*/2);
    in.b = operandFrom("pB", 16, OperandSource::FragAxis::Rows);
    in.kT = 1;
    int counter = 0;
    emitDirectMma(c, body, slots, in, nm, counter);
    CHECK_EQ(countOf(render(body), "simdgroup_load"), 3);
  }

  CASE("the two operands index different axes");
  {
    msl::Context c;
    msl::Block body;
    std::vector<WarpSlot> slots = {
        {SlotCoord::fixed(1), SlotCoord::fixed(1), 0}};
    DirectInputs in = gemmInputs(/*kT=*/2, /*lda=*/64, /*ldb=*/64);
    int counter = 0;
    emitDirectMma(c, body, slots, in, nm, counter);
    const std::string out = render(body);

    CHECK(out.find("pA + 512,") != std::string::npos);
    CHECK(out.find("pB + 8,") != std::string::npos);
    CHECK(out.find("pA + 520,") != std::string::npos);
    CHECK(out.find("pB + 520,") != std::string::npos);
  }

  CASE("banding does not fold an affine row");
  {
    // An affine row is already band-relative, so rowOf leaves it alone.
    OperandSource banded =
        operandFrom("dA", 64, OperandSource::FragAxis::Rows, /*bandFrags=*/2);
    CHECK(!banded.rowOf(SlotCoord::affine(2, 1)).isConst());
    CHECK_EQ(banded.rowOf(SlotCoord::affine(2, 1)).warpScale, 2);
  }

  // ── idle hardware warps ────────────────────────────────────────────────

  CASE("a launch with more warps than the program fences the extras off");
  {
    // An unfenced extra warp loads past the staged tile and stores past the
    // tensor.
    msl::Context c;
    msl::Block body;
    WarpGrid g{1, 2, 2};
    g.hwWarps = 4;
    CHECK(g.guardsIdleWarps());
    const WarpProgram p = planWarpProgram(g);
    emitDirectDot(c, body, p, g, gemmInputs(1, 16, 16),
                  TileView::rowMajor({8, 16}), nm, 0, {}, {});
    const std::string out = render(body);
    CHECK(countOf(out, "if (warp < 2)") > 0);
    const std::size_t guard = out.find("if (warp < 2)");
    CHECK(guard != std::string::npos);
    CHECK(guard < out.find("simdgroup_multiply_accumulate"));
    CHECK(guard < out.find("simdgroup_store"));
  }

  CASE("a launch the program fills needs no fence");
  {
    msl::Context c;
    msl::Block body;
    WarpGrid g{2, 2, 4};
    g.hwWarps = 4;
    CHECK(!g.guardsIdleWarps());
    const WarpProgram p = planWarpProgram(g);
    emitDirectDot(c, body, p, g, gemmInputs(1, 16, 16),
                  TileView::rowMajor({16, 16}), nm, 0, {}, {});
    CHECK_EQ(countOf(render(body), "if (warp <"), 0);
  }

  // ── the device drain ───────────────────────────────────────────────────

  CASE("a device store addresses the window through the leading dimension");
  {
    msl::Context c;
    msl::Block body;
    std::vector<WarpSlot> slots = {
        {SlotCoord::fixed(1), SlotCoord::affine(1, 0), 3}};
    DeviceStoreTarget t;
    t.base = "cptr";
    t.leadingDim = Stride::runtime("ldc");
    t.rowStart = c.var("rs");
    t.colStart = c.var("cs");
    emitAccumDeviceStores(c, body, slots, t, {}, nm);
    const std::string out = render(body);
    CHECK_EQ(countOf(out, "simdgroup_store(acc3,"), 1);
    CHECK(out.find("rs + 8") != std::string::npos);
    CHECK(out.find("* ldc") != std::string::npos);
    CHECK(out.find("cs + warp * 8") != std::string::npos);
    CHECK(out.find(", ldc)") != std::string::npos);
    CHECK_EQ(countOf(out, "pC"), 0);
  }

  CASE("a window at the tensor's corner emits no start terms");
  {
    msl::Context c;
    msl::Block body;
    std::vector<WarpSlot> slots = {
        {SlotCoord::fixed(0), SlotCoord::fixed(2), 0}};
    DeviceStoreTarget t;
    t.base = "cptr";
    t.leadingDim = Stride::runtime("ldc");
    emitAccumDeviceStores(c, body, slots, t, {}, nm);
    const std::string out = render(body);
    CHECK(out.find("simdgroup_store(acc0, cptr + 16, ldc)") !=
          std::string::npos);
  }

  // ── the drain's folded epilogue and bounds ─────────────────────────────

  CASE("the lane mapping matches the measured hardware table");
  {
    static const int64_t grid[8][8] = {
        {0, 0, 1, 1, 8, 8, 9, 9},         {2, 2, 3, 3, 10, 10, 11, 11},
        {4, 4, 5, 5, 12, 12, 13, 13},     {6, 6, 7, 7, 14, 14, 15, 15},
        {16, 16, 17, 17, 24, 24, 25, 25}, {18, 18, 19, 19, 26, 26, 27, 27},
        {20, 20, 21, 21, 28, 28, 29, 29}, {22, 22, 23, 23, 30, 30, 31, 31}};
    for (int64_t lane = 0; lane < 32; ++lane)
      for (int64_t i = 0; i < kFragElemsPerLane; ++i) {
        CHECK_EQ(grid[fragLaneRow(lane)][fragLaneCol(lane, i)], lane);
        CHECK_EQ(fragLaneCol(lane, i) % 2, i);
      }
  }

  CASE("a folded step runs on thread_elements before the store");
  {
    msl::Context c;
    msl::Block body;
    std::vector<WarpSlot> slots = {
        {SlotCoord::fixed(0), SlotCoord::fixed(0), 0}};
    DeviceStoreTarget t;
    t.base = "cptr";
    t.leadingDim = Stride::runtime("ldc");
    t.colStart = c.var("cs");
    DrainStep bias;
    bias.op = "arith.addf";
    bias.operand.kind = DrainOperand::Kind::Row;
    bias.operand.base = "bptr";
    emitAccumDeviceStores(c, body, slots, t, {bias}, nm);
    const std::string out = render(body);
    CHECK_EQ(countOf(out, "acc0.thread_elements()[0] ="), 1);
    CHECK_EQ(countOf(out, "acc0.thread_elements()[1] ="), 1);
    CHECK_EQ(countOf(out, "bptr["), 2);
    CHECK_EQ(countOf(out, "simdgroup_store(acc0,"), 1);
    CHECK(out.find("* ldc]") == std::string::npos);
  }

  CASE("a bounded store guards per fragment and falls back to scalars");
  {
    msl::Context c;
    msl::Block body;
    std::vector<WarpSlot> slots = {
        {SlotCoord::fixed(0), SlotCoord::fixed(1), 1}};
    DeviceStoreTarget t;
    t.base = "cptr";
    t.leadingDim = Stride::runtime("ldc");
    t.rowStart = c.var("rs");
    t.colStart = c.var("cs");
    t.rowBound = c.lit(16384);
    t.colBound = c.lit(30522);
    emitAccumDeviceStores(c, body, slots, t, {}, nm);
    const std::string out = render(body);
    CHECK_EQ(countOf(out, "simdgroup_store(acc1,"), 1);
    CHECK(out.find("+ 8 <= 16384") != std::string::npos);
    CHECK(out.find("+ 8 <= 30522") != std::string::npos);
    CHECK(out.find("else") != std::string::npos);
    CHECK_EQ(countOf(out, "cptr["), 2);
    CHECK_EQ(countOf(out, "< 16384"), 2);
    CHECK_EQ(countOf(out, "< 30522"), 2);
    CHECK_EQ(countOf(out, "threadgroup_barrier"), 0);
    CHECK(out.find("lane >> 1") != std::string::npos);
    CHECK(out.find("lane >> 4") != std::string::npos);
    CHECK(out.find("lane >> 3") != std::string::npos);
  }

  CASE("a splat scale and a residual tile fold as steps");
  {
    msl::Context c;
    msl::Block body;
    std::vector<WarpSlot> slots = {
        {SlotCoord::fixed(0), SlotCoord::fixed(0), 0}};
    DeviceStoreTarget t;
    t.base = "cptr";
    t.leadingDim = Stride::runtime("ldc");
    DrainStep scale;
    scale.op = "arith.mulf";
    scale.operand.kind = DrainOperand::Kind::Splat;
    scale.operand.splat = c.var("alpha");
    DrainStep residual;
    residual.op = "arith.addf";
    residual.operand.kind = DrainOperand::Kind::Tile;
    residual.operand.base = "rptr";
    residual.operand.leadingDim = Stride::runtime("ldr");
    DrainStep relu;
    relu.op = "arith.maxnumf";
    relu.operand.kind = DrainOperand::Kind::Splat;
    relu.operand.splat = c.litF(0.0);
    emitAccumDeviceStores(c, body, slots, t, {scale, residual, relu}, nm);
    const std::string out = render(body);
    CHECK_EQ(countOf(out, "* alpha"), 2);
    CHECK_EQ(countOf(out, "rptr["), 2);
    CHECK_EQ(countOf(out, "metal::max("), 2);
    CHECK(out.find("* ldr") != std::string::npos);
    CHECK_EQ(countOf(out, "simdgroup_store(acc0,"), 1);
  }

  CASE("a memoised operand read is declared at the operand's element");
  {
    // Declared `float` over a `half` operand, the whole-tile arm promotes to
    // f32 while the edge arm stays f16, so two elements of one tensor round
    // differently by whether their tile was whole.
    msl::Context c;
    msl::Block body;
    std::vector<WarpSlot> slots = {
        {SlotCoord::fixed(0), SlotCoord::fixed(0), 0},
        {SlotCoord::fixed(1), SlotCoord::fixed(0), 1}};
    DeviceStoreTarget t;
    t.base = "cptr";
    t.leadingDim = Stride::runtime("ldc");
    t.elem = f16();
    t.rowBound = c.lit(4096);
    t.tileRows = 16;
    t.tileCols = 8;
    DrainStep bias;
    bias.op = "arith.addf";
    bias.operand.kind = DrainOperand::Kind::Row;
    bias.operand.base = "bptr";
    bias.operand.elem = f16();
    bias.roundBefore = true;
    emitAccumDeviceStores(c, body, slots, t, {bias}, nm);
    const std::string out = render(body);
    CHECK(out.find("half fr0 = bptr[") != std::string::npos);
    CHECK(out.find("float fr0") == std::string::npos);
    CHECK_EQ(countOf(out, "+ fr0)"), 2);
  }

  CASE("a narrowing target stores through a fragment of the tensor's type");
  {
    msl::Context c;
    msl::Block body;
    std::vector<WarpSlot> slots = {
        {SlotCoord::fixed(0), SlotCoord::fixed(0), 0}};
    DeviceStoreTarget t;
    t.base = "cptr";
    t.leadingDim = Stride::runtime("ldc");
    t.elem = f16();
    t.colBound = c.lit(100);
    emitAccumDeviceStores(c, body, slots, t, {}, nm);
    const std::string out = render(body);
    CHECK(out.find("simdgroup_half8x8 acc0n") != std::string::npos);
    CHECK_EQ(countOf(out, "acc0n.thread_elements()"), 2);
    CHECK_EQ(countOf(out, "simdgroup_store(acc0n,"), 1);
    CHECK_EQ(countOf(out, "(half)"), 4);
    CHECK_EQ(countOf(out, "simdgroup_store(acc0,"), 0);
  }

  CASE("a relu splat over a narrowing target keeps the target's type");
  {
    msl::Context c;
    msl::Block body;
    std::vector<WarpSlot> slots = {
        {SlotCoord::fixed(0), SlotCoord::fixed(0), 0}};
    DeviceStoreTarget t;
    t.base = "cptr";
    t.leadingDim = Stride::runtime("ldc");
    t.elem = f16();
    DrainStep relu;
    relu.op = "arith.maximumf";
    relu.operand.kind = DrainOperand::Kind::Splat;
    relu.operand.splat = c.litF(0.0, mslTypeOf(f16()));
    emitAccumDeviceStores(c, body, slots, t, {relu}, nm);
    const std::string out = render(body);
    // `metal::max(half, 0.0f)` matches neither overload.
    CHECK_EQ(countOf(out, ", 0.0f)"), 0);
    CHECK_EQ(countOf(out, "metal::max("), 2);
  }

  CASE("a NaN-propagating relu folds with the guarded spelling");
  {
    msl::Context c;
    msl::Block body;
    std::vector<WarpSlot> slots = {
        {SlotCoord::fixed(0), SlotCoord::fixed(0), 0}};
    DeviceStoreTarget t;
    t.base = "cptr";
    t.leadingDim = Stride::runtime("ldc");
    DrainStep relu;
    relu.op = "arith.maximumf";
    relu.operand.kind = DrainOperand::Kind::Splat;
    relu.operand.splat = c.litF(0.0);
    emitAccumDeviceStores(c, body, slots, t, {relu}, nm);
    const std::string out = render(body);
    // A bare metal::max drops a NaN accumulator, hence the guarded ternary.
    CHECK_EQ(countOf(out, "metal::max("), 2);
    CHECK_EQ(countOf(out, "metal::isnan("), 4);
    CHECK_EQ(countOf(out, "?"), 2);
  }

  CASE("a branch hangs off the running value its base names");
  {
    // acc + 2, then (acc + 2) * ((acc + 2) * 3): with branchBase 1 the
    // branch repeats the addition; with 0 it starts from the bare element.
    const auto render_with = [&](int base) {
      msl::Context c;
      msl::Block body;
      std::vector<WarpSlot> slots = {
          {SlotCoord::fixed(0), SlotCoord::fixed(0), 0}};
      DeviceStoreTarget t;
      t.base = "cptr";
      t.leadingDim = Stride::runtime("ldc");
      DrainStep bias;
      bias.op = "arith.addf";
      bias.operand.kind = DrainOperand::Kind::Splat;
      bias.operand.splat = c.litF(2.0);
      DrainStep prod;
      prod.op = "arith.mulf";
      prod.operand.kind = DrainOperand::Kind::AccChain;
      prod.branchBase = base;
      DrainBranchLink scale;
      scale.op = "arith.mulf";
      scale.operand.kind = DrainOperand::Kind::Splat;
      scale.operand.splat = c.litF(3.0);
      prod.branch.push_back(scale);
      emitAccumDeviceStores(c, body, slots, t, {bias, prod}, nm);
      return render(body);
    };
    const std::string rerooted = render_with(1);
    const std::string rooted = render_with(0);
    CHECK_EQ(countOf(rerooted, "2.0f"), 4);
    CHECK_EQ(countOf(rooted, "2.0f"), 2);
    CHECK_EQ(countOf(rerooted, "3.0f"), 2);
    CHECK_EQ(countOf(rooted, "3.0f"), 2);
  }

  // ── the banded C readback ──────────────────────────────────────────────

  CASE("a banded dot slices C into row bands that reuse one region");
  {
    // Each band stores band-relative, both starting at pC and reads back
    // between barriers, so the second band reuses the first's region.
    msl::Context c;
    msl::Block body;
    WarpGrid g{2, 4, 4};
    g.mT = 4;
    g.nT = 2;
    g.bandedC = true;
    const WarpProgram p = planWarpProgram(g);

    int calls = 0;
    auto readbackFor = [&](const Range &rows) {
      ++calls;
      ReadbackInputs back;
      back.actions.push_back(
          StageAction{0, 1, false, {0, 0}, CoordGuard::unguarded()});
      back.names.push_back("r" + std::to_string(rows.lo));
      back.bases.push_back("");
      return back;
    };
    CoordSource cs;
    LayoutBasis row, col;
    row.lane = {1, 2, 4, 8, 16};
    cs.dims = {row, col};
    emitDirectDot(c, body, p, g, gemmInputs(1, 16, 16),
                  TileView::rowMajor({32, 16}), nm, /*bandRows=*/16,
                  readbackFor, cs);
    const std::string out = render(body);

    CHECK_EQ(calls, 2);
    CHECK_EQ(countOf(out, "simdgroup_store(acc0, pC,"), 1);
    CHECK_EQ(countOf(out, "pC + 128, 16"), 2);
    CHECK(out.find("r0 =") != std::string::npos);
    CHECK(out.find("r16 =") != std::string::npos);
    CHECK_EQ(countOf(out, "threadgroup_barrier"), 3);
  }

  CASE("a whole-tile band is today's single pass, unfiltered");
  {
    msl::Context c;
    msl::Block whole, zero;
    WarpGrid g{2, 2, 4};
    const WarpProgram p = planWarpProgram(g);
    emitDirectDot(c, whole, p, g, gemmInputs(1, 16, 16),
                  TileView::rowMajor({16, 16}), nm, /*bandRows=*/16, {}, {});
    emitDirectDot(c, zero, p, g, gemmInputs(1, 16, 16),
                  TileView::rowMajor({16, 16}), nm, /*bandRows=*/0, {}, {});
    CHECK(render(whole) == render(zero));
    CHECK_EQ(countOf(render(whole), "threadgroup_barrier"), 0);
  }

  CASE("a banded grid never takes a warp-dependent row");
  {
    // A banded C is drained by compile-time row, so under bandedC only covers
    // whose warp axis is all-N qualify.
    WarpGrid g{8, 2, 4, /*aDirect=*/true};
    const WarpProgram open = planWarpProgram(g);
    CHECK(open.form == WarpForm::Parameterised);
    CHECK(open.miCount < g.mT);
    g.bandedC = true;
    const WarpProgram banded = planWarpProgram(g);
    for (int64_t w = 0; w < g.numWarps; ++w)
      for (const WarpSlot &s : banded.slots(w, g.mT, g.nT, g.numWarps))
        CHECK(s.mi.isConst());
  }

  return ::agpu_test::report("EmitDirect");
}
