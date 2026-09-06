// End-to-end: plan -> guard -> address -> AST -> MSL text.
#include "agpu/emit/Emit.h"
#include "agpu/emit/EmitDirect.h"
#include "agpu/msl/Printer.h"
#include "fixtures.h"
#include "harness.h"
#include "render.h"

#include <sstream>

using namespace agpu;
using agpu_test::countOf;
using agpu_test::render;

namespace {

// Rows come from the lane id's low bits, columns from its high bits.
CoordSource twoDim() {
  CoordSource s;
  LayoutBasis row, col;
  row.lane = {1, 2, 0, 0, 0};
  col.lane = {0, 0, 4, 8, 16};
  s.dims = {row, col};
  return s;
}

} // namespace

int main() {
  const int ROW = 0, COL = 1;

  CASE("a renamed readback adds the incoming accumulator like a pool one");
  {
    msl::Context c;
    ReadbackPlan plan;
    plan.kind = ReadbackPlan::Kind::Rename;
    plan.regs = {{0, 0}, {0, 1}, {1, 0}};
    msl::SmallVec<msl::Str, 8> names{"c0", "c1", "c2"};
    DirectNames nm;

    msl::Block body;
    emitFragmentReadback(c, body, plan, names, {}, f32(), directAccName(nm));
    const std::string assigned = render(body);
    CHECK(assigned.find("c0 = acc0.thread_elements()[0];") !=
          std::string::npos);
    CHECK(assigned.find("c2 = acc1.thread_elements()[0];") !=
          std::string::npos);
    CHECK(assigned.find("+") == std::string::npos);

    msl::Block added;
    msl::SmallVec<msl::Str, 8> bases{"in0", "in1", "in2"};
    emitFragmentReadback(c, added, plan, names, bases, f32(),
                         directAccName(nm));
    const std::string out = render(added);
    CHECK(out.find("c0 = acc0.thread_elements()[0] + in0;") !=
          std::string::npos);
    CHECK(out.find("c1 = acc0.thread_elements()[1] + in1;") !=
          std::string::npos);
    CHECK(out.find("c2 = acc1.thread_elements()[0] + in2;") !=
          std::string::npos);
  }

  CASE("a register wholly inside the panel stages unguarded");
  {
    msl::Context c;
    CoordSource cs = twoDim();
    TileView pa = TileView::rowMajor({32, 32});
    auto act = planStage(0, {{ROW, 0, 7}, {COL, 0, 31}},
                         {{ROW, 0, 32}, {COL, 0, 32}}, {0, 0});
    CHECK(act.has_value());
    CHECK(act->guard.isUnguarded());

    msl::Block body;
    msl::SmallVec<StageAction, 8> actions{*act};
    msl::SmallVec<msl::Str, 8> names{"v0"};
    emitStage(c, body, pa, "pA", actions, names, cs, f16());
    // The address is per-lane: which lane holds register 0 decides which
    // slot it stages to.
    const std::string out = render(body);
    CHECK(out.find("if (") == std::string::npos);
    CHECK(out.find("pA[") != std::string::npos);
    CHECK(out.find("= v0;") != std::string::npos);
    CHECK(out.find("lane") != std::string::npos);
  }

  CASE("a straddling register stages under exactly the terms it needs");
  {
    msl::Context c;
    CoordSource cs = twoDim();
    TileView pa = TileView::rowMajor({32, 32});
    auto act = planStage(3, {{ROW, 8, 15}, {COL, 0, 63}},
                         {{ROW, 0, 32}, {COL, 0, 32}}, {8, 0});
    CHECK(act.has_value());
    CHECK_EQ(act->guard.terms().size(), 1u);

    msl::Block body;
    emitStage(c, body, pa, "pA", {*act}, {"v0", "v1", "v2", "v3"}, cs, f16());
    const std::string out = render(body);
    CHECK(out.find("if ((lane & 28) < 32) pA[") != std::string::npos);
    CHECK(out.find("= v3;") != std::string::npos);
    CHECK_EQ(countOf(out, ">="), 0); // no lower bound needed
  }

  CASE("a register outside the panel is never planned");
  {
    auto act = planStage(7, {{ROW, 40, 47}, {COL, 0, 31}},
                         {{ROW, 0, 32}, {COL, 0, 32}}, {40, 0});
    CHECK(!act.has_value());
  }

  CASE("both bounds needed emits both terms");
  {
    msl::Context c;
    CoordSource cs = twoDim();
    TileView pa = TileView::rowMajor({16, 32});
    auto act = planStage(0, {{ROW, 0, 63}, {COL, 0, 31}},
                         {{ROW, 16, 32}, {COL, 0, 32}}, {0, 0});
    CHECK(act.has_value());
    CHECK_EQ(act->guard.terms().size(), 2u);
    msl::Block body;
    emitStage(c, body, pa, "pA", {*act}, {"v0"}, cs, f16());
    CHECK(render(body).find("if ((lane & 3) >= 16 && (lane & 3) < 32) pA[") !=
          std::string::npos);
  }

  CASE("eight registers across a 32-row panel");
  {
    msl::Context c;
    CoordSource cs = twoDim();
    TileView pa = TileView::rowMajor({32, 32});

    msl::SmallVec<StageAction, 8> actions;
    msl::SmallVec<msl::Str, 8> names;
    for (int r = 0; r < 8; ++r) {
      names.push_back("v" + std::to_string(r));
      // Register r owns rows [8r, 8r+7], all 32 columns.
      if (auto a = planStage(r, {{ROW, 8 * r, 8 * r + 7}, {COL, 0, 31}},
                             {{ROW, 0, 32}, {COL, 0, 32}}, {8 * r, 0}))
        actions.push_back(*a);
    }
    // Registers 4..7 live in rows 32..63, outside the panel: four survive.
    CHECK_EQ(actions.size(), 4u);

    msl::Block body;
    emitStage(c, body, pa, "pA", actions, names, cs, f16());
    const std::string out = render(body);
    CHECK_EQ(countOf(out, "= v0;"), 1);
    CHECK_EQ(countOf(out, "= v3;"), 1);
    CHECK_EQ(countOf(out, "= v4;"), 0); // outside the panel
    CHECK_EQ(countOf(out, "if ("), 0);
  }

  CASE("a panel at (m0,k0) needs no subtraction at the use site");
  {
    msl::Context c;
    CoordSource cs = twoDim();
    // A is 128x64; the panel is rows 64..95, cols 32..63.
    TileView aFull = TileView::rowMajor({128, 64});
    TileView panel = aFull.subview({64, 32}, {32, 32});
    CHECK_EQ(panel.origin(), 64 * 64 + 32);
    msl::Block body;
    auto act = planStage(0, {{ROW, 64, 71}, {COL, 32, 63}},
                         {{ROW, 64, 96}, {COL, 32, 64}}, {0, 0});
    CHECK(act.has_value());
    emitStage(c, body, panel, "pA", {*act}, {"v0"}, cs, f16());
    CHECK(render(body).find("pA[4128 + ") != std::string::npos);
  }

  CASE("a batch slice removes the rank test entirely");
  {
    msl::Context c;
    CoordSource cs = twoDim();
    // C is 4 x 16 x 32. Emitting batch 2.
    TileView c3 = TileView::rowMajor({4, 16, 32});
    TileView c2 = c3.slice(2);
    CHECK_EQ(c2.rank(), 2);
    auto act = planStage(0, {{ROW, 0, 7}, {COL, 0, 31}},
                         {{ROW, 0, 16}, {COL, 0, 32}}, {0, 0});
    CHECK(act.has_value());
    msl::Block body;
    emitStage(c, body, c2, "pC", {*act}, {"v0"}, cs, f16());
    // Batch 2 of a 4x16x32 tile starts at 2*16*32 = 1024.
    CHECK(render(body).find("pC[1024") != std::string::npos);
  }

  CASE("a batch filter, when the register spans batches, is just a window");
  {
    const int BATCH = 0;
    auto act = planStage(0, {{BATCH, 0, 3}}, {batchWindow(BATCH, 2)}, {0});
    CHECK(act.has_value());
    CHECK_EQ(act->guard.terms().size(), 2u);
  }

  CASE("a zero-origin store emits no offset expression");
  {
    msl::Context c;
    CoordSource cs = twoDim();
    TileView pa = TileView::rowMajor({32, 32});
    auto act = planStage(0, {{ROW, 0, 7}, {COL, 0, 31}},
                         {{ROW, 0, 32}, {COL, 0, 32}}, {0, 0});
    msl::Block body;
    emitStage(c, body, pa, "pA", {*act}, {"v0"}, cs, f16());
    const std::string out = render(body);
    CHECK(out.find("if (") == std::string::npos);
    CHECK(out.find("pA[") != std::string::npos);
    CHECK(out.find("= v0;") != std::string::npos);
    CHECK(out.find("lane") != std::string::npos);
  }

  CASE("a padded pool tile addresses through the padded stride");
  {
    msl::Context c;
    CoordSource cs = twoDim();
    // Four extra elements per row so consecutive rows change bank.
    TileView pa = TileView::rowMajorPadded({32, 32}, 4);
    auto act = planStage(1, {{ROW, 8, 15}, {COL, 0, 31}},
                         {{ROW, 0, 32}, {COL, 0, 32}}, {8, 0});
    msl::Block body;
    emitStage(c, body, pa, "pA", {*act}, {"v0", "v1"}, cs, f16());
    // The pad lives in the stride, so the row term scales by the padded 36.
    const std::string out = render(body);
    CHECK(out.find("* 36") != std::string::npos);
    CHECK(out.find("* 32") == std::string::npos);
  }

  CASE("the staging phase reads as a script");
  {
    msl::Context c;
    CoordSource cs = twoDim();
    TileView pa = TileView::rowMajor({16, 16});
    TileView pb = TileView::rowMajor({16, 16});

    msl::Block body;
    // Barrier order is the caller's: emitStage emits none.
    body.push_back(c.barrier());
    auto a0 = planStage(0, {{ROW, 0, 7}, {COL, 0, 15}},
                        {{ROW, 0, 16}, {COL, 0, 16}}, {0, 0});
    emitStage(c, body, pa, "pA", {*a0}, {"a0"}, cs, f16());
    body.push_back(c.barrier());
    auto b0 = planStage(0, {{ROW, 0, 7}, {COL, 0, 15}},
                        {{ROW, 0, 16}, {COL, 0, 16}}, {0, 0});
    emitStage(c, body, pb, "pB", {*b0}, {"b0"}, cs, f16());
    body.push_back(c.barrier());

    const std::string out = render(body);
    CHECK_EQ(countOf(out, "threadgroup_barrier"), 3);
    CHECK(out.find("= a0;") < out.find("= b0;"));
    CHECK(out.find("pA[") < out.find("pB["));
  }

  CASE("a block-distributed layout reaches emission with its block term");
  {
    // Without it every threadgroup computes the same address and the emitted
    // MSL still compiles.
    msl::Context c;
    CoordSource cs;
    LayoutBasis row;
    row.lane = {1, 2, 4, 0, 0};
    row.block = {8, 16, 32};
    cs.dims = {row};
    CHECK(cs.dims[0].needsBlockId());

    const std::string s =
        render(msl::Block{c.assign(c.var("out"), cs.of(c, 0, ROW))});
    CHECK(s.find("tgpos.x") != std::string::npos);

    const CoordRange r = cs.rangeOf(0, ROW, 256);
    CHECK_EQ(r.lo, 0);
    CHECK_EQ(r.hi, 63);
  }

  CASE("consecutive registers stage as one wide store when the plan merges");
  {
    msl::Context c;
    CoordSource cs = twoDim();
    cs.dims[1].reg = {1, 2}; // registers 0..3 are columns 0..3
    TileView pa = TileView::rowMajor({16, 16});

    msl::SmallVec<StageAction, 8> actions;
    for (int r = 0; r < 4; ++r)
      actions.push_back(*planStage(r, {{ROW, 0, 7}, {COL, 0, 15}},
                                   {{ROW, 0, 16}, {COL, 0, 16}}, {0, r}));
    const AccessPlan w = planStageRuns(actions, cs.dims, pa, 16);
    CHECK_EQ(w.width, 4);
    CHECK_EQ(actions.size(), 1u);
    CHECK_EQ(actions[0].width, 4);

    msl::Block body;
    emitStage(c, body, pa, "pA", actions, {"v0", "v1", "v2", "v3"}, cs, f16());
    const std::string out = render(body);
    // One store of the built vector, through the plain vector type: a
    // 16-column f16 tile's rows are 32-byte multiples, so the slot is
    // half4-aligned.
    CHECK_EQ(countOf(out, "pA["), 1);
    CHECK(out.find("threadgroup half4") != std::string::npos);
    CHECK(out.find("packed_half4") == std::string::npos);
    CHECK(out.find("half4(v0, v1, v2, v3)") != std::string::npos);
  }

  CASE("a guard on any register of the group blocks the merge");
  {
    msl::Context c;
    CoordSource cs = twoDim();
    cs.dims[1].reg = {1, 2};
    TileView pa = TileView::rowMajor({16, 16});

    msl::SmallVec<StageAction, 8> actions;
    for (int r = 0; r < 4; ++r)
      actions.push_back(*planStage(r, {{ROW, 0, 7}, {COL, 0, 15}},
                                   {{ROW, 0, 16}, {COL, 0, 16}}, {0, r}));
    // Register 2 straddles the window and stages under a test.
    actions[2] = *planStage(2, {{ROW, 0, 7}, {COL, 0, 31}},
                            {{ROW, 0, 16}, {COL, 0, 16}}, {0, 2});
    CHECK(actions[2].guard.needsTest());

    planStageRuns(actions, cs.dims, pa, 16);
    CHECK_EQ(actions.size(), 4u);
    for (const StageAction &a : actions)
      CHECK_EQ(a.width, 1);
  }

  CASE("a missing register blocks the merge, an aligned later group still "
       "goes wide");
  {
    msl::Context c;
    CoordSource cs = twoDim();
    cs.dims[1].reg = {1, 2, 4}; // registers 0..7 are columns 0..7
    TileView pa = TileView::rowMajor({16, 16});

    msl::SmallVec<StageAction, 8> actions;
    for (int r = 0; r < 8; ++r)
      if (r != 1) // register 1 fell to a dead guard
        actions.push_back(*planStage(r, {{ROW, 0, 7}, {COL, 0, 15}},
                                     {{ROW, 0, 16}, {COL, 0, 16}}, {0, r}));
    planStageRuns(actions, cs.dims, pa, 16);
    // Group 0..3 is broken by the missing register; group 4..7 merges.
    CHECK_EQ(actions.size(), 4u); // three scalars plus one wide
    CHECK_EQ(actions.back().reg, 4);
    CHECK_EQ(actions.back().width, 4);
  }

  CASE("a strided destination dimension refuses the merge");
  {
    // The registers walk rows: adjacent in the tensor, 16 slots apart in
    // the pool.
    msl::Context c;
    CoordSource cs = twoDim();
    cs.dims[0].reg = {1, 2};
    TileView pa = TileView::rowMajor({16, 16});
    msl::SmallVec<StageAction, 8> actions;
    for (int r = 0; r < 4; ++r)
      actions.push_back(*planStage(r, {{ROW, 0, 15}, {COL, 0, 15}},
                                   {{ROW, 0, 16}, {COL, 0, 16}}, {r, 0}));
    const AccessPlan w = planStageRuns(actions, cs.dims, pa, 16);
    CHECK_EQ(w.width, 1);
    CHECK_EQ(actions.size(), 4u);
  }

  CASE("a merged readback loads once and accumulates per lane");
  {
    msl::Context c;
    CoordSource cs = twoDim();
    cs.dims[1].reg = {1, 2};
    TileView pc = TileView::rowMajor({16, 16});

    msl::SmallVec<StageAction, 8> actions;
    for (int r = 0; r < 4; ++r)
      actions.push_back(*planStage(r, {{ROW, 0, 7}, {COL, 0, 15}},
                                   {{ROW, 0, 16}, {COL, 0, 16}}, {0, r}));
    planStageRuns(actions, cs.dims, pc, 32);
    CHECK_EQ(actions.size(), 1u);

    msl::Block body;
    emitReadback(c, body, pc, "pC", actions, {"o0", "o1", "o2", "o3"},
                 {"b0", "b1", "b2", "b3"}, cs, f32(), f32());
    const std::string out = render(body);
    // One pool read, through the plain vector type; the adds stay scalar,
    // one per destination register.
    CHECK_EQ(countOf(out, "pC["), 1);
    CHECK(out.find("threadgroup float4") != std::string::npos);
    CHECK(out.find("packed_float4") == std::string::npos);
    CHECK(out.find("o0 = o0_w[0] + b0;") != std::string::npos);
    CHECK(out.find("o3 = o0_w[3] + b3;") != std::string::npos);
  }

  CASE("an integer register converts the pooled float before the base add");
  {
    // A float add against an i32 base up to 2^31 is not exact, so the
    // loaded value must convert before the base add.
    msl::Context c;
    CoordSource cs = twoDim();
    TileView pc = TileView::rowMajor({16, 16});
    auto a0 = planStage(0, {{ROW, 0, 7}, {COL, 0, 15}},
                        {{ROW, 0, 16}, {COL, 0, 16}}, {0, 0});
    msl::Block body;
    emitReadback(c, body, pc, "pC", {*a0}, {"o0"}, {"b0"}, cs, f32(), i32());
    const std::string out = render(body);
    CHECK(out.find("(int)pC[") != std::string::npos);
    CHECK(out.find("+ b0") != std::string::npos);

    msl::Block plain;
    emitReadback(c, plain, pc, "pC", {*a0}, {"o0"}, {"b0"}, cs, f32(), f32());
    CHECK(render(plain).find("(int)") == std::string::npos);
  }

  return ::agpu_test::report("Emit");
}
