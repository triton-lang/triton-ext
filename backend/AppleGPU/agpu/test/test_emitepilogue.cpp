// The elementwise region folded into a dot.
#include "agpu/emit/EmitEpilogue.h"
#include "agpu/msl/Printer.h"
#include "harness.h"
#include "render.h"

#include <sstream>

using namespace agpu;
using agpu_test::render;

int main() {
  // ── the two lists agree ────────────────────────────────────────────────

  CASE("every recognised operation has a renderable expression");
  {
    // Over the whole table: a row added without a rendering fails here rather
    // than by passing a nullptr into declStmt.
    msl::Context c;
    for (const EpilogueBinary &e : kEpilogueBinary) {
      CHECK(isEpilogueOp(e.op));
      msl::Expr *r =
          epilogueExpr(c, EpilogueStep{e.op, c.var("b")}, c.var("acc"));
      CHECK(r != nullptr);
      CHECK(!render(r).empty());
    }
    for (const EpilogueUnary &e : kEpilogueUnary) {
      CHECK(isEpilogueOp(e.op));
      msl::Expr *r = epilogueExpr(c, EpilogueStep{e.op}, c.var("acc"));
      CHECK(r != nullptr);
      CHECK(!render(r).empty());
    }
  }

  CASE("an operation outside the table renders nothing");
  {
    msl::Context c;
    CHECK(!isEpilogueOp("tt.dot"));
    CHECK(epilogueExpr(c, EpilogueStep{"tt.dot", c.var("b")}, c.var("acc")) ==
          nullptr);
  }

  // ── the binaries ───────────────────────────────────────────────────────

  CASE("each binary renders its own operator");
  {
    msl::Context c;
    CHECK_EQ(render(epilogueExpr(c, {"arith.addf", c.var("b")}, c.var("a"))),
             std::string("a + b"));
    CHECK_EQ(render(epilogueExpr(c, {"arith.subf", c.var("b")}, c.var("a"))),
             std::string("a - b"));
    CHECK_EQ(render(epilogueExpr(c, {"arith.mulf", c.var("b")}, c.var("a"))),
             std::string("a * b"));
    CHECK_EQ(render(epilogueExpr(c, {"arith.divf", c.var("b")}, c.var("a"))),
             std::string("a / b"));
    CHECK_EQ(
        render(epilogueExpr(c, {"tt.precise_divf", c.var("b")}, c.var("a"))),
        std::string("a / b"));
  }

  CASE("a binary with no right operand asserts as a caller error");
  {
    msl::Context c;
    CHECK(isEpilogueBinary("arith.addf"));
    CHECK(epilogueExpr(c, EpilogueStep{"arith.addf"}, c.var("a")) == nullptr);
  }

  // ── the unaries ────────────────────────────────────────────────────────

  CASE("a unary renders its math function");
  {
    msl::Context c;
    // `abs` is also overloaded for integers, which makes a float argument
    // ambiguous, so the emitter spells it `fabs`.
    CHECK_EQ(render(epilogueExpr(c, {"math.absf"}, c.var("a"))),
             std::string("metal::fabs(a)"));
    CHECK_EQ(render(epilogueExpr(c, {"math.exp"}, c.var("a"))),
             std::string("metal::precise::exp(a)"));
  }

  CASE("fusable is decided by policy");
  {
    // Folding a transcendental into a K loop evaluates it once per step rather
    // than once per output. That is what fusableOnly filters for.
    CHECK(isEpilogueUnary("math.absf", /*fusableOnly=*/true));
    CHECK(!isEpilogueUnary("math.exp", /*fusableOnly=*/true));
    CHECK(isEpilogueUnary("math.exp", /*fusableOnly=*/false));
    msl::Context c;
    CHECK(epilogueExpr(c, {"math.exp"}, c.var("a")) != nullptr);
  }

  CASE("a costly step declines, so nothing is folded silently");
  {
    Decision cheap =
        epilogueDecision({{"arith.mulf", nullptr}, {"math.absf", nullptr}});
    CHECK(cheap.ok());

    Decision costly =
        epilogueDecision({{"arith.mulf", nullptr}, {"math.exp", nullptr}});
    CHECK(costly.isDecline());
    CHECK(!costly.isBug());

    Decision unknown = epilogueDecision({{"tt.dot", nullptr}});
    CHECK(unknown.isDecline());
  }

  // ── the chain ──────────────────────────────────────────────────────────

  CASE("a chain applies its steps in order, each reading the last");
  {
    msl::Context c;
    const std::vector<EpilogueStep> steps = {
        {"arith.mulf", c.var("alpha")},
        {"arith.addf", c.var("bias")},
    };
    CHECK_EQ(render(epilogueChain(c, steps, c.var("acc"))),
             std::string("acc * alpha + bias"));
  }

  CASE("a chain mixing arities threads through both");
  {
    msl::Context c;
    const std::vector<EpilogueStep> steps = {
        {"arith.mulf", c.var("alpha")},
        {"math.absf", nullptr},
        {"arith.addf", c.var("bias")},
    };
    CHECK_EQ(render(epilogueChain(c, steps, c.var("acc"))),
             std::string("metal::fabs(acc * alpha) + bias"));
  }

  CASE("one unrenderable step voids the whole chain");
  {
    msl::Context c;
    const std::vector<EpilogueStep> steps = {
        {"arith.mulf", c.var("alpha")},
        {"tt.dot", nullptr},
        {"arith.addf", c.var("bias")},
    };
    CHECK(epilogueChain(c, steps, c.var("acc")) == nullptr);
  }

  CASE("an empty chain is the accumulator itself");
  {
    msl::Context c;
    CHECK_EQ(render(epilogueChain(c, {}, c.var("acc"))), std::string("acc"));
  }

  return ::agpu_test::report("EmitEpilogue");
}
