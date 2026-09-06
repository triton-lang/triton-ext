#include "agpu/msl/FuncSize.h"
#include "agpu/msl/GuardFuse.h"
#include "agpu/msl/Printer.h"
#include "fixtures.h"
#include "harness.h"

#include <sstream>

using namespace agpu::msl;
using agpu_test::countOf;

namespace {

std::string render(const Block &b) {
  std::ostringstream os;
  Printer p(os);
  p.printBlock(b);
  return os.str();
}

FuncSize sized(int64_t decls, int64_t frags, int64_t branches = 0) {
  FuncSize s;
  s.decls = decls;
  s.fragDecls = frags;
  s.branches = branches;
  return s;
}

Block guardRun(Context &c, int n, const char *cond = "p") {
  Block b;
  for (int i = 0; i < n; ++i)
    b.push_back(
        c.ifStmt(c.var(cond),
                 Block{c.assign(c.var("x" + std::to_string(i)), c.lit(1))}));
  return b;
}

} // namespace

int main() {
  // ── the policy ─────────────────────────────────────────────────────────

  CASE("fusing is never rationed");
  {
    CHECK(planShrink(sized(10, 0, /*branches=*/1)).fuseGuards);
    CHECK(planShrink(sized(999999, 999999, 1)).fuseGuards);
    CHECK(!planShrink(sized(999999, 999999, 0)).fuseGuards);
  }

  CASE("rolling needs both an over-budget function and enough fragments");
  {
    CHECK(planShrink(sized(kDeclBudget + 1, kRollFragFloor)).rollKSteps);
    CHECK(!planShrink(sized(kDeclBudget + 1, kRollFragFloor - 1)).rollKSteps);
    CHECK(!planShrink(sized(kDeclBudget, kRollFragFloor * 10)).rollKSteps);
  }

  CASE("a function inside the budget needs no roll");
  {
    CHECK(withinBudget(sized(kDeclBudget, 0)));
    CHECK(!withinBudget(sized(kDeclBudget + 1, 0)));
    CHECK(!planShrink(sized(100, 100)).rollKSteps);
  }

  CASE("only rolling forces a re-emit");
  {
    ShrinkPlan roll = planShrink(sized(kDeclBudget + 1, kRollFragFloor));
    CHECK(roll.needsReemit());
    ShrinkPlan fuse = planShrink(sized(10, 0, 1));
    CHECK(fuse.any());
    CHECK(!fuse.needsReemit());
  }

  CASE("whether the re-emit helped is checked directly");
  {
    CHECK(shrinkHelped(sized(20000, 5000), sized(9000, 500)));
    CHECK(!shrinkHelped(sized(20000, 5000), sized(20000, 5000)));
  }

  // ── the cost model ─────────────────────────────────────────────────────

  CASE("a run of one is never fused");
  {
    FuseCost cost{8};
    CHECK(!cost.worthFusing(1));
    CHECK(!cost.worthFusing(0));
  }

  CASE("fusing pays as soon as it removes more than it adds");
  {
    FuseCost cost{8};
    CHECK(cost.worthFusing(2));
    CHECK(cost.worthFusing(8));
  }

  CASE("a longer condition makes fusing pay sooner");
  {
    FuseCost wide{40};
    FuseCost narrow{1};
    CHECK(wide.perStatement() > narrow.perStatement());
  }

  // ── fusing ─────────────────────────────────────────────────────────────

  CASE("a run of identical guards becomes one scope");
  {
    Context c;
    Block body = guardRun(c, 4);
    const int64_t fused = fuseGuards(c, body);
    CHECK_EQ(fused, 3);
    CHECK_EQ(body.size(), 1u);
    const std::string out = render(body);
    CHECK_EQ(countOf(out, "if (p)"), 1);
    CHECK_EQ(countOf(out, "x0 = 1;"), 1);
    CHECK_EQ(countOf(out, "x3 = 1;"), 1);
  }

  CASE("different conditions are not fused together");
  {
    Context c;
    Block body;
    body.push_back(c.ifStmt(c.var("p"), Block{c.assign(c.var("a"), c.lit(1))}));
    body.push_back(c.ifStmt(c.var("q"), Block{c.assign(c.var("b"), c.lit(1))}));
    fuseGuards(c, body);
    CHECK_EQ(body.size(), 2u);
  }

  CASE("conditions are compared structurally");
  {
    Context c;
    Block body;
    for (int i = 0; i < 4; ++i)
      body.push_back(
          c.ifStmt(c.binary(BinOp::Lt, c.var("a"), c.var("b")),
                   Block{c.assign(c.var("x" + std::to_string(i)), c.lit(1))}));
    fuseGuards(c, body);
    CHECK_EQ(body.size(), 1u);
  }

  // ── the safety condition ───────────────────────────────────────────────

  CASE("a body that assigns the condition's name breaks the run");
  {
    // `if (p) p = 0; if (p) x = 1;` fusing changes what the second test sees.
    Context c;
    Block body;
    body.push_back(c.ifStmt(c.var("p"), Block{c.assign(c.var("y"), c.lit(1))}));
    body.push_back(c.ifStmt(c.var("p"), Block{c.assign(c.var("p"), c.lit(0))}));
    body.push_back(c.ifStmt(c.var("p"), Block{c.assign(c.var("z"), c.lit(1))}));
    body.push_back(c.ifStmt(c.var("p"), Block{c.assign(c.var("w"), c.lit(1))}));
    fuseGuards(c, body);
    CHECK(body.size() > 1u);
  }

  CASE("a nested assignment to the condition also breaks the run");
  {
    Context c;
    Block body;
    for (int i = 0; i < 2; ++i)
      body.push_back(
          c.ifStmt(c.var("p"), Block{c.assign(c.var("a"), c.lit(1))}));
    body.push_back(c.ifStmt(
        c.var("p"), Block{c.scope(Block{c.assign(c.var("p"), c.lit(0))})}));
    CHECK_EQ(fusableRun(body, 0), 2u);
  }

  CASE("a declaration shadowing the condition breaks the run too");
  {
    Context c;
    Block body;
    body.push_back(c.ifStmt(c.var("p"), Block{c.assign(c.var("a"), c.lit(1))}));
    body.push_back(
        c.ifStmt(c.var("p"), Block{c.declStmt(Context::i32(), "p", c.lit(0))}));
    CHECK_EQ(fusableRun(body, 0), 1u);
  }

  CASE("a condition containing a call is never fused");
  {
    Context c;
    Block body;
    for (int i = 0; i < 4; ++i)
      body.push_back(
          c.ifStmt(c.binary(BinOp::Gt, c.call("f", {c.var("x")}), c.lit(0)),
                   Block{c.assign(c.var("a" + std::to_string(i)), c.lit(1))}));
    CHECK_EQ(fusableRun(body, 0), 0u);
    CHECK_EQ(fuseGuards(c, body), 0);
    CHECK_EQ(body.size(), 4u);

    CHECK(namesRead(c.call("f", {c.var("x")})).opaque);
    CHECK(!namesRead(c.binary(BinOp::Gt, c.var("x"), c.lit(0))).opaque);
  }

  CASE("a body containing a bare call breaks the run");
  {
    Context c;
    Block body;
    body.push_back(c.ifStmt(c.var("p"), Block{c.assign(c.var("a"), c.lit(1))}));
    body.push_back(
        c.ifStmt(c.var("p"), Block{c.exprStmt(c.call("g", {c.var("q")}))}));
    body.push_back(c.ifStmt(c.var("p"), Block{c.assign(c.var("b"), c.lit(1))}));
    CHECK_EQ(fusableRun(body, 0), 1u);
  }

  CASE("a body containing a barrier breaks the run");
  {
    // A barrier in divergent control flow is undefined.
    Context c;
    Block body;
    body.push_back(c.ifStmt(c.var("p"), Block{c.assign(c.var("a"), c.lit(1))}));
    body.push_back(c.ifStmt(c.var("p"), Block{c.barrier()}));
    CHECK_EQ(fusableRun(body, 0), 1u);
  }

  CASE("a write through a pointer breaks the run");
  {
    Context c;
    Block body;
    body.push_back(c.ifStmt(c.var("p"), Block{c.assign(c.var("a"), c.lit(1))}));
    body.push_back(
        c.ifStmt(c.var("p"), Block{c.assign(c.deref(c.var("q")), c.lit(1))}));
    CHECK_EQ(fusableRun(body, 0), 1u);
  }

  CASE("an ordinary run still fuses, so the safety rules are not a veto");
  {
    Context c;
    Block body;
    for (int i = 0; i < 4; ++i)
      body.push_back(
          c.ifStmt(c.var("p"),
                   Block{c.assign(c.var("a" + std::to_string(i)), c.lit(1))}));
    CHECK_EQ(fusableRun(body, 0), 4u);
    fuseGuards(c, body);
    CHECK_EQ(body.size(), 1u);
  }

  CASE("an else arm is never fused");
  {
    Context c;
    Block body;
    body.push_back(c.ifStmt(c.var("p"), Block{c.assign(c.var("a"), c.lit(1))}));
    body.push_back(c.ifElse(c.var("p"), Block{c.assign(c.var("b"), c.lit(1))},
                            Block{c.assign(c.var("d"), c.lit(1))}));
    CHECK_EQ(fusableRun(body, 0), 1u);
  }

  // ── recursion ──────────────────────────────────────────────────────────

  CASE("guards inside a loop are fused as well");
  {
    Context c;
    Block inner = guardRun(c, 4);
    Block body;
    body.push_back(c.forStmt(c.declStmt(Context::i32(), "i", c.lit(0)),
                             c.binary(BinOp::Lt, c.var("i"), c.lit(8)),
                             c.assignOp(BinOp::Add, c.var("i"), c.lit(1)),
                             std::move(inner)));
    const int64_t fused = fuseGuards(c, body);
    CHECK_EQ(fused, 3);
    CHECK_EQ(countOf(render(body), "if (p)"), 1);
  }

  CASE("guards inside both arms of an if are fused");
  {
    Context c;
    Block body;
    body.push_back(c.ifElse(c.var("q"), guardRun(c, 4), guardRun(c, 4)));
    CHECK_EQ(fuseGuards(c, body), 6);
  }

  CASE("fusing does not change what the block computes");
  {
    Context c;
    Block body = guardRun(c, 5);
    fuseGuards(c, body);
    const std::string out = render(body);
    for (int i = 0; i < 5; ++i)
      CHECK_EQ(countOf(out, "x" + std::to_string(i) + " = 1;"), 1);
  }

  // ── the verdict ────────────────────────────────────────────────────────

  CASE("a mitigated kernel and an unfixable one do not read alike");
  {
    const FuncSize under = sized(kDeclBudget - 1, 0);
    const FuncSize over = sized(kDeclBudget + 1, 0);

    CHECK(verdictOf(under, /*reemitted=*/false) == SizeVerdict::Fine);
    CHECK(verdictOf(under, /*reemitted=*/true) == SizeVerdict::Shrunk);
    // Still over after the re-walk.
    CHECK(verdictOf(over, /*reemitted=*/true) == SizeVerdict::Exposed);
    CHECK(verdictOf(over, /*reemitted=*/false) == SizeVerdict::Exposed);
  }

  CASE("the report says why an over-budget kernel was left alone");
  {
    const FuncSize s = sized(kDeclBudget + 500, kRollFragFloor - 1);
    const ShrinkPlan p = planShrink(s);
    CHECK(!p.rollKSteps);

    const std::string r = budgetReport("kern", s, p, /*reemitted=*/false);
    CHECK(r.find("EXPOSED") != std::string::npos);
    CHECK(r.find("below frag floor") != std::string::npos);
  }

  CASE("a healthy kernel's report carries no alarm and no excuse");
  {
    const FuncSize s = sized(12, 0);
    const std::string r = budgetReport("kern", s, planShrink(s), false);
    CHECK(r.find("fine") != std::string::npos);
    CHECK(r.find("EXPOSED") == std::string::npos);
    CHECK(r.find("frag floor") == std::string::npos);
  }

  return ::agpu_test::report("FuncSize");
}
