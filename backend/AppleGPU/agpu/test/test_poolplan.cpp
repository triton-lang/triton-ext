// Whole-function pool accounting and the budget gate.
#include "agpu/plan/PoolPlan.h"
#include "harness.h"

using namespace agpu;

namespace {

PoolRequest req(const char *what, int64_t bytes) {
  return PoolRequest{what, Bytes(bytes)};
}

} // namespace

int main() {
  CASE("a function needs only the largest single request");
  {
    FunctionPool p = planFunctionPool(
        {req("dot", 16384), req("reduce", 4096), req("scan", 8192)});
    CHECK(p.scratch == Bytes(16384));
    CHECK_EQ(std::string(p.driver), std::string("dot"));
  }

  CASE("a function asking for nothing needs nothing");
  {
    FunctionPool p = planFunctionPool({});
    CHECK(p.scratch == Bytes(0));
    CHECK(p.total() == Bytes(0));
  }

  CASE("the driver is named, so an over-budget function says which operation");
  {
    FunctionPool p = planFunctionPool({req("reduce", 1024), req("dot", 40960)});
    CHECK_EQ(std::string(p.driver), std::string("dot"));
  }

  CASE("live buffers each add to the pool");
  {
    FunctionPool p = planFunctionPool({req("dot", 16384)}, Bytes(8192));
    CHECK(p.scratch == Bytes(16384));
    CHECK(p.live == Bytes(8192));
    CHECK(p.total() == Bytes(24576));
  }

  CASE("what is left for one more operation comes from Capacity");
  {
    FunctionPool p = planFunctionPool({req("dot", 1024)}, Bytes(8192));
    Capacity cap = p.capacityFor(Bytes(32768));
    CHECK(cap.available() == Bytes(24576));
  }

  // ── the gate ───────────────────────────────────────────────────────────

  CASE("a function within budget is emitted");
  {
    FunctionPool p = planFunctionPool({req("dot", 16384)}, Bytes(8192));
    CHECK(poolDecision(p, Bytes(32768)).ok());
  }

  CASE("an over-budget function is refused ahead of Metal");
  {
    // Over budget links cleanly, then crashes MTLCompilerService at PSO
    // creation.
    FunctionPool p = planFunctionPool({req("dot", 32768)}, Bytes(8192));
    Decision d = poolDecision(p, Bytes(32768));
    CHECK(d.isDecline());
    CHECK(!d.isBug());
  }

  CASE("the budget is against the total, buffers included");
  {
    FunctionPool p = planFunctionPool({req("dot", 32768)}, Bytes(0));
    CHECK(poolDecision(p, Bytes(32768)).ok());

    FunctionPool q = planFunctionPool({req("dot", 32768)}, Bytes(1));
    CHECK(poolDecision(q, Bytes(32768)).isDecline());
  }

  CASE("a pool that only costs occupancy is not refused");
  {
    // The core budget is twice the per-threadgroup cap.
    FunctionPool p = planFunctionPool({req("dot", kTGResidentBudgetBytes)});
    CHECK(poolDecision(p).ok());
    CHECK_EQ(tgResidency(p.total().count()), 2);

    FunctionPool tight = planFunctionPool({req("dot", 20000)}, Bytes(12000));
    CHECK(poolDecision(tight).ok());
    CHECK(tight.total() == Bytes(32000));
    CHECK_EQ(tgResidency(tight.total().count()), 2);
  }

  CASE("the limit reflects the hardware itself");
  {
    FunctionPool p = planFunctionPool({req("dot", kTGResidentBudgetBytes)});
    CHECK(poolDecision(p).ok());

    FunctionPool over =
        planFunctionPool({req("dot", kTGResidentBudgetBytes + 1)});
    CHECK(poolDecision(over).isDecline());
  }

  CASE("exactly the budget fits");
  {
    FunctionPool p = planFunctionPool({req("dot", 16384)}, Bytes(16384));
    CHECK(p.total() == Bytes(32768));
    CHECK(poolDecision(p, Bytes(32768)).ok());
  }

  // ── the module ─────────────────────────────────────────────────────────

  CASE("a module's pool is the largest any function needs");
  {
    // MSL forbids threadgroup memory outside a kernel: one buffer serves
    // every function.
    FunctionPool a = planFunctionPool({req("dot", 4096)});
    FunctionPool b = planFunctionPool({req("scan", 16384)});
    FunctionPool m = planModulePool({a, b});
    CHECK(m.scratch == Bytes(16384));
    CHECK_EQ(std::string(m.driver), std::string("scan"));
  }

  CASE("a module's live buffers take the max too");
  {
    FunctionPool a = planFunctionPool({req("dot", 1024)}, Bytes(2048));
    FunctionPool b = planFunctionPool({req("dot", 1024)}, Bytes(4096));
    FunctionPool m = planModulePool({a, b});
    CHECK(m.live == Bytes(4096));
    CHECK(m.total() == Bytes(5120));
  }

  CASE("an empty module needs no pool");
  {
    FunctionPool m = planModulePool({});
    CHECK(m.total() == Bytes(0));
  }

  CASE("the module gate refuses what no single function would");
  {
    FunctionPool a = planFunctionPool({req("dot", 30000)}, Bytes(0));
    FunctionPool b = planFunctionPool({req("scan", 1000)}, Bytes(30000));
    CHECK(poolDecision(a, Bytes(32768)).ok());
    CHECK(poolDecision(b, Bytes(32768)).ok());

    FunctionPool m = planModulePool({a, b});
    CHECK(m.total() == Bytes(60000));
    CHECK(poolDecision(m, Bytes(32768)).isDecline());
  }

  return ::agpu_test::report("PoolPlan");
}
