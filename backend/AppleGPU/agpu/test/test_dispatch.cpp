// Which handler lowers an operation and what it means when none does.
#include "agpu/bind/Dispatch.h"
#include "harness.h"

#include <string>

using namespace agpu;

namespace {

OpView op(std::string_view name, std::vector<ValueId> operands = {},
          std::vector<ValueId> results = {}) {
  OpView o;
  o.name = name;
  o.operands = std::move(operands);
  o.results = std::move(results);
  return o;
}

} // namespace

int main() {
  CASE("the first handler that claims an op wins");
  {
    DispatchTable t;
    t.add("first", forOps({"arith.addf"},
                          [](const OpView &) { return Decision::emitted(); }));
    t.add("second", forOps({"arith.addf"},
                           [](const OpView &) { return Decision::failed(); }));

    std::string who;
    const Decision d = t.runNamed(op("arith.addf"), who);
    CHECK(d.ok());
    CHECK_EQ(who, std::string("first"));

    CHECK_EQ(t.order(), (std::vector<std::string>{"first", "second"}));
  }

  CASE("an op nobody claims declines and names itself");
  {
    DispatchTable t;
    t.add("arith", forOps({"arith.addf"},
                          [](const OpView &) { return Decision::emitted(); }));

    std::string who;
    const Decision d = t.runNamed(op("tt.experimental_thing"), who);
    CHECK(d.isDecline());
    CHECK(!d.isBug());
    CHECK(who.empty());
    CHECK_EQ(d.where(), msl::Str("tt.experimental_thing"));
  }

  CASE("a handler that fails stops the search");
  {
    DispatchTable t;
    t.add("broken", forOps({"tt.dot"},
                           [](const OpView &) { return Decision::failed(); }));
    t.add("rescuer", forOps({"tt.dot"}, [](const OpView &) {
            return Decision::emitted();
          }));

    std::string who;
    const Decision d = t.runNamed(op("tt.dot"), who);
    CHECK(d.isBug());
    CHECK_EQ(who, std::string("broken"));
  }

  CASE("a handler that declines also stops the search");
  {
    DispatchTable t;
    t.add("dot", forOps({"tt.dot"}, [](const OpView &) {
            return Decision::declined("dot", "k is not a multiple of 8");
          }));
    t.add("fallback", forOps({"tt.dot"}, [](const OpView &) {
            return Decision::emitted();
          }));

    std::string who;
    const Decision d = t.runNamed(op("tt.dot"), who);
    CHECK(d.isDecline());
    CHECK_EQ(who, std::string("dot"));
    CHECK_EQ(d.why(), msl::Str("k is not a multiple of 8"));
  }

  CASE("notMine is the only way to pass an op along");
  {
    DispatchTable t;
    t.add("passes", forOps({"nothing.matches"},
                           [](const OpView &) { return Decision::failed(); }));
    t.add("claims", forOps({"tt.load"},
                           [](const OpView &) { return Decision::emitted(); }));

    std::string who;
    CHECK(t.runNamed(op("tt.load"), who).ok());
    CHECK_EQ(who, std::string("claims"));
  }

  CASE("running does not run a handler twice");
  {
    int calls = 0;
    DispatchTable t;
    t.add("counting", forOps({"tt.load"}, [&calls](const OpView &) {
            ++calls;
            return Decision::emitted();
          }));

    std::string who;
    t.runNamed(op("tt.load"), who);
    CHECK_EQ(calls, 1);
    CHECK_EQ(who, std::string("counting"));
  }

  CASE("a family claims every name in its list and nothing else");
  {
    DispatchTable t;
    t.add("binop", forOps({"arith.addf", "arith.mulf", "arith.subf"},
                          [](const OpView &) { return Decision::emitted(); }));

    for (const char *n : {"arith.addf", "arith.mulf", "arith.subf"})
      CHECK(t.run(op(n)).ok());
    CHECK(t.run(op("arith.divf")).isDecline());
  }

  CASE("an empty table declines everything");
  {
    DispatchTable t;
    CHECK_EQ(t.size(), (std::size_t)0);
    CHECK(t.run(op("anything")).isDecline());
    CHECK(t.order().empty());
  }

  CASE("an op carries its operands, results and parameters");
  {
    DispatchTable t;
    t.add("range", forOps({"tt.make_range"}, [](const OpView &o) {
            return o.intAt(0) == 64 ? Decision::emitted() : Decision::failed();
          }));

    OpView o = op("tt.make_range", {}, {7});
    o.ints = {64};
    CHECK(t.run(o).ok());
    CHECK_EQ(o.results[0], (ValueId)7);

    OpView bare = op("tt.make_range", {}, {7});
    CHECK_EQ(bare.intAt(0, -1), (int64_t)-1);
    CHECK(t.run(bare).isBug());
  }

  return ::agpu_test::report("Dispatch");
}
