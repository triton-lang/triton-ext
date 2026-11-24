// The environment switches: every one a diagnostic, none of them in the
// compilation cache key.
#include "agpu/core/Gates.h"
#include "harness.h"

#include <fstream>
#include <string>

using namespace agpu;

namespace {

// `CODEGEN_ENV` from `python/triton_apple_backend/compiler.py`, read rather
// than duplicated since that tuple is the cache key's one owner.
//
// `found` and `text` are separate: the tuple is legitimately EMPTY, so "no
// gate keys the cache" and "the file no longer defines CODEGEN_ENV" must not
// read the same.
struct CodegenEnv {
  bool found = false;
  std::string text;
};

CodegenEnv codegenEnvText() {
  std::ifstream in(AGPU_COMPILER_PY);
  CodegenEnv out;
  std::string line;
  bool inTuple = false;
  while (std::getline(in, line)) {
    const std::size_t at = line.find("CODEGEN_ENV = (");
    if (at != std::string::npos) {
      out.found = true;
      if (line.find(')', at) != std::string::npos)
        break;
      inTuple = true;
      continue;
    }
    if (inTuple) {
      if (line.find(')') != std::string::npos)
        break;
      out.text += line + "\n";
    }
  }
  return out;
}

} // namespace

int main() {
  CASE("no gate reaches the compilation cache key");
  {
    // Every gate changes only what is printed, so none of them may key the
    // cache -- a run with logging on must share cache entries with one
    // without.
    const CodegenEnv env = codegenEnvText();
    CHECK(env.found); // the tuple still exists to be checked against
    for (const GateSpec &spec : kGates)
      CHECK(env.text.find("'" + std::string(spec.env) + "'") ==
            std::string::npos);
  }

  CASE("a gate can be set and cleared");
  {
    GateSet s;
    s.set(Gate::DotPlanDebug);
    CHECK(s.on(Gate::DotPlanDebug));
    s.set(Gate::DotPlanDebug, false);
    CHECK(!s.on(Gate::DotPlanDebug));
  }

  CASE("every gate has a distinct name and a stated reason");
  {
    CHECK_EQ(gateCount(), (std::size_t)Gate::Count);
    for (std::size_t i = 0; i < gateCount(); ++i) {
      CHECK(!kGates[i].env.empty());
      CHECK(!kGates[i].because.empty());
      for (std::size_t j = i + 1; j < gateCount(); ++j) {
        CHECK(kGates[i].env != kGates[j].env);
        CHECK(kGates[i].gate != kGates[j].gate);
      }
    }
  }

  CASE("every enumerator has a row");
  {
    for (int i = 0; i < (int)Gate::Count; ++i) {
      const Gate g = (Gate)i;
      CHECK(gateSpec(g).gate == g);
    }
  }

  CASE("each fromEnvironment call reads the environment fresh");
  {
    // A GateSet is a value: two built the same way agree and setting one
    // does not reach the other -- unlike a function-local static getenv cache.
    const GateSet a = GateSet::fromEnvironment();
    const GateSet b = GateSet::fromEnvironment();
    for (const GateSpec &spec : kGates)
      CHECK_EQ(a.on(spec.gate), b.on(spec.gate));

    GateSet manual = a;
    manual.set(Gate::TraceOps);
    CHECK(manual.on(Gate::TraceOps));
    CHECK_EQ(GateSet::fromEnvironment().on(Gate::TraceOps),
             a.on(Gate::TraceOps));

    manual.set(Gate::TraceOps, false);
    CHECK(!manual.on(Gate::TraceOps));
  }

  return ::agpu_test::report("Gates");
}
