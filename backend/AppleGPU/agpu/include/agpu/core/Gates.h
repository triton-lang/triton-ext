// Gates.h - the environment switches. Every gate changes what is printed and
// never what is emitted; the compilation cache is keyed on source+target only.
#ifndef AGPU_GATES_H
#define AGPU_GATES_H

#include "agpu/core/EnumBitset.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <string_view>

namespace agpu {

enum class Gate {
  LogReject,       // the decline log and its teardown summary
  TraceFail,       // a stack-ish trace at the point of a hard failure
  FuncBudgetDebug, // why a function was shrunk and by how much
  DotPlanDebug,    // the chosen dot plan and the ones passed over
  TraceOps,        // every op the walk dispatches
  DeclineLog,      // a file the decline summary is appended to
  Count,
};

struct GateSpec {
  Gate gate;
  std::string_view env;
  std::string_view because;
};

inline constexpr GateSpec kGates[] = {
    {Gate::LogReject, "MSL_LOG_REJECT", "prints what declined; emits nothing"},
    {Gate::TraceFail, "TRITON_MSL_TRACE_FAIL",
     "prints where a hard failure happened"},
    {Gate::FuncBudgetDebug, "MSL_FUNC_BUDGET_DEBUG", "prints shrink decisions"},
    {Gate::DotPlanDebug, "MSL_DOT_PLAN_DEBUG",
     "prints the dot plan and its rejected alternatives"},
    {Gate::TraceOps, "AGPU_TRACE_OPS",
     "logs each op the walk dispatches, with its shape"},
    {Gate::DeclineLog, "AGPU_DECLINE_LOG",
     "names a file the decline summary is appended to"},
};

inline constexpr std::size_t gateCount() {
  return sizeof(kGates) / sizeof(kGates[0]);
}

inline const GateSpec &gateSpec(Gate g) {
  for (const GateSpec &s : kGates)
    if (s.gate == g)
      return s;
  return kGates[0];
}

// Behind an interface so a test can set a gate without touching the process.
class GateSet {
public:
  GateSet() = default;

  static GateSet fromEnvironment() {
    GateSet s;
    for (const GateSpec &spec : kGates)
      if (std::getenv(std::string(spec.env).c_str()) != nullptr)
        s.set(spec.gate, true);
    return s;
  }

  bool on(Gate g) const { return bits_.has(g); }

  void set(Gate g, bool value = true) { bits_.set(g, value); }

private:
  EnumBitset<Gate> bits_;
};

} // namespace agpu

#endif // AGPU_GATES_H
