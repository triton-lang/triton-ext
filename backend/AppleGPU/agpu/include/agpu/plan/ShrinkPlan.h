// ShrinkPlan.h - what to do when a function is too big to compile.
//
// The Metal compiler dies with std::bad_alloc under PromoteMemToReg/SROA on an
// oversized function. Fusing guards is free; rolling K steps changes the
// emitted program and needs a re-walk with the flag set. The measure is
// `msl::FuncSize`; the decision lives here.
#ifndef AGPU_SHRINK_PLAN_H
#define AGPU_SHRINK_PLAN_H

#include "agpu/cost/CodeSize.h"
#include "agpu/msl/Analysis.h"

#include <string>
#include <string_view>

namespace agpu {

using msl::FuncSize;

// A pre-emission forecast of what rolling the K steps would save. The deltas
// cover only the statements `rollK` changes, so the unrolled size can be
// inferred from a measured rolled body without building the unrolled one.
struct RollPrediction {
  bool roll = false;
  int declDelta = 0;
  int fragDelta = 0;
  int mmaDelta = 0;
};

// What a function needs done to it.
struct ShrinkPlan {
  bool fuseGuards = false;
  bool rollKSteps = false;

  bool any() const { return fuseGuards || rollKSteps; }

  bool needsReemit() const { return rollKSteps; }
};

inline ShrinkPlan planShrink(const FuncSize &s) {
  ShrinkPlan p;
  p.fuseGuards = s.branches > 0;
  p.rollKSteps = cost::rollsK(s);
  return p;
}

inline bool shrinkHelped(const FuncSize &before, const FuncSize &after) {
  return after.optimiserLoad() < before.optimiserLoad();
}

// `stmts` and `loops` are left as the rolled body's; nothing downstream of
// the shrink decision reads them.
inline FuncSize unrolledFrom(const FuncSize &rolled, const RollPrediction &p) {
  FuncSize s = rolled;
  s.decls += p.declDelta;
  s.fragDecls += p.fragDelta;
  s.mma += p.mmaDelta;
  return s;
}

inline bool withinBudget(const FuncSize &s) {
  return cost::withinDeclBudget(s);
}

enum class SizeVerdict {
  Fine,    // under budget, nothing to do
  Shrunk,  // was over, the re-walk brought it under
  Exposed, // over budget and not fixable by rolling
};

inline SizeVerdict verdictOf(const FuncSize &s, bool reemitted) {
  if (withinBudget(s))
    return reemitted ? SizeVerdict::Shrunk : SizeVerdict::Fine;
  return SizeVerdict::Exposed;
}

inline const char *name(SizeVerdict v) {
  switch (v) {
  case SizeVerdict::Fine:
    return "fine";
  case SizeVerdict::Shrunk:
    return "shrunk";
  case SizeVerdict::Exposed:
    return "EXPOSED";
  }
  return "?";
}

// One line per function, for MSL_FUNC_BUDGET_DEBUG.
inline std::string budgetReport(std::string_view fn, const FuncSize &s,
                                const ShrinkPlan &plan, bool reemitted) {
  const SizeVerdict v = verdictOf(s, reemitted);
  std::string out;
  out += fn;
  out += " ";
  out += name(v);
  out += " decls=" + std::to_string(s.decls);
  out += "/" + std::to_string(cost::kDeclBudget);
  out += " frags=" + std::to_string(s.fragDecls);
  out += " stmts=" + std::to_string(s.stmts);
  out += " mma=" + std::to_string(s.mma);
  out += " rolled=" + std::string(reemitted ? "yes" : "no");
  out += " fused=" + std::string(plan.fuseGuards ? "yes" : "no");
  if (v == SizeVerdict::Exposed && !plan.rollKSteps &&
      s.fragDecls < cost::kRollFragFloor)
    out += " (below frag floor " + std::to_string(cost::kRollFragFloor) +
           ": rolling would not pay)";
  return out;
}

} // namespace agpu

#endif // AGPU_SHRINK_PLAN_H
