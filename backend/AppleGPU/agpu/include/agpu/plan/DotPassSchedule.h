// DotPassSchedule - which phases one pass of a dot runs.
//
// A single-shot dot declares the accumulator fragments, multiplies into them,
// and drains them (store to the pool, barrier, read back). A dot whose
// accumulators outlive the pass (a Fused plan, run once per K iteration) only
// multiplies; the enclosing loop declares and drains them.
#ifndef AGPU_DOT_PASS_SCHEDULE_H
#define AGPU_DOT_PASS_SCHEDULE_H

#include "agpu/plan/DotPlan.h"

namespace agpu {

struct DotPassSchedule {
  // A rename leaves each value in the lane that already holds it, so it moves
  // no pool bytes.
  enum class Drain { None, Pool, Rename };

  // What a withheld drain omits is emitted once by `emitFusedLoop`.
  bool declareAccums = true;
  Drain drain = Drain::Pool;

  bool drainsC() const { return drain != Drain::None; }

  static DotPassSchedule of(const Plan &p) {
    if (p.accumulatorsOutlivePass())
      return {false, Drain::None};
    if (p.readsBackByRename())
      return {true, Drain::Rename};
    return {};
  }
};

} // namespace agpu

#endif // AGPU_DOT_PASS_SCHEDULE_H
