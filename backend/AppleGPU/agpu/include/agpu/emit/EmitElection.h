// EmitElection - the test that picks one thread out of a redundant set.
//
// Where a layout bit does not move the address, threads differing only in it
// address the same location and an atomic must be applied once.
// `plan::electFor` picks the thread; this spells the test.
#ifndef AGPU_EMIT_ELECTION_H
#define AGPU_EMIT_ELECTION_H

#include "agpu/core/Names.h"
#include "agpu/msl/Builtins.h"
#include "agpu/msl/Context.h"
#include "agpu/plan/AtomicPlan.h"

namespace agpu {

// The condition under which this thread performs the access, or null when
// every thread does.
inline msl::Expr *electionExpr(msl::Context &c, const ThreadElection &e,
                               const ThreadNames &nm) {
  if (e.firstThreadOnly)
    // `.x`: `tid` is the uint3 the ABI declares.
    return c.binary(msl::BinOp::Eq,
                    c.member(c.var(nm.threadId), msl::builtin::comp::X),
                    c.lit(0));

  msl::Expr *cond = nullptr;
  auto freeIsZero = [&](const msl::Str &name, unsigned mask) {
    return c.binary(
        msl::BinOp::Eq,
        c.binary(msl::BinOp::And, c.var(name), c.lit((int64_t)mask)), c.lit(0));
  };
  if (e.needsLaneTest)
    cond = freeIsZero(nm.laneId, e.laneMask);
  if (e.needsWarpTest) {
    msl::Expr *w = freeIsZero(nm.warpId, e.warpMask);
    cond = cond ? c.binary(msl::BinOp::LAnd, cond, w) : w;
  }
  return cond;
}

} // namespace agpu

#endif // AGPU_EMIT_ELECTION_H
