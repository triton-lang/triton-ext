// EmitPoll.h - the spin-wait, emitted.
//
// One thread polls each element; others wait at a hard barrier. The load must
// be volatile or atomic or the compiler hoists it out of the loop.
#ifndef AGPU_EMIT_POLL_H
#define AGPU_EMIT_POLL_H

#include "agpu/core/Names.h"
#include "agpu/emit/EmitElection.h"
#include "agpu/msl/Builtins.h"
#include "agpu/msl/Context.h"
#include "agpu/msl/Printer.h"
#include "agpu/plan/PollPlan.h"

#include <functional>

namespace agpu {

struct PollNames : ThreadNames {
  msl::Str ptr = "flagp";     // the flag's address
  msl::Str expected = "want"; // the value being waited for
  msl::Str result = "ready";  // whether it arrived
  msl::Str flag = "seen";     // the shared answer, for the timeout form
};

// The flag's current value, as an expression that re-reads on evaluation.
// Relaxed; ordering comes from the barrier that follows.
// `isHigh` selects a packed-16 flag's half; empty means the whole word.
inline msl::Expr *pollLoad(msl::Context &c, const PollPlan &p,
                           const msl::Str &ptr, const msl::Str &isHigh) {
  auto word = [&] {
    return c.call(msl::builtin::atomic::Load,
                  {c.var(ptr), c.var(msl::builtin::order::Relaxed)});
  };
  switch (p.load) {
  case PollLoad::AtomicWord:
    return word();
  case PollLoad::VolatileWide:
    return c.deref(c.var(ptr));
  case PollLoad::PackedHalf:
    break;
  }
  msl::Expr *low = c.binary(msl::BinOp::And, word(), c.lit(0xffff));
  if (isHigh.empty())
    return low;
  return c.ternary(c.var(isHigh), c.binary(msl::BinOp::Shr, word(), c.lit(16)),
                   low);
}

// Metal has no 64-bit atomic load; the wide form uses a volatile plain
// pointer and relies on single-copy atomicity of an aligned load.
inline msl::Type pollPtrType(const PollPlan &p) {
  switch (p.load) {
  case PollLoad::VolatileWide:
    return msl::Type::scalar(p.word).pointerTo(msl::AddrSpace::Device,
                                               msl::Type::Volatile);
  case PollLoad::AtomicWord:
  case PollLoad::PackedHalf:
    break;
  }
  return msl::deviceAtomicPtr(p.word);
}

// Declares one element's answer slot and seeds it. Seeding is separate from
// the wait so a barrier can close every seed before any thread polls. An
// element only its own thread reads needs no threadgroup slot.
inline void emitPollSeed(msl::Context &c, msl::Block &body, const PollPlan &p,
                         const PollNames &nm, bool shared = true) {
  if (p.spins)
    return;
  msl::Type ty = msl::Type::scalar(msl::Scalar::Bool);
  if (shared)
    ty = ty.inAddrSpace(msl::AddrSpace::Threadgroup);
  body.push_back(c.declStmt(ty, nm.flag));
  body.push_back(c.assign(c.var(nm.flag), c.litBool(false)));
}

// One element's wait. `isHigh` is the packed-16 half selector, or empty for a
// full-width flag. `owner` elects the thread that polls this element; null
// runs it on every thread, each on its own element.
inline void emitPollElement(msl::Context &c, msl::Block &body,
                            const PollPlan &p, const PollNames &nm,
                            const msl::Str &isHigh = {},
                            msl::Expr *owner = nullptr) {
  const msl::Type wordTy = msl::Type::scalar(p.word);

  msl::Block inner;
  inner.push_back(c.declStmt(wordTy, nm.expected + "_w",
                             c.cast(wordTy, c.var(nm.expected))));

  msl::Expr *loaded = pollLoad(c, p, nm.ptr, isHigh);

  if (p.spins) {
    inner.push_back(
        c.whileStmt(c.binary(msl::BinOp::Ne, loaded, c.var(nm.expected + "_w")),
                    msl::Block{}));
  } else {
    inner.push_back(
        c.assign(c.var(nm.flag),
                 c.binary(msl::BinOp::Eq, loaded, c.var(nm.expected + "_w"))));
  }

  if (!owner) {
    for (msl::Stmt *s : inner)
      body.push_back(s);
    return;
  }
  c.guardedInto(body, owner, std::move(inner));
}

inline void emitPollBarrier(msl::Context &c, msl::Block &body,
                            const PollPlan &p) {
  body.push_back(c.hardBarrier(p.acquire ? msl::Barrier::Scope::Device
                                         : msl::Barrier::Scope::Threadgroup));
}

inline void bindPollResult(msl::Context &c, msl::Block &body, const PollPlan &p,
                           const PollNames &nm) {
  body.push_back(c.declStmt(msl::Type::scalar(msl::Scalar::Bool), nm.result,
                            p.spins ? static_cast<msl::Expr *>(c.litBool(true))
                                    : c.var(nm.flag)));
}

// Every register of a tensor poll. When the election is empty each thread
// holds its own elements and polls them itself; otherwise the elected thread
// polls and the barrier publishes the answer to the threads that share it.
// Replicas take the answer of the register that owns their location.
inline msl::SmallVec<msl::Str, 8>
emitPollTensor(msl::Context &c, msl::Block &body, const PollPlan &p,
               const PollNames &nm, const msl::SmallVec<msl::Str, 8> &ptrs,
               const msl::SmallVec<msl::Str, 8> &expecteds,
               const ReplicaMap &replicas, const ThreadElection &election,
               const msl::SmallVec<msl::Str, 8> &highs = {},
               const std::function<msl::Expr *(int64_t)> &owner = {}) {
  msl::SmallVec<msl::Str, 8> results(ptrs.size());
  const bool shared = election.any();

  msl::SmallVec<PollNames, 8> perReg(ptrs.size());
  for (std::size_t r = 0; r < ptrs.size(); ++r) {
    if (replicas.isReplica((int)r))
      continue;
    PollNames rn = nm;
    rn.ptr = ptrs[r];
    rn.expected = expecteds[r];
    rn.result = nm.result + std::to_string(r);
    rn.flag = nm.flag + std::to_string(r);
    perReg[r] = rn;
    emitPollSeed(c, body, p, rn, shared);
  }

  if (!p.spins && shared)
    body.push_back(c.hardBarrier());

  for (std::size_t r = 0; r < ptrs.size(); ++r) {
    if (replicas.isReplica((int)r))
      continue;
    emitPollElement(c, body, p, perReg[r], r < highs.size() ? highs[r] : "",
                    owner ? owner((int64_t)r) : nullptr);
  }

  if (shared || p.acquire)
    emitPollBarrier(c, body, p);

  for (std::size_t r = 0; r < ptrs.size(); ++r) {
    if (replicas.isReplica((int)r)) {
      results[r] = results[replicas.canonicalOf((int)r)];
      continue;
    }
    bindPollResult(c, body, p, perReg[r]);
    results[r] = perReg[r].result;
  }
  return results;
}

} // namespace agpu

#endif // AGPU_EMIT_POLL_H
