// EmitAtomic.h - one atomic, emitted from its plan.
#ifndef AGPU_EMIT_ATOMIC_H
#define AGPU_EMIT_ATOMIC_H

#include "agpu/core/Names.h"
#include "agpu/emit/EmitElection.h"
#include "agpu/emit/Prelude.h"
#include "agpu/msl/Builtins.h"
#include "agpu/msl/Context.h"
#include "agpu/msl/Printer.h"
#include "agpu/plan/AtomicPlan.h"

#include <functional>

namespace agpu {

struct AtomicNames : ThreadNames {
  msl::Str result = "old";
  msl::Str isHigh = "hi";
  // The pool region an election crossing warps publishes through. Empty
  // where the caller does not need the result back.
  msl::Str scratch;
};

inline msl::Scalar scalarOfWord(AtomicWord w) {
  switch (w) {
  case AtomicWord::I32:
    return msl::Scalar::I32;
  case AtomicWord::U32:
    return msl::Scalar::U32;
  case AtomicWord::I64:
    return msl::Scalar::I64;
  case AtomicWord::F32:
    return msl::Scalar::F32;
  }
  return msl::Scalar::I32;
}

inline const char *spellWord(AtomicWord w) {
  return msl::spell(scalarOfWord(w));
}

inline msl::Type resultTypeOf(const AtomicPlan &p) {
  if (p.resultIsElement())
    return mslTypeOf(p.packedElem);
  return msl::Type::named(spellWord(p.word));
}

// Legal in divergent control flow, unlike a barrier.
inline msl::Stmt *deviceFence(msl::Context &c) {
  return c.exprStmt(c.call(msl::builtin::atomic::ThreadFence,
                           {c.var(msl::builtin::memflags::Device),
                            c.var(msl::builtin::order::SeqCst)}));
}

inline void emitAtomicBody(msl::Context &c, msl::Block &body,
                           const AtomicPlan &p, const msl::Str &ptr,
                           const msl::Str &value, const AtomicNames &nm) {
  switch (p.strategy) {
  case AtomicStrategy::Native:
    body.push_back(
        c.assign(c.var(nm.result),
                 c.call(p.builtin, {c.var(ptr), c.var(value),
                                    c.var(msl::builtin::order::Relaxed)})));
    break;

  case AtomicStrategy::FloatCas:
    body.push_back(c.assign(
        c.var(nm.result),
        c.call(helperName(Helper::AtomicRmwF32),
               {c.var(ptr), c.var(value), c.lit(emuRmwCode(p.emuOp))})));
    break;

  case AtomicStrategy::Packed16:
    body.push_back(
        c.assign(c.var(nm.result),
                 c.call(helperName(Helper::AtomicRmwPacked16),
                        {msl::spell(mslTypeOf(p.packedElem).scalarKind())},
                        {c.var(ptr), c.var(nm.isHigh), c.var(value),
                         c.lit(emuRmwCode(p.emuOp))})));
    break;

  case AtomicStrategy::Unsupported:
    break;
  }
}

// The excluded threads still hold their initialiser, so a reader of the
// result would see zero. The winner publishes and every thread reads back.
// Both barriers sit outside the election: a threadgroup_barrier under
// divergent control flow is undefined in Metal.
inline void emitElectedBroadcast(msl::Context &c, msl::Block &body,
                                 const AtomicPlan &p, const AtomicNames &nm) {
  if (nm.scratch.empty() || !p.election.crossesWarp())
    return;
  msl::Expr *slot = c.subscript(c.var(nm.scratch), c.lit(0));
  // Leading barrier so a reader still on the previous round has finished
  // with the slot before the winner overwrites it.
  body.push_back(c.barrier());
  msl::Block publish;
  publish.push_back(c.assign(slot, c.var(nm.result)));
  c.guardedInto(body, electionExpr(c, p.election, nm), std::move(publish));
  body.push_back(c.barrier());
  body.push_back(c.assign(c.var(nm.result), slot));
  // Closes the scratch epoch: the pool overlays this slot with other regions,
  // so a later write there must not overtake this read.
  body.push_back(c.barrier());
}

// `cond` is ANDed with the election: a masked atomic on a replicated address
// must still elect one issuer.
inline void emitAtomic(msl::Context &c, msl::Block &body, const AtomicPlan &p,
                       const msl::Str &ptr, const msl::Str &value,
                       const AtomicNames &nm, msl::Expr *cond = nullptr) {
  if (!p.usable())
    return;
  // A device seq_cst fence before a uniform-address atomic crashes the AGX3
  // compiler. Outside the election: a barrier there would be divergent.
  if (p.fences.before)
    body.push_back(c.barrier(msl::Barrier::Scope::Device));
  msl::Block inner;
  emitAtomicBody(c, inner, p, ptr, value, nm);
  c.guardedInto(body, c.allOf(cond, electionExpr(c, p.election, nm)),
                std::move(inner));
  if (p.fences.after)
    body.push_back(c.barrier(msl::Barrier::Scope::Device));
  emitElectedBroadcast(c, body, p, nm);
}

// A replica issues nothing; it binds to its canonical register's result.
// `guard(r)` is per-register.
inline msl::SmallVec<msl::Str, 8>
emitAtomicTensor(msl::Context &c, msl::Block &body, const AtomicPlan &p,
                 const msl::SmallVec<msl::Str, 8> &ptrs,
                 const msl::SmallVec<msl::Str, 8> &values,
                 const AtomicNames &nm,
                 const msl::SmallVec<msl::Str, 8> &highs = {},
                 const std::function<msl::Expr *(int64_t)> &guard = {}) {
  msl::SmallVec<msl::Str, 8> results(ptrs.size());
  for (std::size_t r = 0; r < ptrs.size(); ++r) {
    const int canon = p.replicas.canonicalOf((int)r);
    if (p.replicas.isReplica((int)r)) {
      results[r] = results[canon];
      continue;
    }
    AtomicNames rn = nm;
    rn.result = nm.result + std::to_string(r);
    if (r < highs.size())
      rn.isHigh = highs[r];
    body.push_back(c.declStmt(resultTypeOf(p), rn.result, c.lit(0)));
    emitAtomic(c, body, p, ptrs[r], values[r], rn,
               guard ? guard((int64_t)r) : nullptr);
    results[r] = rn.result;
  }
  return results;
}

} // namespace agpu

#endif // AGPU_EMIT_ATOMIC_H
