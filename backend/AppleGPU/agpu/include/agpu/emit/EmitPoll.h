// EmitPoll.h - the spin-wait, emitted.
//
// One thread polls; others wait at a hard barrier. The load must be volatile
// or atomic or the compiler hoists it out of the loop.
#ifndef AGPU_EMIT_POLL_H
#define AGPU_EMIT_POLL_H

#include "agpu/core/Names.h"
#include "agpu/msl/Builtins.h"
#include "agpu/msl/Context.h"
#include "agpu/msl/Printer.h"
#include "agpu/plan/PollPlan.h"

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

// `isHigh` is the packed-16 half selector, or empty for a full-width flag.
inline Decision emitPoll(msl::Context &c, msl::Block &body, const PollPlan &p,
                         const PollNames &nm, const msl::Str &isHigh = {}) {
  if (!p.usable)
    return pollDecision(p);

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

  // Seeded unconditionally to silence a false "used uninitialized" warning.
  // The barrier keeps a lagging warp's seed from landing after the answer.
  if (!p.spins) {
    body.push_back(c.declStmt(msl::Type::scalar(msl::Scalar::Bool)
                                  .inAddrSpace(msl::AddrSpace::Threadgroup),
                              nm.flag));
    body.push_back(c.assign(c.var(nm.flag), c.litBool(false)));
    body.push_back(c.hardBarrier());
  }

  c.guardedInto(body,
                c.binary(msl::BinOp::Eq,
                         c.member(c.var(nm.threadId), msl::builtin::comp::X),
                         c.lit(0)),
                std::move(inner));

  body.push_back(c.hardBarrier(p.acquire ? msl::Barrier::Scope::Device
                                         : msl::Barrier::Scope::Threadgroup));

  body.push_back(c.declStmt(msl::Type::scalar(msl::Scalar::Bool), nm.result,
                            p.spins ? static_cast<msl::Expr *>(c.litBool(true))
                                    : c.var(nm.flag)));
  return Decision::emitted();
}

} // namespace agpu

#endif // AGPU_EMIT_POLL_H
