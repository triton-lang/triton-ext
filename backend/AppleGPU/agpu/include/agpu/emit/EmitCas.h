// EmitCas.h - compare-and-exchange, emitted.
//
// A 16-bit exchange runs as a 32-bit exchange on the containing word, so the
// retry loop re-reads until the failure is a mismatch of the caller's half.
// Through a uniform pointer one thread exchanges and broadcasts the answer via
// threadgroup memory.
#ifndef AGPU_EMIT_CAS_H
#define AGPU_EMIT_CAS_H

#include "agpu/core/Names.h"
#include "agpu/msl/Builtins.h"
#include "agpu/msl/Context.h"
#include "agpu/msl/Printer.h"
#include "agpu/plan/CasPlan.h"

namespace agpu {

struct CasNames : ThreadNames {
  msl::Str ptr = "casp";     // the target, as an atomic_uint*
  msl::Str expected = "cmp"; // what the caller expects to find
  msl::Str desired = "val";  // what to write if it does
  msl::Str result = "old";   // what was actually there
  msl::Str isHigh = "hi";    // packed-16: which half
  msl::Str shared = "casb";  // uniform: the broadcast slot

  CasNames suffixed(const msl::Str &tag) const {
    CasNames n = *this;
    n.ptr += tag;
    n.expected += tag;
    n.desired += tag;
    n.result += tag;
    n.isHigh += tag;
    n.shared += tag;
    return n;
  }
};

// Relaxed on both orders: Metal device atomics are relaxed-only. The plan
// places fences around this for the requested ordering.
inline msl::Expr *casCall(msl::Context &c, const msl::Str &ptr,
                          const msl::Str &expectedVar, msl::Expr *desired) {
  return c.call(msl::builtin::atomic::CompareExchangeWeak,
                {c.var(ptr), c.addrOf(c.var(expectedVar)), desired,
                 c.var(msl::builtin::order::Relaxed),
                 c.var(msl::builtin::order::Relaxed)});
}

// Both directions must use the same crossing: bitcast in, bitcast out.
struct CasWordCrossing {
  msl::Type word;
  bool viaBits = false;

  msl::Expr *into(msl::Context &c, const msl::Str &v) const {
    return viaBits ? c.bitcast(word, c.var(v)) : c.var(v);
  }
  msl::Expr *outOf(msl::Context &c, ElemType elem, msl::Expr *w) const {
    return viaBits ? c.bitcast(mslTypeOf(elem), w) : c.cast(mslTypeOf(elem), w);
  }
};

// The packed form compares one half of the 32-bit word it exchanges.
inline CasWordCrossing crossingFor(const CasPlan &p) {
  return {msl::Type::scalar(p.strategy == CasStrategy::Packed16
                                ? msl::Scalar::U16
                                : msl::Scalar::U32),
          p.viaBits};
}

inline void emitCasCore(msl::Context &c, msl::Block &body, const CasPlan &p,
                        const CasNames &nm, ElemType elem) {
  const msl::Type u32 = msl::Type::scalar(msl::Scalar::U32);
  const CasWordCrossing cross = crossingFor(p);

  switch (p.strategy) {
  case CasStrategy::Word32: {
    // The expected value is in-out: on failure Metal writes what it found,
    // so no second read is needed. The weak form can fail spuriously and
    // leave the expected value in place, so retry only while the value found
    // still equals the one asked for.
    const msl::Str want = nm.expected + "_w";
    body.push_back(c.declStmt(u32, want, cross.into(c, nm.expected)));
    body.push_back(c.declStmt(u32, nm.result + "_w", c.var(want)));
    body.push_back(c.whileStmt(
        c.binary(msl::BinOp::LAnd,
                 c.binary(msl::BinOp::Eq, c.var(nm.result + "_w"), c.var(want)),
                 c.unary(msl::UnOp::LNot, casCall(c, nm.ptr, nm.result + "_w",
                                                  cross.into(c, nm.desired)))),
        msl::Block{}));
    body.push_back(c.declStmt(mslTypeOf(elem), nm.result,
                              cross.outOf(c, elem, c.var(nm.result + "_w"))));
    return;
  }

  case CasStrategy::Packed16: {
    body.push_back(c.declStmt(
        u32, nm.result + "_w",
        c.call(msl::builtin::atomic::Load,
               {c.var(nm.ptr), c.var(msl::builtin::order::Relaxed)})));

    auto halfOf = [&](const msl::Str &word) {
      return c.ternary(c.var(nm.isHigh),
                       c.binary(msl::BinOp::Shr, c.var(word), c.lit(16)),
                       c.binary(msl::BinOp::And, c.var(word), c.lit(0xffff)));
    };

    const msl::Str expectedBits = nm.expected + "_h";
    const msl::Str desiredBits = nm.desired + "_h";
    body.push_back(
        c.declStmt(cross.word, expectedBits, cross.into(c, nm.expected)));
    body.push_back(
        c.declStmt(cross.word, desiredBits, cross.into(c, nm.desired)));

    // Outside the loop: inside, it dies at the closing brace.
    const msl::Str resultBits = nm.result + "_h";
    body.push_back(c.declStmt(cross.word, resultBits));

    msl::Block loop;
    loop.push_back(c.assign(c.var(resultBits),
                            c.cast(cross.word, halfOf(nm.result + "_w"))));
    // A mismatch on our half is the caller's answer.
    loop.push_back(c.ifStmt(
        c.binary(msl::BinOp::Ne, c.var(resultBits), c.var(expectedBits)),
        msl::Block{c.breakStmt()}));

    msl::Expr *merged = c.ternary(
        c.var(nm.isHigh),
        c.binary(
            msl::BinOp::Or,
            c.binary(msl::BinOp::And, c.var(nm.result + "_w"), c.lit(0xffff)),
            c.binary(msl::BinOp::Shl, c.var(desiredBits), c.lit(16))),
        c.binary(msl::BinOp::Or,
                 c.binary(msl::BinOp::And, c.var(nm.result + "_w"),
                          c.lit(int64_t(0xffff0000))),
                 c.var(desiredBits)));

    loop.push_back(c.ifStmt(casCall(c, nm.ptr, nm.result + "_w", merged),
                            msl::Block{c.breakStmt()}));
    body.push_back(c.whileStmt(c.litBool(true), std::move(loop)));

    body.push_back(c.declStmt(mslTypeOf(elem), nm.result,
                              cross.outOf(c, elem, c.var(resultBits))));
    return;
  }

  case CasStrategy::Unsupported:
    return;
  }
}

// `bound` receives the name holding the result: `nm.result`, or under
// electOne a broadcast copy.
inline Decision emitCas(msl::Context &c, msl::Block &body, const CasPlan &p,
                        const CasNames &nm, ElemType elem,
                        msl::Str *bound = nullptr) {
  if (!p.usable())
    return casDecision(p);
  if (bound)
    *bound = nm.result;

  auto fence = [&] {
    return c.exprStmt(c.call(msl::builtin::atomic::ThreadFence,
                             {c.var(msl::builtin::memflags::Device),
                              c.var(msl::builtin::order::SeqCst)}));
  };

  if (p.fences.before)
    body.push_back(fence());

  if (!p.electOne) {
    emitCasCore(c, body, p, nm, elem);
    if (p.fences.after)
      body.push_back(fence());
    return Decision::emitted();
  }

  // Seeded before the guard to silence a "used uninitialized whenever the if
  // condition is false" warning; the compiler cannot see through the barrier.
  body.push_back(c.declStmt(
      mslTypeOf(elem).inAddrSpace(msl::AddrSpace::Threadgroup), nm.shared));
  body.push_back(c.assign(c.var(nm.shared), c.var(nm.expected)));

  // Without this, a lagging warp's seed can land after the electing thread's
  // answer and the group reads `expected` as if the exchange succeeded.
  body.push_back(c.hardBarrier());

  msl::Block inner;
  emitCasCore(c, inner, p, nm, elem);
  inner.push_back(c.assign(c.var(nm.shared), c.var(nm.result)));
  c.guardedInto(body,
                c.binary(msl::BinOp::Eq,
                         c.member(c.var(nm.threadId), msl::builtin::comp::X),
                         c.lit(0)),
                std::move(inner));

  // Separates the exchanging thread's write from the other threads' reads.
  body.push_back(c.hardBarrier());
  const msl::Str broadcast = nm.result + "_b";
  if (bound)
    *bound = broadcast;
  body.push_back(c.declStmt(mslTypeOf(elem), broadcast, c.var(nm.shared)));

  // The slot is reused by every execution, so in a spin loop the electing
  // thread can overwrite it while others still read the previous answer.
  body.push_back(c.hardBarrier());

  // After the broadcast: every lane makes the loads this fence orders.
  if (p.fences.after)
    body.push_back(fence());
  return Decision::emitted();
}

} // namespace agpu

#endif // AGPU_EMIT_CAS_H
