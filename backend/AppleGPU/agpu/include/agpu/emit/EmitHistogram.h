// EmitHistogram.h - counting into threadgroup bins.
#ifndef AGPU_EMIT_HISTOGRAM_H
#define AGPU_EMIT_HISTOGRAM_H

#include "agpu/emit/EmitAtomic.h"
#include "agpu/msl/Builtins.h"
#include "agpu/msl/Context.h"
#include "agpu/plan/HistogramPlan.h"

namespace agpu {

struct HistogramNames : ThreadNames {
  msl::Str bins = "bins";
  msl::Str zi = "zi";
  msl::Str values = "hv";
};

// Bins are atomic, so zeroing goes through atomic_store: `bins[zi] = 0` has
// no viable overload.
inline void emitHistogramZero(msl::Context &c, msl::Block &body,
                              const HistogramPlan &p,
                              const HistogramNames &nm) {
  msl::Block inner;
  inner.push_back(
      c.exprStmt(c.call(msl::builtin::atomic::Store,
                        {c.addrOf(c.subscript(c.var(nm.bins), c.var(nm.zi))),
                         c.lit(0), c.var(msl::builtin::order::Relaxed)})));
  body.push_back(
      c.forStmt(c.declStmt(msl::Context::i32(), nm.zi,
                           c.member(c.var(nm.threadId), msl::builtin::comp::X)),
                c.binary(msl::BinOp::Lt, c.var(nm.zi), c.lit(p.bins)),
                c.assignOp(msl::BinOp::Add, c.var(nm.zi), c.lit(p.threads)),
                std::move(inner)));
  body.push_back(c.barrier());
}

// One atomic increment per source register, guarded by election, by range
// (`tl.histogram` counts [0, bins); out of range would write past the bin
// array) and by the mask when the IR supplies one.
inline void emitHistogramCount(msl::Context &c, msl::Block &body,
                               const HistogramPlan &p,
                               const msl::SmallVec<msl::Str, 8> &srcRegs,
                               const HistogramNames &nm,
                               const msl::SmallVec<msl::Str, 8> &masks = {}) {
  AtomicNames an;
  static_cast<ThreadNames &>(an) = static_cast<const ThreadNames &>(nm);

  msl::Block inner;
  for (std::size_t r = 0; r < srcRegs.size(); ++r) {
    const msl::Str &v = srcRegs[r];
    msl::Expr *guard =
        c.binary(msl::BinOp::LAnd, c.binary(msl::BinOp::Ge, c.var(v), c.lit(0)),
                 c.binary(msl::BinOp::Lt, c.var(v), c.lit(p.bins)));
    if (!masks.empty()) {
      const msl::Str &m = masks[masks.size() == 1 ? 0 : r];
      guard = c.binary(msl::BinOp::LAnd, c.var(m), guard);
    }
    msl::Block one;
    one.push_back(
        c.exprStmt(c.call(msl::builtin::atomic::FetchAdd,
                          {c.addrOf(c.subscript(c.var(nm.bins), c.var(v))),
                           c.lit(1), c.var(msl::builtin::order::Relaxed)})));
    c.guardedInto(inner, guard, std::move(one));
  }

  c.guardedInto(body, electionExpr(c, p.election, an), std::move(inner));
  body.push_back(c.barrier());
}

} // namespace agpu

#endif // AGPU_EMIT_HISTOGRAM_H
