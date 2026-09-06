// EmitReduce.h - a reduction, emitted from its plan. The combine region is
// user TTIR and comes in as a callback.
#ifndef AGPU_EMIT_REDUCE_H
#define AGPU_EMIT_REDUCE_H

#include "agpu/core/Names.h"
#include "agpu/emit/EmitShuffle.h"
#include "agpu/msl/Builtins.h"
#include "agpu/msl/Context.h"
#include "agpu/plan/Combiner.h"
#include "agpu/plan/ReductionPlan.h"

#include <functional>

namespace agpu {

// Lowers the user's combine region. Takes the block to append to and the two
// operand name lists; returns the result names.
using CombineFn = std::function<msl::SmallVec<msl::Str, 4>(
    msl::Block &, const msl::SmallVec<msl::Str, 4> &,
    const msl::SmallVec<msl::Str, 4> &)>;

struct ReduceNames : ScratchNames {
  msl::Str acc = "acc";
  msl::Str peer = "peer";
};

// Metal allows threadgroup declarations only in a kernel, so the pool declares
// the scratch and this file only addresses into it.

// The first slot belonging to `warp`.
inline msl::Expr *anchorExpr(msl::Context &c, const ScratchLayout &slots,
                             msl::Expr *warp) {
  return c.binary(msl::BinOp::Mul, warp, c.lit(slots.warpStride()));
}

inline msl::Expr *slotExpr(msl::Context &c, const ScratchLayout &slots,
                           msl::Expr *warp, const msl::Str &lane) {
  return c.binary(msl::BinOp::Add, anchorExpr(c, slots, warp), c.var(lane));
}

inline msl::Expr *shuffleXor(msl::Context &c, const msl::Str &v, int64_t mask,
                             ElemType elem = i32()) {
  return shuffleOf(c, msl::builtin::simd::ShuffleXor, elem, v,
                   c.lit(mask, msl::Context::u32()));
}

// Fold the registers one thread owns, in order. No lane crossing.
inline msl::SmallVec<msl::Str, 4>
emitLocalFold(msl::Context &c, msl::Block &body, const ReductionPlan &plan,
              const ReductionGroup &g,
              const msl::SmallVec<msl::SmallVec<msl::Str, 8>, 4> &srcNames,
              const ReduceNames &nm, int groupIdx, const CombineFn &combine) {
  const int nOp = (int)srcNames.size();
  msl::SmallVec<msl::Str, 4> accs;
  for (int k = 0; k < nOp; ++k) {
    const msl::Str a =
        nm.acc + std::to_string(groupIdx) + "_" + std::to_string(k);
    body.push_back(c.declStmt(mslTypeOf(plan.elemAt(k)), a,
                              c.var(srcNames[k][g.sourceRegs[0]])));
    accs.push_back(a);
  }
  for (std::size_t i = 1; i < g.sourceRegs.size(); ++i) {
    msl::SmallVec<msl::Str, 4> rhs;
    for (int k = 0; k < nOp; ++k)
      rhs.push_back(srcNames[k][g.sourceRegs[i]]);
    msl::SmallVec<msl::Str, 4> out = combine(body, accs, rhs);
    for (int k = 0; k < nOp; ++k)
      body.push_back(c.assign(c.var(accs[k]), c.var(out[k])));
  }
  return accs;
}

// The lane phase: one XOR shuffle per planned step, high bit first.
inline void emitLaneSteps(msl::Context &c, msl::Block &body,
                          const ReductionPlan &plan,
                          msl::SmallVec<msl::Str, 4> &accs,
                          const ReduceNames &nm, int groupIdx,
                          const CombineFn &combine) {
  const int nOp = (int)accs.size();
  if (const char *fn = plan.laneIntrinsic(plan.scratch.warpSize)) {
    body.push_back(c.assign(c.var(accs[0]), c.call(fn, {c.var(accs[0])})));
    return;
  }
  for (std::size_t si = 0; si < plan.laneSteps.size(); ++si) {
    const ReduceStep &st = plan.laneSteps[si];
    msl::SmallVec<msl::Str, 4> peers;
    for (int k = 0; k < nOp; ++k) {
      const msl::Str p = nm.peer + std::to_string(groupIdx) + "_" +
                         std::to_string(si) + "_" + std::to_string(k);
      body.push_back(
          c.declStmt(mslTypeOf(plan.elemAt(k)), p,
                     shuffleXor(c, accs[k], st.xorOffset, plan.elemAt(k))));
      peers.push_back(p);
    }
    msl::SmallVec<msl::Str, 4> out = combine(body, accs, peers);
    for (int k = 0; k < nOp; ++k)
      body.push_back(c.assign(c.var(accs[k]), c.var(out[k])));
  }
}

// The cross-warp phase: publish to scratch, barrier, then combine the warps
// this reduction spans. Each warp reads its own subset, anchored on its id.
inline void emitWarpSteps(msl::Context &c, msl::Block &body,
                          const ReductionPlan &plan, int64_t numWarps,
                          msl::SmallVec<msl::Str, 4> &accs,
                          const ReduceNames &nm, int groupIdx,
                          const CombineFn &combine) {
  if (!plan.crossWarp())
    return;
  const int nOp = (int)accs.size();
  const ScratchLayout &slots = plan.scratch;

  // Every warp publishes, including those outside this reduction's subset, so
  // the reservation is numWarps slots.
  body.push_back(c.barrier());
  for (int k = 0; k < nOp; ++k) {
    msl::Expr *idx = slotExpr(c, slots, c.var(nm.warpId), nm.laneId);
    body.push_back(c.assign(c.subscript(c.var(nm.scratch[(std::size_t)k]), idx),
                            c.var(accs[k])));
  }
  body.push_back(c.barrier());

  // The executing warp's subset anchor: `(warp & ~warpMask) * 32 + lane`.
  // warpSubset holds XOR offsets, so reads are relative to this anchor.
  msl::Expr *base =
      slotExpr(c, slots,
               c.binary(msl::BinOp::And, c.var(nm.warpId),
                        c.lit((int64_t)plan.anchorMask(numWarps))),
               nm.laneId);

  // Re-seed from the anchor slot; for a non-anchor warp that is not `accs`.
  msl::SmallVec<msl::Str, 4> wacc;
  for (int k = 0; k < nOp; ++k) {
    const msl::Str w =
        nm.acc + "w" + std::to_string(groupIdx) + "_" + std::to_string(k);
    body.push_back(
        c.declStmt(mslTypeOf(plan.elemAt(k)), w,
                   c.subscript(c.var(nm.scratch[(std::size_t)k]), base)));
    wacc.push_back(w);
  }

  for (std::size_t wi = 1; wi < plan.warpSubset.size(); ++wi) {
    msl::SmallVec<msl::Str, 4> peers;
    for (int k = 0; k < nOp; ++k) {
      const msl::Str p = nm.peer + "w" + std::to_string(groupIdx) + "_" +
                         std::to_string(wi) + "_" + std::to_string(k);
      msl::Expr *idx = c.binary(msl::BinOp::Add, base,
                                c.lit(slots.anchorSlots(plan.warpSubset[wi])));
      body.push_back(
          c.declStmt(mslTypeOf(plan.elemAt(k)), p,
                     c.subscript(c.var(nm.scratch[(std::size_t)k]), idx)));
      peers.push_back(p);
    }
    msl::SmallVec<msl::Str, 4> out = combine(body, wacc, peers);
    for (int k = 0; k < nOp; ++k)
      body.push_back(c.assign(c.var(wacc[k]), c.var(out[k])));
  }
  accs = wacc;

  // Closes the scratch epoch: the pool overlays this scratch with other
  // regions, so a later write there must not overtake these reads.
  body.push_back(c.barrier());
}

// A whole reduction: for each survivor group, fold locally, then across lanes,
// then across warps. Returns the accumulator names per group, in plan order.
inline std::vector<msl::SmallVec<msl::Str, 4>>
emitReduce(msl::Context &c, msl::Block &body, const ReductionPlan &plan,
           int64_t numWarps,
           const msl::SmallVec<msl::SmallVec<msl::Str, 8>, 4> &srcNames,
           const ReduceNames &nm, const CombineFn &combine) {
  std::vector<msl::SmallVec<msl::Str, 4>> results;

  if (srcNames.empty() || !plan.operandsShareLayout())
    return results;
  for (std::size_t k = 1; k < srcNames.size(); ++k)
    if (srcNames[k].size() != srcNames[0].size())
      return results;

  if (plan.crossWarp() && (int)nm.scratch.size() < (int)srcNames.size())
    return results;

  for (int gi = 0; gi < plan.groupCount(); ++gi) {
    const ReductionGroup &g = plan.groups[gi];
    msl::SmallVec<msl::Str, 4> accs =
        emitLocalFold(c, body, plan, g, srcNames, nm, gi, combine);
    emitLaneSteps(c, body, plan, accs, nm, gi, combine);
    emitWarpSteps(c, body, plan, numWarps, accs, nm, gi, combine);
    results.push_back(accs);
  }
  return results;
}

} // namespace agpu

#endif // AGPU_EMIT_REDUCE_H
