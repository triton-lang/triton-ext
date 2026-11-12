// EmitReduce.h - a reduction, emitted from its plan. The combine region is
// user TTIR and comes in as a callback.
#ifndef AGPU_EMIT_REDUCE_H
#define AGPU_EMIT_REDUCE_H

#include "agpu/core/Decline.h"
#include "agpu/core/Names.h"
#include "agpu/emit/EmitShuffle.h"
#include "agpu/msl/Builtins.h"
#include "agpu/msl/Context.h"
#include "agpu/plan/Combiner.h"
#include "agpu/plan/ReductionPlan.h"

#include <functional>

namespace agpu {

// Lowers the user's combine region. Takes the block to append to and the two
// operand name lists; returns the result names, or why it could not.
using CombineNames = msl::SmallVec<msl::Str, 4>;
using CombineFn = std::function<Result<CombineNames>(
    msl::Block &, const CombineNames &, const CombineNames &)>;

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
                           msl::Expr *warp, const msl::Str &lane,
                           int groupIdx = 0) {
  msl::Expr *e =
      c.binary(msl::BinOp::Add, anchorExpr(c, slots, warp), c.var(lane));
  if (const int64_t base = slots.groupBase(groupIdx))
    e = c.binary(msl::BinOp::Add, e, c.lit(base));
  return e;
}

inline msl::Expr *shuffleXor(msl::Context &c, const msl::Str &v, int64_t mask,
                             ElemType elem = i32()) {
  return shuffleOf(c, msl::builtin::simd::ShuffleXor, elem, v,
                   c.lit(mask, msl::Context::u32()));
}

// Fold the registers one thread owns, in order. No lane crossing.
inline Result<CombineNames>
emitLocalFold(msl::Context &c, msl::Block &body, const ReductionPlan &plan,
              const ReductionGroup &g,
              const msl::SmallVec<msl::SmallVec<msl::Str, 8>, 4> &srcNames,
              const ReduceNames &nm, int groupIdx, const CombineFn &combine) {
  using R = Result<CombineNames>;
  const int nOp = (int)srcNames.size();
  CombineNames accs;
  for (int k = 0; k < nOp; ++k) {
    const msl::Str a =
        nm.acc + std::to_string(groupIdx) + "_" + std::to_string(k);
    body.push_back(c.declStmt(mslTypeOf(plan.elemAt(k)), a,
                              c.var(srcNames[k][g.sourceRegs[0]])));
    accs.push_back(a);
  }
  for (std::size_t i = 1; i < g.sourceRegs.size(); ++i) {
    CombineNames rhs;
    for (int k = 0; k < nOp; ++k)
      rhs.push_back(srcNames[k][g.sourceRegs[i]]);
    const Result<CombineNames> out = combine(body, accs, rhs);
    if (!out.ok())
      return R::no(out.why);
    for (int k = 0; k < nOp; ++k)
      body.push_back(c.assign(c.var(accs[k]), c.var(out.value[k])));
  }
  return R::of(accs);
}

// The lane phase: one XOR shuffle per planned step, high bit first.
inline Decision emitLaneSteps(msl::Context &c, msl::Block &body,
                              const ReductionPlan &plan, CombineNames &accs,
                              const ReduceNames &nm, int groupIdx,
                              const CombineFn &combine) {
  const int nOp = (int)accs.size();
  if (const char *fn = plan.laneIntrinsic(plan.scratch.warpSize)) {
    body.push_back(c.assign(c.var(accs[0]), c.call(fn, {c.var(accs[0])})));
    return Decision::emitted();
  }
  for (std::size_t si = 0; si < plan.laneSteps.size(); ++si) {
    const ReduceStep &st = plan.laneSteps[si];
    CombineNames peers;
    for (int k = 0; k < nOp; ++k) {
      const msl::Str p = nm.peer + std::to_string(groupIdx) + "_" +
                         std::to_string(si) + "_" + std::to_string(k);
      body.push_back(
          c.declStmt(mslTypeOf(plan.elemAt(k)), p,
                     shuffleXor(c, accs[k], st.xorOffset, plan.elemAt(k))));
      peers.push_back(p);
    }
    const Result<CombineNames> out = combine(body, accs, peers);
    if (!out.ok())
      return out.why;
    for (int k = 0; k < nOp; ++k)
      body.push_back(c.assign(c.var(accs[k]), c.var(out.value[k])));
  }
  return Decision::emitted();
}

// The cross-warp phase for every survivor group at once: each group publishes
// to its own slot range, one barrier, then each warp combines its own subset,
// anchored on its id. Groups of one reduction are independent, so they share
// the opening and closing barriers.
inline Decision emitWarpSteps(msl::Context &c, msl::Block &body,
                              const ReductionPlan &plan, int64_t numWarps,
                              std::vector<CombineNames> &groupAccs,
                              const ReduceNames &nm, const CombineFn &combine) {
  if (!plan.crossWarp() || groupAccs.empty())
    return Decision::emitted();
  const int nOp = (int)groupAccs[0].size();
  const ScratchLayout &slots = plan.scratch;

  // Every warp publishes, including those outside this reduction's subset, so
  // the reservation is numWarps slots per group.
  body.push_back(c.barrier());
  for (std::size_t gi = 0; gi < groupAccs.size(); ++gi)
    for (int k = 0; k < nOp; ++k) {
      msl::Expr *idx = slotExpr(c, slots, c.var(nm.warpId), nm.laneId, (int)gi);
      body.push_back(
          c.assign(c.subscript(c.var(nm.scratch[(std::size_t)k]), idx),
                   c.var(groupAccs[gi][k])));
    }
  body.push_back(c.barrier());

  for (std::size_t gi = 0; gi < groupAccs.size(); ++gi) {
    // The executing warp's subset anchor: `(warp & ~warpMask) * 32 + lane`.
    // warpSubset holds XOR offsets, so reads are relative to this anchor.
    msl::Expr *base =
        slotExpr(c, slots,
                 c.binary(msl::BinOp::And, c.var(nm.warpId),
                          c.lit((int64_t)plan.anchorMask(numWarps))),
                 nm.laneId, (int)gi);

    // Re-seed from the anchor slot; for a non-anchor warp that is not `accs`.
    CombineNames wacc;
    for (int k = 0; k < nOp; ++k) {
      const msl::Str w =
          nm.acc + "w" + std::to_string(gi) + "_" + std::to_string(k);
      body.push_back(
          c.declStmt(mslTypeOf(plan.elemAt(k)), w,
                     c.subscript(c.var(nm.scratch[(std::size_t)k]), base)));
      wacc.push_back(w);
    }

    for (std::size_t wi = 1; wi < plan.warpSubset.size(); ++wi) {
      CombineNames peers;
      for (int k = 0; k < nOp; ++k) {
        const msl::Str p = nm.peer + "w" + std::to_string(gi) + "_" +
                           std::to_string(wi) + "_" + std::to_string(k);
        msl::Expr *idx =
            c.binary(msl::BinOp::Add, base,
                     c.lit(slots.anchorSlots(plan.warpSubset[wi])));
        body.push_back(
            c.declStmt(mslTypeOf(plan.elemAt(k)), p,
                       c.subscript(c.var(nm.scratch[(std::size_t)k]), idx)));
        peers.push_back(p);
      }
      const Result<CombineNames> out = combine(body, wacc, peers);
      if (!out.ok())
        return out.why;
      for (int k = 0; k < nOp; ++k)
        body.push_back(c.assign(c.var(wacc[k]), c.var(out.value[k])));
    }
    groupAccs[gi] = wacc;
  }

  // Closes the scratch epoch: the pool overlays this scratch with other
  // regions, so a later write there must not overtake these reads.
  body.push_back(c.barrier());
  return Decision::emitted();
}

using ReduceResults = std::vector<CombineNames>;

// A whole reduction: for each survivor group, fold locally, then across lanes;
// then all groups across warps. Returns the accumulator names per group, in
// plan order.
inline Result<ReduceResults>
emitReduce(msl::Context &c, msl::Block &body, const ReductionPlan &plan,
           int64_t numWarps,
           const msl::SmallVec<msl::SmallVec<msl::Str, 8>, 4> &srcNames,
           const ReduceNames &nm, const CombineFn &combine) {
  using R = Result<ReduceResults>;
  const auto refused = [] {
    return R::no(Decision::declined(
        "emitReduce", "the emitter refused the plan it was given"));
  };
  if (srcNames.empty() || !plan.operandsShareLayout())
    return refused();
  for (std::size_t k = 1; k < srcNames.size(); ++k)
    if (srcNames[k].size() != srcNames[0].size())
      return refused();

  if (plan.crossWarp() && (int)nm.scratch.size() < (int)srcNames.size())
    return refused();

  ReduceResults results;
  for (int gi = 0; gi < plan.groupCount(); ++gi) {
    const ReductionGroup &g = plan.groups[gi];
    Result<CombineNames> accs =
        emitLocalFold(c, body, plan, g, srcNames, nm, gi, combine);
    if (!accs.ok())
      return R::no(accs.why);
    if (const Decision d =
            emitLaneSteps(c, body, plan, accs.value, nm, gi, combine);
        !d.ok())
      return R::no(d);
    results.push_back(accs.value);
  }
  if (const Decision d =
          emitWarpSteps(c, body, plan, numWarps, results, nm, combine);
      !d.ok())
    return R::no(d);
  return R::of(std::move(results));
}

} // namespace agpu

#endif // AGPU_EMIT_REDUCE_H
