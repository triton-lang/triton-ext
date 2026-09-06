// EmitScan.h - a prefix scan, emitted from its plan.
//
// Three ordered phases: registers, then the lane ladder, then every preceding
// warp's total. A wider axis becomes chained segments, each a whole scan
// taking the running total of the ones before.
#ifndef AGPU_EMIT_SCAN_H
#define AGPU_EMIT_SCAN_H

#include "agpu/core/Names.h"
#include "agpu/emit/EmitReduce.h"
#include "agpu/msl/Builtins.h"
#include "agpu/plan/ScanPlan.h"

namespace agpu {

struct ScanNames : ScratchNames {
  msl::Str acc = "sa";
  msl::Str peer = "sp";
  msl::Str carry = "scarry";
};

// One rung's shuffle: `simd_shuffle_up(v, delta)` or its downward twin, side
// chosen by the plan. Not shuffle_xor; an XOR butterfly mixes partials from
// both sides and gives a reduction.
inline msl::Expr *scanShuffle(msl::Context &c, const ScanPlan &p,
                              const msl::Str &v, int64_t delta, ElemType elem) {
  return shuffleOf(c, p.shuffleName(), elem, v,
                   c.lit(delta, msl::Context::u32()));
}

// A lane's position along the scanned axis; what the guards test. Masked only
// when the axis leaves lane bits free, so a tensor replicated across the warp
// restarts at each copy.
inline msl::Expr *axisLane(msl::Context &c, const ScanPlan &p,
                           const ScanNames &nm) {
  msl::Expr *l = c.var(nm.laneId);
  if (!p.guardNeedsMask(p.scratch.warpSize))
    return l;
  return c.binary(msl::BinOp::And, l, c.lit((int64_t)p.laneMask));
}

// One rung of the lane ladder. Lanes at the far end have no source and the
// shuffle leaves their value undefined, hence the guard.
inline void emitScanStep(msl::Context &c, msl::Block &body, const ScanPlan &p,
                         const ScanStep &step, msl::SmallVec<msl::Str, 4> &accs,
                         const ScanNames &nm, int stepIdx,
                         const CombineFn &combine) {
  const int nOp = (int)accs.size();
  msl::SmallVec<msl::Str, 4> peers;
  for (int k = 0; k < nOp; ++k) {
    const msl::Str pn =
        nm.peer + std::to_string(stepIdx) + "_" + std::to_string(k);
    body.push_back(
        c.declStmt(mslTypeOf(p.elemAt(k)), pn,
                   scanShuffle(c, p, accs[k], step.delta, p.elemAt(k))));
    peers.push_back(pn);
  }

  // Peer first: the combine takes the earlier element as its first argument
  // group and `shuffle_up` fetches from the lane before this one.
  msl::Block inner;
  msl::SmallVec<msl::Str, 4> out = combine(inner, peers, accs);
  for (int k = 0; k < nOp; ++k)
    inner.push_back(c.assign(c.var(accs[k]), c.var(out[k])));

  msl::Expr *cond =
      step.guarded
          ? c.binary(p.guardOp(), axisLane(c, p, nm),
                     c.lit(p.guardBound(step.delta, p.scratch.warpSize)))
          : nullptr;
  c.guardedInto(body, cond, std::move(inner));
}

// The lane phase: the ladder, in increasing delta.
inline void emitScanLanes(msl::Context &c, msl::Block &body, const ScanPlan &p,
                          msl::SmallVec<msl::Str, 4> &accs, const ScanNames &nm,
                          const CombineFn &combine) {
  if (const char *fn = p.laneIntrinsic(p.scratch.warpSize)) {
    body.push_back(c.assign(c.var(accs[0]), c.call(fn, {c.var(accs[0])})));
    return;
  }
  for (std::size_t i = 0; i < p.laneSteps.size(); ++i)
    emitScanStep(c, body, p, p.laneSteps[i], accs, nm, (int)i, combine);
}

// ── the cross-warp phase ──────────────────────────────────────────────────
// Each warp publishes its total, then adds the totals of every warp before it.
// A prefix over warps: warp w takes warps [0, w).

// One publish per scan regardless of register count; a total belongs to the
// warp.
inline void emitScanPublish(msl::Context &c, msl::Block &body,
                            const ScanPlan &p,
                            const msl::SmallVec<msl::Str, 4> &accs,
                            const ScanNames &nm) {
  if (!p.crossWarp)
    return;
  const int nOp = (int)accs.size();
  const ScratchLayout &slots = p.scratch;

  // After the ladder one lane holds the warp's total: the last along the axis
  // for a forward scan, the first for a reverse one. That serves the whole
  // warp only when the axis covers every lane bit; otherwise each lane
  // publishes its own slot unguarded. See `publishesPerLane`.
  body.push_back(c.barrier());
  for (int k = 0; k < nOp; ++k) {
    msl::Expr *idx = p.publishesPerLane(slots.warpSize)
                         ? slotExpr(c, slots, c.var(nm.warpId), nm.laneId)
                         : anchorExpr(c, slots, c.var(nm.warpId));
    msl::Expr *who = p.publishesPerLane(slots.warpSize)
                         ? nullptr
                         : c.binary(msl::BinOp::Eq, c.var(nm.laneId),
                                    c.lit(p.totalLane(slots.warpSize)));
    c.guardedInto(
        body, who,
        msl::Block{c.assign(c.subscript(c.var(nm.scratch[(std::size_t)k]), idx),
                            c.var(accs[k]))});
  }
  body.push_back(c.barrier());
}

// Warp `w`'s published value for operand `k`, read from the executing warp's
// own subset. The slot is relative to the subset anchor (`warpId` masked by
// what the axis does not traverse), since `carryWarps` yields positions within
// the axis.
inline msl::Expr *scanTotalSlot(msl::Context &c, const ScanPlan &p,
                                const ScanNames &nm, int64_t numWarps,
                                int64_t w, int k) {
  const ScratchLayout &slots = p.scratch;
  msl::Expr *anchor = c.binary(msl::BinOp::And, c.var(nm.warpId),
                               c.lit((int64_t)p.anchorMask(numWarps)));
  msl::Expr *warpTerm = c.binary(msl::BinOp::Add, anchor, c.lit(w));
  if (!p.publishesPerLane(slots.warpSize))
    return c.subscript(c.var(nm.scratch[(std::size_t)k]),
                       anchorExpr(c, slots, warpTerm));

  msl::Expr *column =
      c.binary(msl::BinOp::And, c.var(nm.laneId),
               c.lit((int64_t)p.carryLaneKeepMask(slots.warpSize)));
  msl::Expr *slot =
      c.binary(msl::BinOp::Or, column,
               c.lit(p.totalLane(slots.warpSize) & (int64_t)p.laneMask));
  return c.subscript(
      c.var(nm.scratch[(std::size_t)k]),
      c.binary(msl::BinOp::Add, anchorExpr(c, slots, warpTerm), slot));
}

// The preceding warps' totals folded into one value, then applied to every
// register. The first `carryFoldOrder` entry seeds the carry unguarded, each
// later one folds in under its own guard, and the application runs once under
// the seed's guard.
inline void emitScanCarry(msl::Context &c, msl::Block &body, const ScanPlan &p,
                          int64_t numWarps,
                          const std::vector<msl::SmallVec<msl::Str, 8>> &acc,
                          const ScanNames &nm, const CombineFn &combine) {
  if (!p.crossWarp)
    return;
  const std::vector<int64_t> fold = p.carryFoldOrder(numWarps);
  if (fold.empty() || acc.empty())
    return;
  const int nOp = (int)acc.size();

  // This warp's position within its subset: `warpId` masked by what the axis
  // does traverse.
  msl::Expr *position =
      c.binary(msl::BinOp::And, c.var(nm.warpId), c.lit((int64_t)p.warpMask));

  msl::SmallVec<msl::Str, 4> carry;
  for (int k = 0; k < nOp; ++k) {
    const msl::Str v = nm.carry + std::to_string(k);
    body.push_back(c.declStmt(mslTypeOf(p.elemAt(k)), v,
                              scanTotalSlot(c, p, nm, numWarps, fold[0], k)));
    carry.push_back(v);
  }
  for (std::size_t i = 1; i < fold.size(); ++i) {
    const int64_t w = fold[i];
    msl::Block inner;
    msl::SmallVec<msl::Str, 4> peers;
    for (int k = 0; k < nOp; ++k) {
      const msl::Str v = nm.carry + std::to_string(w) + "_" + std::to_string(k);
      inner.push_back(c.declStmt(mslTypeOf(p.elemAt(k)), v,
                                 scanTotalSlot(c, p, nm, numWarps, w, k)));
      peers.push_back(v);
    }
    // The carry so far is the earlier value, so it leads.
    msl::SmallVec<msl::Str, 4> out = combine(inner, carry, peers);
    for (int k = 0; k < nOp; ++k)
      inner.push_back(c.assign(c.var(carry[k]), c.var(out[k])));
    c.guardedInto(body, c.binary(p.carryOp(), position, c.lit(w)),
                  std::move(inner));
  }

  // Every register, every lane. A preceding warp's total belongs to all 32
  // lanes here, including the lane the in-warp prefix guard rejects.
  msl::Block apply;
  const std::size_t regs = acc[0].size();
  for (std::size_t r = 0; r < regs; ++r) {
    msl::SmallVec<msl::Str, 4> regAcc;
    for (int k = 0; k < nOp; ++k)
      regAcc.push_back(acc[(std::size_t)k][r]);
    msl::SmallVec<msl::Str, 4> out = combine(apply, carry, regAcc);
    for (int k = 0; k < nOp; ++k)
      apply.push_back(c.assign(c.var(acc[(std::size_t)k][r]), c.var(out[k])));
  }
  c.guardedInto(body, c.binary(p.carryOp(), position, c.lit(fold[0])),
                std::move(apply));

  // Closes the scratch epoch: the pool overlays this scratch with other
  // regions, so a later write there must not overtake these reads.
  body.push_back(c.barrier());
}

// The running segment total between a chained scan's windows, carried through
// the chain slots. Readers take the other parity's row, so no slot is read and
// written in one barrier epoch and the chain needs no barriers of its own.

// This thread's slot in `parity`'s row.
inline msl::Expr *scanChainSlot(msl::Context &c, const ScanPlan &p,
                                const ScanNames &nm, int64_t numWarps,
                                int64_t parity, int k) {
  msl::Expr *idx = c.lit(p.chainBase(numWarps, parity));
  if (p.anchorMask(numWarps) != 0)
    idx =
        c.binary(msl::BinOp::Add, idx,
                 anchorExpr(c, p.scratch,
                            c.binary(msl::BinOp::And, c.var(nm.warpId),
                                     c.lit((int64_t)p.anchorMask(numWarps)))));
  if (p.carryLaneKeepMask(p.scratch.warpSize) != 0)
    idx = c.binary(
        msl::BinOp::Add, idx,
        c.binary(msl::BinOp::And, c.var(nm.laneId),
                 c.lit((int64_t)p.carryLaneKeepMask(p.scratch.warpSize))));
  return c.subscript(c.var(nm.scratch[(std::size_t)k]), idx);
}

// The previous window's running total, read by every thread from its slot.
inline msl::SmallVec<msl::Str, 4>
emitScanChainRead(msl::Context &c, msl::Block &body, const ScanPlan &p,
                  int64_t numWarps, int nOp, int64_t parity,
                  const ScanNames &nm) {
  msl::SmallVec<msl::Str, 4> carry;
  for (int k = 0; k < nOp; ++k) {
    const msl::Str v = nm.carry + "s" + std::to_string(k);
    body.push_back(c.declStmt(mslTypeOf(p.elemAt(k)), v,
                              scanChainSlot(c, p, nm, numWarps, parity, k)));
    carry.push_back(v);
  }
  return carry;
}

// This window's running total, written by the thread that holds it: the one at
// the axis's final warp and lane position, in its last axis-order register.
// The previous segments' carry is already applied, so no fold is needed.
inline void
emitScanChainWrite(msl::Context &c, msl::Block &body, const ScanPlan &p,
                   int64_t numWarps, int64_t parity,
                   const msl::SmallVec<msl::SmallVec<msl::Str, 8>, 4> &results,
                   const ScanNames &nm) {
  const int nOp = (int)results.size();
  msl::Block write;
  for (int k = 0; k < nOp; ++k)
    write.push_back(c.assign(scanChainSlot(c, p, nm, numWarps, parity, k),
                             c.var(results[(std::size_t)k].back())));

  msl::Expr *holder = c.binary(
      msl::BinOp::Eq,
      c.binary(msl::BinOp::And, c.var(nm.warpId), c.lit((int64_t)p.warpMask)),
      c.lit(p.finalWarp()));
  if (p.laneMask != 0)
    holder = c.binary(
        msl::BinOp::And, holder,
        c.binary(msl::BinOp::Eq,
                 c.binary(msl::BinOp::And, c.var(nm.laneId),
                          c.lit((int64_t)p.laneMask)),
                 c.lit(p.totalLane(p.scratch.warpSize) & (int64_t)p.laneMask)));
  c.guardedInto(body, holder, std::move(write));
}

// The same hand-off within one warp, which needs no slot: the running total
// sits at the axis's final lane position in this lane's column and a shuffle
// broadcasts it.
inline msl::SmallVec<msl::Str, 4> emitScanChainBroadcast(
    msl::Context &c, msl::Block &body, const ScanPlan &p,
    const msl::SmallVec<msl::SmallVec<msl::Str, 8>, 4> &results,
    const ScanNames &nm) {
  const int nOp = (int)results.size();
  msl::SmallVec<msl::Str, 4> carry;
  for (int k = 0; k < nOp; ++k) {
    msl::Expr *src = c.binary(
        msl::BinOp::Or,
        c.binary(msl::BinOp::And, c.var(nm.laneId),
                 c.lit((int64_t)p.carryLaneKeepMask(p.scratch.warpSize))),
        c.lit(p.totalLane(p.scratch.warpSize) & (int64_t)p.laneMask));
    const msl::Str v = nm.carry + "s" + std::to_string(k);
    body.push_back(
        c.declStmt(mslTypeOf(p.elemAt(k)), v,
                   shuffleOf(c, msl::builtin::simd::Shuffle, p.elemAt(k),
                             results[(std::size_t)k].back(), src)));
    carry.push_back(v);
  }
  return carry;
}

// A whole scan over one thread's registers. `srcNames[k][r]` is operand k's
// register r in axis order, reversed by the caller for a reverse scan; that
// ordering applies to the local pass only. Returns `results[k][r]`.
inline msl::SmallVec<msl::SmallVec<msl::Str, 8>, 4>
emitScan(msl::Context &c, msl::Block &body, const ScanPlan &p, int64_t numWarps,
         const msl::SmallVec<msl::SmallVec<msl::Str, 8>, 4> &srcNames,
         const ScanNames &nm, const CombineFn &combine) {
  msl::SmallVec<msl::SmallVec<msl::Str, 8>, 4> results;
  if (!p.usable || srcNames.empty() || srcNames[0].empty())
    return results;

  const int nOp = (int)srcNames.size();
  const int64_t regs = (int64_t)srcNames[0].size();

  // The plan refuses operands that disagree (operandsShareLayout), so a
  // mismatch here means the caller's names don't match what it planned.
  for (int k = 1; k < nOp; ++k)
    if ((int64_t)srcNames[k].size() != regs)
      return results;

  // A thread's registers are not always one window: `windowRegs` of them go
  // through the shuffle-and-carry machinery, the rest are other columns or
  // further segments of the same axis. Each window gets its own local fold,
  // lane ladder and cross-warp carry; `chainedWindows` consecutive windows
  // also fold in the running total of the segments before them. Emitted by
  // recursion into the single-scan path below.
  if (p.windowRegs < regs) {
    results.resize((std::size_t)nOp);
    // The running segment total when it travels in registers; across warps
    // it travels through the chain slots instead and this stays empty.
    msl::SmallVec<msl::Str, 4> segCarry;
    int64_t windowIdx = 0;
    for (int64_t base = 0; base < regs; base += p.windowRegs, ++windowIdx) {
      const int64_t inChain = windowIdx % p.chainedWindows;
      if (inChain == 0)
        segCarry.clear();
      const int64_t n = std::min<int64_t>(p.windowRegs, regs - base);
      msl::SmallVec<msl::SmallVec<msl::Str, 8>, 4> window;
      for (int k = 0; k < nOp; ++k) {
        msl::SmallVec<msl::Str, 8> one;
        for (int64_t i = 0; i < n; ++i)
          one.push_back(srcNames[k][(std::size_t)(base + i)]);
        window.push_back(std::move(one));
      }

      ScanNames rn = nm;
      const msl::Str tag = "c" + std::to_string(base);
      rn.acc = nm.acc + tag;
      rn.peer = nm.peer + tag;
      rn.carry = nm.carry + tag;

      const auto got = emitScan(c, body, p, numWarps, window, rn, combine);
      if (got.size() != (std::size_t)nOp)
        return {};

      // The previous segments' running total, into every register of this
      // one. Unguarded: a preceding segment precedes every element here.
      if (inChain > 0) {
        const msl::SmallVec<msl::Str, 4> chain =
            p.crossWarp ? emitScanChainRead(c, body, p, numWarps, nOp,
                                            (inChain - 1) & 1, rn)
                        : segCarry;
        for (int64_t i = 0; i < n; ++i) {
          msl::SmallVec<msl::Str, 4> regAcc;
          for (int k = 0; k < nOp; ++k)
            regAcc.push_back(got[(std::size_t)k][(std::size_t)i]);
          msl::SmallVec<msl::Str, 4> out = combine(body, chain, regAcc);
          for (int k = 0; k < nOp; ++k)
            body.push_back(c.assign(c.var(got[(std::size_t)k][(std::size_t)i]),
                                    c.var(out[k])));
        }
      }

      // After the carry above, the holder's last register ends with the
      // running total through this segment, so hand the chain forward here.
      if (inChain + 1 < p.chainedWindows && base + n < regs) {
        if (p.crossWarp)
          emitScanChainWrite(c, body, p, numWarps, inChain & 1, got, rn);
        else
          segCarry = emitScanChainBroadcast(c, body, p, got, rn);
      }

      for (int k = 0; k < nOp; ++k)
        for (const msl::Str &s : got[(std::size_t)k])
          results[(std::size_t)k].push_back(s);
    }
    return results;
  }

  if (p.crossWarp && (int)nm.scratch.size() < nOp)
    return {};

  // `acc[k][r]` is operand k's accumulator for register r; it ends holding
  // that register's own prefix.
  std::vector<msl::SmallVec<msl::Str, 8>> acc((std::size_t)nOp);
  for (int k = 0; k < nOp; ++k)
    for (int64_t r = 0; r < regs; ++r) {
      const msl::Str a = nm.acc + std::to_string(k) + "_" + std::to_string(r);
      body.push_back(
          c.declStmt(mslTypeOf(p.elemAt(k)), a, c.var(srcNames[k][r])));
      acc[(std::size_t)k].push_back(a);
    }

  auto at = [&](int64_t r) {
    msl::SmallVec<msl::Str, 4> out;
    for (int k = 0; k < nOp; ++k)
      out.push_back(acc[(std::size_t)k][(std::size_t)r]);
    return out;
  };
  auto storeAt = [&](msl::Block &into, int64_t r,
                     const msl::SmallVec<msl::Str, 4> &from) {
    for (int k = 0; k < nOp; ++k)
      into.push_back(
          c.assign(c.var(acc[(std::size_t)k][(std::size_t)r]), c.var(from[k])));
  };

  // The local pass: register r absorbs register r-1's running total in place
  // and keeps its own partial.
  for (int64_t r = 1; r < regs; ++r)
    storeAt(body, r, combine(body, at(r - 1), at(r)));

  // The cross-lane phases run on a separate accumulator, seeded from the last
  // register, which holds this thread's whole running total.
  msl::SmallVec<msl::Str, 4> laneScan;
  for (int k = 0; k < nOp; ++k) {
    const msl::Str a = nm.acc + "x" + std::to_string(k);
    body.push_back(
        c.declStmt(mslTypeOf(p.elemAt(k)), a,
                   c.var(acc[(std::size_t)k][(std::size_t)(regs - 1)])));
    laneScan.push_back(a);
  }

  emitScanLanes(c, body, p, laneScan, nm, combine);

  // Published straight after the ladder: a warp publishes its whole total,
  // which the ladder just left in its top lane.
  emitScanPublish(c, body, p, laneScan, nm);

  // The preceding lanes' contribution, folded into every register, before the
  // cross-warp carry. This fold is guarded (the first lane along the axis has
  // no predecessor) while the warp carry applies to every lane, so the order
  // matters. Shuffled by the smallest delta, one hop back.
  if (const char *fn = p.prefixIntrinsic(p.scratch.warpSize)) {
    const msl::Str a = nm.acc + "p0";
    body.push_back(
        c.declStmt(mslTypeOf(p.elemAt(0)), a,
                   c.call(fn, {c.var(acc[0][(std::size_t)(regs - 1)])})));
    const msl::SmallVec<msl::Str, 4> prefix = {a};
    for (int64_t r = 0; r < regs; ++r)
      storeAt(body, r, combine(body, prefix, at(r)));
  } else if (!p.laneSteps.empty()) {
    const int64_t low = p.laneSteps.front().delta;
    msl::SmallVec<msl::Str, 4> prefix;
    for (int k = 0; k < nOp; ++k) {
      const msl::Str a = nm.acc + "p" + std::to_string(k);
      body.push_back(
          c.declStmt(mslTypeOf(p.elemAt(k)), a,
                     scanShuffle(c, p, laneScan[k], low, p.elemAt(k))));
      prefix.push_back(a);
    }
    msl::Block inner;
    for (int64_t r = 0; r < regs; ++r)
      storeAt(inner, r, combine(inner, prefix, at(r)));
    c.guardedInto(body,
                  c.binary(p.guardOp(), axisLane(c, p, nm),
                           c.lit(p.guardBound(low, p.scratch.warpSize))),
                  std::move(inner));
  }

  // The warps before this one: one carry, folded into every register.
  emitScanCarry(c, body, p, numWarps, acc, nm, combine);

  for (int k = 0; k < nOp; ++k) {
    msl::SmallVec<msl::Str, 8> one;
    for (int64_t r = 0; r < regs; ++r)
      one.push_back(acc[(std::size_t)k][(std::size_t)r]);
    results.push_back(std::move(one));
  }
  return results;
}

} // namespace agpu

#endif // AGPU_EMIT_SCAN_H
