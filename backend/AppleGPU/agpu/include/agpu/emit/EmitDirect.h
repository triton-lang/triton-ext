// EmitDirect.h - the direct dot path.
#ifndef AGPU_EMIT_DIRECT_H
#define AGPU_EMIT_DIRECT_H

#include "agpu/core/Names.h"
#include "agpu/emit/EmitEpilogue.h"
#include "agpu/emit/EmitPanel.h"
#include "agpu/emit/primitives/DeviceStoreTarget.h"
#include "agpu/emit/primitives/FragLane.h"
#include "agpu/emit/primitives/OperandSource.h"
#include "agpu/emit/primitives/SlotExpr.h"
#include "agpu/emit/primitives/Stride.h"
#include "agpu/msl/Builtins.h"
#include "agpu/msl/Context.h"
#include "agpu/plan/DotPassSchedule.h"
#include "agpu/plan/DotPlan.h"
#include "agpu/plan/ReadbackPlan.h"
#include "agpu/plan/WarpSlots.h"

#include <functional>
#include <map>
#include <utility>

namespace agpu {

struct DirectInputs {
  OperandSource a, b;
  int64_t kT = 1;     // fragments along K
  bool rollK = false; // emit a K loop, leaving it rolled
};

// How C comes back out of the pool, in the caller's layout. Produced per band
// by `readbackFor`. A null callback means the caller addresses `poolC` itself.
struct ReadbackInputs {
  msl::SmallVec<StageAction, 8> actions;
  msl::SmallVec<msl::Str, 8> names;
  msl::SmallVec<msl::Str, 8> bases;

  // What poolC holds.
  ElemType elem = f32();

  // What the destination registers hold: i32 for the lifted integer dot,
  // whose pool stays f32.
  ElemType regElem = f32();

  ReadbackPlan plan;

  bool empty() const { return actions.empty(); }
};

// The fused path uses this too: emitAccumDecls once before its K loop, this
// inside it.
inline void emitDirectMma(msl::Context &c, msl::Block &body,
                          const std::vector<WarpSlot> &slots,
                          const DirectInputs &in, const DirectNames &nm,
                          int &counter, FragShare share = {}) {
  const bool shared = share.active() && !in.rollK;
  FragCache localA, localB;
  FragCache &aCache = shared ? *share.a : localA;
  FragCache &bCache = shared ? *share.b : localB;

  // Fold each operand's warp term into one base pointer and load at literal
  // offsets: a `simdgroup_load` whose address mentions the warp id compiles
  // measurably slower than base-plus-literal.
  DirectInputs ind = in;
  std::vector<WarpSlot> lslots = slots;
  const auto rebase = [&](OperandSource &src, bool isA) {
    SlotCoord w{};
    bool any = false;
    for (const WarpSlot &s : slots) {
      const SlotCoord &p = isA ? s.mi : s.ni;
      if (p.isConst())
        continue;
      SlotCoord t = p;
      t.constant = 0;
      if (any && !(t == w))
        return;
      w = t;
      any = true;
    }
    if (!any)
      return;
    const msl::Str name = "p" + nm.frag + std::to_string(counter++);
    body.push_back(c.declStmt(msl::Type::named(nm.opElem).pointerTo(src.space),
                              name,
                              c.binary(msl::BinOp::Add, c.var(src.buffer),
                                       src.axisOffsetOf(c, w, nm.warpId))));
    src.buffer = name;
    for (WarpSlot &s : lslots) {
      SlotCoord &p = isA ? s.mi : s.ni;
      if (!p.isConst())
        p = SlotCoord::fixed(p.constant);
    }
  };
  rebase(ind.a, true);
  rebase(ind.b, false);
  const DirectInputs &inr = ind;
  const std::vector<WarpSlot> &rslots = lslots;

  auto accumulate = [&](msl::Block &into, KStep k, int64_t kIndex) {
    msl::Block &decls = shared ? *share.decls : into;
    // Two passes: every distinct fragment's load, then the MMAs. The second
    // pass hits the cache the first filled.
    const auto frags = [&](const WarpSlot &s) {
      msl::Expr *kFrags = kTermExpr(c, k.kOffset(1), nm.kVar);
      return std::pair(loadFrag(c, into, decls, aCache, inr.a, s.mi, kIndex,
                                inr.a.kOffsetOf(c, kFrags), nm, counter),
                       loadFrag(c, into, decls, bCache, inr.b, s.ni, kIndex,
                                inr.b.kOffsetOf(c, kFrags), nm, counter));
    };
    for (const WarpSlot &s : rslots)
      frags(s);
    for (const WarpSlot &s : rslots) {
      const auto [fa, fb] = frags(s);
      into.push_back(sgMma(c, nm.acc + std::to_string(s.acc), fa, fb));
    }
  };

  if (in.rollK) {
    msl::Block inner;
    accumulate(inner, KStep::rolled(), 0);
    body.push_back(c.forStmt(
        c.declStmt(msl::Context::i32(), nm.kVar, c.lit(0)),
        c.binary(msl::BinOp::Lt, c.var(nm.kVar), c.lit(in.kT * kSgFragDim)),
        c.assignOp(msl::BinOp::Add, c.var(nm.kVar), c.lit(kSgFragDim)),
        std::move(inner)));
    return;
  }

  for (int64_t ki = 0; ki < in.kT; ++ki)
    accumulate(body, KStep::unrolled(ki), ki);
}

inline void emitAccumDecls(msl::Context &c, msl::Block &body,
                           const std::vector<WarpSlot> &slots,
                           const DirectNames &nm) {
  for (const WarpSlot &s : slots)
    body.push_back(c.declStmt(
        kSimdgroup8x8.mslTypeNode(nm.accElem), nm.acc + std::to_string(s.acc),
        c.call(kSimdgroup8x8.zeroCtor(nm.accElem), {c.litF(0.0)})));
}

// Store each accumulator into the C pool at its slot's position, relative to
// the band that pool region currently holds. `rowFrag0` is the band's first
// fragment row; an unbanded dot passes 0 and the subtraction folds away.
inline void emitAccumStores(msl::Context &c, msl::Block &body,
                            const std::vector<WarpSlot> &slots,
                            const TileView &cv, const DirectNames &nm,
                            int64_t rowFrag0) {
  for (const WarpSlot &s : slots) {
    const SlotCoord mi =
        s.mi.isConst() ? SlotCoord::fixed(s.mi.constant - rowFrag0) : s.mi;
    msl::Expr *off = fragOffset(c, cv, {mi, s.ni}, nullptr, nm.warpId);
    body.push_back(
        c.exprStmt(c.call(msl::builtin::sg::Store,
                          {c.var(nm.acc + std::to_string(s.acc)),
                           c.binary(msl::BinOp::Add, c.var(nm.poolC), off),
                           c.lit(cv.strideAt(0))})));
  }
}

inline msl::Expr *wholeTileInBounds(msl::Context &c,
                                    const DeviceStoreTarget &t) {
  if (!t.bounded() || t.tileRows <= 0 || t.tileCols <= 0)
    return nullptr;
  const auto axisIn = [&](msl::Expr *start, int64_t extent, msl::Expr *bound) {
    msl::Expr *end =
        start ? c.binary(msl::BinOp::Add, start, c.lit(extent)) : c.lit(extent);
    return c.binary(msl::BinOp::Le, end, bound);
  };
  msl::Expr *allIn = nullptr;
  if (t.rowBound)
    allIn = axisIn(t.rowStart, t.tileRows, t.rowBound);
  if (t.colBound) {
    msl::Expr *cIn = axisIn(t.colStart, t.tileCols, t.colBound);
    allIn = allIn ? c.binary(msl::BinOp::LAnd, allIn, cIn) : cIn;
  }
  return allIn;
}

inline void declareEdgeScratch(msl::Context &c, msl::Block &slow,
                               const DeviceStoreTarget &t,
                               const DirectNames &nm) {
  const msl::Str scr = directScratchName(nm.frag);
  const msl::Type ptr = mslTypeOf(f32()).pointerTo(msl::AddrSpace::Threadgroup);
  slow.push_back(
      c.declStmt(ptr, scr,
                 c.binary(msl::BinOp::Add, c.cast(ptr, c.var(t.edgeScratch)),
                          c.binary(msl::BinOp::Mul, c.var(nm.warpId),
                                   c.lit(kSgFragDim * kSgFragDim)))));
}

inline msl::Expr *fragmentInBounds(msl::Context &c, const DeviceStoreTarget &t,
                                   msl::Expr *row, msl::Expr *col) {
  msl::Expr *inside = nullptr;
  const auto also = [&](msl::Expr *cond) {
    inside = inside ? c.binary(msl::BinOp::LAnd, inside, cond) : cond;
  };
  if (t.rowBound)
    also(c.binary(msl::BinOp::Le,
                  c.binary(msl::BinOp::Add, row, c.lit(kSgFragDim)),
                  t.rowBound));
  if (t.colBound)
    also(c.binary(msl::BinOp::Le,
                  c.binary(msl::BinOp::Add, col, c.lit(kSgFragDim)),
                  t.colBound));
  return inside;
}

using DrainValueAt = std::function<msl::Expr *(msl::Expr *value, msl::Expr *row,
                                               msl::Expr *col)>;

// Rounds through a per-warp 8x8 pool scratch so the scalar copies can spell
// literal rows: a lane-varying row index in the store address costs a register
// allocation step. Stored row-major and copied a row at a time, so the lane is
// the column. A simdgroup barrier is enough between fragments since each warp
// has its own scratch.
inline void emitScratchEdgeStores(msl::Context &c, msl::Block &edge,
                                  const DeviceStoreTarget &t,
                                  const DirectNames &nm, const WarpSlot &s,
                                  msl::Expr *rowBase, msl::Expr *colBase,
                                  const DrainValueAt &chainAt) {
  const msl::Str scr = directScratchName(nm.frag);
  edge.push_back(c.exprStmt(
      c.call(msl::builtin::sg::Store, {c.var(nm.acc + std::to_string(s.acc)),
                                       c.var(scr), c.lit(kSgFragDim)})));
  edge.push_back(c.barrier(msl::Barrier::Scope::Simdgroup));
  for (int64_t r = 0; r < kSgFragDim; ++r) {
    msl::Expr *rowAbs = c.binary(msl::BinOp::Add, rowBase, c.lit(r));
    msl::Expr *colAbs = c.binary(msl::BinOp::Add, colBase, c.var(nm.laneId));
    msl::Expr *guard =
        c.binary(msl::BinOp::Lt, c.var(nm.laneId), c.lit(kSgFragDim));
    if (t.rowBound)
      guard = c.binary(msl::BinOp::LAnd, guard,
                       c.binary(msl::BinOp::Lt, rowAbs, t.rowBound));
    if (t.colBound)
      guard = c.binary(msl::BinOp::LAnd, guard,
                       c.binary(msl::BinOp::Lt, colAbs, t.colBound));
    msl::Expr *off =
        c.binary(msl::BinOp::Add, t.leadingDim.scale(c, rowAbs), colAbs);
    msl::Expr *value = chainAt(
        c.subscript(c.var(scr), c.binary(msl::BinOp::Add, c.lit(r * kSgFragDim),
                                         c.var(nm.laneId))),
        rowAbs, colAbs);
    msl::Block one;
    one.push_back(
        c.assign(c.subscript(basePtr(c, t.base, t.baseOffset), off), value));
    c.guardedInto(edge, guard, std::move(one));
  }
  edge.push_back(c.barrier(msl::Barrier::Scope::Simdgroup));
}

inline void
emitGuardedScalarEdgeStores(msl::Context &c, msl::Block &edge,
                            const DeviceStoreTarget &t, const msl::Str &acc,
                            const std::function<msl::Expr *(int64_t)> &elemRow,
                            const std::function<msl::Expr *(int64_t)> &elemCol,
                            const DrainValueAt &chainAt) {
  for (int64_t i = 0; i < kFragElemsPerLane; ++i) {
    msl::Expr *row = elemRow(i);
    msl::Expr *col = elemCol(i);
    msl::Expr *guard = nullptr;
    const auto guardAlso = [&](msl::Expr *cond) {
      guard = guard ? c.binary(msl::BinOp::LAnd, guard, cond) : cond;
    };
    if (t.rowBound)
      guardAlso(c.binary(msl::BinOp::Lt, elemRow(i), t.rowBound));
    if (t.colBound)
      guardAlso(c.binary(msl::BinOp::Lt, elemCol(i), t.colBound));
    msl::Expr *off = c.binary(msl::BinOp::Add, t.leadingDim.scale(c, row), col);
    msl::Expr *value = chainAt(fragElemExpr(c, acc, i), row, col);
    msl::Block one;
    one.push_back(
        c.assign(c.subscript(basePtr(c, t.base, t.baseOffset), off), value));
    c.guardedInto(edge, guard, std::move(one));
  }
}

// Drain each accumulator straight to its fragment's place in the device
// tensor, for `Plan::storesCDirect`: no pool, no readback, no barrier.
//
// A bounded target (`t.rowBound`/`colBound`) is tested once for the whole tile
// where the extents allow it, so interior tiles run bare simdgroup_stores
// behind a single test. A tile that fails the test splits per fragment: whole
// fragments store whole, one crossing the edge stores guarded scalars.
inline void emitAccumDeviceStores(msl::Context &c, msl::Block &body,
                                  const std::vector<WarpSlot> &slots,
                                  const DeviceStoreTarget &t,
                                  const std::vector<DrainStep> &steps,
                                  const DirectNames &nm) {
  msl::Block drained;
  msl::Block &into = t.uniformGuard ? drained : body;

  msl::Expr *allIn = wholeTileInBounds(c, t);

  msl::Block fast, slow;
  // The unguarded arms' operand reads, memoised so a store between two slots'
  // reads cannot force a reload. Only valid where the arm is unguarded and
  // covers the whole tile; guarded arms read in place. `memoInto` is whichever
  // unguarded block is being built.
  std::map<std::string, msl::Str> fastReads;
  int fastReadSeq = 0;
  msl::Block *memoInto = &fast;
  // Memo arms' stores, deferred until after all reads, same reason.
  msl::Block memoStores;
  if (allIn && !t.edgeScratch.empty())
    declareEdgeScratch(c, slow, t, nm);
  for (const WarpSlot &s : slots) {
    const auto rowE = [&]() {
      msl::Expr *row = c.binary(msl::BinOp::Mul, coordOf(c, s.mi, nm.warpId),
                                c.lit(kSgFragDim));
      return t.rowStart ? c.binary(msl::BinOp::Add, t.rowStart, row) : row;
    };
    const auto colE = [&]() {
      msl::Expr *col = c.binary(msl::BinOp::Mul, coordOf(c, s.ni, nm.warpId),
                                c.lit(kSgFragDim));
      return t.colStart ? c.binary(msl::BinOp::Add, t.colStart, col) : col;
    };

    const msl::Str acc = nm.acc + std::to_string(s.acc);
    const auto sgStore = [&](msl::Block &into, const msl::Str &frag) {
      msl::Expr *off =
          c.binary(msl::BinOp::Add, t.leadingDim.scale(c, rowE()), colE());
      into.push_back(c.exprStmt(c.call(
          msl::builtin::sg::Store,
          {c.var(frag),
           c.binary(msl::BinOp::Add, basePtr(c, t.base, t.baseOffset), off),
           t.leadingDim.expr(c)})));
    };

    const auto chainAt = [&](msl::Expr *value, msl::Expr *row, msl::Expr *col,
                             const std::string &memoElem =
                                 std::string()) -> msl::Expr * {
      // `key` keeps a spine step's memoised read distinct from a branch
      // link's.
      const auto operandRead = [&](const DrainOperand &od,
                                   const std::string &keyAt) -> msl::Expr * {
        msl::Expr *rhs = nullptr;
        switch (od.kind) {
        default:
          break;
        case DrainOperand::Kind::Splat:
          rhs = od.splat;
          break;
        case DrainOperand::Kind::Row:
          rhs = c.subscript(basePtr(c, od.base, od.baseOffset), col);
          break;
        case DrainOperand::Kind::Col:
          rhs = c.subscript(basePtr(c, od.base, od.baseOffset), row);
          break;
        case DrainOperand::Kind::Tile:
          rhs = c.subscript(
              basePtr(c, od.base, od.baseOffset),
              c.binary(msl::BinOp::Add, od.leadingDim.scale(c, row), col));
          break;
        }
        if (rhs && !memoElem.empty() && od.kind != DrainOperand::Kind::Splat) {
          const auto coord = [](const SlotCoord &p) {
            return std::to_string(p.constant) + ":" +
                   std::to_string(p.warpScale) + ":" +
                   std::to_string(p.warpDiv) + ":" + std::to_string(p.warpMod);
          };
          std::string key = keyAt + "|";
          if (od.kind != DrainOperand::Kind::Col)
            key += coord(s.ni) + "@" + memoElem;
          if (od.kind != DrainOperand::Kind::Row)
            key += "|" + coord(s.mi);
          auto it = fastReads.find(key);
          if (it == fastReads.end()) {
            const msl::Str rn = nm.frag + "r" + std::to_string(fastReadSeq++);
            // The operand's own element: a wider memo would promote the step
            // consuming it and round differently.
            memoInto->push_back(c.declStmt(mslTypeOf(od.elem), rn, rhs));
            it = fastReads.emplace(key, rn).first;
          }
          rhs = c.var(it->second);
        }
        return rhs;
      };

      msl::Expr *cur = value;
      ElemType curElem = f32();
      int stepIdx = 0;
      std::vector<msl::Expr *> ran{value};
      for (const DrainStep &st : steps) {
        if (st.roundBefore) {
          cur = c.cast(mslTypeOf(t.elem), cur);
          curElem = t.elem;
        }
        msl::Expr *rhs = nullptr;
        if (st.operand.kind == DrainOperand::Kind::AccChain) {
          const int at = st.branchBase < stepIdx ? st.branchBase : stepIdx;
          msl::Expr *b = ran[(std::size_t)(at < 0 ? 0 : at)];
          int li = 0;
          for (const DrainBranchLink &lk : st.branch)
            b = epilogueExpr(
                c,
                {std::string_view(lk.op),
                 operandRead(lk.operand, std::to_string(stepIdx) + "b" +
                                             std::to_string(li++))},
                b);
          rhs = b;
        } else {
          rhs = operandRead(st.operand, std::to_string(stepIdx));
        }
        cur = epilogueExpr(c, {std::string_view(st.op), rhs}, cur, curElem);
        ++stepIdx;
        ran.push_back(cur);
      }
      if (t.narrows())
        cur = c.cast(mslTypeOf(t.elem), cur);
      return cur;
    };

    const auto elemRow = [&](int64_t) {
      return c.binary(msl::BinOp::Add, rowE(), fragLaneRowExpr(c, nm.laneId));
    };
    const auto elemCol = [&](int64_t i) {
      return c.binary(msl::BinOp::Add, colE(),
                      fragLaneColExpr(c, nm.laneId, i));
    };

    // Steps applied in place, then one simdgroup_store. A narrowing target
    // converts into a fragment of the tensor's element first:
    // `simdgroup_store` deduces its pointer type from the fragment.
    const auto wholeFragment = [&](msl::Block &into, bool memo) {
      if (memo)
        memoInto = &into;
      const auto memoFor = [&](int64_t i) {
        return memo ? std::to_string(i) : std::string();
      };
      if (!t.narrows()) {
        if (!steps.empty())
          for (int64_t i = 0; i < kFragElemsPerLane; ++i)
            into.push_back(c.assign(fragElemExpr(c, acc, i),
                                    chainAt(fragElemExpr(c, acc, i), elemRow(i),
                                            elemCol(i), memoFor(i))));
        sgStore(memo ? memoStores : into, acc);
        return;
      }
      const msl::Str narrow = acc + "n";
      into.push_back(c.declStmt(
          kSimdgroup8x8.mslTypeNode(msl::spell(mslTypeOf(t.elem).scalarKind())),
          narrow, nullptr));
      for (int64_t i = 0; i < kFragElemsPerLane; ++i)
        into.push_back(c.assign(fragElemExpr(c, narrow, i),
                                chainAt(fragElemExpr(c, acc, i), elemRow(i),
                                        elemCol(i), memoFor(i))));
      sgStore(memo ? memoStores : into, narrow);
    };

    if (!t.bounded()) {
      wholeFragment(into, /*memo=*/true);
      continue;
    }
    if (allIn)
      wholeFragment(fast, /*memo=*/true);

    // Only asked when there's no whole-tile test. A whole-fragment sub-arm
    // for the edge tile costs a register allocation step, so edge tiles store
    // as guarded scalars.
    msl::Expr *inside =
        allIn ? nullptr : fragmentInBounds(c, t, rowE(), colE());

    msl::Block whole;
    if (inside)
      wholeFragment(whole, /*memo=*/false);

    const DrainValueAt valueAt = [&](msl::Expr *value, msl::Expr *row,
                                     msl::Expr *col) {
      return chainAt(value, row, col);
    };

    msl::Block edge;
    if (allIn && !t.edgeScratch.empty())
      emitScratchEdgeStores(c, edge, t, nm, s, rowE(), colE(), valueAt);
    else
      emitGuardedScalarEdgeStores(c, edge, t, acc, elemRow, elemCol, valueAt);

    if (inside)
      slow.push_back(c.ifElse(inside, std::move(whole), std::move(edge)));
    else
      for (msl::Stmt *st : edge)
        slow.push_back(st);
  }
  for (msl::Stmt *st : memoStores)
    memoInto->push_back(st);
  if (allIn)
    into.push_back(c.ifElse(allIn, std::move(fast), std::move(slow)));
  else
    for (msl::Stmt *st : slow)
      into.push_back(st);
  if (t.uniformGuard)
    c.guardedInto(body, t.uniformGuard, std::move(drained));
}

// The whole direct path: for each band of C rows the pool can hold and each
// block the program calls for, declare accumulators, accumulate, store, then
// read that band back out. A C that fits whole is one band, one pass.
inline void
emitDirectDot(msl::Context &c, msl::Block &body, const WarpProgram &prog,
              const WarpGrid &grid, const DirectInputs &in, const TileView &cv,
              const DirectNames &nm, int64_t bandRows,
              const std::function<ReadbackInputs(const Range &)> &readbackFor,
              const CoordSource &cCoords, DotPassSchedule sched = {}) {
  const int64_t rows = cv.extentAt(0);
  if (bandRows <= 0 || bandRows > rows)
    bandRows = rows;

  int counter = 0;
  std::map<int64_t, std::pair<FragCache, FragCache>> bandCaches;
  const bool acrossBands = bandRows < rows;
  for (int64_t r0 = 0; r0 < rows; r0 += bandRows) {
    const Range band{r0, std::min(r0 + bandRows, rows)};
    const bool whole = band.lo == 0 && band.hi == rows;

    // The row coordinate is compile-time here: `planWarpProgram` withholds
    // the row-affine form from banded grids.
    const auto inBand = [&](const std::vector<WarpSlot> &slots) {
      if (whole)
        return slots;
      std::vector<WarpSlot> out;
      for (const WarpSlot &s : slots)
        if (s.mi.isConst() && s.mi.constant * kSgFragDim >= band.lo &&
            s.mi.constant * kSgFragDim < band.hi)
          out.push_back(s);
      return out;
    };

    const bool renaming =
        sched.drain == DotPassSchedule::Drain::Rename && readbackFor;
    const ReadbackInputs renamed =
        renaming ? readbackFor(band) : ReadbackInputs{};

    emitWarpBlocks(
        c, body, prog, grid, nm.warpId,
        [&](msl::Block &inner, const std::vector<WarpSlot> &all, int64_t w) {
          const std::vector<WarpSlot> slots = inBand(all);
          if (sched.declareAccums)
            emitAccumDecls(c, inner, slots, nm);
          FragShare share;
          if (acrossBands) {
            auto &fc = bandCaches[prog.guardWarp(w).value_or(-1)];
            share = {&body, &fc.first, &fc.second, &counter};
          }
          emitDirectMma(c, inner, slots, in, nm, counter, share);
          if (sched.drain == DotPassSchedule::Drain::Pool)
            emitAccumStores(c, inner, slots, cv, nm, band.lo / kSgFragDim);
          else if (renaming)
            emitFragmentReadback(c, inner, renamed.plan, renamed.names,
                                 renamed.bases, renamed.regElem,
                                 directAccName(nm));
        });

    // No drain: fragments stay live for the caller. Such a pass is one band.
    if (!sched.drainsC() || sched.drain == DotPassSchedule::Drain::Rename)
      return;

    // Drained but no readback requested: remaining bands still need MMAs.
    if (!readbackFor)
      continue;

    // Outside the warp guard, after a barrier: a thread's destination layout
    // may pull elements other warps wrote. The band buffer holds rows
    // [band.lo, band.hi) while a register's coordinate is in the whole tile's
    // frame; `originAt` absorbs the difference.
    const ReadbackInputs back = readbackFor(band);
    body.push_back(c.barrier());
    emitReadback(c, body, cv.originAt({band.lo, 0}), nm.poolC, back.actions,
                 back.names, back.bases, cCoords, back.elem, back.regElem);

    if (band.hi < rows)
      body.push_back(c.barrier());
  }
}

} // namespace agpu

#endif // AGPU_EMIT_DIRECT_H
