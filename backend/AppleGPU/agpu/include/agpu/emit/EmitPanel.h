// EmitPanel.h - one panel tile, emitted.
#ifndef AGPU_EMIT_PANEL_H
#define AGPU_EMIT_PANEL_H

#include "agpu/core/Names.h"
#include "agpu/emit/Emit.h"
#include "agpu/emit/primitives/FragLane.h"
#include "agpu/emit/primitives/OperandSource.h"
#include "agpu/emit/primitives/SlotExpr.h"
#include "agpu/msl/Builtins.h"
#include "agpu/msl/Context.h"
#include "agpu/plan/CanonicalFragment.h"
#include "agpu/plan/PanelSchedule.h"
#include "agpu/plan/ReadbackPlan.h"
#include "agpu/plan/WarpSlots.h"

#include <functional>

#include <array>
#include <map>

namespace agpu {

struct MmaNames : ThreadNames {
  // The accumulator is float on every path Metal offers. Operands follow the
  // dot's own element type.
  msl::Str accElem = "float";
  msl::Str opElem = "half";

  msl::Str poolA = "pA";
  msl::Str poolB = "pB";
  msl::Str poolC = "pC";
  msl::Str poolE = "pE";
  msl::Str acc = "acc";
  msl::Str kVar = "kv";
};

struct PanelNames : MmaNames {
  msl::Str fragA = "fa";
  msl::Str fragB = "fb";
};

// Everything one tile needs that this file does not decide.
struct PanelInputs {
  msl::SmallVec<StageAction, 8> aActions, bActions;
  msl::SmallVec<msl::Str, 8> aNames, bNames;

  ElemType aElem = f16(), bElem = f16(), cElem = f32();

  // What C's registers hold: i32 for the lifted integer dot, whose pool stays
  // f32. The readback converts before adding `cBases`.
  ElemType cRegElem = f32();

  // `cBases` are the incoming accumulator names; the readback adds to them.
  // An empty entry means no incoming value.
  msl::SmallVec<StageAction, 8> cActions;
  msl::SmallVec<msl::Str, 8> cNames, cBases;
  ReadbackPlan cRename;

  // Where the MMA reads A's fragments: the staged pool tile, or the device
  // tensor with this tile's corner as origin.
  OperandSource a;

  bool rollK = false;
};

// A, B and C have three different layouts, so they need separate CoordSources.
struct PanelCoords {
  CoordSource a, b, c;

  static PanelCoords forAll(const CoordSource &s) { return {s, s, s}; }
};

struct DirectNames : MmaNames {
  msl::Str frag = "f";
};

struct PanelMmaSize {
  int decls = 0;
  int fragDecls = 0;
  int mma = 0;

  int load() const { return decls + mma; }
};

// A and B each need their own instance; the keyspace is not shared.
class FragCache {
public:
  msl::Str lookup(SlotCoord pos, int64_t k) {
    auto it = names_.find(Key{pos, k});
    return it == names_.end() ? msl::Str() : it->second;
  }
  void put(SlotCoord pos, int64_t k, const msl::Str &name) {
    names_[Key{pos, k}] = name;
  }

private:
  struct Key {
    SlotCoord pos;
    int64_t k;
    bool operator<(const Key &o) const {
      if (!(pos == o.pos))
        return pos < o.pos;
      return k < o.k;
    }
  };
  std::map<Key, msl::Str> names_;
};

struct FragShare {
  msl::Block *decls = nullptr;
  FragCache *a = nullptr;
  FragCache *b = nullptr;
  int *counter = nullptr;

  bool active() const { return decls != nullptr; }
};

class FragReuse {
public:
  explicit FragReuse(msl::Block &decls) : decls_(decls) {}

  FragShare shareFor(const PanelTile &t, int64_t block) {
    return {&decls_, &a_[{block, t.batch, t.m.lo, t.k.lo, t.n.lo}],
            &b_[{block, t.batch, t.n.lo, t.k.lo, 0}], &counter_};
  }

private:
  msl::Block &decls_;
  std::map<std::array<int64_t, 5>, FragCache> a_, b_;
  int counter_ = 0;
};

inline msl::Str loadFrag(msl::Context &c, msl::Block &into, msl::Block &decls,
                         FragCache &cache, const OperandSource &src,
                         SlotCoord pos, int64_t kIndex, msl::Expr *kTerm,
                         const DirectNames &nm, int &counter) {
  const SlotCoord row = src.rowOf(pos);
  if (msl::Str hit = cache.lookup(row, kIndex); !hit.empty())
    return hit;

  const msl::Str name = nm.frag + std::to_string(counter++);
  decls.push_back(c.declStmt(kSimdgroup8x8.mslTypeNode(nm.opElem), name));
  msl::Expr *off = src.fragOffsetOf(c, row, nm.warpId);
  if (kTerm)
    off = c.binary(msl::BinOp::Add, off, kTerm);
  into.push_back(c.exprStmt(
      c.call(msl::builtin::sg::Load,
             {c.var(name), c.binary(msl::BinOp::Add, c.var(src.buffer), off),
              src.leadingDim.expr(c)})));
  cache.put(row, kIndex, name);
  return name;
}

inline msl::Expr *kTermExpr(msl::Context &c, KStep::Offset o,
                            const msl::Str &loopVar) {
  if (!o.fromLoopVar)
    return c.lit(o.constant);
  return c.binary(msl::BinOp::Mul, c.var(loopVar), c.lit(o.scale));
}

inline msl::Stmt *sgLoad(msl::Context &c, const msl::Str &frag,
                         const msl::Str &buf, msl::Expr *off, msl::Expr *ld) {
  return c.exprStmt(
      c.call(msl::builtin::sg::Load,
             {c.var(frag), c.binary(msl::BinOp::Add, c.var(buf), off), ld}));
}
inline msl::Stmt *sgLoad(msl::Context &c, const msl::Str &frag,
                         const msl::Str &buf, msl::Expr *off, int64_t ld) {
  return sgLoad(c, frag, buf, off, (msl::Expr *)c.lit(ld));
}

inline msl::Stmt *sgStore(msl::Context &c, const msl::Str &frag,
                          const msl::Str &buf, msl::Expr *off, int64_t ld) {
  return c.exprStmt(c.call(
      msl::builtin::sg::Store,
      {c.var(frag), c.binary(msl::BinOp::Add, c.var(buf), off), c.lit(ld)}));
}

inline msl::Stmt *sgMma(msl::Context &c, const msl::Str &acc, const msl::Str &a,
                        const msl::Str &b) {
  return c.exprStmt(c.call(msl::builtin::sg::MultiplyAccumulate,
                           {c.var(acc), c.var(a), c.var(b), c.var(acc)}));
}

inline void emitAccumStore(msl::Context &c, msl::Block &body,
                           const PanelTile &t, const PanelNames &nm,
                           const WarpSlot &s, const msl::Str &accName) {
  const TileView cv = t.cView();
  msl::Expr *off = fragOffset(c, cv, {s.mi, s.ni}, nullptr, nm.warpId);
  body.push_back(sgStore(c, accName, nm.poolC, off, cv.strideAt(0)));
}

// Empty for the single-tile case. Includes the batch, since two slices' tiles
// share m/n/k.
inline msl::Str tileTag(const PanelTile &t) {
  msl::Str tag;
  if (t.batch != 0)
    tag = "_s" + std::to_string(t.batch);
  if (t.m.lo == 0 && t.n.lo == 0 && t.k.lo == 0)
    return tag;
  return tag + "_" + std::to_string(t.m.lo) + "_" + std::to_string(t.n.lo) +
         "_" + std::to_string(t.k.lo);
}

inline msl::Str panelAccName(const PanelTile &t, const PanelNames &nm,
                             int acc) {
  msl::Str tag;
  if (t.batch != 0)
    tag = "_s" + std::to_string(t.batch);
  if (t.m.lo != 0 || t.n.lo != 0)
    tag += "_" + std::to_string(t.m.lo) + "_" + std::to_string(t.n.lo);
  const msl::Str join = tag.empty() ? msl::Str() : tag + "_";
  return nm.acc + join + std::to_string(acc);
}

// Individually named because an array of fragments defeats SROA and lands in
// stack memory.
inline void emitPanelAccumDecls(msl::Context &c, msl::Block &body,
                                const PanelTile &t, const PanelNames &nm,
                                const WarpProgram &prog, const WarpGrid &grid) {
  int accCount = 0;
  for (int64_t w = 0; w < prog.blockCount(grid.numWarps); ++w)
    for (const WarpSlot &s : prog.slots(w, grid.mT, grid.nT, grid.numWarps))
      accCount = std::max(accCount, s.acc + 1);
  for (int a = 0; a < accCount; ++a)
    body.push_back(c.declStmt(
        kSimdgroup8x8.mslTypeNode(nm.accElem), panelAccName(t, nm, a),
        c.call(kSimdgroup8x8.zeroCtor(nm.accElem), {c.litF(0.0)})));
}

// K steps outermost, slots within. An A fragment serves every slot in its row,
// a B fragment every slot in its column.
inline void emitPanelMma(msl::Context &c, msl::Block &body, const PanelTile &t,
                         const PanelNames &nm, const OperandSource &a,
                         const std::vector<WarpSlot> &slots, bool rollK,
                         FragShare share = {}) {
  const msl::Str tag = tileTag(t);
  const msl::Str join = tag.empty() ? msl::Str() : tag + "_";
  const auto accNameOf = [&](const WarpSlot &s) {
    return panelAccName(t, nm, s.acc);
  };

  // B's K axis is its row axis, so its fragments move along columns.
  const TileView bv = t.bView();
  OperandSource b;
  b.buffer = nm.poolB;
  b.leadingDim = Stride(bv.strideAt(0));
  b.fragAxis = OperandSource::FragAxis::Cols;

  DirectNames na, nb;
  static_cast<MmaNames &>(na) = nm;
  static_cast<MmaNames &>(nb) = nm;
  na.frag = nm.fragA + join;
  nb.frag = nm.fragB + join;

  const bool shared = share.active() && !rollK;
  FragCache localA, localB;
  int localCounter = 0;
  FragCache &aCache = shared ? *share.a : localA;
  FragCache &bCache = shared ? *share.b : localB;
  int &counter = shared ? *share.counter : localCounter;

  auto step = [&](msl::Block &into, KStep k, int64_t kIndex) {
    msl::Block &decls = shared ? *share.decls : into;
    for (const WarpSlot &s : slots) {
      msl::Expr *kFrags = kTermExpr(c, k.kOffset(1), nm.kVar);
      const msl::Str fa = loadFrag(c, into, decls, aCache, a, s.mi, kIndex,
                                   a.kOffsetOf(c, kFrags), na, counter);
      const msl::Str fb = loadFrag(c, into, decls, bCache, b, s.ni, kIndex,
                                   b.kOffsetOf(c, kFrags), nb, counter);
      into.push_back(sgMma(c, accNameOf(s), fa, fb));
    }
  };

  if (rollK) {
    msl::Block inner;
    step(inner, KStep::rolled(), 0);
    msl::For *loop = c.forStmt(
        c.declStmt(msl::Context::i32(), nm.kVar, c.lit(0)),
        c.binary(msl::BinOp::Lt, c.var(nm.kVar), c.lit(t.k.size())),
        c.assignOp(msl::BinOp::Add, c.var(nm.kVar), c.lit(kSgFragDim)),
        std::move(inner));
    const int64_t perTrip = std::max<int64_t>(1, (int64_t)slots.size());
    loop->unrollCount = std::min<int64_t>(
        std::max<int64_t>(1, msl::kUnrollCount / perTrip), t.kSteps());
    body.push_back(loop);
  } else {
    for (int64_t ki = 0; ki < t.kSteps(); ++ki)
      step(body, KStep::unrolled(ki), ki);
  }
}

// Must count exactly what the emitDot panel walk makes emitPanelMma emit for
// this form; emitKernel infers the unrolled size from these counts and a
// measured rolled body.
inline PanelMmaSize predictPanelDotSize(const DotFacts &f, const Panel &p,
                                        bool rollK) {
  PanelMmaSize out;
  std::map<std::array<int64_t, 5>, FragCache> sharedA, sharedB;
  const msl::Str mark = "x";
  forEachPanelTile(f, p, [&](const PanelTile &t) {
    const WarpGrid grid = panelWarpGrid(t, warpsFor(f), f.numWarps);
    const WarpProgram prog = planWarpProgram(grid);
    if (t.k.lo == 0) {
      int accCount = 0;
      for (int64_t w = 0; w < prog.blockCount(grid.numWarps); ++w)
        for (const WarpSlot &s : prog.slots(w, grid.mT, grid.nT, grid.numWarps))
          accCount = std::max(accCount, s.acc + 1);
      out.decls += accCount;
      out.fragDecls += accCount;
    }
    for (int64_t w = 0; w < prog.blockCount(grid.numWarps); ++w) {
      const std::vector<WarpSlot> slots =
          prog.slots(w, grid.mT, grid.nT, grid.numWarps);
      if (slots.empty())
        continue;
      const int64_t block = prog.guardWarp(w).value_or(-1);
      FragCache localA, localB;
      FragCache &a =
          rollK ? localA : sharedA[{block, t.batch, t.m.lo, t.k.lo, t.n.lo}];
      FragCache &b =
          rollK ? localB : sharedB[{block, t.batch, t.n.lo, t.k.lo, 0}];
      const int64_t steps = rollK ? 1 : t.kSteps();
      if (rollK)
        out.decls += 1;
      for (int64_t ki = 0; ki < steps; ++ki)
        for (const WarpSlot &sl : slots) {
          if (a.lookup(sl.mi, ki).empty()) {
            a.put(sl.mi, ki, mark);
            out.decls += 1;
            out.fragDecls += 1;
          }
          if (b.lookup(sl.ni, ki).empty()) {
            b.put(sl.ni, ki, mark);
            out.decls += 1;
            out.fragDecls += 1;
          }
          out.mma += 1;
        }
    }
  });
  return out;
}

inline void
emitFragmentReadback(msl::Context &c, msl::Block &body,
                     const ReadbackPlan &plan,
                     const msl::SmallVec<msl::Str, 8> &names,
                     const msl::SmallVec<msl::Str, 8> &bases, ElemType regElem,
                     const std::function<msl::Str(int64_t)> &accName) {
  for (std::size_t r = 0; r < plan.regs.size() && r < names.size(); ++r) {
    if (plan.regs[r].acc < 0)
      continue;
    msl::Expr *value =
        fragElemExpr(c, accName(plan.regs[r].acc), plan.regs[r].elem);
    if (regElem.kind == ElemType::Kind::Int)
      value = c.cast(mslTypeOf(regElem), value);
    if (r < bases.size() && !bases[r].empty())
      value = c.binary(msl::BinOp::Add, value, c.var(bases[r]));
    body.push_back(c.assign(c.var(names[r]), value));
  }
}

inline std::function<msl::Str(int64_t)> directAccName(const MmaNames &nm) {
  return [acc = nm.acc](int64_t a) { return acc + std::to_string(a); };
}

inline void emitPanelTile(msl::Context &c, msl::Block &body, const PanelTile &t,
                          const PanelNames &nm, const PanelInputs &in,
                          const PanelCoords &coords, const WarpGrid &grid,
                          const WarpProgram &prog, FragReuse *reuse = nullptr) {
  for (PanelPhase ph : phasesOf(t)) {
    if (needsBarrierBefore(ph))
      body.push_back(c.barrier());

    switch (ph) {
    // Staging and readback use the staged views; the MMA uses the plain ones.
    case PanelPhase::StageA:
      emitStage(c, body, t.aStagedView(), nm.poolA, in.aActions, in.aNames,
                coords.a, in.aElem);
      break;
    case PanelPhase::StageB:
      emitStage(c, body, t.bStagedView(), nm.poolB, in.bActions, in.bNames,
                coords.b, in.bElem);
      break;
    case PanelPhase::Mma:
      if (t.k.lo == 0)
        emitPanelAccumDecls(c, body, t, nm, prog, grid);
      emitWarpBlocks(
          c, body, prog, grid, nm.warpId,
          [&](msl::Block &inner, const std::vector<WarpSlot> &slots,
              int64_t w) {
            FragShare share;
            if (reuse)
              share = reuse->shareFor(t, prog.guardWarp(w).value_or(-1));
            emitPanelMma(c, inner, t, nm, in.a, slots, in.rollK, share);
          });
      break;
    case PanelPhase::Drain:
      emitWarpBlocks(
          c, body, prog, grid, nm.warpId,
          [&](msl::Block &inner, const std::vector<WarpSlot> &slots, int64_t) {
            for (const WarpSlot &s : slots)
              emitAccumStore(c, inner, t, nm, s, panelAccName(t, nm, s.acc));
          });
      break;
    case PanelPhase::Readback:
      emitReadback(c, body, t.cStagedView(), nm.poolC, in.cActions, in.cNames,
                   in.cBases, coords.c, in.cElem, in.cRegElem);
      break;
    case PanelPhase::Rename:
      emitWarpBlocks(
          c, body, prog, grid, nm.warpId,
          [&](msl::Block &inner, const std::vector<WarpSlot> &, int64_t) {
            emitFragmentReadback(
                c, inner, in.cRename, in.cNames, in.cBases, in.cRegElem,
                [&](int64_t a) { return panelAccName(t, nm, (int)a); });
          });
      break;
    }
  }
}

} // namespace agpu

#endif // AGPU_EMIT_PANEL_H
