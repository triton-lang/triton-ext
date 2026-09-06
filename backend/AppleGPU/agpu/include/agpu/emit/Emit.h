// Emit.h - turning plans into MSL.
#ifndef AGPU_EMIT_H
#define AGPU_EMIT_H

#include "agpu/core/CoordGuard.h"
#include "agpu/core/Names.h"
#include "agpu/core/TileView.h"
#include "agpu/emit/LayoutExpr.h"
#include "agpu/emit/primitives/CoordHoist.h"
#include "agpu/emit/primitives/VectorSpelling.h"
#include "agpu/msl/Context.h"
#include "agpu/plan/StagePlan.h"

namespace agpu {

struct CoordSource : ThreadNames {
  std::vector<LayoutBasis> dims; // one per output dimension

  // Null builds the expression inline. Set to share hoisted names across all
  // consumers of this source.
  CoordHoist *hoist = nullptr;

  msl::Expr *of(msl::Context &c, int reg, int dim) const {
    if (hoist)
      return hoist->coord(c, dims[dim], reg);
    return coordExpr(c, dims[dim], reg, laneId, warpId, blockId);
  }
  CoordRange rangeOf(int reg, int dim, int64_t dimSize) const {
    return dims[dim].rangeOf(reg, dim, dimSize);
  }
};

// A guard's terms as one conjunction, or null when none is needed.
template <class CoordOf>
inline msl::Expr *guardCond(msl::Context &c, const CoordGuard &g,
                            CoordOf coordOf) {
  if (!g.needsTest())
    return nullptr;
  msl::SmallVec<msl::Expr *, 4> terms;
  for (const GuardTerm &t : g.terms()) {
    msl::BinOp op = t.op == GuardTerm::Op::Ge ? msl::BinOp::Ge : msl::BinOp::Lt;
    terms.push_back(c.binary(op, coordOf(t.dim), c.lit(t.bound)));
  }
  return c.chain(msl::BinOp::LAnd, {terms.begin(), terms.end()});
}

inline msl::Expr *guardExpr(msl::Context &c, const CoordGuard &g, int reg,
                            const CoordSource &src) {
  return guardCond(c, g, [&](int dim) { return src.of(c, reg, dim); });
}

// The runtime twin of `TileView::offsetOf`. Takes no `StageAction::coord`:
// `src.of` is already the complete coordinate.
inline msl::Expr *offsetExprOf(msl::Context &c, const TileView &v, int reg,
                               const CoordSource &src) {
  std::vector<msl::Expr *> coord;
  for (int d = 0; d < v.rank(); ++d)
    coord.push_back(src.of(c, reg, d));

  return v.linearize<msl::Expr *>(
      coord,
      [&](msl::Expr *t, int64_t s) {
        return c.binary(msl::BinOp::Mul, t, c.lit(s));
      },
      [&](msl::Expr *a, msl::Expr *b) {
        return c.binary(msl::BinOp::Add, a, b);
      },
      [&](int64_t k) { return c.lit(k); });
}

// The address is a runtime expression: the slot depends on the lane holding
// the register.
inline msl::Stmt *stageStore(msl::Context &c, const TileView &dst,
                             const msl::Str &buf, const msl::Str &srcName,
                             const StageAction &a, const CoordSource &src) {
  msl::Stmt *store =
      c.assign(c.subscript(c.var(buf), offsetExprOf(c, dst, a.reg, src)),
               c.var(srcName));
  return c.guarded(guardExpr(c, a.guard, a.reg, src), store);
}

// A run of `a.width` registers as one pool store. Slots
// `a.reg .. a.reg+width-1` are consecutive and share a guard, so the
// predicate from the first register covers all of them.
inline msl::Stmt *stageStoreWide(msl::Context &c, const TileView &dst,
                                 const msl::Str &buf,
                                 const msl::SmallVec<msl::Str, 8> &srcNames,
                                 const StageAction &a, const CoordSource &src,
                                 ElemType elem) {
  msl::SmallVec<msl::Expr *, 4> lanes;
  for (int i = 0; i < a.width; ++i)
    lanes.push_back(c.var(srcNames[(std::size_t)(a.reg + i)]));
  msl::Expr *slot = c.subscript(c.var(buf), offsetExprOf(c, dst, a.reg, src));
  msl::Stmt *store = c.assign(
      wideLValue(c, slot, elem, a.width, a.packed, msl::AddrSpace::Threadgroup),
      c.call(vecCtorName(elem, a.width), lanes));
  return c.guarded(guardExpr(c, a.guard, a.reg, src), store);
}

// Dead actions were dropped during planning.
inline void emitStage(msl::Context &c, msl::Block &body, const TileView &dst,
                      const msl::Str &buf,
                      const msl::SmallVec<StageAction, 8> &actions,
                      const msl::SmallVec<msl::Str, 8> &srcNames,
                      const CoordSource &src, ElemType elem) {
  for (const StageAction &a : actions) {
    msl::Stmt *s = a.width > 1
                       ? stageStoreWide(c, dst, buf, srcNames, a, src, elem)
                       : stageStore(c, dst, buf, srcNames[a.reg], a, src);
    if (s)
      body.push_back(s);
  }
}

// A lifted integer dot pools f32 while its registers are i32. Convert before
// the base add: a float add against an i32 base up to 2^31 is not exact.
inline msl::Expr *readbackValueExpr(msl::Context &c, msl::Expr *loaded,
                                    ElemType poolElem, ElemType regElem) {
  if (regElem.kind == ElemType::Kind::Int &&
      poolElem.kind == ElemType::Kind::Float)
    return c.cast(mslTypeOf(regElem), loaded);
  return loaded;
}

inline msl::Stmt *readbackLoad(msl::Context &c, const TileView &src,
                               const msl::Str &buf, const msl::Str &dstName,
                               const msl::Str &baseName, const StageAction &a,
                               const CoordSource &coords, ElemType poolElem,
                               ElemType regElem) {
  msl::Expr *slot = readbackValueExpr(
      c, c.subscript(c.var(buf), offsetExprOf(c, src, a.reg, coords)), poolElem,
      regElem);
  msl::Expr *value = baseName.empty()
                         ? slot
                         : c.binary(msl::BinOp::Add, slot, c.var(baseName));
  msl::Stmt *load = c.assign(c.var(dstName), value);
  return c.guarded(guardExpr(c, a.guard, a.reg, coords), load);
}

// A run of `a.width` registers read back as one pool load. The accumulate is
// still per lane.
inline void readbackLoadWide(msl::Context &c, msl::Block &body,
                             const TileView &src, const msl::Str &buf,
                             const msl::SmallVec<msl::Str, 8> &dstNames,
                             const msl::SmallVec<msl::Str, 8> &baseNames,
                             const StageAction &a, const CoordSource &coords,
                             ElemType elem, ElemType regElem) {
  msl::Expr *slot =
      c.subscript(c.var(buf), offsetExprOf(c, src, a.reg, coords));
  const msl::Str v = dstNames[(std::size_t)a.reg] + "_w";
  msl::Block inner;
  inner.push_back(c.declStmt(vectorTypeOf(elem, a.width, a.packed), v,
                             wideLValue(c, slot, elem, a.width, a.packed,
                                        msl::AddrSpace::Threadgroup)));
  for (int i = 0; i < a.width; ++i) {
    const std::size_t r = (std::size_t)(a.reg + i);
    const msl::Str base = r < baseNames.size() ? baseNames[r] : msl::Str{};
    msl::Expr *lane =
        readbackValueExpr(c, c.subscript(c.var(v), c.lit(i)), elem, regElem);
    msl::Expr *value =
        base.empty() ? lane : c.binary(msl::BinOp::Add, lane, c.var(base));
    inner.push_back(c.assign(c.var(dstNames[r]), value));
  }
  // The run shares one guard; without it a second panel's readback both
  // redeclares the temp and clobbers registers it does not own.
  c.guardedInto(body, guardExpr(c, a.guard, a.reg, coords), std::move(inner));
}

// `elem` is what the pool holds, `regElem` what the registers hold. They
// differ only for the lifted integer dot: pool f32, registers i32.
inline void emitReadback(msl::Context &c, msl::Block &body, const TileView &src,
                         const msl::Str &buf,
                         const msl::SmallVec<StageAction, 8> &actions,
                         const msl::SmallVec<msl::Str, 8> &dstNames,
                         const msl::SmallVec<msl::Str, 8> &baseNames,
                         const CoordSource &coords, ElemType elem,
                         ElemType regElem) {
  for (const StageAction &a : actions) {
    if (a.width > 1) {
      readbackLoadWide(c, body, src, buf, dstNames, baseNames, a, coords, elem,
                       regElem);
      continue;
    }
    const msl::Str base =
        a.reg < (int)baseNames.size() ? baseNames[a.reg] : msl::Str{};
    if (msl::Stmt *s = readbackLoad(c, src, buf, dstNames[a.reg], base, a,
                                    coords, elem, regElem))
      body.push_back(s);
  }
}

} // namespace agpu

#endif // AGPU_EMIT_H
