// EmitMove.h - loads and stores, emitted from their plan.
//
// The site hands over expressions: accesses are `base[off]`, no pointer
// materialised. Callbacks mint fresh AST per call, so
// no node is shared between statements.
#ifndef AGPU_EMIT_MOVE_H
#define AGPU_EMIT_MOVE_H

#include "agpu/emit/EmitPoison.h"
#include "agpu/emit/primitives/VectorSpelling.h"
#include "agpu/msl/Context.h"
#include "agpu/msl/Equal.h"
#include "agpu/msl/Printer.h"
#include "agpu/plan/AccessPlan.h"
#include "agpu/plan/Elementwise.h"

#include <functional>

namespace agpu {

struct MoveSite {
  // The lvalue for register r's element: `base[off]`, or `*p`.
  std::function<msl::Expr *(int64_t)> elem;

  // The condition register r's access runs under, or null for "always".
  std::function<msl::Expr *(int64_t)> guard;

  // The IR-supplied value for a lane the mask excludes. Read only under
  // MaskedInit::Other.
  std::function<msl::Expr *(int64_t)> other;

  // One name per register, broadcasting at size one. Loads declare these;
  // stores read them.
  msl::SmallVec<msl::Str, 8> values;

  msl::AddrSpace space = msl::AddrSpace::Device;
};

inline const msl::Str &at(const msl::SmallVec<msl::Str, 8> &v, int64_t r) {
  return v[v.size() == 1 ? 0 : (std::size_t)r];
}

// A buffer both stored and loaded by one kernel needs coherent accesses, or a
// load can be served a stale line another threadgroup wrote. Coherence costs
// bandwidth, so `p.coherent` comes from a whole-function analysis.
inline unsigned accessQuals(const MovePlan &p) {
  return p.coherent ? msl::Type::Coherent : msl::Type::QualNone;
}

inline msl::Expr *accessLValue(msl::Context &c, const MovePlan &p,
                               const MoveSite &site, int64_t r, ElemType elem,
                               int64_t width, bool packed) {
  msl::Expr *lv = site.elem(r);
  const unsigned q = accessQuals(p);
  if (width == 1 && q == msl::Type::QualNone)
    return lv;
  if (width == 1)
    return c.deref(
        c.cast(mslTypeOf(elem).pointerTo(site.space, q), c.addrOf(lv)));
  return wideLValue(c, lv, elem, width, packed, site.space, q);
}

// `declare`: an unmasked load declares its registers here; a masked load's
// were declared by the init phase and are assigned instead.
inline void emitLoadRun(msl::Context &c, msl::Block &body, const MovePlan &p,
                        const MoveSite &site, ElemType elem, int64_t base,
                        bool declare) {
  if (!p.vectorised()) {
    msl::Expr *rhs = accessLValue(c, p, site, base, elem, 1, false);
    if (declare)
      body.push_back(c.declStmt(mslTypeOf(elem), at(site.values, base), rhs));
    else
      body.push_back(c.assign(c.var(at(site.values, base)), rhs));
    return;
  }
  const msl::Str v = at(site.values, base) + "_w";
  body.push_back(c.declStmt(
      vectorTypeOf(elem, p.width(), p.runs.packed), v,
      accessLValue(c, p, site, base, elem, p.width(), p.runs.packed)));
  for (int64_t i = 0; i < p.width(); ++i) {
    msl::Expr *lane = c.subscript(c.var(v), c.lit(i));
    if (declare)
      body.push_back(
          c.declStmt(mslTypeOf(elem), at(site.values, base + i), lane));
    else
      body.push_back(c.assign(c.var(at(site.values, base + i)), lane));
  }
}

inline void emitStoreRun(msl::Context &c, msl::Block &body, const MovePlan &p,
                         const MoveSite &site, ElemType elem, int64_t base) {
  if (!p.vectorised()) {
    body.push_back(c.assign(accessLValue(c, p, site, base, elem, 1, false),
                            c.var(at(site.values, base))));
    return;
  }
  msl::SmallVec<msl::Expr *, 4> lanes;
  for (int64_t i = 0; i < p.width(); ++i)
    lanes.push_back(c.var(at(site.values, base + i)));
  body.push_back(
      c.assign(accessLValue(c, p, site, base, elem, p.width(), p.runs.packed),
               c.call(vecCtorName(elem, p.width()), lanes)));
}

inline void emitMaskedScalar(msl::Context &c, msl::Block &body,
                             const MoveFacts &f, const MovePlan &p,
                             const MoveSite &site, int64_t r, ElemType elem) {
  msl::Expr *cond = site.guard ? site.guard(r) : nullptr;
  msl::Stmt *access =
      f.isStore ? c.assign(accessLValue(c, p, site, r, elem, 1, false),
                           c.var(at(site.values, r)))
                : c.assign(c.var(at(site.values, r)),
                           accessLValue(c, p, site, r, elem, 1, false));
  if (msl::Stmt *s = c.guarded(cond, access))
    body.push_back(s);
}

// The guard every register of a run shares, or null when one has none or
// they differ.
inline msl::Expr *sharedRunGuard(const MoveSite &site, int64_t base,
                                 int64_t width) {
  msl::Expr *g = site.guard ? site.guard(base) : nullptr;
  for (int64_t i = 1; g && i < width; ++i) {
    msl::Expr *o = site.guard(base + i);
    if (!o || !msl::exprsEqual(g, o))
      return nullptr;
  }
  return g;
}

// A run whose registers share one guard moves as one access under it; any
// other run moves register by register.
inline void emitGuardedRun(msl::Context &c, msl::Block &body,
                           const MoveFacts &f, const MovePlan &p,
                           const MoveSite &site, ElemType elem, int64_t base) {
  bool anyDead = false;
  for (int64_t i = 0; i < p.width(); ++i)
    anyDead = anyDead || p.guards.deadAt(base + i);
  if (p.runGuarded && anyDead)
    return;
  if (p.vectorised() && !anyDead)
    if (msl::Expr *g = p.runGuarded && site.guard
                           ? site.guard(base)
                           : sharedRunGuard(site, base, p.width())) {
      msl::Block run;
      if (f.isStore)
        emitStoreRun(c, run, p, site, elem, base);
      else
        emitLoadRun(c, run, p, site, elem, base, /*declare=*/false);
      c.guardedInto(body, g, std::move(run));
      return;
    }
  for (int64_t i = 0; i < p.width(); ++i)
    if (!p.guards.deadAt(base + i))
      emitMaskedScalar(c, body, f, p, site, base + i, elem);
}

inline void emitMove(msl::Context &c, msl::Block &body, const MoveFacts &f,
                     const MovePlan &p, const MoveSite &site, ElemType elem) {
  // Every register must be defined before the mask is consulted: the mask is
  // a runtime value, so a lane it excludes still reads its name.
  if (p.init != MaskedInit::None)
    for (int64_t r = 0; r < f.regCount; ++r) {
      msl::Expr *seed =
          p.init == MaskedInit::Other ? site.other(r) : poisonValue(c, elem);
      body.push_back(c.declStmt(mslTypeOf(elem), at(site.values, r), seed));
    }
  const bool declare = !f.isStore && p.init == MaskedInit::None;

  if (!p.peel) {
    for (int64_t base = 0; base < f.regCount; base += p.width()) {
      if (runIsDead(p.guards, base, p.width()))
        continue;
      if (f.hasMask && !runIsUnguarded(p.guards, base, p.width())) {
        emitGuardedRun(c, body, f, p, site, elem, base);
        continue;
      }
      if (f.isStore)
        emitStoreRun(c, body, p, site, elem, base);
      else
        emitLoadRun(c, body, p, site, elem, base, declare);
    }
    return;
  }

  // The peel condition is every distinct conjunct of every guard. Splitting
  // it per run would re-serialise the runs.
  msl::Expr *allTrue = nullptr;
  msl::SmallVec<msl::Expr *, 8> seen;
  msl::SmallVec<msl::Expr *, 8> pending;
  for (int64_t r = 0; r < f.regCount; r += p.runGuarded ? p.width() : 1) {
    if (msl::Expr *g = site.guard ? site.guard(r) : nullptr)
      pending.push_back(g);
    while (!pending.empty()) {
      msl::Expr *t = pending.back();
      pending.pop_back();
      if (t->kind == msl::ExprKind::Binary &&
          static_cast<msl::Binary *>(t)->op == msl::BinOp::LAnd) {
        pending.push_back(static_cast<msl::Binary *>(t)->rhs);
        pending.push_back(static_cast<msl::Binary *>(t)->lhs);
        continue;
      }
      bool dup = false;
      for (msl::Expr *s : seen)
        dup = dup || msl::exprsEqual(s, t);
      if (dup)
        continue;
      seen.push_back(t);
      allTrue = allTrue ? c.binary(msl::BinOp::LAnd, allTrue, t) : t;
    }
  }

  msl::Block hot, cold;
  for (int64_t base = 0; base < f.regCount; base += p.width()) {
    if (f.isStore)
      emitStoreRun(c, hot, p, site, elem, base);
    else
      emitLoadRun(c, hot, p, site, elem, base, /*declare=*/false);
  }

  if (!allTrue) {
    for (msl::Stmt *s : hot)
      body.push_back(s);
    return;
  }

  // Every register guarded by the one conjunct the peel tests: where it is
  // false, no access runs, so there is no cold arm.
  bool everyGuarded = true;
  for (int64_t r = 0; r < f.regCount && everyGuarded; ++r)
    everyGuarded = site.guard(r) != nullptr;
  if (everyGuarded && seen.size() == 1) {
    body.push_back(c.ifStmt(allTrue, std::move(hot)));
    return;
  }

  for (int64_t base = 0; base < f.regCount; base += p.width())
    emitGuardedRun(c, cold, f, p, site, elem, base);

  body.push_back(c.ifElse(allTrue, std::move(hot), std::move(cold)));
}

} // namespace agpu

#endif // AGPU_EMIT_MOVE_H
