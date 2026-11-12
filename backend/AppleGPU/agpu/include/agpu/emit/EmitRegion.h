// EmitRegion.h - an unstructured region, emitted as a dispatch loop.
//
// Order inside an edge:
//   1. copy the edge's arguments into the destination's parameters
//   2. assign the state
//   3. `continue`
// The copies are simultaneous in the source language, so they go through
// temporaries when they overlap.
#ifndef AGPU_EMIT_REGION_H
#define AGPU_EMIT_REGION_H

#include "agpu/msl/AstWalk.h"
#include "agpu/msl/Context.h"
#include "agpu/plan/RegionPlan.h"

#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace agpu {

// A tensor value is held in N registers, each its own MSL variable.
struct RegionNames {
  msl::Str stateVar = "__state";
  // The names holding value `v`, in register order. What a copy reads.
  std::function<ValueNames(ValueId)> namesOf;
  // The type of one of `v`'s registers, for the hoisted declarations.
  std::function<msl::Type(ValueId)> typeOf;

  // The variables value `v` owns, each with its own type. Not a subset of
  // `namesOf`: a pointer's offset is storage the value owns without holding.
  // Optional; null means use `namesOf` at `typeOf`.
  std::function<std::vector<std::pair<msl::Str, msl::Type>>(ValueId)> storageOf;

  std::vector<std::pair<msl::Str, msl::Type>> storage(ValueId v) const {
    if (storageOf)
      return storageOf(v);
    std::vector<std::pair<msl::Str, msl::Type>> out;
    for (const msl::Str &n : namesOf(v))
      out.emplace_back(n, typeOf(v));
    return out;
  }
};

// A value defined in one case body and read in another must live outside both;
// an MSL variable dies at its closing brace.
inline void emitHoisted(msl::Context &c, msl::Block &body, const RegionPlan &p,
                        const RegionNames &nm) {
  for (ValueId v : p.hoisted)
    for (const auto &s : nm.storage(v))
      body.push_back(c.declStmt(s.second, s.first));
}

// A declaration inside a case body for a hoisted value shadows the outer
// variable, so the cross-loop reader sees one nothing wrote. Declarations with
// an initialiser become assignments; the rest are dropped.
inline void dropShadowingDeclsIn(msl::Context &c, msl::Block &body,
                                 const msl::PtrSet<msl::Str> &hoistedNames) {
  msl::Block kept;
  for (msl::Stmt *s : body) {
    // Recurse first: a value defined inside a loop, guard, or nested state
    // machine has its shadowing declaration nested too.
    msl::forEachChildBlock(
        s, [&](msl::Block &b) { dropShadowingDeclsIn(c, b, hoistedNames); });
    if (s->kind != msl::StmtKind::Decl) {
      kept.push_back(s);
      continue;
    }
    msl::Decl *d = static_cast<msl::Decl *>(s);
    if (!hoistedNames.count(d->name)) {
      kept.push_back(s);
      continue;
    }
    if (d->init)
      kept.push_back(c.assign(c.var(d->name), d->init));
  }
  body = std::move(kept);
}

inline void dropShadowingDecls(msl::Context &c, msl::Block &body,
                               const RegionPlan &p, const RegionNames &nm) {
  // Storage, matching what `emitHoisted` declared.
  msl::PtrSet<msl::Str> hoistedNames;
  for (ValueId v : p.hoisted)
    for (const auto &s : nm.storage(v))
      hoistedNames.insert(s.first);
  dropShadowingDeclsIn(c, body, hoistedNames);
}

// Sequential copies are wrong when a destination is also a source, e.g. a
// self-branch with `(a, b) -> (b, a)`. The overlap test covers the whole edge,
// since a cycle can be longer than two.
inline void emitEdgeCopies(msl::Context &c, msl::Block &body, const Edge &e,
                           const std::vector<ValueId> &params,
                           const RegionNames &nm, int &tmpCounter) {
  bool overlaps = false;
  for (ValueId src : e.args)
    if (detail::contains(params, src))
      overlaps = true;

  // One parameter/argument pair is N copies; the overlap decision is per
  // value.
  if (!overlaps) {
    for (std::size_t i = 0; i < e.args.size(); ++i) {
      const ValueNames dst = nm.namesOf(params[i]);
      const ValueNames src = nm.namesOf(e.args[i]);
      for (std::size_t r = 0; r < dst.size() && r < src.size(); ++r)
        body.push_back(c.assign(c.var(dst[r]), c.var(src[r])));
    }
    return;
  }

  std::vector<ValueNames> tmps;
  for (std::size_t i = 0; i < e.args.size(); ++i) {
    ValueNames held;
    for (const msl::Str &s : nm.namesOf(e.args[i])) {
      const msl::Str t = "__phi" + std::to_string(tmpCounter++);
      body.push_back(c.declStmt(nm.typeOf(e.args[i]), t, c.var(s)));
      held.push_back(t);
    }
    tmps.push_back(std::move(held));
  }
  for (std::size_t i = 0; i < e.args.size(); ++i) {
    const ValueNames dst = nm.namesOf(params[i]);
    for (std::size_t r = 0; r < dst.size() && r < tmps[i].size(); ++r)
      body.push_back(c.assign(c.var(dst[r]), c.var(tmps[i][r])));
  }
}

// The `continue` is required inside a conditional arm, where the code after
// the `if` would otherwise run too.
inline msl::Block emitEdge(msl::Context &c, const RegionFacts &f,
                           const RegionPlan &p, const Edge &e,
                           const RegionNames &nm, int &tmpCounter) {
  msl::Block out;
  emitEdgeCopies(c, out, e, f.blocks[(std::size_t)e.to].params, nm, tmpCounter);
  out.push_back(
      c.assign(c.var(nm.stateVar), c.lit(p.stateOf[(std::size_t)e.to])));
  out.push_back(c.continueStmt());
  return out;
}

// The statements a block's terminator becomes.
inline void emitTerminator(msl::Context &c, msl::Block &body,
                           const RegionFacts &f, const RegionPlan &p,
                           const BlockFacts &b, const RegionNames &nm,
                           msl::Expr *condition, int &tmpCounter) {
  switch (b.term) {
  case TermKind::Return:
    // The loop's own test reads the state.
    body.push_back(c.assign(c.var(nm.stateVar), c.lit(p.exitState)));
    return;

  case TermKind::Branch:
    for (msl::Stmt *s : emitEdge(c, f, p, b.edges[0], nm, tmpCounter))
      body.push_back(s);
    return;

  case TermKind::CondBranch:
    body.push_back(c.ifElse(condition,
                            emitEdge(c, f, p, b.edges[0], nm, tmpCounter),
                            emitEdge(c, f, p, b.edges[1], nm, tmpCounter)));
    return;
  }
}

// `bodyOf` builds one block's non-terminator statements; `condOf` supplies a
// conditional branch's condition.
inline void
emitRegion(msl::Context &c, msl::Block &out, const RegionFacts &f,
           const RegionPlan &p, const RegionNames &nm,
           const std::function<msl::Block(BlockId)> &bodyOf,
           const std::function<msl::Expr *(BlockId)> &condOf = nullptr) {
  emitHoisted(c, out, p, nm);

  msl::StateMachine *m =
      c.stateMachine(nm.stateVar, p.stateOf[(std::size_t)f.entry]);
  m->exitState = p.exitState;

  int tmpCounter = 0;
  for (BlockId b = 0; b < (BlockId)f.blocks.size(); ++b) {
    msl::Block body = bodyOf(b);
    dropShadowingDecls(c, body, p, nm);
    emitTerminator(c, body, f, p, f.blocks[(std::size_t)b], nm,
                   condOf ? condOf(b) : nullptr, tmpCounter);
    m->cases.push_back({p.stateOf[(std::size_t)b], std::move(body)});
  }
  out.push_back(m);
}

} // namespace agpu

#endif // AGPU_EMIT_REGION_H
