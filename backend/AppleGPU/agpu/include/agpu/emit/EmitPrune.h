// EmitPrune.h - deleting the statements nothing reads.
//
// `Analysis.h` finds them. Metal's optimiser drops dead registers but not the
// barriers a redundant `convert_layout` emits ahead of a dot.
#ifndef AGPU_EMIT_PRUNE_H
#define AGPU_EMIT_PRUNE_H

#include "agpu/msl/Analysis.h"
#include "agpu/msl/AstWalk.h"

#include <functional>

namespace agpu {

// An `if` with nothing left in either arm runs nothing; it goes when its
// condition has no effect of its own. Inner blocks first, so an `if` emptied
// by the ones inside it goes too.
inline bool dropEmptyIfs(msl::Block &body) {
  bool dropped = false;
  msl::PtrSet<msl::Stmt *> drop;
  for (msl::Stmt *s : body) {
    msl::forEachChildBlock(s, [&](msl::Block &child) {
      dropped = dropEmptyIfs(child) || dropped;
    });
    if (s->kind != msl::StmtKind::If)
      continue;
    auto *arm = static_cast<msl::If *>(s);
    if (arm->thenBody.empty() && arm->elseBody.empty() &&
        !msl::hasSideEffect(arm->cond))
      drop.insert(s);
  }
  if (drop.empty())
    return dropped;
  msl::eraseStmts(body, drop);
  return true;
}

// A dropped `if` can leave its condition's names unread, and a dropped name
// can empty an `if`, so the two alternate until neither finds anything.
inline void pruneDead(msl::Block &body) {
  for (;;) {
    const msl::SmallVec<msl::Stmt *, 8> dead = msl::findDeadDecls(body);
    if (!dead.empty())
      msl::eraseStmts(body, msl::PtrSet<msl::Stmt *>(dead.begin(), dead.end()));
    if (!dropEmptyIfs(body) && dead.empty())
      return;
  }
}

// A threadgroup barrier with no threadgroup memory touched since the last one
// in its block orders nothing that one did not. Touching is any use of storage
// declared threadgroup, of a pointer into it, or of a cast to one (the pool is
// declared outside the body). A hard barrier is never dropped but still fences
// what follows it; one ordering device memory neither goes nor fences.
inline void pruneRedundantBarriers(msl::Block &body) {
  const msl::PtrSet<msl::Str> shared = msl::threadgroupNames(body);

  bool hit = false;
  const auto onExpr = [&](msl::Expr *e) {
    if (e->kind == msl::ExprKind::VarRef)
      hit = hit || shared.count(static_cast<msl::VarRef *>(e)->name);
    else if (e->kind == msl::ExprKind::Cast)
      hit = hit || static_cast<msl::Cast *>(e)->to.addrSpace() ==
                       msl::AddrSpace::Threadgroup;
  };
  const auto touches = [&](msl::Stmt *s) {
    hit = false;
    msl::visitBlock(msl::Block{s}, [](msl::Stmt *) {}, onExpr);
    return hit;
  };
  const auto condTouches = [&](msl::If *s) {
    hit = false;
    msl::visitExprs(s->cond, onExpr);
    return hit;
  };

  // Whether the block leaves no threadgroup access after its last barrier,
  // given whether it is entered that way. Each arm of an `if` is entered as
  // the `if` is; a loop body, re-entered from its own end, is not assumed.
  const std::function<bool(msl::Block &, bool)> prune = [&](msl::Block &b,
                                                            bool fenced) {
    msl::PtrSet<msl::Stmt *> drop;
    for (msl::Stmt *s : b) {
      if (s->kind == msl::StmtKind::If) {
        auto *arm = static_cast<msl::If *>(s);
        const bool in = fenced && !condTouches(arm);
        const bool thenOut = prune(arm->thenBody, in);
        const bool elseOut =
            arm->elseBody.empty() ? in : prune(arm->elseBody, in);
        fenced = thenOut && elseOut;
        continue;
      }
      if (s->kind != msl::StmtKind::Barrier) {
        msl::forEachChildBlock(s,
                               [&](msl::Block &child) { prune(child, false); });
        fenced = fenced && !touches(s);
        continue;
      }
      auto *bar = static_cast<msl::Barrier *>(s);
      const bool fences = bar->scope == msl::Barrier::Scope::Threadgroup;
      if (fences && !bar->hard && fenced)
        drop.insert(s);
      else
        fenced = fences;
    }
    if (!drop.empty())
      msl::eraseStmts(b, drop);
    return fenced;
  };
  prune(body, false);
}

} // namespace agpu

#endif // AGPU_EMIT_PRUNE_H
