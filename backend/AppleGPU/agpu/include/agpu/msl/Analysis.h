// Analysis.h - the AST analyses.
#ifndef AGPU_MSL_ANALYSIS_H
#define AGPU_MSL_ANALYSIS_H

#include "Ast.h"
#include "AstWalk.h"
#include "Builtins.h"
#include "Containers.h"

namespace agpu::msl {

// One statement's contribution to the dead-cycle fixpoint. `guard` is the name
// whose death lets the statement be skipped, empty when nothing can skip it.
struct CycleStep {
  Str guard;
  SmallVec<Str, 4> observes;
  bool pending = false;
};

// ── size ──────────────────────────────────────────────────────────────────

// AGX runs SROA over every simdgroup fragment alloca and throws bad_alloc past
// roughly ten thousand of them in one function, so declaration count is the
// metric.
struct FuncSize {
  int stmts = 0;
  int decls = 0;     // every Decl and ArrayDecl, at any depth
  int fragDecls = 0; // decls whose type is an opaque matrix
  int loops = 0;
  int branches = 0;
  int barriers = 0;
  int mma = 0; // simdgroup_multiply_accumulate calls

  // Each MMA is a use of three fragments the register allocator must keep
  // live, so a body can be far past what the optimiser will chew through on
  // calls alone while its declaration count still looks small.
  int optimiserLoad() const { return decls + mma; }
};

inline FuncSize measure(const Block &body) {
  FuncSize s;
  visitBlock(
      body,
      [&](Stmt *st) {
        ++s.stmts;
        switch (st->kind) {
        case StmtKind::Decl:
          ++s.decls;
          if (static_cast<Decl *>(st)->type.isMatrix())
            ++s.fragDecls;
          break;
        case StmtKind::ArrayDecl: {
          // Counted per element: an array of N fragments is N allocas.
          // Threadgroup memory is skipped; SROA never promotes it.
          const ArrayDecl *a = static_cast<ArrayDecl *>(st);
          if (a->elem.addrSpace() != AddrSpace::Threadgroup)
            s.decls += static_cast<int>(a->count);
          break;
        }
        case StmtKind::For:
        case StmtKind::While:
          ++s.loops;
          break;
        case StmtKind::If:
          ++s.branches;
          break;
        case StmtKind::Barrier:
          ++s.barriers;
          break;
        case StmtKind::ExprStmt: {
          auto *e = static_cast<ExprStmt *>(st)->expr;
          if (e && e->kind == ExprKind::Call &&
              static_cast<Call *>(e)->callee == builtin::sg::MultiplyAccumulate)
            ++s.mma;
          break;
        }
        default:
          break;
        }
      },
      [](Expr *) {});
  return s;
}

// Names read anywhere in the block, at any depth.
//
// A plain `v = rhs` does not read v, but `v[i] = rhs`, `*v = rhs` and
// `v += rhs` all do. Reads from statements in `ignore` do not count.
inline PtrSet<Str> collectReads(const Block &body,
                                const PtrSet<Stmt *> &ignore = {}) {
  PtrSet<Str> used;

  auto readExpr = [&](Expr *e) {
    visitExprs(e, [&](Expr *x) {
      if (x->kind == ExprKind::VarRef)
        used.insert(static_cast<VarRef *>(x)->name);
    });
  };

  for (Stmt *top : body) {
    visitStmtsOnly(top, [&](Stmt *s) {
      if (ignore.count(s))
        return;
      switch (s->kind) {
      case StmtKind::Decl:
        readExpr(static_cast<Decl *>(s)->init);
        return;
      case StmtKind::ArrayDecl:
        for (Expr *e : static_cast<ArrayDecl *>(s)->init)
          readExpr(e);
        return;
      case StmtKind::Assign: {
        auto *a = static_cast<Assign *>(s);
        readExpr(a->value);
        // A compound assignment or a non-VarRef target reads the target too.
        if (a->compound || a->target->kind != ExprKind::VarRef)
          readExpr(a->target);
        return;
      }
      default:
        forEachOwnExpr(s, readExpr);
        return;
      }
    });
  }
  return used;
}

// A call counts as an effect: an atomic whose result is ignored still has to
// run.
inline bool hasSideEffect(const Expr *e) {
  if (!e)
    return false;
  bool impure = false;
  visitExprs(const_cast<Expr *>(e), [&](Expr *n) {
    if (n->kind == ExprKind::Call)
      impure = true;
  });
  return impure;
}

// The initialiser of a declaration, or null for one without.
inline const Expr *initOf(const Stmt *s) {
  if (s->kind == StmtKind::Decl)
    return static_cast<const Decl *>(s)->init;
  return nullptr;
}

// The VarRef a write goes through: `buf[i]` and `buf[i][j]` reach a named
// base. Null for anything unresolvable (member write, cast target).
//
// Subscript only. `*p = v` also resolves to a name, but a read of `*q`
// (distinct pointer, same memory) would not name p, so reporting it would be
// unsound.
inline const Expr *writtenThrough(const Expr *target) {
  for (const Expr *e = target; e;) {
    if (e->kind == ExprKind::VarRef)
      return e;
    if (e->kind != ExprKind::Subscript)
      return nullptr;
    e = static_cast<const Subscript *>(e)->base;
  }
  return nullptr;
}

// The name one statement writes, or empty when it writes no single name.
// `escapes` is set when the write goes through a pointer or an unresolvable
// subscript.
inline Str writtenName(const Stmt *s, bool &escapes) {
  escapes = false;
  switch (s->kind) {
  case StmtKind::Assign: {
    auto *a = static_cast<const Assign *>(s);
    if (a->target && a->target->kind == ExprKind::VarRef)
      return static_cast<const VarRef *>(a->target)->name;
    if (a->target)
      if (const Expr *base = writtenThrough(a->target))
        return static_cast<const VarRef *>(base)->name;
    escapes = true;
    return {};
  }
  case StmtKind::Decl:
    return static_cast<const Decl *>(s)->name;
  case StmtKind::ArrayDecl:
    return static_cast<const ArrayDecl *>(s)->name;
  default:
    return {};
  }
}

// Defaults to true for anything not modelled (bare call, barrier, return), so
// a new statement kind inherits the safe answer.
inline bool writesTo(Stmt *s, const PtrSet<Str> &names) {
  bool hit = false;
  visitStmtsOnly(s, [&](Stmt *n) {
    switch (n->kind) {
    case StmtKind::Assign:
    case StmtKind::Decl:
    case StmtKind::ArrayDecl: {
      bool escapes = false;
      const Str w = writtenName(n, escapes);
      if (escapes)
        hit = true;
      else if (!w.empty() && names.count(w))
        hit = true;
      return;
    }

    // Control flow is transparent; only its contents count.
    case StmtKind::If:
    case StmtKind::For:
    case StmtKind::While:
    case StmtKind::Scope:
    case StmtKind::StateMachine:
    case StmtKind::Function:
    case StmtKind::Break:
    case StmtKind::Continue:
      return;

    default:
      hit = true;
      return;
    }
  });
  return hit;
}

// Names kept alive only by a self-sustaining cycle such as `i = i + 1`. Plain
// liveness marks those used forever, so this is a greatest fixpoint: assume
// every local dead, then revive any name read by a statement that is not a
// pure assignment to a currently-dead name.
inline PtrSet<Str> findDeadCycle(const Block &body,
                                 const PtrSet<Stmt *> &ignore) {
  PtrSet<Str> candidate;
  SmallVec<CycleStep, 32> steps;
  Map<Str, SmallVec<unsigned, 4>> guardedBy;

  for (Stmt *top : body)
    visitStmtsOnly(top, [&](Stmt *s) {
      if (ignore.count(s))
        return;
      if (s->kind == StmtKind::Decl)
        candidate.insert(static_cast<Decl *>(s)->name);
      else if (s->kind == StmtKind::ArrayDecl)
        candidate.insert(static_cast<ArrayDecl *>(s)->name);

      CycleStep step;
      if (s->kind == StmtKind::Assign) {
        auto *a = static_cast<Assign *>(s);
        if (a->target->kind == ExprKind::VarRef && !hasSideEffect(a->value))
          step.guard = static_cast<VarRef *>(a->target)->name;
      } else if (s->kind == StmtKind::Decl) {
        auto *d = static_cast<Decl *>(s);
        if (!hasSideEffect(d->init))
          step.guard = d->name;
      }
      forEachOwnExpr(s, [&](Expr *e) {
        visitExprs(e, [&](Expr *x) {
          if (x->kind == ExprKind::VarRef)
            step.observes.push_back(static_cast<VarRef *>(x)->name);
        });
      });
      if (step.observes.empty())
        return;
      if (!step.guard.empty()) {
        step.pending = true;
        guardedBy[step.guard].push_back(static_cast<unsigned>(steps.size()));
      }
      steps.push_back(std::move(step));
    });

  SmallVec<Str, 32> work;
  auto kill = [&](const Str &n) {
    if (candidate.erase(n))
      work.push_back(n);
  };

  for (CycleStep &step : steps)
    if (!step.pending || !candidate.count(step.guard)) {
      step.pending = false;
      for (const Str &n : step.observes)
        kill(n);
    }

  while (!work.empty()) {
    const Str dead = std::move(work.back());
    work.pop_back();
    auto it = guardedBy.find(dead);
    if (it == guardedBy.end())
      continue;
    for (unsigned i : it->second) {
      CycleStep &step = steps[i];
      if (!step.pending)
        continue;
      step.pending = false;
      for (const Str &n : step.observes)
        kill(n);
    }
  }
  return candidate;
}

inline SmallVec<Stmt *, 8> findDeadDecls(const Block &body) {
  SmallVec<Stmt *, 8> dead;
  PtrSet<Stmt *> removed;

  for (;;) {
    const PtrSet<Str> used = collectReads(body, removed);
    const PtrSet<Str> cyclic = findDeadCycle(body, removed);

    // Names whose declaration cannot go because an assignment to them must
    // stay: `v = f(x);` runs for its effect even when nothing reads v.
    PtrSet<Str> pinned;
    for (Stmt *top : body)
      visitStmtsOnly(top, [&](Stmt *s) {
        if (removed.count(s) || s->kind != StmtKind::Assign)
          return;
        auto *a = static_cast<Assign *>(s);
        if (a->target->kind != ExprKind::VarRef)
          return;
        if (hasSideEffect(a->value))
          pinned.insert(static_cast<VarRef *>(a->target)->name);
      });

    SmallVec<Stmt *, 8> round;
    PtrSet<Str> condemned;
    for (Stmt *top : body)
      visitStmtsOnly(top, [&](Stmt *s) {
        if (removed.count(s))
          return;
        Str name;
        if (s->kind == StmtKind::Decl)
          name = static_cast<Decl *>(s)->name;
        else if (s->kind == StmtKind::ArrayDecl)
          name = static_cast<ArrayDecl *>(s)->name;
        else
          return;
        if ((used.count(name) && !cyclic.count(name)) || pinned.count(name))
          return;
        if (hasSideEffect(initOf(s)))
          return;
        round.push_back(s);
        condemned.insert(name);
      });

    // An assignment to a condemned name goes with it.
    for (Stmt *top : body)
      visitStmtsOnly(top, [&](Stmt *s) {
        if (removed.count(s) || s->kind != StmtKind::Assign)
          return;
        auto *a = static_cast<Assign *>(s);
        if (a->target->kind != ExprKind::VarRef)
          return;
        if (condemned.count(static_cast<VarRef *>(a->target)->name))
          round.push_back(s);
      });

    if (round.empty())
      break;
    for (Stmt *s : round) {
      removed.insert(s);
      dead.push_back(s);
    }
  }
  return dead;
}

inline void eraseStmts(Block &body, const PtrSet<Stmt *> &drop) {
  Block kept;
  kept.reserve(body.size());
  for (Stmt *s : body) {
    if (drop.count(s))
      continue;
    forEachChildBlock(s, [&](Block &child) { eraseStmts(child, drop); });
    kept.push_back(s);
  }
  body.swap(kept);
}

} // namespace agpu::msl

#endif // AGPU_MSL_ANALYSIS_H
