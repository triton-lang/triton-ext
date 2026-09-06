// GuardFuse.h - merging a run of identically-guarded statements into one
// scope.
//
// `if (c) a; if (c) b; if (c) d;` becomes `if (c) { a; b; d; }`. Requires
// structurally equal conditions and no statement in the run writing a name the
// condition reads.
#ifndef AGPU_MSL_FUSE_H
#define AGPU_MSL_FUSE_H

#include "Analysis.h"
#include "Context.h"
#include "Equal.h"

namespace agpu::msl {

struct FuseCost {
  int64_t condChars = 0;  // rendered width of the condition
  int64_t braceChars = 2; // `{` and `}` on their own lines

  // `if (` + cond + `) ` per guarded statement.
  int64_t perStatement() const { return 4 + condChars + 2; }

  bool worthFusing(int64_t runLength) const {
    if (runLength < 2)
      return false;
    const int64_t saved = (runLength - 1) * perStatement();
    return saved > braceChars;
  }
};

inline bool clobbers(Stmt *s, const PtrSet<Str> &names) {
  return writesTo(s, names);
}

// `opaque` is set when the expression contains a call: `f(x) > 0` reads more
// than `x`.
struct ReadNames {
  PtrSet<Str> names;
  bool opaque = false;
};

inline ReadNames namesRead(Expr *e) {
  ReadNames out;
  visitExprs(e, [&](Expr *n) {
    switch (n->kind) {
    case ExprKind::VarRef:
      out.names.insert(static_cast<VarRef *>(n)->name);
      return;
    case ExprKind::Call:
      out.opaque = true;
      return;
    default:
      return;
    }
  });
  return out;
}

inline std::size_t fusableRun(const Block &b, std::size_t i) {
  if (i >= b.size() || b[i]->kind != StmtKind::If)
    return 0;
  auto *first = static_cast<If *>(b[i]);
  if (!first->cond || !first->elseBody.empty())
    return 0;

  const ReadNames cond = namesRead(first->cond);
  if (cond.opaque)
    return 0;

  const PtrSet<Str> &condNames = cond.names;
  std::size_t j = i;
  while (j < b.size()) {
    if (b[j]->kind != StmtKind::If)
      break;
    auto *cur = static_cast<If *>(b[j]);
    if (!cur->elseBody.empty() || !exprsEqual(cur->cond, first->cond))
      break;
    bool clobbered = false;
    for (Stmt *s : cur->thenBody)
      clobbered = clobbered || clobbers(s, condNames);
    if (clobbered)
      break;
    ++j;
  }
  return j - i;
}

inline int64_t fuseGuards(Context &c, Block &body, int64_t condWidth = 8);

inline int64_t fuseGuardsIn(Context &c, Stmt *s, int64_t condWidth) {
  int64_t n = 0;
  forEachChildBlock(
      s, [&](Block &nested) { n += fuseGuards(c, nested, condWidth); });
  return n;
}

inline int64_t fuseGuards(Context &c, Block &body, int64_t condWidth) {
  int64_t fused = 0;
  Block kept;
  kept.reserve(body.size());

  const FuseCost cost{condWidth};
  for (std::size_t i = 0; i < body.size();) {
    fused += fuseGuardsIn(c, body[i], condWidth);

    const std::size_t run = fusableRun(body, i);
    if (!cost.worthFusing((int64_t)run)) {
      kept.push_back(body[i]);
      ++i;
      continue;
    }

    Block merged;
    for (std::size_t k = i; k < i + run; ++k) {
      if (k > i)
        fused += fuseGuardsIn(c, body[k], condWidth);
      auto *g = static_cast<If *>(body[k]);
      for (Stmt *inner : g->thenBody)
        merged.push_back(inner);
    }
    kept.push_back(
        c.ifStmt(static_cast<If *>(body[i])->cond, std::move(merged)));
    fused += (int64_t)run - 1;
    i += run;
  }
  body.swap(kept);
  return fused;
}

} // namespace agpu::msl

#endif // AGPU_MSL_FUSE_H
