// EmitSchedule.h - the order of a block's statements.
//
// Elementwise ops are emitted one op at a time over every register, so a
// thread holds each op's results for all of its registers at once. A wide tile
// in a large threadgroup then spills. Placing each pure declaration right
// before its first reader runs one register's chain to its end before the next
// begins.
#ifndef AGPU_EMIT_SCHEDULE_H
#define AGPU_EMIT_SCHEDULE_H

#include "agpu/msl/Analysis.h"
#include "agpu/msl/AstWalk.h"

#include <algorithm>
#include <unordered_map>
#include <vector>

namespace agpu {

namespace detail {

inline bool sinkable(const msl::Stmt *s, const msl::PtrSet<msl::Str> &shared) {
  if (s->kind != msl::StmtKind::Decl)
    return false;
  const auto *d = static_cast<const msl::Decl *>(s);
  const msl::Type::Form form = d->type.form();
  if (form != msl::Type::Form::Scalar && form != msl::Type::Form::Vector)
    return false;
  msl::Expr *init = d->init;
  if (!init)
    return false;
  bool pure = true;
  msl::visitExprs(init, [&](msl::Expr *e) {
    if (e->kind == msl::ExprKind::Subscript || e->kind == msl::ExprKind::Deref)
      pure = false;
    if (e->kind == msl::ExprKind::VarRef &&
        shared.count(static_cast<msl::VarRef *>(e)->name))
      pure = false;
    if (e->kind == msl::ExprKind::Call &&
        msl::callEffect(static_cast<msl::Call *>(e)->callee) !=
            msl::CallEffect::None)
      pure = false;
  });
  return pure;
}

inline bool readsOtherLanes(const msl::Stmt *s) {
  if (s->kind != msl::StmtKind::Decl)
    return false;
  bool lanes = false;
  msl::visitExprs(static_cast<const msl::Decl *>(s)->init, [&](msl::Expr *e) {
    lanes = lanes || (e->kind == msl::ExprKind::Call &&
                      msl::callEffect(static_cast<msl::Call *>(e)->callee) ==
                          msl::CallEffect::Lanes);
  });
  return lanes;
}

// The names a statement assigns, at any depth.
inline msl::PtrSet<msl::Str> namesWritten(msl::Stmt *s) {
  msl::PtrSet<msl::Str> out;
  msl::visitStmtsOnly(s, [&](msl::Stmt *n) {
    bool escapes = false;
    const msl::Str w = msl::writtenName(n, escapes);
    if (!w.empty())
      out.insert(w);
  });
  return out;
}

inline void sinkIn(msl::Block &body, const msl::PtrSet<msl::Str> &shared) {
  for (msl::Stmt *s : body)
    msl::forEachChildBlock(s,
                           [&](msl::Block &child) { sinkIn(child, shared); });

  const std::size_t n = body.size();
  std::vector<msl::PtrSet<msl::Str>> reads, writes;
  reads.reserve(n);
  writes.reserve(n);
  for (msl::Stmt *s : body) {
    reads.push_back(msl::collectReads(msl::Block{s}));
    writes.push_back(namesWritten(s));
  }

  // The order as a circular list over the original positions, `n` its
  // sentinel, so a move relinks instead of shifting the rest.
  std::vector<std::size_t> next(n + 1), prev(n + 1);
  for (std::size_t k = 0; k <= n; ++k) {
    next[k] = (k + 1) % (n + 1);
    prev[k] = (k + n) % (n + 1);
  }

  const auto nameOf = [&](std::size_t k) -> const msl::Str & {
    return static_cast<msl::Decl *>(body[k])->name;
  };
  std::vector<int> laneReaders(n, 0);
  std::unordered_map<msl::Str, std::size_t> laneDecl;
  for (std::size_t k = 0; k < n; ++k)
    if (readsOtherLanes(body[k]))
      laneDecl.emplace(nameOf(k), k);
  for (std::size_t k = 0; k < n; ++k)
    for (const msl::Str &r : reads[k])
      if (const auto it = laneDecl.find(r); it != laneDecl.end())
        ++laneReaders[it->second];

  for (std::size_t i = n; i-- > 0;) {
    if (!sinkable(body[i], shared))
      continue;
    const msl::Str &name = nameOf(i);
    // Cross-lane values passed but still to be read.
    std::vector<std::pair<std::size_t, int>> open;
    std::size_t dest = next[i];
    bool found = false, blocked = false;
    for (std::size_t j = next[i]; j != n && !blocked; j = next[j]) {
      if (open.empty())
        dest = j;
      if (reads[j].count(name) || writes[j].count(name)) {
        found = true;
        break;
      }
      for (const msl::Str &op : reads[i])
        blocked = blocked || writes[j].count(op);
      for (auto &[k, left] : open)
        left -= (int)reads[j].count(nameOf(k));
      open.erase(std::remove_if(open.begin(), open.end(),
                                [](const auto &o) { return o.second == 0; }),
                 open.end());
      if (laneReaders[j] > 0)
        open.emplace_back(j, laneReaders[j]);
    }
    if (!found || blocked || dest == next[i])
      continue;
    const std::size_t j = dest;
    next[prev[i]] = next[i];
    prev[next[i]] = prev[i];
    prev[i] = prev[j];
    next[i] = j;
    next[prev[j]] = i;
    prev[j] = i;
  }

  std::vector<msl::Stmt *> order;
  order.reserve(n);
  for (std::size_t k = next[n]; k != n; k = next[k])
    order.push_back(body[k]);
  body.clear();
  for (msl::Stmt *s : order)
    body.push_back(s);
}

} // namespace detail

// Pure declarations move toward their first reader, never leaving a cross-lane
// value live across them.
inline void sinkToFirstReader(msl::Block &body) {
  detail::sinkIn(body, msl::threadgroupNames(body));
}

} // namespace agpu

#endif // AGPU_EMIT_SCHEDULE_H
