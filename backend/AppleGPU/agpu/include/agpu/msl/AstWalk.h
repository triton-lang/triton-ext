// AstWalk.h - child iteration, generated from Nodes.def.
#ifndef AGPU_MSL_AST_WALK_H
#define AGPU_MSL_AST_WALK_H

#include "Ast.h"

namespace agpu::msl {
namespace detail {

struct Visitor {
  void (*onExpr)(void *, Expr *) = nullptr;
  void (*onStmt)(void *, Stmt *) = nullptr;
  void *ctx = nullptr;

  void expr(Expr *e) const {
    if (e && onExpr)
      onExpr(ctx, e);
  }
  void stmt(Stmt *s) const {
    if (s && onStmt)
      onStmt(ctx, s);
  }
};

#define EXPR_NODE(Name, Class)                                                 \
  inline void children([[maybe_unused]] Class *n,                              \
                       [[maybe_unused]] const Visitor &v) {
#define EXPR_CHILD(Class, field) v.expr(n->field);
#define EXPR_CHILD_LIST(Class, field)                                          \
  for (Expr * c : n->field)                                                    \
    v.expr(c);
#define STMT_NODE(Name, Class)                                                 \
  inline void children([[maybe_unused]] Class *n,                              \
                       [[maybe_unused]] const Visitor &v) {
#define STMT_CHILD_EXPR(Class, field) v.expr(n->field);
#define STMT_CHILD_EXPR_LIST(Class, field)                                     \
  for (Expr * c : n->field)                                                    \
    v.expr(c);
#define STMT_CHILD_STMT(Class, field) v.stmt(n->field);
#define STMT_CHILD_BLOCK(Class, field)                                         \
  for (Stmt * c : n->field)                                                    \
    v.stmt(c);
#define STMT_CHILD_BLOCK_LIST(Class, field, member)                            \
  for (auto &entry : n->field)                                                 \
    for (Stmt * c : entry.member)                                              \
      v.stmt(c);
#define NODE_END }
#include "NodesClosed.def"

} // namespace detail

inline void forEachChild(Expr *e, const detail::Visitor &v) {
  if (!e)
    return;
  switch (e->kind) {
#define EXPR_NODE(Name, Class)                                                 \
  case ExprKind::Name:                                                         \
    detail::children(static_cast<Class *>(e), v);                              \
    return;
#include "Nodes.def"
  }
}

inline void forEachChild(Stmt *s, const detail::Visitor &v) {
  if (!s)
    return;
  switch (s->kind) {
#define STMT_NODE(Name, Class)                                                 \
  case StmtKind::Name:                                                         \
    detail::children(static_cast<Class *>(s), v);                              \
    return;
#include "Nodes.def"
  }
}

// Expression nodes own no Blocks, but the table emits NODE_END for them too,
// so they still need an open body.
#define EXPR_NODE(Name, Class)                                                 \
  template <class F> inline void blocksOf(Class *, F &) {
#define STMT_NODE(Name, Class)                                                 \
  template <class F>                                                           \
  inline void blocksOf([[maybe_unused]] Class *n, [[maybe_unused]] F &fn) {
#define STMT_CHILD_BLOCK(Class, field) fn(n->field);
#define STMT_CHILD_BLOCK_LIST(Class, field, member)                            \
  for (auto &entry : n->field)                                                 \
    fn(entry.member);
#define NODE_END }
#include "NodesClosed.def"

// Every Block a statement owns, by reference so a caller can rewrite it.
template <class F> void forEachChildBlock(Stmt *s, F fn) {
  if (!s)
    return;
  switch (s->kind) {
#define STMT_NODE(Name, Class)                                                 \
  case StmtKind::Name:                                                         \
    blocksOf(static_cast<Class *>(s), fn);                                     \
    return;
#include "Nodes.def"
  }
}

namespace detail {
template <class F> Visitor exprVisitor(F &f) {
  Visitor v;
  v.ctx = &f;
  v.onExpr = [](void *c, Expr *e) { (*static_cast<F *>(c))(e); };
  return v;
}
template <class P> Visitor pairVisitor(P &p) {
  Visitor v;
  v.ctx = &p;
  v.onExpr = [](void *c, Expr *e) { static_cast<P *>(c)->onExpr(e); };
  v.onStmt = [](void *c, Stmt *s) { static_cast<P *>(c)->onStmt(s); };
  return v;
}
} // namespace detail

template <class F> void forEachChildExpr(Expr *e, F fn) {
  auto v = detail::exprVisitor(fn);
  forEachChild(e, v);
}

template <class F> void forEachOwnExpr(Stmt *s, F fn) {
  auto v = detail::exprVisitor(fn);
  forEachChild(s, v);
}

template <class F> void visitExprs(Expr *e, F fn) {
  if (!e)
    return;
  fn(e);
  forEachChildExpr(e, [&](Expr *c) { visitExprs(c, fn); });
}

template <class StmtFn, class ExprFn>
void visitStmts(Stmt *s, StmtFn onStmt, ExprFn onExpr) {
  if (!s)
    return;
  onStmt(s);
  struct Pair {
    StmtFn *sf;
    ExprFn *ef;
    void onExpr(Expr *e) { visitExprs(e, *ef); }
    void onStmt(Stmt *c) { visitStmts(c, *sf, *ef); }
  } p{&onStmt, &onExpr};
  auto v = detail::pairVisitor(p);
  forEachChild(s, v);
}

template <class StmtFn, class ExprFn>
void visitBlock(const Block &b, StmtFn onStmt, ExprFn onExpr) {
  for (Stmt *s : b)
    visitStmts(s, onStmt, onExpr);
}

template <class F> void visitStmtsOnly(Stmt *s, F fn) {
  if (!s)
    return;
  fn(s);
  struct Only {
    F *f;
    void onExpr(Expr *) {}
    void onStmt(Stmt *c) { visitStmtsOnly(c, *f); }
  } o{&fn};
  auto v = detail::pairVisitor(o);
  v.onExpr = nullptr;
  forEachChild(s, v);
}

} // namespace agpu::msl

#endif // AGPU_MSL_AST_WALK_H
