// Equal.h - structural equality of expressions.
#ifndef AGPU_MSL_EQUAL_H
#define AGPU_MSL_EQUAL_H

#include "Ast.h"
#include "AstWalk.h"

namespace agpu::msl {

bool exprsEqual(const Expr *a, const Expr *b);

namespace detail {

inline bool payloadEqual(const Expr *a, const Expr *b) {
  switch (a->kind) {
  case ExprKind::VarRef:
    return static_cast<const VarRef *>(a)->name ==
           static_cast<const VarRef *>(b)->name;

  case ExprKind::Literal: {
    const auto *x = static_cast<const Literal *>(a);
    const auto *y = static_cast<const Literal *>(b);
    return x->type == y->type && x->sameValueAs(*y);
  }

  case ExprKind::Unary:
    return static_cast<const Unary *>(a)->op ==
           static_cast<const Unary *>(b)->op;

  case ExprKind::Binary:
    return static_cast<const Binary *>(a)->op ==
           static_cast<const Binary *>(b)->op;

  case ExprKind::Call: {
    const auto *x = static_cast<const Call *>(a);
    const auto *y = static_cast<const Call *>(b);
    return x->callee == y->callee && x->templateArgs == y->templateArgs;
  }

  case ExprKind::Member:
    return static_cast<const Member *>(a)->field ==
           static_cast<const Member *>(b)->field;

  case ExprKind::Cast: {
    auto *x = static_cast<const Cast *>(a);
    auto *y = static_cast<const Cast *>(b);
    return x->to == y->to && x->style == y->style;
  }

  // Kinds whose entire content is their children.
  case ExprKind::Ternary:
  case ExprKind::Subscript:
  case ExprKind::Deref:
  case ExprKind::AddrOf:
    return true;
  }
  return false;
}

inline SmallVec<Expr *, 4> childList(const Expr *e) {
  SmallVec<Expr *, 4> out;
  forEachChildExpr(const_cast<Expr *>(e), [&](Expr *c) { out.push_back(c); });
  return out;
}

} // namespace detail

// Conservative: a false answer never claims two different addresses are the
// same.
inline bool exprsEqual(const Expr *a, const Expr *b) {
  if (a == b)
    return true;
  if (!a || !b || a->kind != b->kind)
    return false;
  if (!detail::payloadEqual(a, b))
    return false;

  const SmallVec<Expr *, 4> ca = detail::childList(a);
  const SmallVec<Expr *, 4> cb = detail::childList(b);
  if (ca.size() != cb.size())
    return false;
  for (std::size_t i = 0; i < ca.size(); ++i)
    if (!exprsEqual(ca[i], cb[i]))
      return false;
  return true;
}

} // namespace agpu::msl

#endif // AGPU_MSL_EQUAL_H
