// Context.h - node arena and builder. Owns every node and hands out raw
// pointers. The builder folds integer constants as it goes.
#ifndef AGPU_MSL_CONTEXT_H
#define AGPU_MSL_CONTEXT_H

#include "Ast.h"

#include <deque>
#include <memory>
#include <utility>

namespace agpu::msl {

class Context {
public:
  // ── types ───────────────────────────────────────────────────────────────
  static Type i32() { return Type::scalar(Scalar::I32); }
  static Type u32() { return Type::scalar(Scalar::U32); }
  static Type f32() { return Type::scalar(Scalar::F32); }
  static Type boolTy() { return Type::scalar(Scalar::Bool); }

  // ── expressions ─────────────────────────────────────────────────────────

  VarRef *var(std::string name) { return make<VarRef>(std::move(name)); }

  Literal *lit(int64_t v, Type t = i32()) {
    auto *l = make<Literal>();
    l->form = Literal::Form::Int;
    l->intValue = v;
    l->type = std::move(t);
    return l;
  }
  Literal *litHex(int64_t v, Type t = u32()) {
    Literal *l = lit(v, std::move(t));
    l->hex = true;
    return l;
  }
  Literal *litF(double v, Type t = f32()) {
    auto *l = make<Literal>();
    l->form = Literal::Form::Float;
    l->floatValue = v;
    l->type = std::move(t);
    return l;
  }
  Literal *litNull(Type t) {
    auto *l = make<Literal>();
    l->form = Literal::Form::Null;
    l->type = std::move(t);
    return l;
  }
  Literal *litBool(bool v) {
    auto *l = make<Literal>();
    l->form = Literal::Form::Bool;
    l->intValue = v ? 1 : 0;
    l->type = boolTy();
    return l;
  }

  // Null means "no condition" throughout the emitters.
  Expr *allOf(Expr *a, Expr *b) {
    if (!a)
      return b;
    if (!b)
      return a;
    return binary(BinOp::LAnd, a, b);
  }

  Expr *binary(BinOp op, Expr *lhs, Expr *rhs) {
    if (auto *f = foldBinary(op, lhs, rhs))
      return f;
    if (Expr *s = simplifyBinary(op, lhs, rhs))
      return s;
    return make<Binary>(op, lhs, rhs);
  }

  Expr *add(Expr *a, Expr *b) { return binary(BinOp::Add, a, b); }
  Expr *mul(Expr *a, Expr *b) { return binary(BinOp::Mul, a, b); }

  Expr *chain(BinOp op, const std::vector<Expr *> &parts) {
    if (parts.empty())
      return lit(0);
    Expr *e = parts.front();
    for (std::size_t i = 1; i < parts.size(); ++i)
      e = binary(op, e, parts[i]);
    return e;
  }

  Expr *unary(UnOp op, Expr *e) { return make<Unary>(op, e); }
  Expr *ternary(Expr *c, Expr *t, Expr *f) { return make<Ternary>(c, t, f); }
  Expr *cast(Type to, Expr *e) { return make<Cast>(std::move(to), e); }
  Expr *cast(Type to, Expr *e, Cast::Style s) {
    return make<Cast>(std::move(to), e, s);
  }
  Expr *bitcast(Type to, Expr *e) {
    return make<Cast>(std::move(to), e, Cast::Style::Bits);
  }
  Expr *construct(Type to, Expr *e) {
    return make<Cast>(std::move(to), e, Cast::Style::Functional);
  }
  Expr *subscript(Expr *b, Expr *i) { return make<Subscript>(b, i); }
  Expr *member(Expr *b, std::string f) { return make<Member>(b, std::move(f)); }
  Expr *deref(Expr *e) { return make<Deref>(e); }
  Expr *addrOf(Expr *e) {
    if (e->kind == ExprKind::Deref)
      return static_cast<Deref *>(e)->operand;
    return make<AddrOf>(e);
  }

  Call *call(std::string callee, SmallVec<Expr *, 4> args) {
    return make<Call>(std::move(callee), std::move(args));
  }

  // `f<half>(...)`, for a callee whose type parameter appears only in its
  // return type.
  Call *call(std::string callee, SmallVec<Str, 2> targs,
             SmallVec<Expr *, 4> args) {
    Call *c = call(std::move(callee), std::move(args));
    c->templateArgs = std::move(targs);
    return c;
  }

  // ── statements ──────────────────────────────────────────────────────────

  Decl *declStmt(Type t, std::string n, Expr *init = nullptr) {
    return make<Decl>(std::move(t), std::move(n), init);
  }
  ArrayDecl *arrayDecl(Type elem, std::string n, SmallVec<Expr *, 4> init) {
    auto *d =
        make<ArrayDecl>(std::move(elem), std::move(n), (int64_t)init.size());
    d->init = std::move(init);
    return d;
  }
  ArrayDecl *arrayDecl(Type elem, std::string n, int64_t count) {
    return make<ArrayDecl>(std::move(elem), std::move(n), count);
  }
  Assign *assign(Expr *target, Expr *value) {
    return make<Assign>(target, value);
  }
  Assign *assignOp(BinOp op, Expr *target, Expr *value) {
    auto *a = make<Assign>(target, value);
    a->compound = true;
    a->compoundOp = op;
    return a;
  }
  ExprStmt *exprStmt(Expr *e) { return make<ExprStmt>(e); }
  Break *breakStmt() { return make<Break>(); }
  Continue *continueStmt() { return make<Continue>(); }
  Barrier *barrier(Barrier::Scope s = Barrier::Scope::Threadgroup) {
    return make<Barrier>(s);
  }
  // Must not merge with an adjacent barrier, e.g. the pair around a
  // double-buffer swap.
  Barrier *hardBarrier(Barrier::Scope s = Barrier::Scope::Threadgroup) {
    auto *b = make<Barrier>(s);
    b->hard = true;
    return b;
  }
  Return *returnStmt(Expr *v = nullptr) {
    auto *r = make<Return>();
    r->value = v;
    return r;
  }
  Return *returnStructStmt(SmallVec<Expr *, 4> fields) {
    auto *r = make<Return>();
    r->structFields = std::move(fields);
    return r;
  }

  If *ifStmt(Expr *cond, Block thenBody) {
    return make<If>(cond, std::move(thenBody));
  }
  If *ifElse(Expr *cond, Block thenBody, Block elseBody) {
    auto *n = make<If>(cond, std::move(thenBody));
    n->elseBody = std::move(elseBody);
    return n;
  }

  enum class GuardFold {
    Always,  // no condition, or a literal true: run unconditionally
    Never,   // a literal false: the body is unreachable
    Runtime, // a real test
  };

  static GuardFold foldGuard(Expr *cond) {
    if (!cond)
      return GuardFold::Always;
    if (auto *l = asLiteral(cond); l && l->form == Literal::Form::Bool)
      return l->intValue ? GuardFold::Always : GuardFold::Never;
    return GuardFold::Runtime;
  }

  // A null condition means "always".
  Stmt *guarded(Expr *cond, Stmt *s) {
    switch (foldGuard(cond)) {
    case GuardFold::Always:
      return s;
    case GuardFold::Never:
      return nullptr; // drop the body
    case GuardFold::Runtime:
      break;
    }
    return ifStmt(cond, Block{s});
  }

  // A null condition splices the statements in unguarded, with no scope
  // wrapped around them.
  void guardedInto(Block &into, Expr *cond, Block body) {
    switch (foldGuard(cond)) {
    case GuardFold::Always:
      for (Stmt *s : body)
        into.push_back(s);
      return;
    case GuardFold::Never:
      return; // the body is unreachable
    case GuardFold::Runtime:
      into.push_back(ifStmt(cond, std::move(body)));
      return;
    }
  }

  For *forStmt(Stmt *init, Expr *cond, Stmt *step, Block body) {
    return make<For>(init, cond, step, std::move(body));
  }
  While *whileStmt(Expr *cond, Block body) {
    return make<While>(cond, std::move(body));
  }
  Scope *scope(Block body) { return make<Scope>(std::move(body)); }
  StateMachine *stateMachine(std::string stateVar, int64_t entry = 0) {
    auto *m = make<StateMachine>();
    m->stateVar = std::move(stateVar);
    m->entry = entry;
    return m;
  }
  Function *function() { return make<Function>(); }
  StructDecl *structDecl() { return make<StructDecl>(); }

  std::size_t nodeCount() const { return nodes_.size(); }

private:
  using Owned = std::unique_ptr<void, void (*)(void *)>;

  template <class T, class... A> T *make(A &&...a) {
    T *raw = new T(std::forward<A>(a)...);
    nodes_.emplace_back(raw, [](void *p) { delete static_cast<T *>(p); });
    return raw;
  }

  static Literal *asLiteral(Expr *e) {
    return e && e->kind == ExprKind::Literal ? static_cast<Literal *>(e)
                                             : nullptr;
  }

  Expr *foldBinary(BinOp op, Expr *lhs, Expr *rhs) {
    Literal *a = asLiteral(lhs), *b = asLiteral(rhs);
    if (!a || !b)
      return nullptr;
    if (a->form != Literal::Form::Int || b->form != Literal::Form::Int)
      return nullptr;
    const int64_t x = a->intValue, y = b->intValue;
    switch (op) {
    case BinOp::Add:
      return lit(x + y, a->type);
    case BinOp::Sub:
      return lit(x - y, a->type);
    case BinOp::Mul:
      return lit(x * y, a->type);
    case BinOp::Div:
      return y ? lit(x / y, a->type) : nullptr;
    case BinOp::Rem:
      return y ? lit(x % y, a->type) : nullptr;
    case BinOp::And:
      return lit(x & y, a->type);
    case BinOp::Or:
      return lit(x | y, a->type);
    case BinOp::Xor:
      return lit(x ^ y, a->type);
    case BinOp::Shl:
      return lit(x << y, a->type);
    case BinOp::Shr:
      return lit(x >> y, a->type);
    case BinOp::Eq:
      return litBool(x == y);
    case BinOp::Ne:
      return litBool(x != y);
    case BinOp::Lt:
      return litBool(x < y);
    case BinOp::Le:
      return litBool(x <= y);
    case BinOp::Gt:
      return litBool(x > y);
    case BinOp::Ge:
      return litBool(x >= y);
    default:
      return nullptr;
    }
  }

  Expr *simplifyBinary(BinOp op, Expr *lhs, Expr *rhs) {
    Literal *a = asLiteral(lhs), *b = asLiteral(rhs);
    auto isInt = [](Literal *l, int64_t v) {
      return l && l->form == Literal::Form::Int && l->intValue == v;
    };
    switch (op) {
    case BinOp::Add:
      if (isInt(a, 0))
        return rhs;
      if (isInt(b, 0))
        return lhs;
      return nullptr;
    case BinOp::Sub:
      if (isInt(b, 0))
        return lhs;
      return nullptr;
    case BinOp::Mul:
      if (isInt(a, 1))
        return rhs;
      if (isInt(b, 1))
        return lhs;
      if (isInt(a, 0) || isInt(b, 0))
        return lit(0);
      return nullptr;
    case BinOp::Div:
      if (isInt(b, 1))
        return lhs;
      return nullptr;
    case BinOp::Shl:
    case BinOp::Shr:
      if (isInt(b, 0))
        return lhs;
      return nullptr;
    default:
      return nullptr;
    }
  }

  std::deque<Owned> nodes_;
};

} // namespace agpu::msl

#endif // AGPU_MSL_CONTEXT_H
