// ValueNumber.h - one declaration per value.
//
// Registers that replicate or broadcast one element repeat each other's
// chains. A declaration whose value a name in scope already holds is dropped
// and its readers read that name; values are numbered through assignments.
#ifndef AGPU_MSL_VALUENUMBER_H
#define AGPU_MSL_VALUENUMBER_H

#include "Analysis.h"
#include "AstWalk.h"
#include "Printer.h"

#include <cstdint>
#include <cstring>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace agpu::msl {

namespace detail {

inline const VarRef *baseVar(const Expr *e) {
  while (e && (e->kind == ExprKind::Subscript || e->kind == ExprKind::Member))
    e = e->kind == ExprKind::Subscript ? static_cast<const Subscript *>(e)->base
                                       : static_cast<const Member *>(e)->base;
  return e && e->kind == ExprKind::VarRef ? static_cast<const VarRef *>(e)
                                          : nullptr;
}

inline bool writesOperands(const Expr *e) {
  return e->kind == ExprKind::AddrOf ||
         (e->kind == ExprKind::Unary &&
          static_cast<const Unary *>(e)->op == UnOp::PreInc) ||
         (e->kind == ExprKind::Call &&
          callEffect(static_cast<const Call *>(e)->callee) ==
              CallEffect::State);
}

// Assigned, or reached through a reference argument, an address or `++`.
template <class F> void eachWritten(Stmt *s, F fn) {
  const auto args = [&](Expr *e) {
    forEachChildExpr(e, [&](Expr *c) {
      if (const VarRef *v = baseVar(c))
        fn(v->name);
    });
  };
  if (s->kind == StmtKind::Assign) {
    if (const VarRef *v = baseVar(static_cast<Assign *>(s)->target))
      fn(v->name);
  } else if (s->kind == StmtKind::ExprStmt) {
    Expr *e = static_cast<ExprStmt *>(s)->expr;
    if (e && e->kind == ExprKind::Call)
      args(e);
  }
  forEachOwnExpr(s, [&](Expr *own) {
    visitExprs(own, [&](Expr *e) {
      if (writesOperands(e))
        args(e);
    });
  });
}

class ValueNumbering {
public:
  explicit ValueNumbering(const Block &body)
      : shared_(threadgroupNames(body)), changing_(shared_) {
    for (Stmt *top : body)
      visitStmtsOnly(top, [&](Stmt *s) {
        eachWritten(s, [&](const Str &n) { changing_.insert(n); });
      });
  }

  void run(Block &body) {
    block(body);
    eraseStmts(body, dropped_);
  }

private:
  using Id = std::uint32_t;

  struct Frame {
    std::vector<Id> held;
    std::vector<std::pair<Str, std::optional<Str>>> aliased;
    std::vector<std::pair<Str, std::optional<Type>>> declared;
  };

  struct Reinterpreted {
    Id source;
    Type type;
  };

  Id fresh() { return next_++; }

  Id valueOf(const Str &name) {
    auto [it, added] = value_.try_emplace(name, 0);
    if (added)
      it->second = fresh();
    return it->second;
  }

  void forget(const Str &name) { value_[name] = fresh(); }

  const Str &spell(const Type &t) {
    for (const auto &[type, spelled] : spelled_)
      if (type == t)
        return spelled;
    std::ostringstream os;
    Printer(os).printType(t);
    return spelled_.emplace_back(t, os.str()).second;
  }

  bool plainValue(const Str &name) const {
    const auto t = type_.find(name);
    if (t == type_.end())
      return false;
    const Type::Form form = t->second.form();
    return form == Type::Form::Scalar || form == Type::Form::Vector;
  }

  // False when `e` reads memory or changes state; a pointer argument would let
  // a call read memory.
  bool key(const Expr *e, Str &k, bool &lanes) {
    switch (e->kind) {
    case ExprKind::VarRef: {
      const Str &n = static_cast<const VarRef *>(e)->name;
      if (shared_.count(n))
        return false;
      k += 'v' + std::to_string(valueOf(n));
      return true;
    }
    case ExprKind::Literal: {
      const auto *l = static_cast<const Literal *>(e);
      std::uint64_t bits;
      std::memcpy(&bits, &l->floatValue, sizeof bits);
      k += 'l' + std::to_string((int)l->form) + ':' +
           std::to_string(l->intValue) + ':' + std::to_string(bits) +
           spell(l->type);
      return true;
    }
    case ExprKind::Unary: {
      const UnOp op = static_cast<const Unary *>(e)->op;
      if (op == UnOp::PreInc)
        return false;
      k += 'u' + std::to_string((int)op);
      break;
    }
    case ExprKind::Binary:
      k += 'b' + std::to_string((int)static_cast<const Binary *>(e)->op);
      break;
    case ExprKind::Ternary:
      k += 't';
      break;
    case ExprKind::Call: {
      const auto *c = static_cast<const Call *>(e);
      const CallEffect fx = callEffect(c->callee);
      if (fx == CallEffect::State)
        return false;
      for (const Expr *a : c->args)
        if (a->kind == ExprKind::VarRef &&
            !plainValue(static_cast<const VarRef *>(a)->name))
          return false;
      lanes = lanes || fx == CallEffect::Lanes;
      k += 'c' + c->callee;
      for (const Str &t : c->templateArgs)
        k += '<' + t;
      break;
    }
    case ExprKind::Member:
      k += 'm' + static_cast<const Member *>(e)->field;
      break;
    case ExprKind::Cast: {
      const auto *c = static_cast<const Cast *>(e);
      k += 'x' + std::to_string((int)c->style) + spell(c->to);
      break;
    }
    default:
      return false;
    }
    bool ok = true;
    k += '(';
    forEachChildExpr(const_cast<Expr *>(e), [&](Expr *c) {
      ok = ok && key(c, k, lanes);
      k += ',';
    });
    k += ')';
    return ok;
  }

  // A lane call answers for the lanes active where it runs, so it matches only
  // in its own block.
  Id valueFrom(const Type &type, const Expr *init, const Block &at) {
    if (init->kind == ExprKind::VarRef) {
      const Str &src = static_cast<const VarRef *>(init)->name;
      const auto t = type_.find(src);
      if (!shared_.count(src) && t != type_.end() && t->second == type)
        return valueOf(src);
    }
    const VarRef *bitsOf = nullptr;
    if (init->kind == ExprKind::Cast) {
      const auto *c = static_cast<const Cast *>(init);
      if (c->style == Cast::Style::Bits && c->operand->kind == ExprKind::VarRef)
        bitsOf = static_cast<const VarRef *>(c->operand);
    }
    if (bitsOf)
      if (const auto r = reinterpreted_.find(valueOf(bitsOf->name));
          r != reinterpreted_.end() && r->second.type == type)
        return r->second.source;

    Str k = spell(type) + '=';
    bool lanes = false;
    if (!key(init, k, lanes))
      return fresh();
    if (lanes)
      k += '@' + std::to_string(reinterpret_cast<std::uintptr_t>(&at));
    auto [it, added] = computed_.try_emplace(std::move(k), 0);
    if (added)
      it->second = fresh();
    if (bitsOf && plainValue(bitsOf->name))
      reinterpreted_.try_emplace(
          it->second,
          Reinterpreted{valueOf(bitsOf->name), type_.at(bitsOf->name)});
    return it->second;
  }

  // A read of a variable assigned more than once goes to the declaration
  // holding its current value, unless the expression writes its operands.
  void rename(Expr *e, bool forward) {
    visitExprs(e, [&](Expr *x) { forward = forward && !writesOperands(x); });
    visitExprs(e, [&](Expr *x) {
      if (x->kind != ExprKind::VarRef)
        return;
      auto *v = static_cast<VarRef *>(x);
      if (const auto a = alias_.find(v->name); a != alias_.end()) {
        v->name = a->second;
        return;
      }
      if (!forward || !changing_.count(v->name) || shared_.count(v->name))
        return;
      if (const auto h = holder_.find(valueOf(v->name)); h != holder_.end())
        v->name = h->second;
    });
  }

  void renameOwn(Stmt *s, bool forward) {
    forEachOwnExpr(s, [&](Expr *e) { rename(e, forward); });
  }

  void alias(const Str &name, std::optional<Str> to) {
    const auto it = alias_.find(name);
    frames_.back().aliased.emplace_back(
        name,
        it == alias_.end() ? std::nullopt : std::optional<Str>(it->second));
    if (to)
      alias_[name] = *to;
    else if (it != alias_.end())
      alias_.erase(it);
  }

  bool holdsOneValue(const Decl &d) const {
    const Type::Form form = d.type.form();
    const AddrSpace as = d.type.addrSpace();
    return d.init && !changing_.count(d.name) &&
           (form == Type::Form::Scalar || form == Type::Form::Vector) &&
           (as == AddrSpace::None || as == AddrSpace::Thread);
  }

  void decl(Decl *d, const Block &at) {
    if (d->init)
      rename(d->init, true);
    const Id id = d->init ? valueFrom(d->type, d->init, at) : fresh();

    const auto prior = type_.find(d->name);
    frames_.back().declared.emplace_back(
        d->name, prior == type_.end() ? std::nullopt
                                      : std::optional<Type>(prior->second));
    type_.insert_or_assign(d->name, d->type);
    value_[d->name] = id;
    if (alias_.count(d->name))
      alias(d->name, std::nullopt);

    if (!holdsOneValue(*d))
      return;
    if (const auto h = holder_.find(id); h != holder_.end()) {
      alias(d->name, h->second);
      dropped_.insert(d);
      return;
    }
    holder_.emplace(id, d->name);
    frames_.back().held.push_back(id);
  }

  void assign(Assign *a, const Block &at) {
    rename(a->value, true);
    rename(a->target, false);
    if (a->target->kind != ExprKind::VarRef) {
      if (const VarRef *v = baseVar(a->target))
        forget(v->name);
      return;
    }
    const Str &n = static_cast<const VarRef *>(a->target)->name;
    const auto t = type_.find(n);
    value_[n] = a->compound || t == type_.end()
                    ? fresh()
                    : valueFrom(t->second, a->value, at);
  }

  // What a loop or branch writes has no single value in any arm or after it.
  void nested(Stmt *s) {
    PtrSet<Str> written;
    visitStmtsOnly(s, [&](Stmt *n) {
      eachWritten(n, [&](const Str &w) { written.insert(w); });
    });
    const auto forgetWritten = [&] {
      for (const Str &w : written)
        forget(w);
    };
    forgetWritten();
    renameOwn(s, true);
    if (s->kind == StmtKind::For) {
      auto *f = static_cast<For *>(s);
      if (f->init)
        renameOwn(f->init, false);
      if (f->step)
        renameOwn(f->step, false);
    }
    forEachChildBlock(s, [&](Block &b) {
      forgetWritten();
      block(b);
    });
    forgetWritten();
  }

  void statement(Stmt *s, const Block &at) {
    if (s->kind == StmtKind::Decl)
      return decl(static_cast<Decl *>(s), at);
    if (s->kind == StmtKind::Assign)
      return assign(static_cast<Assign *>(s), at);
    bool hasBlock = false;
    forEachChildBlock(s, [&](Block &) { hasBlock = true; });
    if (hasBlock || s->kind == StmtKind::For)
      return nested(s);
    renameOwn(s, s->kind != StmtKind::ExprStmt);
    eachWritten(s, [&](const Str &w) { forget(w); });
  }

  void block(Block &b) {
    frames_.emplace_back();
    for (Stmt *s : b)
      statement(s, b);
    Frame &f = frames_.back();
    for (Id id : f.held)
      holder_.erase(id);
    for (auto it = f.aliased.rbegin(); it != f.aliased.rend(); ++it)
      if (it->second)
        alias_[it->first] = *it->second;
      else
        alias_.erase(it->first);
    for (auto it = f.declared.rbegin(); it != f.declared.rend(); ++it) {
      if (it->second)
        type_.insert_or_assign(it->first, *it->second);
      else
        type_.erase(it->first);
      forget(it->first);
    }
    frames_.pop_back();
  }

  const PtrSet<Str> shared_;
  PtrSet<Str> changing_;
  std::unordered_map<Str, Id> value_;
  std::unordered_map<Str, Type> type_;
  std::unordered_map<Str, Id> computed_;
  std::unordered_map<Id, Reinterpreted> reinterpreted_;
  std::unordered_map<Id, Str> holder_;
  std::unordered_map<Str, Str> alias_;
  std::vector<Frame> frames_;
  std::vector<std::pair<Type, Str>> spelled_;
  PtrSet<Stmt *> dropped_;
  Id next_ = 0;
};

} // namespace detail

inline void reuseEqualValues(Block &body) {
  detail::ValueNumbering(body).run(body);
}

} // namespace agpu::msl

#endif // AGPU_MSL_VALUENUMBER_H
