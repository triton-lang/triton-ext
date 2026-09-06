// Ast.h - the MSL syntax tree. Kinds come from Nodes.def.
#ifndef AGPU_MSL_AST_H
#define AGPU_MSL_AST_H

#include "Containers.h"

#include <algorithm>
#include <cstdint>
#include <memory>

namespace agpu::msl {

// ── kinds, from the one table ─────────────────────────────────────────────

enum class ExprKind {
#define EXPR_NODE(Name, Class) Name,
#include "Nodes.def"
};

enum class StmtKind {
#define STMT_NODE(Name, Class) Name,
#include "Nodes.def"
};

const char *name(ExprKind k);
const char *name(StmtKind k);

// ── types ─────────────────────────────────────────────────────────────────

enum class Scalar {
  Bool,
  I8,
  U8,
  I16,
  U16,
  I32,
  U32,
  I64,
  U64,
  F16,
  BF16,
  F32
};
enum class AddrSpace { None, Device, Threadgroup, Thread, Constant };

// A type is a scalar, a vector of one, a pointer to one, or a named opaque
// (simdgroup_half8x8 and friends).
class Type {
public:
  // Matrix is separate from Named so the size analysis does not conflate
  // simdgroup fragments with other opaque types like `atomic_uint`.
  enum class Form { Scalar, Vector, Pointer, Named };

  static Type scalar(Scalar s) {
    return Type(Form::Scalar, s, 0, {}, AddrSpace::None);
  }
  static Type vector(Scalar s, int n) {
    return Type(Form::Vector, s, n, {}, AddrSpace::None);
  }

  bool isPacked() const { return packed_; }
  static Type named(Str n) {
    return Type(Form::Named, Scalar::I32, 0, std::move(n), AddrSpace::None);
  }
  Type pointerTo(AddrSpace as) const {
    Type t = *this;
    t.pointee_ = std::make_shared<Type>(*this);
    t.form_ = Form::Pointer;
    t.addrSpace_ = as;
    return t;
  }

  Form form() const { return form_; }
  Scalar scalarKind() const { return scalar_; }
  int lanes() const { return lanes_; }
  const Str &namedText() const { return named_; }
  AddrSpace addrSpace() const { return addrSpace_; }
  const Type &pointee() const { return *pointee_; }

  bool operator==(const Type &o) const {
    if (form_ != o.form_ || addrSpace_ != o.addrSpace_)
      return false;
    switch (form_) {
    case Form::Scalar:
      return scalar_ == o.scalar_;
    case Form::Vector:
      return scalar_ == o.scalar_ && lanes_ == o.lanes_ && packed_ == o.packed_;
    case Form::Named:
      return named_ == o.named_;
    case Form::Pointer:
      return *pointee_ == *o.pointee_;
    }
    return false;
  }

private:
  Type(Form f, Scalar s, int n, Str nm, AddrSpace as)
      : form_(f), scalar_(s), lanes_(n), named_(std::move(nm)), addrSpace_(as) {
  }

  Form form_ = Form::Scalar;
  Scalar scalar_ = Scalar::I32;
  int lanes_ = 0;
  Str named_;
  AddrSpace addrSpace_ = AddrSpace::None;
  bool packed_ = false;
  std::shared_ptr<Type> pointee_;
};

// ── operators ─────────────────────────────────────────────────────────────

enum class BinOp {
  Add,
  Sub,
  Mul,
  Div,
  Rem,
  And,
  Or,
  Xor,
  Shl,
  Shr,
  Eq,
  Ne,
  Lt,
  Le,
  Gt,
  Ge,
  LAnd,
  LOr,
};

enum class UnOp { Neg, Not, LNot, PreInc };

// ── expressions ───────────────────────────────────────────────────────────

struct Expr {
  const ExprKind kind;
  explicit Expr(ExprKind k) : kind(k) {}
};

struct VarRef : Expr {
  Str name;
  explicit VarRef(Str n) : Expr(ExprKind::VarRef), name(std::move(n)) {}
};

struct Literal : Expr {
  // Null is its own form: MSL does not accept `(T*)0` everywhere it wants a
  // pointer constant.
  enum class Form { Int, Float, Bool, Null };
  Form form = Form::Int;
  int64_t intValue = 0;
  double floatValue = 0.0;
  // Spelling only: a bit mask reads as hex and compares equal to its decimal.
  bool hex = false;
  Type type = Type::scalar(Scalar::I32);

  static Literal *makeInt(int64_t v, Type t);
  static Literal *makeFloat(double v, Type t);
  static Literal *makeBool(bool v);

  Literal() : Expr(ExprKind::Literal) {}
};

struct Binary : Expr {
  BinOp op;
  Expr *lhs;
  Expr *rhs;
  Binary(BinOp o, Expr *l, Expr *r)
      : Expr(ExprKind::Binary), op(o), lhs(l), rhs(r) {}
};

struct Cast : Expr {
  // `Functional` is the value conversion spelled `T(v)`, for a named type
  // that cannot take the parenthesised form.
  enum class Style {
    Value,      // (T)v
    Static,     // static_cast<T>(v)
    Bits,       // as_type<T>(v): reinterpretation
    Functional, // T(v)
  };
  Type to;
  Expr *operand;
  Style style = Style::Value;
  Cast(Type t, Expr *e, Style s = Style::Value)
      : Expr(ExprKind::Cast), to(std::move(t)), operand(e), style(s) {}
};

struct Subscript : Expr {
  Expr *base;
  Expr *index;
  Subscript(Expr *b, Expr *i) : Expr(ExprKind::Subscript), base(b), index(i) {}
};

struct Member : Expr {
  Expr *base;
  Str field;
  Member(Expr *b, Str f)
      : Expr(ExprKind::Member), base(b), field(std::move(f)) {}
};

struct Deref : Expr {
  Expr *operand;
  explicit Deref(Expr *e) : Expr(ExprKind::Deref), operand(e) {}
};

struct Stmt {
  const StmtKind kind;
  explicit Stmt(StmtKind k) : kind(k) {}
};

using Block = SmallVec<Stmt *, 8>;

struct Decl : Stmt {
  Type type;
  Str name;
  Expr *init = nullptr;
  Decl(Type t, Str n, Expr *i)
      : Stmt(StmtKind::Decl), type(std::move(t)), name(std::move(n)), init(i) {}
};

struct Assign : Stmt {
  Expr *target;
  Expr *value;
  bool compound = false;
  BinOp compoundOp = BinOp::Add;
  Assign(Expr *t, Expr *v) : Stmt(StmtKind::Assign), target(t), value(v) {}
};

struct If : Stmt {
  Expr *cond;
  Block thenBody;
  Block elseBody;
  If(Expr *c, Block t) : Stmt(StmtKind::If), cond(c), thenBody(std::move(t)) {}
  bool hasElse() const { return !elseBody.empty(); }
};

class Attribute {
public:
  enum class Kind {
    None,
    Buffer, // takes an index
    ThreadgroupPositionInGrid,
    ThreadPositionInThreadgroup,
    ThreadgroupsPerGrid,
  };

  Attribute() = default;
  static Attribute buffer(int64_t index) {
    return Attribute(Kind::Buffer, index);
  }
  static Attribute builtin(Kind k) { return Attribute(k, 0); }

  Kind kind() const { return kind_; }
  int64_t value() const { return value_; }
  bool present() const { return kind_ != Kind::None; }

  bool operator==(const Attribute &o) const {
    return kind_ == o.kind_ && value_ == o.value_;
  }

private:
  Attribute(Kind k, int64_t v) : kind_(k), value_(v) {}
  Kind kind_ = Kind::None;
  int64_t value_ = 0;
};

struct Function : Stmt {
  struct Param {
    Type type;
    Str name;
    Attribute attribute;
  };
  bool isKernel = false;
  bool isInline = false;
  // Signature terminated by `;`, no body. Distinct from an empty body, which
  // is a legal function that does nothing.
  bool isPrototype = false;
  // Type parameters, printed as `template <typename A, typename B>`. Empty
  // with `isSpecialization` set prints the explicit form, `template <>`.
  SmallVec<Str, 2> templateParams;
  bool isSpecialization = false;
  // Arguments on the function's own name, as in `f<half>(...)`.
  SmallVec<Str, 2> templateArgs;
  Type returnType = Type::named("void");
  Str name;
  SmallVec<Param, 8> params;
  Block body;
  // Printed before the function, in brackets.
  Attribute qualifier;
  Function() : Stmt(StmtKind::Function) {}
};

} // namespace agpu::msl

#endif // AGPU_MSL_AST_H
