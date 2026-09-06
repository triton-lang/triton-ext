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
  enum class Form { Scalar, Vector, Pointer, Named, Matrix };

  static Type scalar(Scalar s) {
    return Type(Form::Scalar, s, 0, {}, AddrSpace::None);
  }
  static Type vector(Scalar s, int n) {
    return Type(Form::Vector, s, n, {}, AddrSpace::None);
  }

  // `packed_floatN`, which aligns only to the element.
  static Type packedVector(Scalar s, int n) {
    Type t(Form::Vector, s, n, {}, AddrSpace::None);
    t.packed_ = true;
    return t;
  }
  bool isPacked() const { return packed_; }
  static Type named(Str n) {
    return Type(Form::Named, Scalar::I32, 0, std::move(n), AddrSpace::None);
  }
  static Type matrix(Str n) {
    return Type(Form::Matrix, Scalar::I32, 0, std::move(n), AddrSpace::None);
  }
  bool isMatrix() const { return form_ == Form::Matrix; }
  // MSL accepts combinations of these on one pointer.
  //   Coherent   bypasses the cache, which can serve a stale line after
  //              another threadgroup's store.
  //   Volatile   forbids eliding or reordering the access.
  //   Const      the pointee is not written through this pointer.
  enum Qualifier : unsigned {
    QualNone = 0,
    Coherent = 1u << 0,
    Volatile = 1u << 1,
    Const = 1u << 2,
  };

  Type pointerTo(AddrSpace as, unsigned quals = QualNone) const {
    Type t = *this;
    t.pointee_ = std::make_shared<Type>(*this);
    t.form_ = Form::Pointer;
    t.addrSpace_ = as;
    t.quals_ = quals;
    return t;
  }
  unsigned quals() const { return quals_; }
  bool hasQual(Qualifier q) const { return (quals_ & q) != 0; }
  bool isCoherent() const { return hasQual(Coherent); }

  // The address space of storage: `threadgroup float pool[1024];` is not a
  // pointer.
  Type inAddrSpace(AddrSpace as) const {
    Type t = *this;
    t.addrSpace_ = as;
    return t;
  }

  Type withQual(Qualifier q) const {
    Type t = *this;
    t.quals_ |= q;
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
    case Form::Matrix:
      return named_ == o.named_;
    case Form::Pointer:
      return *pointee_ == *o.pointee_ && quals_ == o.quals_;
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
  unsigned quals_ = QualNone;
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

  bool sameValueAs(const Literal &o) const {
    if (form != o.form)
      return false;
    switch (form) {
    case Form::Int:
      return intValue == o.intValue;
    case Form::Bool:
      return intValue == o.intValue;
    case Form::Float:
      return floatValue == o.floatValue;
    case Form::Null:
      return true;
    }
    return false;
  }
};

struct Unary : Expr {
  UnOp op;
  Expr *operand;
  Unary(UnOp o, Expr *e) : Expr(ExprKind::Unary), op(o), operand(e) {}
};

struct Binary : Expr {
  BinOp op;
  Expr *lhs;
  Expr *rhs;
  Binary(BinOp o, Expr *l, Expr *r)
      : Expr(ExprKind::Binary), op(o), lhs(l), rhs(r) {}
};

struct Ternary : Expr {
  Expr *cond;
  Expr *whenTrue;
  Expr *whenFalse;
  Ternary(Expr *c, Expr *t, Expr *f)
      : Expr(ExprKind::Ternary), cond(c), whenTrue(t), whenFalse(f) {}
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

struct Call : Expr {
  Str callee;
  SmallVec<Str, 2> templateArgs;
  SmallVec<Expr *, 4> args;
  Call(Str c, SmallVec<Expr *, 4> a)
      : Expr(ExprKind::Call), callee(std::move(c)), args(std::move(a)) {}
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

struct AddrOf : Expr {
  Expr *operand;
  explicit AddrOf(Expr *e) : Expr(ExprKind::AddrOf), operand(e) {}
};

// ── statements ────────────────────────────────────────────────────────────

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

struct ArrayDecl : Stmt {
  Type elem;
  Str name;
  int64_t count = 0;
  SmallVec<Expr *, 4> init;
  ArrayDecl(Type e, Str n, int64_t c)
      : Stmt(StmtKind::ArrayDecl), elem(std::move(e)), name(std::move(n)),
        count(c) {}
};

struct Assign : Stmt {
  Expr *target;
  Expr *value;
  bool compound = false;
  BinOp compoundOp = BinOp::Add;
  Assign(Expr *t, Expr *v) : Stmt(StmtKind::Assign), target(t), value(v) {}
};

struct ExprStmt : Stmt {
  Expr *expr;
  explicit ExprStmt(Expr *e) : Stmt(StmtKind::ExprStmt), expr(e) {}
};

struct Return : Stmt {
  Expr *value = nullptr;
  SmallVec<Expr *, 4> structFields;
  Return() : Stmt(StmtKind::Return) {}
};

struct Break : Stmt {
  Break() : Stmt(StmtKind::Break) {}
};
struct Continue : Stmt {
  Continue() : Stmt(StmtKind::Continue) {}
};

struct Barrier : Stmt {
  enum class Scope { Threadgroup, Device, Simdgroup };
  Scope scope = Scope::Threadgroup;

  // Must not merge with an adjacent barrier. A double-buffered pipeline puts
  // one before a buffer swap and one after; merging them lets a thread read a
  // buffer another is still writing.
  bool hard = false;

  explicit Barrier(Scope s) : Stmt(StmtKind::Barrier), scope(s) {}

  // Enum declaration order does not match breadth, so do not compare Scope
  // values directly.
  static int breadth(Scope s) {
    switch (s) {
    case Scope::Simdgroup:
      return 0;
    case Scope::Threadgroup:
      return 1;
    case Scope::Device:
      return 2;
    }
    return 0;
  }

  // Merging must never narrow the scope.
  static Scope widest(Scope a, Scope b) {
    return std::max(a, b,
                    [](Scope x, Scope y) { return breadth(x) < breadth(y); });
  }
};

// The printer decides inline vs braced from the body.
struct If : Stmt {
  Expr *cond;
  Block thenBody;
  Block elseBody;
  If(Expr *c, Block t) : Stmt(StmtKind::If), cond(c), thenBody(std::move(t)) {}
  bool hasElse() const { return !elseBody.empty(); }
};

inline constexpr int64_t kUnrollCount = 8;

struct For : Stmt {
  Stmt *init;
  Expr *cond;
  Stmt *step;
  Block body;
  int64_t unrollCount = 0;
  For(Stmt *i, Expr *c, Stmt *s, Block b)
      : Stmt(StmtKind::For), init(i), cond(c), step(s), body(std::move(b)) {}
};

struct While : Stmt {
  Expr *cond;
  Block body;
  While(Expr *c, Block b)
      : Stmt(StmtKind::While), cond(c), body(std::move(b)) {}
};

struct Scope : Stmt {
  Block body;
  explicit Scope(Block b) : Stmt(StmtKind::Scope), body(std::move(b)) {}
};

// An unstructured control-flow region, lowered to a dispatch loop: each block
// is a numbered state and a block's terminator assigns the next state rather
// than jumping.
struct StateMachine : Stmt {
  struct Case {
    int64_t value = 0;
    Block body;
  };
  Str stateVar = "__state";
  Type stateType = Type::scalar(Scalar::I32);
  int64_t entry = 0;
  SmallVec<Case, 4> cases;

  // Distinct from every block number.
  int64_t exitState = -1;

  StateMachine() : Stmt(StmtKind::StateMachine) {}
};

// An MSL attribute: `[[buffer(0)]]`, `[[thread_position_in_threadgroup]]`,
// `[[max_total_threads_per_threadgroup(256)]]`. A kind plus its value.
class Attribute {
public:
  enum class Kind {
    None,
    Buffer,                        // takes an index
    MaxTotalThreadsPerThreadgroup, // takes a thread count
    ThreadgroupPositionInGrid,
    ThreadPositionInThreadgroup,
    ThreadgroupsPerGrid,
    ThreadIndexInThreadgroup,
  };

  Attribute() = default;
  static Attribute buffer(int64_t index) {
    return Attribute(Kind::Buffer, index);
  }
  static Attribute maxThreads(int64_t n) {
    return Attribute(Kind::MaxTotalThreadsPerThreadgroup, n);
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

struct StructDecl : Stmt {
  Str name;
  SmallVec<std::pair<Type, Str>, 4> fields;
  StructDecl() : Stmt(StmtKind::StructDecl) {}
};

} // namespace agpu::msl

#endif // AGPU_MSL_AST_H
