#include "agpu/msl/Printer.h"

#include "agpu/msl/Builtins.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ios>

namespace agpu::msl {

const char *spell(Scalar s) {
  switch (s) {
  case Scalar::Bool:
    return "bool";
  case Scalar::I8:
    return "char";
  case Scalar::U8:
    return "uchar";
  case Scalar::I16:
    return "short";
  case Scalar::U16:
    return "ushort";
  case Scalar::I32:
    return "int";
  case Scalar::U32:
    return "uint";
  case Scalar::I64:
    return "long";
  case Scalar::U64:
    return "ulong";
  case Scalar::F16:
    return "half";
  case Scalar::BF16:
    return "bfloat";
  case Scalar::F32:
    return "float";
  }
  return "int";
}

// No trailing space; callers add one.
const char *spell(AddrSpace a) {
  switch (a) {
  case AddrSpace::None:
    return "";
  case AddrSpace::Device:
    return "device";
  case AddrSpace::Threadgroup:
    return "threadgroup";
  case AddrSpace::Thread:
    return "thread";
  case AddrSpace::Constant:
    return "constant";
  }
  return "";
}

const char *spell(BinOp op) {
  switch (op) {
  case BinOp::Add:
    return "+";
  case BinOp::Sub:
    return "-";
  case BinOp::Mul:
    return "*";
  case BinOp::Div:
    return "/";
  case BinOp::Rem:
    return "%";
  case BinOp::And:
    return "&";
  case BinOp::Or:
    return "|";
  case BinOp::Xor:
    return "^";
  case BinOp::Shl:
    return "<<";
  case BinOp::Shr:
    return ">>";
  case BinOp::Eq:
    return "==";
  case BinOp::Ne:
    return "!=";
  case BinOp::Lt:
    return "<";
  case BinOp::Le:
    return "<=";
  case BinOp::Gt:
    return ">";
  case BinOp::Ge:
    return ">=";
  case BinOp::LAnd:
    return "&&";
  case BinOp::LOr:
    return "||";
  }
  return "+";
}

// C precedence. Higher binds tighter.
int precedence(BinOp op) {
  switch (op) {
  case BinOp::Mul:
  case BinOp::Div:
  case BinOp::Rem:
    return 10;
  case BinOp::Add:
  case BinOp::Sub:
    return 9;
  case BinOp::Shl:
  case BinOp::Shr:
    return 8;
  case BinOp::Lt:
  case BinOp::Le:
  case BinOp::Gt:
  case BinOp::Ge:
    return 7;
  case BinOp::Eq:
  case BinOp::Ne:
    return 6;
  case BinOp::And:
    return 5;
  case BinOp::Xor:
    return 4;
  case BinOp::Or:
    return 3;
  case BinOp::LAnd:
    return 2;
  case BinOp::LOr:
    return 1;
  }
  return 0;
}

// Built with `pointerTo` because Metal rejects `device atomic_int` as an
// automatic variable.
Type atomicPtr(Scalar s, AddrSpace as) {
  return Type::named(std::string("atomic_") + spell(s)).pointerTo(as);
}

Type deviceAtomicPtr(Scalar s) { return atomicPtr(s, AddrSpace::Device); }

void Printer::printType(const Type &t) {
  // A non-pointer type may still carry an address space: `threadgroup float
  // pool[1024]`.
  if (t.form() != Type::Form::Pointer) {
    const char *as = spell(t.addrSpace());
    if (*as)
      os_ << as << " ";
  }
  switch (t.form()) {
  case Type::Form::Scalar:
    os_ << spell(t.scalarKind());
    return;
  case Type::Form::Vector:
    if (t.isPacked())
      os_ << "packed_";
    os_ << spell(t.scalarKind()) << t.lanes();
    return;
  case Type::Form::Named:
    os_ << t.namedText();
    return;
  case Type::Form::Pointer: {
    const char *as = spell(t.addrSpace());
    if (*as)
      os_ << as << " ";
    printType(t.pointee());
    os_ << " *";
    return;
  }
  }
}

void Printer::printLiteral(const Literal *l) {
  switch (l->form) {
  case Literal::Form::Bool:
    os_ << (l->intValue ? "true" : "false");
    return;
  case Literal::Form::Null:
    os_ << "nullptr";
    return;
  case Literal::Form::Int: {
    if (l->hex)
      os_ << "0x" << std::hex << l->intValue << std::dec;
    else
      os_ << l->intValue;
    // Needed or MSL narrows the comparison.
    if (l->type.form() == Type::Form::Scalar &&
        (l->type.scalarKind() == Scalar::U32 ||
         l->type.scalarKind() == Scalar::U64))
      os_ << "u";
    return;
  }
  case Literal::Form::Float: {
    const double v = l->floatValue;
    const bool isF32 = l->type.form() == Type::Form::Scalar &&
                       l->type.scalarKind() == Scalar::F32;

    // MSL has no suffix for half or bfloat and an unsuffixed literal is a
    // double that does not convert implicitly, so wrap it: `bfloat x = 0.0`
    // is an error.
    const bool narrow = l->type.form() == Type::Form::Scalar &&
                        (l->type.scalarKind() == Scalar::F16 ||
                         l->type.scalarKind() == Scalar::BF16);
    const auto wrapped = [&](const char *text) {
      if (narrow)
        os_ << spell(l->type.scalarKind()) << "(" << text << ")";
      else
        os_ << text;
    };

    // `%g` would spell inf/nan in a form that is not valid MSL.
    if (std::isnan(v)) {
      wrapped("NAN");
      return;
    }
    if (std::isinf(v)) {
      wrapped(v < 0 ? "(-INFINITY)" : "INFINITY");
      return;
    }

    // The shortest spelling that round-trips at the literal's own width.
    char buf[40];
    for (int prec = 6; prec <= 17; ++prec) {
      std::snprintf(buf, sizeof(buf), "%.*g", prec, v);
      const double back = std::strtod(buf, nullptr);
      if (isF32 ? (float)back == (float)v : back == v)
        break;
    }
    std::string s(buf);
    if (s.find('.') == std::string::npos && s.find('e') == std::string::npos)
      s += ".0";

    if (isF32)
      s += "f";
    wrapped(s.c_str());
    return;
  }
  }
}

void Printer::printExpr(const Expr *e) { printExprAt(e, 0); }

// `outerPrec` is the precedence of the surrounding context. Parens appear when
// the tree would otherwise re-associate.
void Printer::printExprAt(const Expr *e, int outerPrec) {
  if (!e) {
    os_ << "/*null*/";
    return;
  }
  switch (e->kind) {
  case ExprKind::VarRef:
    os_ << static_cast<const VarRef *>(e)->name;
    return;
  case ExprKind::Literal:
    printLiteral(static_cast<const Literal *>(e));
    return;
  case ExprKind::Binary: {
    auto *b = static_cast<const Binary *>(e);
    const int p = precedence(b->op);
    const bool paren = p < outerPrec;
    if (paren)
      os_ << "(";
    printExprAt(b->lhs, p);
    os_ << " " << spell(b->op) << " ";
    // +1 or `a - (b - c)` prints as `a - b - c`.
    printExprAt(b->rhs, p + 1);
    if (paren)
      os_ << ")";
    return;
  }
  case ExprKind::Cast: {
    auto *c = static_cast<const Cast *>(e);
    switch (c->style) {
    case Cast::Style::Value:
      os_ << "(";
      printType(c->to);
      os_ << ")";
      printExprAt(c->operand, 12);
      return;
    // Bracketed forms delimit the operand themselves, so no precedence guard.
    case Cast::Style::Static:
      os_ << "static_cast<";
      printType(c->to);
      os_ << ">(";
      printExpr(c->operand);
      os_ << ")";
      return;
    case Cast::Style::Bits:
      os_ << "as_type<";
      printType(c->to);
      os_ << ">(";
      printExpr(c->operand);
      os_ << ")";
      return;
    case Cast::Style::Functional:
      printType(c->to);
      os_ << "(";
      printExpr(c->operand);
      os_ << ")";
      return;
    }
    return;
  }
  case ExprKind::Subscript: {
    auto *s = static_cast<const Subscript *>(e);
    printExprAt(s->base, 12);
    os_ << "[";
    printExprAt(s->index, 0);
    os_ << "]";
    return;
  }
  case ExprKind::Member: {
    auto *m = static_cast<const Member *>(e);
    printExprAt(m->base, 12);
    os_ << "." << m->field;
    return;
  }
  case ExprKind::Deref: {
    auto *d = static_cast<const Deref *>(e);
    os_ << "*";
    printExprAt(d->operand, 12);
    return;
  }
  }
}

void Printer::indent() {
  if (skipIndent_) {
    skipIndent_ = false;
    return;
  }
  for (int i = 0; i < depth_; ++i)
    os_ << "  ";
}

void Printer::printBlock(const Block &b) {
  for (const Stmt *s : b)
    printStmt(s);
}

void Printer::printInline(const Stmt *s) {
  if (!s)
    return;
  const Indented flatten(*this, -depth_);
  printStmt(s);
}

// A statement inside a `for` header: no indent, no trailing `;\n`.
void Printer::printHeaderStmt(const Stmt *s) {
  if (!s)
    return;
  switch (s->kind) {
  case StmtKind::Decl: {
    auto *d = static_cast<const Decl *>(s);
    printType(d->type);
    os_ << " " << d->name;
    if (d->init) {
      os_ << " = ";
      printExpr(d->init);
    }
    return;
  }
  case StmtKind::Assign: {
    auto *a = static_cast<const Assign *>(s);
    printExpr(a->target);
    os_ << " " << (a->compound ? std::string(spell(a->compoundOp)) + "=" : "=")
        << " ";
    printExpr(a->value);
    return;
  }
  default:
    assert(false && "statement kind cannot be a loop header");
    return;
  }
}

bool isHeaderStmt(const Stmt *s) {
  if (!s)
    return true; // an absent clause is legal: `for (;;)`
  return s->kind == StmtKind::Decl || s->kind == StmtKind::Assign;
}

void Printer::printBraced(const Block &body) {
  os_ << "{\n";
  {
    const Indented in(*this);
    printBlock(body);
  }
  indent();
  os_ << "}";
}

static void printAttribute(std::ostream &os, const Attribute &a) {
  using K = Attribute::Kind;
  switch (a.kind()) {
  case K::None:
    return;
  case K::Buffer:
    os << "[[buffer(" << a.value() << ")]]";
    return;
  case K::ThreadgroupPositionInGrid:
    os << "[[threadgroup_position_in_grid]]";
    return;
  case K::ThreadPositionInThreadgroup:
    os << "[[thread_position_in_threadgroup]]";
    return;
  case K::ThreadgroupsPerGrid:
    os << "[[threadgroups_per_grid]]";
    return;
  }
}

void Printer::printParams(const Function *f) {
  printList(f->params, [&](const Function::Param &p, std::size_t) {
    printType(p.type);
    // A prototype may leave parameters unnamed.
    if (!p.name.empty())
      os_ << " " << p.name;
    if (p.attribute.present()) {
      os_ << " ";
      printAttribute(os_, p.attribute);
    }
  });
}

void Printer::printStmt(const Stmt *s) {
  if (!s)
    return;
  switch (s->kind) {
  case StmtKind::Decl:
  case StmtKind::Assign:
    indent();
    printHeaderStmt(s);
    os_ << ";\n";
    return;

  case StmtKind::If: {
    auto *n = static_cast<const If *>(s);
    indent();
    os_ << "if (";
    printExpr(n->cond);
    os_ << ")";
    const bool inlineable = n->thenBody.size() == 1 && !n->hasElse() &&
                            n->thenBody[0]->kind == StmtKind::Assign;
    if (inlineable) {
      os_ << " ";
      printInline(n->thenBody[0]);
      return;
    }
    os_ << " ";
    printBraced(n->thenBody);
    if (n->hasElse()) {
      os_ << " else ";
      if (n->elseBody.size() == 1 && n->elseBody[0]->kind == StmtKind::If) {
        skipIndent_ = true;
        printStmt(n->elseBody[0]);
        return;
      }
      printBraced(n->elseBody);
    }
    os_ << "\n";
    return;
  }
  case StmtKind::Function: {
    auto *f = static_cast<const Function *>(s);
    // On its own line before the declaration; after the parameter list it
    // does not parse.
    if (f->qualifier.present()) {
      indent();
      printAttribute(os_, f->qualifier);
      os_ << "\n";
    }
    if (!f->templateParams.empty() || f->isSpecialization) {
      indent();
      os_ << "template <";
      printList(f->templateParams,
                [&](const Str &p, std::size_t) { os_ << "typename " << p; });
      os_ << ">\n";
    }
    indent();
    if (f->isKernel)
      os_ << "kernel ";
    if (f->isInline)
      os_ << "inline ";
    printType(f->returnType);
    os_ << " " << f->name;
    if (!f->templateArgs.empty()) {
      os_ << "<";
      printList(f->templateArgs, [&](const Str &a, std::size_t) { os_ << a; });
      os_ << ">";
    }
    os_ << "(";
    printParams(f);
    os_ << ")";
    if (f->isPrototype) {
      os_ << ";\n";
      return;
    }
    os_ << " ";
    printBraced(f->body);
    os_ << "\n";
    return;
  }
  }
}

} // namespace agpu::msl
