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

const char *spell(UnOp op) {
  switch (op) {
  case UnOp::Neg:
    return "-";
  case UnOp::Not:
    return "~";
  case UnOp::LNot:
    return "!";
  case UnOp::PreInc:
    return "++";
  }
  return "-";
}

// Whether an expression renders starting with `-`.
static bool leadsWithMinus(const Expr *e) {
  if (!e)
    return false;
  if (e->kind == ExprKind::Literal) {
    auto *l = static_cast<const Literal *>(e);
    switch (l->form) {
    case Literal::Form::Int:
      return l->intValue < 0;
    case Literal::Form::Float:
      return l->floatValue < 0;
    case Literal::Form::Bool:
      return false;
    case Literal::Form::Null:
      return false;
    }
    return false;
  }
  if (e->kind == ExprKind::Unary)
    return static_cast<const Unary *>(e)->op == UnOp::Neg;
  return false;
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
    if (t.hasQual(Type::Const))
      os_ << "const ";
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
  case Type::Form::Matrix:
    os_ << t.namedText();
    return;
  case Type::Form::Pointer: {
    // Order fixed by the grammar: `volatile device coherent(device) const
    // float *`.
    if (t.hasQual(Type::Volatile))
      os_ << "volatile ";
    const char *as = spell(t.addrSpace());
    if (*as)
      os_ << as << " ";
    if (t.hasQual(Type::Coherent))
      os_ << "coherent(" << as << ") ";
    if (t.hasQual(Type::Const))
      os_ << "const ";
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
  case ExprKind::Unary: {
    auto *u = static_cast<const Unary *>(e);
    // `--x` would lex as a predecrement.
    os_ << spell(u->op);
    if (u->op == UnOp::Neg && leadsWithMinus(u->operand)) {
      os_ << '(';
      printExprAt(u->operand, 0);
      os_ << ')';
      return;
    }
    printExprAt(u->operand, 12);
    return;
  }
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
  case ExprKind::Ternary: {
    auto *t = static_cast<const Ternary *>(e);
    const bool paren = outerPrec > 0;
    if (paren)
      os_ << "(";
    printExprAt(t->cond, 2);
    os_ << " ? ";
    printExprAt(t->whenTrue, 0);
    os_ << " : ";
    printExprAt(t->whenFalse, 0);
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
  case ExprKind::Call: {
    auto *c = static_cast<const Call *>(e);
    os_ << c->callee;
    if (!c->templateArgs.empty())
      printWrapped(
          "<",
          [&] {
            printList(c->templateArgs,
                      [&](const Str &a, std::size_t) { os_ << a; });
          },
          ">");
    printWrapped(
        "(",
        [&] {
          printList(c->args,
                    [&](const Expr *a, std::size_t) { printExprAt(a, 0); });
        },
        ")");
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
  case ExprKind::AddrOf: {
    auto *a = static_cast<const AddrOf *>(e);
    os_ << "&";
    printExprAt(a->operand, 12);
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

namespace {
struct BarrierForm {
  const char *fn;
  const char *flags;
};

BarrierForm barrierForm(Barrier::Scope s) {
  namespace b = builtin::barrier;
  namespace mf = builtin::memflags;
  using S = Barrier::Scope;
  switch (s) {
  case S::Simdgroup:
    return {b::Simdgroup, mf::Threadgroup};
  case S::Device:
    return {b::Threadgroup, mf::DeviceAndThreadgroup};
  case S::Threadgroup:
    return {b::Threadgroup, mf::Threadgroup};
  }
  return {b::Threadgroup, mf::Threadgroup};
}
} // namespace

void Printer::flushBarrier() {
  if (!barrierPending_)
    return;
  barrierPending_ = false;
  const BarrierForm f = barrierForm(pendingScope_);
  indent();
  os_ << f.fn << "(" << f.flags << ");\n";
}

void Printer::printBlock(const Block &b) {
  for (const Stmt *s : b) {
    if (s && s->kind == StmtKind::Barrier) {
      // Adjacent barriers collapse to the widest scope requested. A hard
      // barrier neither absorbs a pending barrier nor is absorbed by one.
      auto *bar = static_cast<const Barrier *>(s);
      if (bar->hard) {
        flushBarrier();
        indent();
        const BarrierForm f = barrierForm(bar->scope);
        os_ << f.fn << "(" << f.flags << ");\n";
        continue;
      }
      if (barrierPending_) {
        pendingScope_ = Barrier::widest(pendingScope_, bar->scope);
      } else {
        barrierPending_ = true;
        pendingScope_ = bar->scope;
      }
      continue;
    }
    flushBarrier();
    printStmt(s);
  }
  flushBarrier();
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
  case StmtKind::ExprStmt:
    printExpr(static_cast<const ExprStmt *>(s)->expr);
    return;
  default:
    assert(false && "statement kind cannot be a loop header");
    return;
  }
}

bool isHeaderStmt(const Stmt *s) {
  if (!s)
    return true; // an absent clause is legal: `for (;;)`
  return s->kind == StmtKind::Decl || s->kind == StmtKind::Assign ||
         s->kind == StmtKind::ExprStmt;
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

void Printer::printBracedWithBreak(const Block &body) {
  os_ << "{\n";
  {
    const Indented in(*this);
    printBlock(body);
    // Flush first, or a pending barrier lands after the break.
    flushBarrier();
    // `break` after a statement that already leaves is unreachable.
    const bool alreadyLeaves =
        !body.empty() && (body.back()->kind == StmtKind::Continue ||
                          body.back()->kind == StmtKind::Break ||
                          body.back()->kind == StmtKind::Return);
    if (!alreadyLeaves) {
      indent();
      os_ << "break;\n";
    }
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
  case K::MaxTotalThreadsPerThreadgroup:
    os << "[[max_total_threads_per_threadgroup(" << a.value() << ")]]";
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
  case K::ThreadIndexInThreadgroup:
    os << "[[thread_index_in_threadgroup]]";
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
  case StmtKind::ExprStmt:
    indent();
    printHeaderStmt(s);
    os_ << ";\n";
    return;

  case StmtKind::ArrayDecl: {
    auto *d = static_cast<const ArrayDecl *>(s);
    indent();
    printType(d->elem);
    os_ << " " << d->name << "[" << d->count << "]";
    if (!d->init.empty()) {
      printWrapped(
          " = {",
          [&] {
            printList(d->init,
                      [&](const Expr *e, std::size_t) { printExpr(e); });
          },
          "}");
    }
    os_ << ";\n";
    return;
  }
  case StmtKind::Return: {
    auto *r = static_cast<const Return *>(s);
    indent();
    os_ << "return";
    if (!r->structFields.empty()) {
      os_ << " {";
      for (std::size_t i = 0; i < r->structFields.size(); ++i) {
        if (i)
          os_ << ", ";
        printExpr(r->structFields[i]);
      }
      os_ << "}";
    } else if (r->value) {
      os_ << " ";
      printExpr(r->value);
    }
    os_ << ";\n";
    return;
  }
  case StmtKind::Break:
    indent();
    os_ << "break;\n";
    return;
  case StmtKind::Continue:
    indent();
    os_ << "continue;\n";
    return;
  case StmtKind::Barrier:
    // Only reached when a barrier is printed outside printBlock.
    barrierPending_ = true;
    pendingScope_ = static_cast<const Barrier *>(s)->scope;
    flushBarrier();
    return;
  case StmtKind::If: {
    auto *n = static_cast<const If *>(s);
    indent();
    os_ << "if (";
    printExpr(n->cond);
    os_ << ")";
    const bool inlineable = n->thenBody.size() == 1 && !n->hasElse() &&
                            (n->thenBody[0]->kind == StmtKind::Assign ||
                             n->thenBody[0]->kind == StmtKind::ExprStmt ||
                             n->thenBody[0]->kind == StmtKind::Break ||
                             n->thenBody[0]->kind == StmtKind::Continue);
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
  case StmtKind::For: {
    auto *f = static_cast<const For *>(s);
    if (f->unrollCount > 1) {
      indent();
      os_ << "#pragma clang loop unroll_count(" << f->unrollCount << ")\n";
    }
    indent();
    printWrapped(
        "for (",
        [&] {
          printHeaderStmt(f->init);
          os_ << "; ";
          // An absent condition is legal: the exit test may live in
          // the body.
          if (f->cond)
            printExpr(f->cond);
          os_ << "; ";
          printHeaderStmt(f->step);
        },
        ") ");
    printBraced(f->body);
    os_ << "\n";
    return;
  }
  case StmtKind::While: {
    auto *w = static_cast<const While *>(s);
    indent();
    os_ << "while (";
    printExpr(w->cond);
    os_ << ") ";
    printBraced(w->body);
    os_ << "\n";
    return;
  }
  case StmtKind::Scope: {
    auto *sc = static_cast<const Scope *>(s);
    indent();
    printBraced(sc->body);
    os_ << "\n";
    return;
  }
  case StmtKind::StateMachine: {
    auto *m = static_cast<const StateMachine *>(s);
    indent();
    printType(m->stateType);
    os_ << " " << m->stateVar << " = " << m->entry << ";\n";

    indent();
    os_ << "while (" << m->stateVar << " != " << m->exitState << ") {\n";
    {
      const Indented in(*this);
      indent();
      os_ << "switch (" << m->stateVar << ") {\n";
      for (const auto &c : m->cases) {
        indent();
        os_ << "case " << c.value << ": ";
        printBracedWithBreak(c.body);
        os_ << "\n";
      }
      // A state with no case would spin forever.
      indent();
      os_ << "default: " << m->stateVar << " = " << m->exitState
          << "; break;\n";
      indent();
      os_ << "}\n";
    }
    indent();
    os_ << "}\n";
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
  case StmtKind::StructDecl: {
    auto *d = static_cast<const StructDecl *>(s);
    indent();
    os_ << "struct " << d->name << " {\n";
    {
      const Indented in(*this);
      for (const auto &f : d->fields) {
        indent();
        printType(f.first);
        os_ << " " << f.second << ";\n";
      }
    }
    indent();
    os_ << "};\n";
    return;
  }
  }
}

} // namespace agpu::msl
