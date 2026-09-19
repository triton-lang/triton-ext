// Printer.h - AST to MSL text. Parens come from precedence. Adjacent soft
// barriers collapse.
#ifndef AGPU_MSL_PRINTER_H
#define AGPU_MSL_PRINTER_H

#include "Ast.h"

#include <ostream>
#include <string>

namespace agpu::msl {

class Printer {
public:
  explicit Printer(std::ostream &os) : os_(os) {}

  void printBlock(const Block &b);
  void printStmt(const Stmt *s);
  void printExpr(const Expr *e);
  void printType(const Type &t);

  void printInline(const Stmt *s);

  // A statement inside a `for` header: no indent, no trailing semicolon.
  void printHeaderStmt(const Stmt *s);

  void printBraced(const Block &body);
  void printBracedWithBreak(const Block &body);

private:
  class Indented {
  public:
    explicit Indented(Printer &p, int by = 1) : p_(p), by_(by) {
      p_.depth_ += by_;
    }
    ~Indented() { p_.depth_ -= by_; }
    Indented(const Indented &) = delete;
    Indented &operator=(const Indented &) = delete;

  private:
    Printer &p_;
    int by_;
  };

  void indent();
  void printExprAt(const Expr *e, int outerPrec);
  void printLiteral(const Literal *l);
  void printParams(const Function *f);

  template <class Item, class PrintOne>
  void printCallLike(const char *callee, const Item &items, PrintOne one) {
    os_ << callee << "(";
    printList(items, one);
    os_ << ")";
  }

  template <class Items, class PrintOne>
  void printList(const Items &items, PrintOne one) {
    for (std::size_t i = 0; i < items.size(); ++i) {
      if (i)
        os_ << ", ";
      one(items[i], i);
    }
  }

  template <class PrintInner>
  void printWrapped(const char *open, PrintInner inner, const char *close) {
    os_ << open;
    inner();
    os_ << close;
  }

  std::ostream &os_;
  int depth_ = 0;
  // Cleared by the next indent(), so an `else if` stays on one line.
  bool skipIndent_ = false;
  bool barrierPending_ = false;
  Barrier::Scope pendingScope_ = Barrier::Scope::Threadgroup;

  void flushBarrier();
};

// A null clause is legal: `for (;;)`.
bool isHeaderStmt(const Stmt *s);

const char *spell(Scalar s);
const char *spell(AddrSpace a);
const char *spell(BinOp op);
const char *spell(UnOp op);

Type atomicPtr(Scalar s, AddrSpace as);
Type deviceAtomicPtr(Scalar s);

int precedence(BinOp op);

} // namespace agpu::msl

#endif // AGPU_MSL_PRINTER_H
